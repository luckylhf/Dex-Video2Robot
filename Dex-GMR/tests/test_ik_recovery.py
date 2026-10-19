"""Exercise normal, fallback and rollback paths using real MuJoCo integration."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mink
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from general_motion_retargeting import GeneralMotionRetargeting


class IkRecoveryTest(unittest.TestCase):
    def robot(self, mode='table-retry'):
        obj = GeneralMotionRetargeting('smplx', 'dex_evt', use_velocity_limit=True,
                                       ik_recovery=mode)
        obj.tasks1, obj.tasks2 = ['table1'], ['table2']
        obj.shoulder_posture_task = None
        obj.scaled_human_data = {}
        obj.update_targets = lambda *a: None
        return obj

    def solve_plan(self, obj, plan):
        outcomes = iter(plan)
        def solve(*args, **kwargs):
            value = next(outcomes)
            if isinstance(value, Exception):
                raise value
            velocity = np.zeros(obj.model.nv)
            velocity[0] = value
            return velocity
        return solve

    def test_each_successful_retry_is_applied_and_other_table_is_still_attempted(self):
        fail = lambda: mink.NoSolutionFound('daqp')
        cases = [
            ([1, 10], 11, 'normal', ['normal', 'normal']),
            ([fail(), 2, 10], 12, 'recovered', ['recovered', 'normal']),
            ([1, fail(), 20], 21, 'recovered', ['normal', 'recovered']),
            ([fail(), 2, fail(), 20], 22, 'recovered', ['recovered', 'recovered']),
            ([fail(), fail(), 10], 10, 'partial', ['failed', 'normal']),
            ([1, fail(), fail()], 1, 'partial', ['normal', 'failed']),
            ([fail(), fail(), fail(), fail()], 0, 'held', ['failed', 'failed']),
        ]
        for plan, displacement, status, stages in cases:
            with self.subTest(status=status, stages=stages):
                obj = self.robot()
                original_limits = list(obj.ik_limits)
                before = obj.configuration.data.qpos.copy()
                with patch.object(mink, 'solve_ik', side_effect=self.solve_plan(obj, plan)) as solve:
                    obj.retarget({}, framesIdx=109)
                self.assertEqual(solve.call_count, len(plan))
                self.assertAlmostEqual(obj.configuration.data.qpos[0] - before[0],
                                       displacement * obj.retarget_dt / 2)
                self.assertEqual(obj.last_ik_result['status'], status)
                self.assertEqual([s['status'] for s in obj.last_ik_result['stages']], stages)
                self.assertEqual(obj.ik_limits, original_limits)
                np.testing.assert_allclose(obj.configuration.data.xpos[obj.model.body('pelvis').id],
                                           obj.configuration.data.qpos[:3])
                for call in solve.call_args_list:
                    if call.args[4] == obj.damping * 50:
                        self.assertTrue(all(isinstance(x, mink.ConfigurationLimit)
                                            for x in call.kwargs['limits']))

    def test_strict_mode_does_not_retry_and_rolls_back_partial_frame(self):
        obj = self.robot('strict')
        before = obj.configuration.data.qpos.copy()
        with patch.object(mink, 'solve_ik', side_effect=self.solve_plan(
                obj, [1, mink.NoSolutionFound('daqp')])) as solve:
            with self.assertRaises(mink.NoSolutionFound):
                obj.retarget({}, framesIdx=109)
        self.assertEqual(solve.call_count, 2)
        np.testing.assert_array_equal(obj.configuration.data.qpos, before)
        self.assertEqual(obj.ik_recovery_events, [])

    def test_programming_errors_and_nonfinite_velocities_are_not_retried(self):
        for outcome, error in [(RuntimeError('unexpected backend error'), RuntimeError),
                               (float('nan'), ValueError)]:
            with self.subTest(error=error):
                obj = self.robot()
                before = obj.configuration.data.qpos.copy()
                with patch.object(mink, 'solve_ik', side_effect=self.solve_plan(obj, [outcome])) as solve:
                    with self.assertRaises(error):
                        obj.retarget({}, framesIdx=109)
                self.assertEqual(solve.call_count, 1)
                np.testing.assert_array_equal(obj.configuration.data.qpos, before)

    def test_held_frame_restores_contacts_and_reset_clears_audit(self):
        obj = self.robot()
        contact = SimpleNamespace(_in_stance=True, _contact_xy=np.array([1., 2.]),
                                  reset=lambda: None)
        obj.foot_contact_tasks = [contact]
        def change_contact(_):
            contact._in_stance = False
            contact._contact_xy = None
        obj._update_foot_contact_state = change_contact
        with patch.object(mink, 'solve_ik', side_effect=mink.NoSolutionFound('daqp')):
            obj.retarget({}, framesIdx=109)
        self.assertTrue(contact._in_stance)
        np.testing.assert_array_equal(contact._contact_xy, [1., 2.])
        self.assertEqual(obj.ik_status_counts, {'held': 1})
        obj.reset()
        self.assertEqual(obj.ik_status_counts, {})
        self.assertEqual(obj.ik_recovery_events, [])
        self.assertIsNone(obj.last_ik_result)

    def test_invalid_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown IK recovery mode'):
            GeneralMotionRetargeting('smplx', 'dex_evt', ik_recovery='typo')


if __name__ == '__main__':
    unittest.main()
