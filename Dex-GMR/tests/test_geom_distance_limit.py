"""Collision normals and certified handling of contradictory CCD results."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from general_motion_retargeting.geom_distance_limit import GeomDistanceLimit


class GeomDistanceTest(unittest.TestCase):
    def scene(self, center_distance):
        model = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
          <geom name="a" type="sphere" size=".1"/>
          <body pos="{center_distance} 0 0"><freejoint/>
            <geom name="b" type="sphere" size=".1"/>
          </body></worldbody></mujoco>''')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        limit = GeomDistanceLimit(model, [(['a'], ['b'])],
                                  minimum_distance_from_collisions=.05,
                                  collision_detection_distance=.5)
        return model, data, limit

    def test_false_zero_uses_certified_gap_not_witness_length(self):
        model, data, limit = self.scene(.4)
        # Witnesses claim a .4 m gap; the actual surface gap is only .2 m.
        with patch.object(limit, '_geom_distance', return_value=(0., np.zeros(3), np.array([.4, 0, 0]))):
            distance, p1, p2, normal = limit._distance_with_normal(model, data, 0, 1, .5)
        self.assertAlmostEqual(distance, .2)
        self.assertAlmostEqual(normal @ (p2 - p1), .2)
        self.assertEqual(limit.distance_fallback_count, 1)

    def test_uncertified_bad_result_must_not_hide_a_collision(self):
        model, data, limit = self.scene(.15)
        with patch.object(limit, '_geom_distance', return_value=(0., np.zeros(3), np.array([.4, 0, 0]))):
            with self.assertRaisesRegex(ValueError, 'without a separation certificate'):
                limit._distance_with_normal(model, data, 0, 1, .5)

    def test_penetration_normal_pushes_bodies_apart(self):
        model, data, limit = self.scene(.15)
        c = limit.compute_qp_inequalities(SimpleNamespace(model=model, data=data), .02)
        distance_before = mujoco.mj_geomDistance(model, data, 0, 1, .5, np.zeros(6))
        self.assertLess(distance_before, 0)
        # Minimum-norm displacement satisfying this one separation constraint.
        delta = c.G[0] * c.h[0] / (c.G[0] @ c.G[0])
        self.assertGreater(delta[0], 0)
        mujoco.mj_integratePos(model, data.qpos, delta, 1.)
        mujoco.mj_forward(model, data)
        distance_after = mujoco.mj_geomDistance(model, data, 0, 1, .5, np.zeros(6))
        self.assertGreater(distance_after, distance_before)

    def test_real_contact_is_not_treated_as_a_false_zero(self):
        model, data, limit = self.scene(.2)
        distance, _, _, _ = limit._distance_with_normal(model, data, 0, 1, .5)
        self.assertAlmostEqual(distance, 0., places=7)
        self.assertEqual(limit.distance_fallback_count, 0)


if __name__ == '__main__':
    unittest.main()
