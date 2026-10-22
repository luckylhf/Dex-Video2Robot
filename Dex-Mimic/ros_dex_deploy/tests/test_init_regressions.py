"""D/init and motion handoff regressions. No hardware, ROS executor or GUI."""
import contextlib
import io
import threading
import time
import unittest
from pathlib import Path
from queue import Queue
from types import SimpleNamespace as NS
import sys

import mujoco
import numpy as np

WS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(WS / 'src/rl_control'), str(WS / 'src/mujoco_node')]
from common.robot_data import RobotData
from common.body_id_map import BodyServoIdMap
from common.joystick import ControlFlag, JoystickHumanoid, YUNZHUOMap
from common.robot_interface import RobotInterfaceImpl
from policy.mimic.fsm_mimic import FSMStateMimic, FSMStateMimicDefault
from mujoco_node.simulator_view_asyn import RobotSimulator
from mujoco_node.elastic_band import ElasticBand
from bodyctrl_msgs.msg import CmdMotorCtrl, MotorCtrl


def interface():
    obj = RobotInterfaceImpl.__new__(RobotInterfaceImpl)
    obj.robot_data_ = RobotData()
    obj.robot_data_.sim_ = True
    obj.sim = True
    obj.id_map = BodyServoIdMap()
    obj.id_map.body_can_id_map_init()
    return obj


def messages(obj, generation=10, running=False, sim_time=0.):
    groups = (range(12), range(15, 29), range(12, 15))
    return [NS(header=NS(frame_id=f'dex_sim:{generation}:{int(running)}:{sim_time:.9f}'),
               status=[NS(name=obj.id_map.get_id_by_index(i), pos=0., speed=0., current=0.)
                       for i in group]) for group in groups]


def default(rd):
    obj = FSMStateMimicDefault.__new__(FSMStateMimicDefault)
    obj.robot_data_ = rd
    obj.motor_nums = 29
    obj.interp_step = .01
    obj._sim_interp_duration_s = 1.0
    obj._sim_interp_start_s = None
    obj.default_xml = np.ones(29, dtype=np.float32) * .2
    obj.kps_xml, obj.kds_xml = np.ones(29), np.ones(29)
    obj._q_factor = .8
    obj._start_pose = np.ones(29, dtype=np.float32) * .7
    return obj


def joystick():
    obj = JoystickHumanoid.__new__(JoystickHumanoid)
    obj.data_mutex = threading.RLock()
    obj.joy_map = YUNZHUOMap()
    obj.joy_flag = ControlFlag()
    obj.joy_flag.x_speed_command = obj.joy_flag.y_speed_command = obj.joy_flag.yaw_speed_command = 0.
    obj._last_b, obj._a_press_t = 0., None
    obj.get_x_y_yaw_speed_command = lambda: None
    obj.get_walk_height_command = lambda: None
    return obj


class InitRegressionTest(unittest.TestCase):
    def test_plain_d_without_body_reset_uses_measured_pose(self):
        j = joystick()
        j.joy_map.f = j.joy_map.d = 1.
        j.joy_flag_update()
        obj = interface()
        obj.robot_data_.q_a_[6:] = .7
        obj.update_robot_cmd(j.joy_flag)
        state = default(obj.robot_data_)
        state.on_enter()
        state.run(obj.robot_data_.control_flag)
        np.testing.assert_allclose(obj.robot_data_.q_d_[6:], .7)

    def test_d_release_outside_fdown_does_not_latch_an_edge(self):
        j = joystick()
        j.joy_map.f = j.joy_map.d = 1.
        j.joy_flag_update()
        j.joy_map.f = j.joy_map.d = 0.
        j.joy_flag_update()
        j.joy_map.c = 1.
        j.joy_flag_update()
        j.joy_map.c = 0.
        j.joy_map.f = j.joy_map.d = 1.
        j.joy_flag_update()
        self.assertEqual(j.joy_flag.fsm_state_command, 'gotoMIMICDEFAULT')
        self.assertFalse(getattr(j, '_d_prev', False))

    def test_actual_reset_same_state_reinitializes_and_paused_time_does_not_advance(self):
        obj = interface()
        obj._update_sim_init_state(messages(obj))
        state = default(obj.robot_data_)
        # Deliberately do not call on_enter: a reset while already in this state.
        for _ in range(200):
            state.run(obj.robot_data_.control_flag)
        np.testing.assert_array_equal(state._start_pose, np.zeros(29))
        np.testing.assert_array_equal(obj.robot_data_.q_d_[6:], np.zeros(29))
        self.assertEqual(state._q_factor, 0.)
        self.assertTrue(obj.robot_data_.sim_init_ready_)
        for i in range(101):
            obj._update_sim_init_state(messages(obj, running=True, sim_time=i * .01))
            state.run(obj.robot_data_.control_flag)
        np.testing.assert_allclose(obj.robot_data_.q_d_[6:], .2, atol=1e-7)
        obj._update_sim_init_state(messages(obj, generation=11))
        state.run(obj.robot_data_.control_flag)
        self.assertEqual(state._q_factor, 0.)
        np.testing.assert_array_equal(obj.robot_data_.q_d_[6:], np.zeros(29))

    def test_mixed_incomplete_nonfinite_and_stale_feedback_are_not_consumed(self):
        obj = interface()
        bad = messages(obj)
        bad[1].header.frame_id = 'dex_sim:9:0:0.000000000'
        obj._update_sim_init_state(bad)
        self.assertFalse(obj.robot_data_.sim_feedback_coherent_)
        bad = messages(obj)
        bad[1].status.pop()
        obj._update_sim_init_state(bad)
        self.assertFalse(obj.robot_data_.sim_feedback_coherent_)
        bad = messages(obj)
        bad[0].status[0].pos = float('nan')
        obj._update_sim_init_state(bad)
        self.assertFalse(obj.robot_data_.sim_feedback_coherent_)
        obj._update_sim_init_state(messages(obj))
        obj.robot_data_.sim_init_pending_ = False
        obj._update_sim_init_state(messages(obj))
        self.assertFalse(obj.robot_data_.sim_init_pending_)
        obj._update_sim_init_state(messages(obj, generation=9))
        self.assertEqual(obj.robot_data_.sim_init_id_, 10)
        self.assertFalse(obj.robot_data_.sim_feedback_coherent_)

    def test_slow_physics_and_duplicate_feedback_do_not_finish_early(self):
        obj = interface()
        obj._update_sim_init_state(messages(obj, running=True))
        state = default(obj.robot_data_)
        state.run(obj.robot_data_.control_flag)
        # Arbitrarily many controller cycles with no physical progress do nothing.
        for _ in range(500): state.run(obj.robot_data_.control_flag)
        self.assertEqual(state._q_factor, 0.)
        obj._update_sim_init_state(messages(obj, running=True, sim_time=.25))
        state.run(obj.robot_data_.control_flag)
        self.assertAlmostEqual(state._q_factor, .25)
        np.testing.assert_allclose(obj.robot_data_.q_d_[6:], .05)
        for _ in range(500): state.run(obj.robot_data_.control_flag)
        self.assertAlmostEqual(state._q_factor, .25)
        obj._update_sim_init_state(messages(obj, running=False, sim_time=.25))
        for _ in range(500): state.run(obj.robot_data_.control_flag)
        self.assertAlmostEqual(state._q_factor, .25)
        obj._update_sim_init_state(messages(obj, running=True, sim_time=1.0))
        state.run(obj.robot_data_.control_flag)
        self.assertEqual(state._q_factor, 1.)
        np.testing.assert_allclose(obj.robot_data_.q_d_[6:], .2, atol=1e-7)

    def test_physics_time_uses_slowest_group_and_rejects_bad_time(self):
        obj = interface()
        group = messages(obj, sim_time=.3)
        group[1].header.frame_id = 'dex_sim:10:0:0.310000000'
        group[2].header.frame_id = 'dex_sim:10:0:0.290000000'
        obj._update_sim_init_state(group)
        self.assertTrue(obj.robot_data_.sim_feedback_coherent_)
        self.assertAlmostEqual(obj.robot_data_.sim_time_, .29)
        for bad in [float('nan'), float('inf'), -.1, .2]:
            obj._update_sim_init_state(messages(obj, sim_time=bad))
            self.assertFalse(obj.robot_data_.sim_feedback_coherent_)
            self.assertAlmostEqual(obj.robot_data_.sim_time_, .29)
        # New reset generation legitimately restarts data.time at zero.
        obj._update_sim_init_state(messages(obj, generation=11, sim_time=0.))
        self.assertTrue(obj.robot_data_.sim_init_pending_)
        self.assertEqual(obj.robot_data_.sim_time_, 0.)

    def test_ordinary_sim_transition_starts_at_current_physics_time(self):
        obj = interface()
        obj._update_sim_init_state(messages(obj, running=True, sim_time=5.))
        obj.robot_data_.sim_init_pending_ = False
        obj.robot_data_.sim_init_ready_ = True
        obj.robot_data_.q_a_[6:] = .7
        state = default(obj.robot_data_)
        state.on_enter()
        state.run(obj.robot_data_.control_flag)
        np.testing.assert_allclose(obj.robot_data_.q_d_[6:], .7)
        obj._update_sim_init_state(messages(obj, running=True, sim_time=5.25))
        state.run(obj.robot_data_.control_flag)
        np.testing.assert_allclose(obj.robot_data_.q_d_[6:], .575, atol=1e-7)

    def test_real_interpolation_still_uses_measured_pose(self):
        rd = RobotData()
        rd.q_a_[6:] = .7
        state = default(rd)
        state.on_enter()
        state.run(rd.control_flag)
        np.testing.assert_allclose(rd.q_d_[6:], .7)
        self.assertAlmostEqual(state._q_factor, .01)
        for _ in range(100): state.run(rd.control_flag)
        np.testing.assert_allclose(rd.q_d_[6:], .2, atol=1e-7)

    def test_unready_sim_does_not_publish_commands(self):
        obj = interface()
        obj.initialized = True
        obj.convert_to_motor_commands = lambda: self.fail('converted an unready command')
        obj.publish_motor_commands = lambda flag: self.fail('published an unready command')
        obj.send_motor_commands(obj.robot_data_.control_flag)

    def test_coordinate_validator_rejects_wrong_array_size(self):
        obj = interface()
        obj.node = None
        obj.whole_joint_nums, obj.floating_base_dof = 35, 6
        obj.zero_pos, obj.zero_offset, obj.motor_dir = np.zeros(1), np.zeros(29), np.ones(29)
        with self.assertRaises(ValueError): obj._validate_sim_coordinate_contract()

    def test_mimic_reentry_clears_previous_heading_and_targets(self):
        obj = FSMStateMimic.__new__(FSMStateMimic)
        obj.obs_dim, obj.num_actions, obj.motor_nums = 216, 29, 29
        obj.physical_dt, obj.decimation, obj._warm_start_steps = .01, 1, 0
        obj._stand_total_steps, obj.warm_start_time = 30, .5
        obj.default_joint_pos_xml = obj.kps_xml = obj.kds_xml = np.zeros(29)
        obj.last_action = np.zeros(29)
        obj._ref_yaw_inv = np.array([1., 0., 0., 0.])
        obj._stand_anchor_yaw, obj._last_motion_target_xml = 1.2, np.ones(29)
        obj._prev_target_xml = np.ones(29)
        with contextlib.redirect_stdout(io.StringIO()): obj.on_enter()
        self.assertIsNone(obj._stand_anchor_yaw)
        self.assertIsNone(obj._last_motion_target_xml)
        self.assertIsNone(obj._prev_target_xml)

    def test_handoff_starts_from_final_rate_limited_target(self):
        obj = FSMStateMimic.__new__(FSMStateMimic)
        rd = RobotData()
        obj.robot_data_ = rd
        obj.mj2lab, obj._ref_yaw_inv = np.arange(29), np.array([1., 0., 0., 0.])
        obj.obs_dim, obj.motor_nums, obj.last_action = 216, 29, np.zeros(29)
        obj.action_scale_seq = np.ones(29)
        obj._q_lower, obj._q_upper = np.full(29, -10.), np.full(29, 10.)
        obj.output_names, obj.input_obs_name, obj.input_step_name = ['action'], 'obs', 'step'
        obj.max_target_step_motion, obj.max_target_step_standing = .01, None
        obj._prev_target_xml, obj._playback_phase = np.zeros(29), 'play_motion'
        obj._stand_step_count, obj._stand_total_steps = 0, 30
        action = np.ones(29, dtype=np.float32)
        obj.ort_session = NS(run=lambda *args: [np.array([action])])
        zeros = np.zeros(29)
        obj._run_policy(zeros, zeros, zeros, zeros, 0, 100, 1)
        np.testing.assert_allclose(obj._last_motion_target_xml, rd.q_d_[6:])
        np.testing.assert_allclose(rd.q_d_[6:], .01)
        obj._playback_phase, action[:] = 'return_stand', 0.
        obj._run_policy(zeros, zeros, zeros, zeros, 0, 1, 0)
        self.assertLess(float(np.max(np.abs(rd.q_d_[6:] - .01))), .001)

    def test_reset_and_callback_share_lock_and_reject_previous_generation(self):
        model = mujoco.MjModel.from_xml_path(str(WS / 'src/mujoco_node/resources/dex/mjcf/dex_29.xml'))
        obj = RobotSimulator.__new__(RobotSimulator)
        obj.model, obj.data = model, mujoco.MjData(model)
        obj.joint_cmd_lock = threading.Lock()
        obj.raw_joint_commands = Queue()
        obj.joint_commands, obj.last_raw_joint_commands = {}, {}
        obj.band_attached_link = model.body('waist_pitch_link').id
        obj.elastic_band = ElasticBand()
        obj.simulation_started = False
        obj._init_stamp_ns, obj._sim_init_id = 0, 'old'
        stamp = NS(sec=1, nanosec=0)
        obj.node = NS(get_clock=lambda: NS(now=lambda: NS(nanoseconds=10**9, to_msg=lambda: stamp)))
        snapshots = []
        obj.read_sensors = lambda: snapshots.append((obj._sim_init_id, obj.data.qpos.copy()))
        done = threading.Event()
        with obj.joint_cmd_lock:
            thread = threading.Thread(target=lambda: (obj._reset_robot(), done.set()))
            thread.start()
            self.assertFalse(done.wait(.03))
        thread.join(2.)
        self.assertTrue(done.is_set())
        self.assertEqual(snapshots[-1][0], obj._sim_init_id)
        msg = CmdMotorCtrl()
        msg.header.stamp.sec = 1
        msg.header.frame_id = 'dex_sim:old'
        cmd = MotorCtrl()
        cmd.name = 51
        msg.cmds.append(cmd)
        obj.joint_cmd_callback(msg)
        self.assertTrue(obj.raw_joint_commands.empty())
        msg.header.frame_id = f'dex_sim:{obj._sim_init_id}'
        obj.joint_cmd_callback(msg)
        self.assertEqual(obj.raw_joint_commands.qsize(), 1)


if __name__ == '__main__': unittest.main()
