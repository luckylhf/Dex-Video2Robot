"""使用真实 Dex FSM 方法离线核对时长与仿真观测，不启动 ROS 或硬件。"""
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'ros_dex_deploy/src/mujoco_node'))
sys.path.insert(0, str(ROOT / 'ros_dex_deploy/src/rl_control'))
joystick = types.ModuleType('common.joystick')
joystick.ControlFlag = SimpleNamespace
with patch.dict(sys.modules, {'common.joystick': joystick}):
    from policy.mimic.fsm_mimic import FSMStateMimic
from sim2sim_dex import MujocoSim2Sim


class DeployTest(unittest.TestCase):
    def test_both_mimic_configs_use_100hz_inference(self):
        config_dir = ROOT / 'ros_dex_deploy/src/rl_control/policy/mimic/config'
        for name in ['mimic.yaml', 'mimic_real.yaml']:
            with self.subTest(config=name):
                cfg = yaml.safe_load((config_dir / name).read_text())
                self.assertAlmostEqual(cfg['physical_dt'], .01)
                self.assertEqual(cfg['decimation'], 1)

    def test_motion_duration_at_training_policy_rate(self):
        self.check_duration(2, 50)

    def test_motion_duration_at_100hz(self):
        self.check_duration(1, 100)

    def check_duration(self, decimation, expected):
        obj = object.__new__(FSMStateMimic)
        zeros = np.zeros(29, dtype=np.float32)
        clip = SimpleNamespace(name="fixture", T=50, fps=50, joint_pos_at=lambda i: zeros,
            joint_vel_at=lambda i: zeros, anchor_quat_at=lambda i: np.array([1., 0, 0, 0]))
        obj.clips, obj._active_clip_idx = [clip], 0
        obj._motion_step = obj._motion_substep = obj._ws_counter = obj._warm_start_steps = 0
        obj._motion_elapsed = 0.
        obj.physical_dt, obj.decimation = .01, decimation
        obj.finished = False
        frames = []
        obj._run_policy = lambda *args, **kwargs: frames.append(kwargs['motion_step'])
        obj._start_return_stand = lambda: setattr(obj, 'finished', True)
        calls = 0
        while not obj.finished and calls < 300:
            obj._run_motion_step()
            calls += 1
        self.assertTrue(obj.finished)
        self.assertEqual(calls, expected)
        self.assertEqual(frames, [i for i in range(50) for _ in range(expected // 50)])

    def test_100hz_fsm_recomputes_from_fresh_state_every_call(self):
        obj = object.__new__(FSMStateMimic)
        obj._call_counter, obj.decimation = 0, 1
        obj._last_run_time, obj._playback_phase, obj._motion_step = 0., 'default_pose', 0
        obj._set_kp_kd = lambda: None
        observed = []
        obj._step_playlist = lambda: observed.append(obj.sensor_tick)
        for tick in range(100):
            obj.sensor_tick = tick
            obj.run(SimpleNamespace(motion_cmd=''))
        self.assertEqual(observed, list(range(100)))

    def test_sim2sim_defaults_to_engineer_50hz_and_preserves_motion_duration(self):
        from sim2sim_dex import build_argparser
        self.assertAlmostEqual(build_argparser().parse_args([]).control_dt, .02)
        for playlist in (False, True):
            self.check_sim2sim_duration(playlist, .02, 50)
            self.check_sim2sim_duration(playlist, .01, 100)

    def check_sim2sim_duration(self, playlist, period, count):
        with self.subTest(playlist=playlist, period=period):
            obj = object.__new__(MujocoSim2Sim)
            obj.control_dt, obj._motion_elapsed, obj.current_step = period, 0., 0
            obj.motion_stream = SimpleNamespace(total_steps=50, fps=50.)
            obj.motion_library = SimpleNamespace() if playlist else None
            obj._playback_phase, obj._stand_total_steps, obj.stand_duration_s = obj.PLAY_MOTION, 100, 1.
            obj._active_motion_name = 'fixture'
            obj._consume_next_motion_request = lambda: False
            obj._reference_output = lambda i: i
            frames = []
            obj._run_policy_step = lambda ref, future, step, **kwargs: frames.append(step)
            obj.last_action = np.zeros(29)
            obj._current_joint_positions = lambda: np.zeros(29)
            finished = False
            for _ in range(count):
                finished = obj.step()
            self.assertEqual(frames, [i for i in range(50) for _ in range(count // 50)])
            if playlist:
                self.assertEqual(obj._playback_phase, obj.DEFAULT_POSE)
            else:
                self.assertTrue(finished)
            self.assertAlmostEqual(obj._motion_elapsed, 1.)

    def test_real_and_sim_observation_agree(self):
        rng = np.random.default_rng(216)
        real = object.__new__(FSMStateMimic)
        sim = object.__new__(MujocoSim2Sim)
        q, qd, qref, qdref, laq, laqd = rng.normal(size=(6, 29)).astype(np.float32)
        robot_quat = Rotation.from_euler('xyz', [.1, -.2, .3]).as_quat()[[3, 0, 1, 2]]
        ref_quat = Rotation.from_euler('xyz', [-.2, .15, .4]).as_quat()[[3, 0, 1, 2]]
        omega = np.array([.2, -.4, .1], dtype=np.float32)
        real.robot_data_ = SimpleNamespace(get_joint_pos=lambda: q, get_joint_vel=lambda: qd,
            get_robot_quat=lambda: robot_quat, get_angular_velocity=lambda: omega,
            q_d_=np.zeros(35), q_dot_d_=np.zeros(35), tau_d_=np.zeros(35))
        real.mj2lab = np.arange(29)
        real.obs_dim, real.motor_nums = 216, 29
        real.last_action = rng.normal(size=29).astype(np.float32)
        real.action_scale_seq = np.ones(29)
        real._q_lower, real._q_upper = np.full(29, -np.inf), np.full(29, np.inf)
        real._ref_yaw_inv = np.array([1., 0, 0, 0])
        real._playback_phase = 'play_motion'
        real.max_target_step_motion = real.max_target_step_standing = None
        real.input_obs_name, real.input_step_name, real.output_names = 'obs', 'time_step', ['actions']
        captured = {}
        def infer(outputs, inputs):
            captured['obs'] = inputs['obs'][0]
            return [np.zeros((1, 29), dtype=np.float32)]
        real.ort_session = SimpleNamespace(run=infer)
        real._run_policy(qref, qdref, laq, laqd, 7, 50, 1, ref_anchor_quat=ref_quat)
        sim.data = SimpleNamespace(qpos=q, qvel=qd)
        sim.joint_qpos_indices = sim.joint_qvel_indices = np.arange(29)
        sim.metadata = SimpleNamespace(obs_dim=216)
        sim.motion_stream = SimpleNamespace(total_steps=50)
        sim._sensor = lambda name: robot_quat if name == 'orientation' else omega
        sim.last_action = real.last_action.copy()
        # The real method updates last_action after inference; compare using the input history.
        sim.last_action = captured['obs'][125:154].copy()
        ref = {'joint_pos': qref, 'joint_vel': qdref, 'body_quat_w': np.array([ref_quat])}
        future = {'joint_pos': laq, 'joint_vel': laqd}
        np.testing.assert_allclose(sim.build_observation(7, ref, future), captured['obs'], atol=1e-6)


if __name__ == '__main__':
    unittest.main()
