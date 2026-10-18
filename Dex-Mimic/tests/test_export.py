"""CPU 验证真实导出器和历史权重的双头路由；仿真环境接口用离线数据代替。"""
import importlib.util
import json
import sys
import shutil
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import torch
from rsl_rl.modules import ActorCritic, EmpiricalNormalization

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from csv_to_npz_mujoco import ROBOT_MJCF, ROBOT_JOINT_NAMES, POLICY_BODY_NAMES, ROBOT_BODY_NAMES, STANDING_JOINT_POS_MAP, run_fk
from prepare_deploy import validate_policy, OBSERVATIONS


class ExportTest(unittest.TestCase):
    def test_real_exporter_numerics_and_contract(self):
        # Only Isaac's environment/type imports are replaced. The exporter,
        # RSL actors/normalizers, supplied checkpoints and ONNX Runtime are real.
        modules = {}
        for name in ['isaaclab.envs', 'isaaclab_rl.rsl_rl.exporter', 'whole_body_tracking.tasks.tracking.mdp']:
            modules[name] = types.ModuleType(name)
        modules['isaaclab.envs'].ManagerBasedRLEnv = object
        modules['isaaclab_rl.rsl_rl.exporter']._OnnxPolicyExporter = torch.nn.Module
        modules['whole_body_tracking.tasks.tracking.mdp'].MotionCommand = object
        file = ROOT / 'source/whole_body_tracking/whole_body_tracking/utils/exporter.py'
        spec = importlib.util.spec_from_file_location('offline_dex_exporter', file)
        exporter = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(exporter)

        checkpoint = torch.load(ROOT / 'policy/dex/wbt_model_46400.pt', map_location='cpu', weights_only=False)
        critic_dim = checkpoint['model_state_dict']['critic.0.weight'].shape[1]
        actor = ActorCritic(214, critic_dim, 29, actor_hidden_dims=[768, 384, 192],
            critic_hidden_dims=[768, 384, 192], activation='elu')
        actor.load_state_dict(checkpoint['model_state_dict'])
        actor.eval()
        norm = EmpiricalNormalization(shape=(214,))
        norm.load_state_dict(checkpoint['obs_norm_state_dict'])
        norm.eval()
        names, bodies = ROBOT_JOINT_NAMES['dex_evt'], POLICY_BODY_NAMES['dex_evt']
        default = torch.tensor([STANDING_JOINT_POS_MAP['dex_evt'].get(n, 0.) for n in names])
        manifest = json.loads((Path(ROBOT_MJCF['dex_evt']).parents[1] / 'model.json').read_text())
        lo = torch.tensor([float(manifest['joint_limits'][n]['lower']) for n in names])
        hi = torch.tensor([float(manifest['joint_limits'][n]['upper']) for n in names])
        motion = SimpleNamespace(joint_pos=torch.stack([default, default]), joint_vel=torch.zeros((2, 29)),
            body_pos_w=torch.zeros((2, 30, 3)), body_quat_w=torch.tensor([1., 0, 0, 0]).repeat(2, 30, 1),
            body_lin_vel_w=torch.zeros((2, 30, 3)), body_ang_vel_w=torch.zeros((2, 30, 3)))
        command = SimpleNamespace(motion=motion, cfg=SimpleNamespace(anchor_body_name='pelvis', body_names=bodies))
        group = SimpleNamespace(stiffness={'.*': 100.}, damping={'.*': 5.}, effort_limit_sim={'.*': 40.}, joint_names_expr=['.*'])
        robot = SimpleNamespace(data=SimpleNamespace(joint_names=names, default_joint_pos_nominal=default,
            soft_joint_pos_limits=torch.stack([lo, hi], dim=1).unsqueeze(0)), cfg=SimpleNamespace(actuators={'all': group}))
        env = SimpleNamespace(scene={'robot': robot}, step_dt=.02,
            command_manager=SimpleNamespace(get_term=lambda _: command, active_terms=['motion']),
            action_manager=SimpleNamespace(get_term=lambda _: SimpleNamespace(_scale=torch.full((1, 29), .05))),
            observation_manager=SimpleNamespace(active_terms={'policy': OBSERVATIONS},
                cfg=SimpleNamespace(policy=SimpleNamespace(history_length=1))))

        with tempfile.TemporaryDirectory() as temp:
            exporter.export_motion_policy_merged_as_onnx(env, actor, norm,
                str(ROOT / 'policy/dex/standing_model_2000.pt'), temp)
            exporter.attach_onnx_metadata(env, 'offline_interface_test_only', temp, 'policy_merged.onnx')
            path = Path(temp) / 'policy_merged.onnx'
            validate_policy(path)
            session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
            rng = np.random.default_rng(216)
            obs = rng.normal(0, .05, (1, 216)).astype(np.float32)
            obs[:, -2:] = [0, 1]
            got = session.run(['actions'], {'obs': obs, 'time_step': np.zeros((1, 1), dtype=np.float32)})[0]
            with torch.no_grad():
                expected = actor.actor(norm(torch.from_numpy(obs[:, :-2]))).numpy()
            np.testing.assert_allclose(got, expected, atol=3e-5, rtol=3e-5)
            # Test the actual package CLI and standalone physics path as well.
            q = np.tile(default.numpy(), (50, 1))
            all_bodies = ROBOT_BODY_NAMES['dex_evt']
            pos, quat = run_fk(np.tile([0., 0, 1.], (50, 1)), np.tile([1., 0, 0, 0], (50, 1)),
                q, names, all_bodies, ROBOT_MJCF['dex_evt'])
            reference = Path(temp) / 'reference.npz'
            np.savez(reference, fps=50, model_sha256=manifest['model_sha256'], quat_order='wxyz',
                joint_names=names, body_names=all_bodies, joint_pos=q, joint_vel=np.zeros((50, 29)),
                body_pos_w=pos, body_quat_w=quat,
                body_lin_vel_w=np.zeros((50, len(all_bodies), 3)),
                body_ang_vel_w=np.zeros((50, len(all_bodies), 3)))
            deployment = Path(temp) / 'ros_dex_deploy'
            bundle = deployment / 'src/rl_control/policy/mimic/out'
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/prepare_deploy.py'),
                str(path), str(reference), '--output', str(bundle)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            model_dir = deployment / 'src/mujoco_node/resources/dex'
            shutil.copytree(Path(ROBOT_MJCF['dex_evt']).parents[1], model_dir)
            helper_dir = deployment / 'src/mujoco_node/mujoco_node'
            helper_dir.mkdir(parents=True)
            shutil.copy2(ROOT / 'ros_dex_deploy/src/mujoco_node/mujoco_node/policy_contract.py', helper_dir)
            shutil.copy2(ROOT / 'ros_dex_deploy/check_bundle.py', deployment / 'check_bundle.py')
            shutil.copytree(ROOT / 'ros_dex_deploy/src/rl_control/policy/mimic/config', bundle.parent / 'config')
            result = subprocess.run([sys.executable, str(deployment / 'check_bundle.py')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            from sim2sim_dex import OnnxMotionPolicy, NpzMotionStream, MujocoSim2Sim
            policy = OnnxMotionPolicy(path)
            stream = NpzMotionStream(str(reference), names, bodies)
            simulation = MujocoSim2Sim(Path(ROBOT_MJCF['dex_evt']), policy, .02, .0025,
                zero_xy_ref=True, max_target_step=None, motion_stream=stream)
            for _ in range(10):
                simulation.step()
            self.assertTrue(np.isfinite(simulation.data.qpos).all())


if __name__ == '__main__':
    unittest.main()
