"""回归：显式关节顺序、非平凡四元数、末帧和非法输入。"""
import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from pkl_to_npz import load_pkl, convert_single
from csv_to_npz_mujoco import ROBOT_JOINT_NAMES, ROBOT_MJCF, run_fk, add_transition_to_npz
from check_motion import validate_motion


class ConversionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'motion.pkl'
        manifest = json.loads((ROOT / 'source/whole_body_tracking/whole_body_tracking/assets/dex_evt/model.json').read_text())
        self.names = ROBOT_JOINT_NAMES['dex_evt']
        self.values = np.linspace(-0.1, 0.1, 29)
        self.motion = {
            'fps': 30., 'root_pos': np.tile([0, 0, 1.], (4, 1)),
            'root_rot': np.tile(Rotation.from_euler('xyz', [0.2, -0.1, 0.3]).as_quat(), (4, 1)),
            'dof_pos': np.tile(self.values[::-1], (4, 1)),
            'joint_names': self.names[::-1], 'model_sha256': manifest['model_sha256'],
            'quat_order': 'xyzw',
        }

    def tearDown(self):
        self.tmp.cleanup()

    def load(self):
        self.path.write_bytes(pickle.dumps(self.motion))
        return load_pkl(str(self.path), 50, 'dex_evt')

    def test_explicit_joint_order(self):
        out = self.load()
        np.testing.assert_allclose(out['joint_pos'][0], self.values, atol=1e-7)

    def test_quaternion_and_last_frame(self):
        out = self.load()
        self.assertEqual(len(out['base_pos']), 6)
        np.testing.assert_allclose(out['base_quat'][0], self.motion['root_rot'][0][[3, 0, 1, 2]], atol=1e-7)

    def test_invalid_quaternion_rejected(self):
        self.motion['root_rot'][:] = 0
        with self.assertRaises(ValueError):
            self.load()

    def test_wrong_model_rejected(self):
        self.motion['model_sha256'] = 'old_model'
        with self.assertRaises(ValueError):
            self.load()

    def test_wrong_quaternion_order_rejected(self):
        self.motion['quat_order'] = 'wxyz'
        with self.assertRaises(ValueError):
            self.load()

    def test_duplicate_joint_rejected(self):
        self.motion['joint_names'][0] = self.motion['joint_names'][1]
        with self.assertRaises(ValueError):
            self.load()

    def test_transition_fk_and_model_tag(self):
        self.path.write_bytes(pickle.dumps(self.motion))
        raw = Path(self.tmp.name) / 'raw.npz'
        final = Path(self.tmp.name) / 'final.npz'
        convert_single(str(self.path), str(raw), 50, 'dex_evt', ROBOT_MJCF['dex_evt'])
        add_transition_to_npz(str(raw), str(final), hold_duration=0.1, trans_duration=0.2,
            tail_trans_duration=0.2, tail_hold_duration=0.1, stand_height=1.0,
            head_skip_frames=0, tail_skip_frames=0, skip_threshold=1.5, mjcf_path=ROBOT_MJCF['dex_evt'])
        data = np.load(final)
        pos, quat = run_fk(data['body_pos_w'][:, 0], data['body_quat_w'][:, 0],
            data['joint_pos'], data['joint_names'].tolist(), data['body_names'].tolist(), ROBOT_MJCF['dex_evt'])
        np.testing.assert_allclose(pos, data['body_pos_w'], atol=1e-6)
        self.assertEqual(str(data['model_sha256']), self.motion['model_sha256'])

    def test_csv_single_cli_output_passes_motion_check(self):
        csv = Path(self.tmp.name) / 'motion.csv'
        header = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'] + self.names
        frames = np.tile([0., 0., 1., 0., 0., 0., 1.] + [0.] * 29, (16, 1))
        np.savetxt(csv, frames, delimiter=',', header=','.join(header), comments='')
        raw = Path(self.tmp.name) / 'csv_motion'
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/csv_to_npz_mujoco.py'),
            '--input_file', str(csv), '--output_name', str(raw), '--add_transition'],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(validate_motion(raw.with_suffix('.npz'))['fps'], 50)
        self.assertGreater(validate_motion(raw.with_name(raw.name + '_with_transition.npz'))['frames'], 26)

    def test_transition_rejects_clip_cropped_below_two_frames(self):
        self.path.write_bytes(pickle.dumps(self.motion))
        raw = Path(self.tmp.name) / 'raw.npz'
        convert_single(str(self.path), str(raw), 50, 'dex_evt', ROBOT_MJCF['dex_evt'])
        for head, tail in [(5, 5), (5, 0)]:
            with self.subTest(head=head, tail=tail):
                with self.assertRaisesRegex(ValueError, 'fewer than 2 frames'):
                    add_transition_to_npz(str(raw), str(Path(self.tmp.name) / 'invalid.npz'),
                        hold_duration=1., trans_duration=1., tail_trans_duration=2.5,
                        tail_hold_duration=1., stand_height=.87, head_skip_frames=head,
                        tail_skip_frames=tail, skip_threshold=1.5, mjcf_path=ROBOT_MJCF['dex_evt'])

    def test_csv_batch_cli_output_passes_motion_check(self):
        csv_dir = Path(self.tmp.name) / 'csv'
        csv_dir.mkdir()
        header = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'] + self.names
        frames = np.tile([0., 0., 1., 0., 0., 0., 1.] + [0.] * 29, (16, 1))
        np.savetxt(csv_dir / 'motion.csv', frames, delimiter=',', header=','.join(header), comments='')
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/csv_to_npz_mujoco.py'),
            '--batch_dir', str(csv_dir), '--add_transition'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(validate_motion(csv_dir.parent / 'npz_dex_raw/motion.npz')['fps'], 50)
        self.assertGreater(validate_motion(csv_dir.parent / 'npz_dex/motion_with_transition.npz')['frames'], 26)


if __name__ == '__main__':
    unittest.main()
