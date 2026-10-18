"""随包 SMPL-X 资源必须能被实际人体参数加载器读取。"""
import os
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from general_motion_retargeting.utils.smpl import (
    load_smplx_file, load_gvhmr_pred_file, get_smplx_data_offline_fast,
)


@unittest.skipUnless(all((ROOT / 'assets/body_models/smplx' / f'SMPLX_{gender}.npz').is_file()
                        for gender in ['NEUTRAL', 'MALE', 'FEMALE']),
                     'Prepare third-party SMPL-X resources before running model integration tests')
class SmplxModelsTest(unittest.TestCase):
    def test_three_bundled_genders_load_real_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            motion = Path(tmp) / 'human.npz'
            for gender in ['neutral', 'male', 'female']:
                with self.subTest(gender=gender):
                    np.savez(motion, gender=gender, betas=np.zeros(16), pose_body=np.zeros((2, 63)),
                        root_orient=np.zeros((2, 3)), trans=np.zeros((2, 3)), mocap_frame_rate=30.)
                    data, body, output, height = load_smplx_file(str(motion), str(ROOT / 'assets/body_models'))
                    self.assertEqual(tuple(output.vertices.shape), (2, 10475, 3))
                    self.assertTrue(torch.isfinite(output.joints).all())
                    self.assertEqual(body.num_betas, 16)
                    frames, fps = get_smplx_data_offline_fast(data, body, output, tgt_fps=30)
                    self.assertEqual((len(frames), fps), (2, 30))
                    self.assertIn('pelvis', frames[0])
                    data.close()

    def test_gvhmr_params_load_bundled_neutral(self):
        with tempfile.TemporaryDirectory() as tmp:
            motion = Path(tmp) / 'hmr4d_results.pt'
            torch.save({'smpl_params_global': {'betas': torch.zeros((2, 10)),
                'body_pose': torch.zeros((2, 63)), 'global_orient': torch.zeros((2, 3)),
                'transl': torch.zeros((2, 3))}}, motion)
            data, body, output, height = load_gvhmr_pred_file(str(motion), str(ROOT / 'assets/body_models'))
            self.assertEqual(tuple(output.vertices.shape), (2, 10475, 3))
            self.assertTrue(torch.isfinite(output.joints).all())
            self.assertEqual(body.num_betas, 16)
            self.assertEqual(data['betas'].shape, (16,))

    def test_retarget_model_path_default_env_and_cli(self):
        bundled = ROOT / 'assets/body_models'
        cases = [({}, [], bundled),
                 ({'SMPLX_MODEL_DIR': '/custom/env_models'}, [], Path('/custom/env_models')),
                 ({'SMPLX_MODEL_DIR': '/custom/env_models'}, ['--body-models', '/custom/cli_models'], Path('/custom/cli_models'))]
        for env, options, expected in cases:
            with self.subTest(expected=expected), patch.dict(os.environ, env, clear=True):
                argv = ['retarget_motion.py', 'prediction.pt', '--output', 'unused.pkl', *options]
                with patch.object(sys, 'argv', argv), patch(
                    'general_motion_retargeting.utils.smpl.load_gvhmr_pred_file',
                    side_effect=RuntimeError('loader reached')) as loader:
                    with self.assertRaisesRegex(RuntimeError, 'loader reached'):
                        runpy.run_path(str(ROOT / 'scripts/retarget_motion.py'), run_name='__main__')
                    self.assertEqual(loader.call_args.args[1], str(expected))


if __name__ == '__main__':
    unittest.main()
