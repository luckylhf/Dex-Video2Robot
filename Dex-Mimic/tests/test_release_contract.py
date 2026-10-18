"""Release regressions: asset drift rejection and consistent bundle selection."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ros_dex_deploy/src/mujoco_node'))
from mujoco_node.policy_contract import asset_sha256, resolve_mimic_config, validate_identity, verify_model, validate_deployment_timing


class ReleaseContractTest(unittest.TestCase):
    def test_training_provenance_and_hardware_timing_are_distinct(self):
        meta = {'control_dt': '0.02'}
        self.assertAlmostEqual(validate_deployment_timing(meta, .01, 1), .01)
        with self.assertRaisesRegex(ValueError, 'at least 100 Hz'):
            validate_deployment_timing(meta, .01, 2)
        with self.assertRaisesRegex(ValueError, 'training control_dt'):
            validate_deployment_timing({'control_dt': '0.01'}, .01, 1)

    def test_changed_urdf_is_rejected_even_if_manifest_and_mjcf_stay_old(self):
        source = ROOT / 'source/whole_body_tracking/whole_body_tracking/assets/dex_evt'
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'robot'
            shutil.copytree(source, target, ignore=shutil.ignore_patterns('*.usd', '*.usda', '*.usdc'))
            verify_model(target)
            with (target / 'urdf/dex_29.urdf').open('a') as stream:
                stream.write('\n<!-- changed after export -->\n')
            with self.assertRaisesRegex(ValueError, 'Working URDF'):
                verify_model(target)

    def test_selected_bundle_overrides_both_paths_without_changing_yaml(self):
        path = ROOT / 'ros_dex_deploy/src/rl_control/policy/mimic/config/mimic.yaml'
        original = path.read_bytes()
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'MIMIC_BUNDLE_DIR': temp}):
            config = resolve_mimic_config(path)
            self.assertEqual(config['onnx_path'], str(Path(temp) / 'policy_merged.onnx'))
            self.assertEqual(config['npz_dir'], str(Path(temp) / 'npz'))
        self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
