"""回归：恢复训练必须保留基类恢复出的迭代数。"""
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ros_dex_deploy/src/mujoco_node'))


class ResumeTest(unittest.TestCase):
    def test_resumed_iteration_is_preserved(self):
        class BaseRunner:
            def load(self, path, load_optimizer=True):
                self.current_learning_iteration = 74
                return {'restored': True}
        modules = {}
        for name in ['rsl_rl.runners.on_policy_runner', 'isaaclab_rl.rsl_rl', 'wandb', 'whole_body_tracking.utils.exporter']:
            modules[name] = types.ModuleType(name)
        modules['rsl_rl.runners.on_policy_runner'].OnPolicyRunner = BaseRunner
        modules['isaaclab_rl.rsl_rl'].export_policy_as_onnx = lambda *a, **k: None
        modules['whole_body_tracking.utils.exporter'].attach_onnx_metadata = lambda *a, **k: None
        modules['whole_body_tracking.utils.exporter'].export_motion_policy_as_onnx = lambda *a, **k: None
        path = Path(__file__).resolve().parents[1] / 'source/whole_body_tracking/whole_body_tracking/utils/my_on_policy_runner.py'
        spec = importlib.util.spec_from_file_location('offline_runner_test', path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(module)
        obj = object.__new__(module.MotionOnPolicyRunner)
        obj._dex_contract = lambda: {}
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / 'checkpoint.pt'
            module.torch.save({'infos': {'dex_contract': {}}}, checkpoint)
            self.assertEqual(obj.load(checkpoint), {'restored': True})
            with self.subTest('unversioned checkpoint'):
                module.torch.save({'infos': None}, checkpoint)
                with self.assertRaisesRegex(ValueError, 'Unversioned checkpoint'):
                    obj.load(checkpoint)
            with self.subTest('changed model/dynamics'):
                module.torch.save({'infos': {'dex_contract': {'model': 'changed'}}}, checkpoint)
                with self.assertRaisesRegex(ValueError, 'model/dynamics differ'):
                    obj.load(checkpoint)
        self.assertEqual(obj.current_learning_iteration, 74)


if __name__ == '__main__':
    unittest.main()
