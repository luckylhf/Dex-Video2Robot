"""部署入口与弹力带局部回归，不启动 ROS、GUI 或物理仿真。"""
import ast
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

WS = Path(__file__).resolve().parents[1] / 'ros_dex_deploy'
sys.path.insert(0, str(WS / 'src/mujoco_node'))
from mujoco_node.elastic_band import ElasticBand


class DeployEnvironmentTest(unittest.TestCase):
    def test_band_damping_uses_attachment_origin_velocity(self):
        source = WS / 'src/mujoco_node/mujoco_node/simulator_view_asyn.py'
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RobotSimulator')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'add_virtual_force')
        namespace = {'mujoco': mujoco, 'np': np,
                     'time': SimpleNamespace(time=lambda: 10.0), 'BAND_RELEASE_SECONDS': 0.8}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
        model = mujoco.MjModel.from_xml_path(str(WS / 'src/mujoco_node/resources/dex/mjcf/dex_29.xml'))
        data = mujoco.MjData(model)
        data.qpos[0] = 0.2
        data.qvel[4] = 1.0
        mujoco.mj_forward(model, data)
        body = model.body('waist_pitch_link').id
        band = ElasticBand()
        obj = SimpleNamespace(model=model, data=data, band_attached_link=body,
                              elastic_band=band, _band_release_t=None)
        jac = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jac, None, body)
        expected = band.Advance(data.xpos[body], jac @ data.qvel)
        namespace['add_virtual_force'](obj)
        np.testing.assert_allclose(data.xfrc_applied[body, :3], expected, atol=1e-10)
        target = np.zeros(model.nv)
        mujoco.mj_applyFT(model, data, expected, np.zeros(3), data.xpos[body], body, target)
        with self.subTest(phase='force_at_attachment_origin'):
            actual = np.zeros(model.nv)
            mujoco.mj_applyFT(model, data, data.xfrc_applied[body, :3],
                             data.xfrc_applied[body, 3:], data.xipos[body], body, actual)
            np.testing.assert_allclose(actual, target, atol=1e-10)
        torque = np.cross(data.xpos[body] - data.xipos[body], expected)
        with self.subTest(phase='half_released'):
            obj._band_release_t = 9.6
            namespace['add_virtual_force'](obj)
            np.testing.assert_allclose(data.xfrc_applied[body],
                                       np.r_[expected, torque] * 0.5, atol=1e-10)
        with self.subTest(phase='fully_released'):
            obj._band_release_t = 9.0
            namespace['add_virtual_force'](obj)
            np.testing.assert_array_equal(data.xfrc_applied[body], np.zeros(6))
            self.assertFalse(band.enable)
            self.assertIsNone(obj._band_release_t)
        with self.subTest(phase='disabled_with_stale_wrench'):
            band.enable = False
            data.xfrc_applied[body] = 1.0
            namespace['add_virtual_force'](obj)
            np.testing.assert_array_equal(data.xfrc_applied[body], np.zeros(6))

    def test_coincident_band_points_do_not_divide_by_zero(self):
        for mode in ('spring', 'elastic'):
            with self.subTest(mode=mode):
                band = ElasticBand(property=mode)
                with np.errstate(divide='raise', invalid='raise'):
                    force = band.Advance(band.point.copy(), np.zeros(3))
                np.testing.assert_array_equal(force, np.zeros(3))

    def test_only_expected_entries_are_checked_and_all_are_required(self):
        entries = ('mujoco_node/mujoco_node', 'rl_control/rl_control_node',
                   'rl_control/rl_control_node_sim', 'sim_joy/teleop_gui', 'sim_joy/teleop_joy')
        for script in ('run_sim.sh', 'run_real.sh'):
            with self.subTest(script=script), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                required = entries if script == 'run_sim.sh' else ('rl_control/rl_control_node',)
                for entry in required:
                    package, name = entry.split('/')
                    path = root / 'install' / package / 'lib' / package / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text('#!/test/venv/bin/python\n')
                unrelated = root / 'install/other/lib/other/native_node'
                unrelated.parent.mkdir(parents=True)
                unrelated.write_bytes(b'\x7fELF')
                source = (WS / script).read_text()
                self.assertIn('entry_paths=(', source)
                checks = source[source.index('entry_paths=('):source.index('# 构建前：')]
                command = checks + '\ncheck_entry_shebangs\n'
                result = subprocess.run(['bash', '-c', command], env={'WS_DIR': temp,
                    'venv_py': '/test/venv/bin/python'}, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                package, name = required[-1].split('/')
                target = root / 'install' / package / 'lib' / package / name
                target.write_text('#!/usr/bin/python3\n')
                result = subprocess.run(['bash', '-c', command], env={'WS_DIR': temp,
                    'venv_py': '/test/venv/bin/python'}, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                target.unlink()
                result = subprocess.run(['bash', '-c', command], env={'WS_DIR': temp,
                    'venv_py': '/test/venv/bin/python'}, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)

    def test_build_uses_selected_python_despite_path_and_shebang(self):
        for script in ('run_sim.sh', 'run_real.sh'):
            with self.subTest(script=script), tempfile.TemporaryDirectory() as temp:
                colcon = Path(temp) / 'colcon'
                colcon.write_text('#!/missing/system/python\nimport sys\nprint(sys.executable)\n')
                colcon.chmod(0o755)
                command = next(line for line in (WS / script).read_text().splitlines()
                               if '--symlink-install --cmake-args' in line)
                result = subprocess.run(['/bin/bash', '-c', command], env={
                    'PATH': '/nonexistent', 'venv_py': sys.executable, 'colcon_bin': str(colcon)},
                    capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), sys.executable)


if __name__ == '__main__':
    unittest.main()
