"""回归：Isaac 保留圆柱，standing 按真实倾斜脚底对齐。"""
import ast
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from csv_to_npz_mujoco import _build_standing_pose_fk
from sim2sim_dex import MujocoSim2Sim, DEFAULT_MJCF, FOOT_GEOM_NAMES


class FootGeometryTest(unittest.TestCase):
    def test_sim2sim_resolves_all_fourteen_cylinders(self):
        sim = object.__new__(MujocoSim2Sim)
        sim.model = mujoco.MjModel.from_xml_path(str(DEFAULT_MJCF))
        self.assertEqual(len(sim._build_foot_geom_ids()), 14)
        with patch('sim2sim_dex.FOOT_GEOM_NAMES', FOOT_GEOM_NAMES + ('missing_foot',)):
            with self.assertRaisesRegex(RuntimeError, 'expected all fourteen'):
                sim._build_foot_geom_ids()

    def test_isaac_preserves_cylinders(self):
        tree = ast.parse((ROOT / 'source/whole_body_tracking/whole_body_tracking/robots/dex_evt.py').read_text())
        values = [ast.literal_eval(n.value) for n in ast.walk(tree)
                  if isinstance(n, ast.keyword) and n.arg == 'replace_cylinders_with_capsules']
        self.assertEqual(values, [False])

    def test_tilted_standing_height_matches_mujoco_distance(self):
        xml = '''<mujoco><compiler angle="radian"/><worldbody>
          <geom name="floor" type="plane" size="5 5 .1"/>
          <body name="pelvis"><freejoint/><inertial mass="1" pos="0 0 0" diaginertia="1 1 1"/>
            <body name="ankle_roll_l_link" pos="0 0 -.8">
              <joint name="ankle_pitch_l_joint" axis="0 1 0"/>
              <geom name="sole" type="cylinder" size=".012 .12" euler="0 1.5707963267948966 0" group="1"/>
            </body></body></worldbody></mujoco>'''
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'cylinder.xml'
            path.write_text(xml)
            _, _, _, got = _build_standing_pose_fk(['pelvis', 'ankle_roll_l_link'],
                ['ankle_pitch_l_joint'], 1., str(path), standing_joint_pos_map={'ankle_pitch_l_joint': .6})
            model = mujoco.MjModel.from_xml_path(str(path))
            data = mujoco.MjData(model)
            data.qpos[2] = 1.
            data.qpos[3] = 1.
            data.qpos[7] = .6
            mujoco.mj_forward(model, data)
            expected = mujoco.mj_geomDistance(model, data, model.geom('sole').id,
                model.geom('floor').id, 10., np.zeros(6))
            self.assertAlmostEqual(got, expected, places=8)


if __name__ == '__main__':
    unittest.main()
