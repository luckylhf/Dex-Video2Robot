"""回归：源圆柱脚底、倾斜最低点与实际支撑点。"""
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from general_motion_retargeting import GeneralMotionRetargeting
from general_motion_retargeting.collision_config import COLLISION_CONFIG
from general_motion_retargeting.geom_distance_limit import FootGroundContactLimit


class FootGeometryTest(unittest.TestCase):
    def test_urdf_and_mjcf_have_matching_seven_cylinders(self):
        urdf = ET.parse(ROOT / 'assets/dex/urdf/dex_29.urdf').getroot()
        model = mujoco.MjModel.from_xml_path(str(ROOT / 'assets/dex/mjcf/dex_29.xml'))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        for side in ['l', 'r']:
            name = f'ankle_roll_{side}_link'
            collisions = urdf.find(f"./link[@name='{name}']").findall('collision')
            self.assertEqual(len(collisions), 7)
            bid = model.body(name).id
            ids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid and model.geom_group[g] == 1]
            self.assertEqual(len(ids), 7)
            for collision, gid in zip(collisions, ids):
                self.assertEqual(model.geom_type[gid], mujoco.mjtGeom.mjGEOM_CYLINDER)
                cylinder = collision.find('geometry/cylinder')
                self.assertIsNotNone(cylinder)
                origin = collision.find('origin')
                np.testing.assert_allclose(model.geom_pos[gid], np.fromstring(origin.get('xyz'), sep=' '), atol=1e-10)
                np.testing.assert_allclose(model.geom_size[gid, :2],
                    [float(cylinder.get('radius')), float(cylinder.get('length')) / 2], atol=1e-10)
                axis = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy'), sep=' ')).as_matrix()[:, 2]
                actual = Rotation.from_quat(model.geom_quat[gid], scalar_first=True).as_matrix()[:, 2]
                self.assertLess(min(np.linalg.norm(axis - actual), np.linalg.norm(axis + actual)), 1e-6)
                witness = np.zeros(6)
                z = mujoco.mj_geomDistance(model, data, gid, model.geom('floor').id, 10., witness)
                self.assertGreaterEqual(z, -1e-8)
            self.assertLess(min(mujoco.mj_geomDistance(model, data, g, model.geom('floor').id, 10., np.zeros(6))
                                for g in ids), 1e-8)

    def test_all_cylinders_are_constrained(self):
        cfg = COLLISION_CONFIG['dex_evt']
        self.assertEqual(len(cfg['foot_ground_geoms']), 21)
        for side in ['l', 'r']:
            for i in range(7):
                self.assertIn(f'ankle_roll_{side}_link_collision_{i}', cfg['foot_ground_geoms'])

    def test_tilted_cylinder_height_and_support_point(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <geom name="floor" type="plane" size="5 5 .1"/>
          <body name="ankle_roll_l_link" pos="0 0 1"><freejoint/>
            <geom name="sole" type="cylinder" size=".012 .12" euler="0 40 0"/>
          </body></worldbody></mujoco>''')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        gid = model.geom('sole').id
        expected = mujoco.mj_geomDistance(model, data, gid, model.geom('floor').id, 10., np.zeros(6))
        obj = object.__new__(GeneralMotionRetargeting)
        obj.model = model
        obj.configuration = SimpleNamespace(data=data)
        obj._foot_geom_ids = [gid]
        self.assertAlmostEqual(obj._compute_foot_min_z(), expected, places=8)
        limit = FootGroundContactLimit(model, ['sole'])
        z, offset = limit._lowest_point(model, data, gid)
        self.assertAlmostEqual(z, expected, places=8)
        local = data.geom_xmat[gid].reshape(3, 3).T @ offset
        self.assertAlmostEqual(abs(local[2]), .12, places=8)
        self.assertAlmostEqual(np.linalg.norm(local[:2]), .012, places=8)

    def test_support_point_jacobian_matches_finite_difference(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <geom name="floor" type="plane" size="5 5 .1"/>
          <body pos="0 0 1"><freejoint/>
            <geom name="sole" type="cylinder" size=".012 .12" euler="0 40 0"/>
          </body></worldbody></mujoco>''')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        limit = FootGroundContactLimit(model, ['sole'], margin=2.)
        constraint = limit.compute_qp_inequalities(SimpleNamespace(model=model, data=data), .02)
        original = data.qpos.copy()
        numerical = []
        for axis in range(model.nv):
            distances = []
            for sign in [-1, 1]:
                data.qpos[:] = original
                velocity = np.zeros(model.nv)
                velocity[axis] = sign
                mujoco.mj_integratePos(model, data.qpos, velocity, 1e-6)
                mujoco.mj_forward(model, data)
                distances.append(mujoco.mj_geomDistance(model, data, model.geom('sole').id,
                    model.geom('floor').id, 10., np.zeros(6)))
            numerical.append((distances[1] - distances[0]) / 2e-6)
        np.testing.assert_allclose(-constraint.G[0], numerical, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
