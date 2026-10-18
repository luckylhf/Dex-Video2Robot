"""离线回归：锁头不丢质量、URDF/MJCF 运动学一致、三处模型一致。"""
import hashlib
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'source/whole_body_tracking/whole_body_tracking/assets/dex_evt'


class ModelTest(unittest.TestCase):
    def test_only_head_joint_types_changed(self):
        original = ET.parse(MODEL / 'urdf/tiangong3_source.urdf').getroot()
        locked = ET.parse(MODEL / 'urdf/dex_29.urdf').getroot()
        before = {j.get('name'): j for j in original.findall('joint')}
        after = {j.get('name'): j for j in locked.findall('joint')}
        changed = [n for n in before if before[n].get('type') != after[n].get('type')]
        self.assertEqual(set(changed), {'head_yaw_joint', 'head_pitch_joint'})
        self.assertEqual(sum(j.get('type') == 'revolute' for j in after.values()), 29)
        for a, b in zip(original.findall('link'), locked.findall('link')):
            self.assertEqual(ET.tostring(a.find('inertial')), ET.tostring(b.find('inertial')))

    def test_mass_and_full_inertia(self):
        urdf = ET.parse(MODEL / 'urdf/dex_29.urdf').getroot()
        xml = ET.parse(MODEL / 'mjcf/dex_29.xml').getroot()
        bodies = {b.get('name'): b for b in xml.findall('.//body')}
        for link in urdf.findall('link'):
            src = link.find('inertial')
            dst = bodies[link.get('name')].find('inertial')
            self.assertIsNotNone(dst, link.get('name'))
            self.assertAlmostEqual(float(src.find('mass').get('value')), float(dst.get('mass')))
            i = src.find('inertia').attrib
            full = [float(i[k]) for k in ['ixx', 'iyy', 'izz', 'ixy', 'ixz', 'iyz']]
            np.testing.assert_allclose(np.fromstring(dst.get('fullinertia'), sep=' '), full, atol=1e-12)
        model = mujoco.MjModel.from_xml_path(str(MODEL / 'mjcf/dex_29.xml'))
        self.assertEqual((model.nq, model.nv, model.nu), (36, 35, 29))
        self.assertAlmostEqual(sum(float(x.find('inertial/mass').get('value')) for x in urdf.findall('link')), model.body_mass.sum(), places=6)

    def test_random_pose_fk_against_urdf(self):
        model = mujoco.MjModel.from_xml_path(str(MODEL / 'mjcf/dex_29.xml'))
        data = mujoco.MjData(model)
        urdf = ET.parse(MODEL / 'urdf/dex_29.urdf').getroot()
        rng = np.random.default_rng(29)
        for _ in range(8):
            data.qpos[:] = model.qpos0
            data.qpos[:3] = rng.normal(size=3)
            quat = Rotation.random(random_state=rng).as_quat()
            data.qpos[3:7] = quat[[3, 0, 1, 2]]
            poses = {'pelvis': np.eye(4)}
            poses['pelvis'][:3, :3] = Rotation.from_quat(quat).as_matrix()
            poses['pelvis'][:3, 3] = data.qpos[:3]
            for j in urdf.findall('joint'):
                origin = j.find('origin')
                transform = np.eye(4)
                transform[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
                transform[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
                if j.get('type') != 'fixed':
                    angle = rng.uniform(-0.1, 0.1)
                    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j.get('name'))
                    data.qpos[model.jnt_qposadr[jid]] = angle
                    axis = np.fromstring(j.find('axis').get('xyz'), sep=' ')
                    transform[:3, :3] = transform[:3, :3] @ Rotation.from_rotvec(axis * angle).as_matrix()
                poses[j.find('child').get('link')] = poses[j.find('parent').get('link')] @ transform
            mujoco.mj_forward(model, data)
            for name, pose in poses.items():
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                np.testing.assert_allclose(data.xpos[bid], pose[:3, 3], atol=1e-7)
                np.testing.assert_allclose(data.xmat[bid].reshape(3, 3), pose[:3, :3], atol=1e-7)

    def test_three_copies_match(self):
        peers = [ROOT.parent / 'Dex-GMR/assets/dex', ROOT / 'ros_dex_deploy/src/mujoco_node/resources/dex']
        for subpath in ['mjcf/dex_29.xml', 'urdf/dex_29.urdf', 'model.json']:
            expected = hashlib.sha256((MODEL / subpath).read_bytes()).hexdigest()
            for peer in peers:
                self.assertEqual(expected, hashlib.sha256((peer / subpath).read_bytes()).hexdigest())

    def test_deployment_motors_and_sensors_exist(self):
        model = mujoco.MjModel.from_xml_path(str(MODEL / 'mjcf/dex_29.xml'))
        data = mujoco.MjData(model)
        for j in [model.joint(i).name for i in range(1, model.njnt)]:
            self.assertGreaterEqual(model.actuator('motor_' + j).id, 0)
            for suffix in ['_pos', '_vel', '_torque']:
                self.assertEqual(int(model.sensor(j + suffix).dim[0]), 1)
        for _ in range(10):
            mujoco.mj_step(model, data)
        self.assertTrue(np.isfinite(data.qpos).all())
        self.assertEqual(int(model.sensor('orientation').dim[0]), 4)


if __name__ == '__main__':
    unittest.main()
