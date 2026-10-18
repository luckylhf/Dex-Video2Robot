"""回归：最新模型只有网格足底时，落地高度不能为 Inf。"""
import sys
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from general_motion_retargeting import GeneralMotionRetargeting
from general_motion_retargeting.params import ROBOT_XML_DICT
from general_motion_retargeting.collision_config import COLLISION_CONFIG
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from retarget_motion import retarget_frames


class RetargetTest(unittest.TestCase):
    def test_configured_ground_geoms_are_all_active(self):
        obj = GeneralMotionRetargeting('smplx', 'dex_evt')
        self.assertEqual(len(obj.foot_ground_limit.geom_ids), 21)

    def test_missing_ground_geom_rejected(self):
        cfg = {**COLLISION_CONFIG['dex_evt'], 'foot_ground_geoms': ['missing_tcp_collision']}
        with patch.dict(COLLISION_CONFIG, {'dex_evt': cfg}):
            with self.assertRaisesRegex(ValueError, 'missing_tcp_collision'):
                GeneralMotionRetargeting('smplx', 'dex_evt')

    def test_missing_pair_geom_rejected(self):
        cfg = {**COLLISION_CONFIG['dex_evt'],
               'collision_avoidance_pairs1': [(['missing_wrist_collision'], ['pelvis_collision'])]}
        with patch.dict(COLLISION_CONFIG, {'dex_evt': cfg}):
            with self.assertRaisesRegex(ValueError, 'missing_wrist_collision'):
                GeneralMotionRetargeting('smplx', 'dex_evt')

    def test_foot_grounding_is_finite(self):
        obj = object.__new__(GeneralMotionRetargeting)
        obj.model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT['dex_evt']))
        data = mujoco.MjData(obj.model)
        mujoco.mj_forward(obj.model, data)
        obj.configuration = SimpleNamespace(data=data)
        obj._foot_geom_ids = obj._identify_foot_geoms()
        self.assertTrue(np.isfinite(obj._compute_foot_min_z()))
        self.assertAlmostEqual(obj._compute_foot_min_z(), 0, places=7)

    def test_real_ik_exports_named_pkl(self):
        root = Path(__file__).resolve().parents[1]
        cfg = json.loads((root / 'general_motion_retargeting/ik_configs/smplx_todex_v3.json').read_text())
        model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT['dex_evt']))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        pelvis = data.xpos[model.body('pelvis').id].copy()
        human_root = pelvis.copy()
        human_root[2] /= cfg['human_scale_table']['pelvis']
        frame = {}
        for table in [cfg['ik_match_table1'], cfg['ik_match_table2']]:
            for body, (human, _, _, offset, quat) in table.items():
                if human in frame:
                    continue
                bid = model.body(body).id
                rotation = Rotation.from_quat(data.xquat[bid], scalar_first=True)
                hrotation = rotation * Rotation.from_quat(quat, scalar_first=True).inv()
                target = data.xpos[bid] - rotation.apply(offset)
                pos = human_root if human == 'pelvis' else human_root + (target - pelvis) / cfg['human_scale_table'][human]
                frame[human] = (pos, hrotation.as_quat(scalar_first=True))
        with tempfile.TemporaryDirectory() as tmp:
            out = retarget_frames([frame for _ in range(8)], 30, 'smplx', None, Path(tmp) / 'ik.pkl')
            motion = pickle.loads(out.read_bytes())
            self.assertEqual(motion['dof_pos'].shape, (8, 29))
            self.assertEqual(len(set(motion['joint_names'])), 29)
            self.assertTrue(np.isfinite(motion['root_pos']).all())


if __name__ == '__main__':
    unittest.main()
