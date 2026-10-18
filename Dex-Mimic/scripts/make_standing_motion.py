"""为同一 tracking 任务生成最新版 Dex 的静态参考，供训练新的 standing 头。"""
import json
from pathlib import Path
import numpy as np
from csv_to_npz_mujoco import ROBOT_JOINT_NAMES, ROBOT_BODY_NAMES, ROBOT_MJCF, STANDING_JOINT_POS_MAP, _build_standing_pose_fk

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    names, bodies = ROBOT_JOINT_NAMES['dex_evt'], ROBOT_BODY_NAMES['dex_evt']
    q, pos, quat, foot_z = _build_standing_pose_fk(bodies, names, 1.0, ROBOT_MJCF['dex_evt'],
        standing_joint_pos_map=STANDING_JOINT_POS_MAP['dex_evt'])
    pos[:, 2] -= foot_z
    n = 250
    out = root / 'dataset/standing/standing.npz'
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(ROBOT_MJCF['dex_evt']).parents[1] / 'model.json').read_text())
    np.savez(out, fps=50, joint_names=names, body_names=bodies,
        joint_pos=np.tile(q, (n, 1)), joint_vel=np.zeros((n, 29), dtype=np.float32),
        body_pos_w=np.tile(pos, (n, 1, 1)), body_quat_w=np.tile(quat, (n, 1, 1)),
        body_lin_vel_w=np.zeros((n, len(bodies), 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((n, len(bodies), 3), dtype=np.float32),
        model_sha256=manifest['model_sha256'], quat_order='wxyz')
    print(out)
