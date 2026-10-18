"""训练/打包前检查最新 Dex NPZ；验证名称、模型、帧率、数值、限位及抽样 FK。"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from csv_to_npz_mujoco import ROBOT_MJCF, ROBOT_JOINT_NAMES, ROBOT_BODY_NAMES, run_fk

MODEL = Path(ROBOT_MJCF['dex_evt']).parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ros_dex_deploy/src/mujoco_node'))
from mujoco_node.policy_contract import verify_model


def validate_motion(path):
    manifest = verify_model(MODEL)
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    if str(data.get('model_sha256', '')) != manifest['model_sha256']:
        raise ValueError(f'{path}: robot model mismatch')
    if float(data['fps']) != 50 or str(data.get('quat_order', '')) != 'wxyz':
        raise ValueError(f'{path}: deployment requires 50 fps and wxyz quaternions')
    joints, bodies = data['joint_names'].tolist(), data['body_names'].tolist()
    if len(joints) != 29 or len(set(joints)) != 29 or set(joints) != set(ROBOT_JOINT_NAMES['dex_evt']):
        raise ValueError(f'{path}: invalid joint names')
    n_bodies = len(ROBOT_BODY_NAMES['dex_evt'])
    if len(bodies) != n_bodies or bodies[0] != 'pelvis' or set(bodies) != set(ROBOT_BODY_NAMES['dex_evt']):
        raise ValueError(f'{path}: invalid body names')
    n = len(data['joint_pos'])
    shapes = {'joint_pos': (n, 29), 'joint_vel': (n, 29), 'body_pos_w': (n, n_bodies, 3),
              'body_quat_w': (n, n_bodies, 4), 'body_lin_vel_w': (n, n_bodies, 3),
              'body_ang_vel_w': (n, n_bodies, 3)}
    if n < 2:
        raise ValueError(f'{path}: too few frames')
    for name, shape in shapes.items():
        if data[name].shape != shape or not np.isfinite(data[name]).all():
            raise ValueError(f'{path}: invalid {name}')
    if not np.allclose(np.linalg.norm(data['body_quat_w'], axis=-1), 1, atol=1e-3):
        raise ValueError(f'{path}: invalid quaternion norm')
    for col, name in enumerate(joints):
        lim = manifest['joint_limits'][name]
        q = data['joint_pos'][:, col]
        if np.any(q < float(lim['lower']) - 1e-4) or np.any(q > float(lim['upper']) + 1e-4):
            raise ValueError(f'{path}: joint position exceeds URDF limits: {name}')
        if np.max(np.abs(data['joint_vel'][:, col])) > float(lim['velocity']) + 1e-3:
            raise ValueError(f'{path}: joint velocity exceeds URDF limits: {name}; slow the source motion')
    idx = np.unique(np.linspace(0, n - 1, min(n, 12), dtype=int))
    pos, quat = run_fk(data['body_pos_w'][idx, 0], data['body_quat_w'][idx, 0],
                       data['joint_pos'][idx], joints, bodies, ROBOT_MJCF['dex_evt'])
    error = float(np.max(np.abs(pos - data['body_pos_w'][idx])))
    rotation_error = float(np.max(1 - np.abs(np.sum(quat * data['body_quat_w'][idx], axis=-1))))
    if error > 1e-5 or rotation_error > 1e-5:
        raise ValueError(f'{path}: FK mismatch ({error:g} m, quaternion error {rotation_error:g})')
    return {'file': str(path), 'frames': n, 'fps': 50, 'duration_s': n/50,
            'fk_max_error_m': error, 'model_sha256': manifest['model_sha256']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('motion', type=Path)
    args = parser.parse_args()
    files = sorted(args.motion.glob('*.npz')) if args.motion.is_dir() else [args.motion]
    if not files:
        parser.error('No NPZ files found')
    for path in files:
        print(json.dumps(validate_motion(path), ensure_ascii=False))
