"""SMPL-X/GVHMR 或 BVH → 带明确模型标识和关节名的 Dex PKL。"""
import argparse
import os
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def retarget_frames(frames, fps, source, height, output, view=False,
                    use_velocity_limit=True, use_collision_avoidance=True):
    from general_motion_retargeting import GeneralMotionRetargeting
    from general_motion_retargeting.params import ROBOT_XML_DICT
    import mujoco

    if len(frames) < 2 or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Motion needs >=2 frames and a positive fps')
    manifest = json.loads((ROOT / 'assets/dex/model.json').read_text())
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}; choose another --output')
    retarget = GeneralMotionRetargeting(source, 'dex_evt', actual_human_height=height,
                                      use_velocity_limit=use_velocity_limit,
                                      use_collision_avoidance=use_collision_avoidance,
                                      motion_fps=fps)
    viewer = None
    if view:
        from general_motion_retargeting import RobotMotionViewer
        viewer = RobotMotionViewer(robot_type='dex_evt', motion_fps=fps)
    qpos = []
    try:
        for i, frame in enumerate(frames):
            pose = retarget.retarget(frame, framesIdx=i)
            if not np.isfinite(pose).all():
                raise ValueError(f'IK frame {i} has NaN/Inf')
            if pose.shape != (36,):
                raise ValueError(f'Expected 36 qpos, got {pose.shape}')
            qpos.append(pose.copy())
            if viewer:
                viewer.step(root_pos=pose[:3], root_rot=pose[3:7], dof_pos=pose[7:], rate_limit=True)
    finally:
        if viewer:
            viewer.close()
    qpos = np.asarray(qpos)
    joint_names = retarget.robot_joint_names
    for col, name in enumerate(joint_names):
        limit = manifest['joint_limits'][name]
        if np.any(qpos[:, 7 + col] < float(limit['lower']) - 1e-5) or np.any(qpos[:, 7 + col] > float(limit['upper']) + 1e-5):
            raise ValueError(f'IK exceeds URDF position limits: {name}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as f:
        pickle.dump({'fps': float(fps), 'root_pos': qpos[:, :3],
                     'root_rot': qpos[:, 3:7][:, [1, 2, 3, 0]], 'quat_order': 'xyzw',
                     'dof_pos': qpos[:, 7:], 'joint_names': joint_names,
                     'model_sha256': manifest['model_sha256'], 'robot': manifest['robot'],
                     'local_body_pos': None, 'link_body_list': None}, f)
    print(f'Saved {output}: {len(qpos)} frames, {fps:g} fps, {len(joint_names)} joints; model={Path(ROBOT_XML_DICT["dex_evt"]).name}')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--body-models', type=Path, default=os.environ.get('SMPLX_MODEL_DIR', ROOT / 'assets/body_models'),
                        help='SMPL-X parent directory (default: bundled assets/body_models; SMPLX_MODEL_DIR overrides)')
    parser.add_argument('--bvh-format', choices=['bvh_lafan1', 'bvh_soma'], default='bvh_lafan1')
    parser.add_argument('--view', action='store_true')
    parser.add_argument('--no-velocity-limit', action='store_true',
                        help='Disable the IK velocity constraint when the QP is infeasible; '
                             'collision avoidance stays enabled and check_motion still enforces URDF limits')
    parser.add_argument('--no-collision-avoidance', action='store_true',
                        help='Disable IK self-collision pairs when the QP is infeasible; '
                             'the velocity limit stays enabled')
    args = parser.parse_args()
    if args.input.suffix.lower() == '.bvh':
        from general_motion_retargeting.utils.lafan1 import load_lafan1_file, load_soma_file
        loader = load_soma_file if args.bvh_format == "bvh_soma" else load_lafan1_file
        frames, height, fps = loader(str(args.input))
        if fps is None:
            parser.error('BVH must contain Frame Time')
        source = args.bvh_format
    else:
        from check_body_models import require_body_models
        from general_motion_retargeting.utils.smpl import (load_smplx_file, load_gvhmr_pred_file,
            get_smplx_data_offline_fast, get_gvhmr_data_offline_fast)
        if args.input.suffix.lower() == '.pt':
            require_body_models(args.body_models, ('neutral',))
            motion, body, result, height = load_gvhmr_pred_file(str(args.input), str(args.body_models))
            frames, fps = get_gvhmr_data_offline_fast(motion, body, result, tgt_fps=30)
        elif args.input.suffix.lower() == '.npz':
            with np.load(args.input, allow_pickle=True) as smplx_data:
                gender = str(smplx_data['gender']) if 'gender' in smplx_data.files else 'neutral'
            require_body_models(args.body_models, (gender,))
            motion, body, result, height = load_smplx_file(str(args.input), str(args.body_models))
            source_fps = float(np.asarray(motion['mocap_frame_rate']).item())
            frames, fps = get_smplx_data_offline_fast(motion, body, result, tgt_fps=source_fps)
        else:
            parser.error('Expected .pt, .npz, or .bvh')
        source = 'smplx'
    retarget_frames(frames, fps, source, height, args.output, args.view,
                    use_velocity_limit=not args.no_velocity_limit,
                    use_collision_avoidance=not args.no_collision_avoidance)
