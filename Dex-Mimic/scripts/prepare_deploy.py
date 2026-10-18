"""校验并打包双头 ONNX 与同模型 NPZ；不编译 ROS、不连接或启动机器人。"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from check_motion import MODEL, validate_motion
from csv_to_npz_mujoco import POLICY_BODY_NAMES, ROBOT_BODY_NAMES

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ros_dex_deploy/src/mujoco_node'))
from mujoco_node.policy_contract import validate_identity, DEPLOYMENT_CONTROL_DT
OBSERVATIONS = ['command', 'motion_anchor_ori_b', 'base_ang_vel', 'joint_pos',
                'joint_vel', 'actions', 'motion_phase', 'vel_lookahead_near', 'pos_lookahead_near']


def validate_policy(policy):
    manifest = json.loads((MODEL / 'model.json').read_text())
    model = onnx.load(str(policy))
    onnx.checker.check_model(model)
    meta = {p.key: p.value for p in model.metadata_props}
    validate_identity(meta, MODEL)
    if meta.get('model_sha256') != manifest['model_sha256']:
        raise ValueError('ONNX model identity mismatch; export with this Dex-Mimic package')
    if meta.get('quat_order') != 'wxyz':
        raise ValueError('ONNX must declare wxyz quaternions')
    bodies = meta.get('body_names', '').split(',')
    # ONNX carries the 30 policy body names (a subset of the 39 model bodies);
    # the NPZ carries all 39, so compare against the policy list, not the model.
    if (len(bodies) != 30 or bodies[0] != 'pelvis'
            or set(bodies) != set(POLICY_BODY_NAMES['dex_evt'])
            or not set(bodies) <= set(ROBOT_BODY_NAMES['dex_evt'])):
        raise ValueError('Invalid ONNX body mapping')
    names = meta.get('joint_names', '').split(',')
    if len(names) != 29 or len(set(names)) != 29 or set(names) != set(manifest['joint_names']):
        raise ValueError('ONNX must name exactly the 29 Dex joints')
    if meta.get('observation_names', '').split(',') != OBSERVATIONS:
        raise ValueError('ONNX observation order differs from the Dex FSM')
    if not np.isclose(float(meta.get('control_dt', 0)), .02) or float(meta.get('motion_fps', 0)) != 50:
        raise ValueError('Expected 50 Hz training metadata and 50 fps reference data; deployment inference is 100 Hz')
    for field in ['default_joint_pos', 'action_scale', 'joint_stiffness', 'joint_damping',
                  'joint_effort_limit', 'joint_pos_lower', 'joint_pos_upper']:
        values = np.fromstring(meta.get(field, ''), sep=',')
        if len(values) != 29 or not np.isfinite(values).all():
            raise ValueError(f'Invalid ONNX metadata: {field}')
    lo = np.fromstring(meta['joint_pos_lower'], sep=',')
    hi = np.fromstring(meta['joint_pos_upper'], sep=',')
    default = np.fromstring(meta['default_joint_pos'], sep=',')
    effort = np.fromstring(meta['joint_effort_limit'], sep=',')
    for col, name in enumerate(names):
        limit = manifest['joint_limits'][name]
        if lo[col] > hi[col] or lo[col] < float(limit['lower']) - 1e-6 or hi[col] > float(limit['upper']) + 1e-6:
            raise ValueError('Exported command limits exceed the URDF bounds')
        if default[col] < lo[col] or default[col] > hi[col] or effort[col] <= 0 or effort[col] > float(limit['effort']) + 1e-6:
            raise ValueError('Invalid default position or effort limit')
    session = ort.InferenceSession(str(policy), providers=['CPUExecutionProvider'])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if [(v.name, v.shape) for v in inputs] != [('obs', [1, 216]), ('time_step', [1, 1])]:
        raise ValueError('Deployment requires merged inputs obs(1,216), time_step(1,1)')
    if outputs[0].name != 'actions' or outputs[0].shape != [1, 29]:
        raise ValueError('Expected actions(1,29)')
    for phase in [[1., 0.], [0., 1.]]:
        obs = np.zeros((1, 216), dtype=np.float32)
        obs[0, -2:] = phase
        out = session.run(['actions'], {'obs': obs, 'time_step': np.zeros((1, 1), dtype=np.float32)})[0]
        if not np.isfinite(out).all():
            raise ValueError('Nonfinite output in standing/tracking head')
    return meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('policy', type=Path)
    parser.add_argument('motion', type=Path, help='A matching NPZ file or a directory')
    parser.add_argument('--output', type=Path, default=ROOT / 'ros_dex_deploy/src/rl_control/policy/mimic/out')
    parser.add_argument('--purpose', choices=['tracking', 'standing-demo'], default='tracking')
    args = parser.parse_args()
    meta = validate_policy(args.policy)
    files = sorted(args.motion.glob('*.npz')) if args.motion.is_dir() else [args.motion]
    if not files:
        parser.error('No reference motions to package')
    reports = [validate_motion(path) for path in files]
    for report, path in zip(reports, files):
        report['file'] = path.name
        report['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    if args.output.exists() and any(p.is_file() and p.name != 'README.md' for p in args.output.rglob('*')):
        parser.error('A bundle already exists; choose a fresh --output directory or archive the previous bundle')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'npz').mkdir(exist_ok=True)
    shutil.copy2(args.policy, args.output / 'policy_merged.onnx')
    for path in files:
        shutil.copy2(path, args.output / 'npz' / path.name)
    report = {'model_sha256': meta['model_sha256'], 'onnx_sha256': hashlib.sha256(args.policy.read_bytes()).hexdigest(),
              'contract_version': meta['contract_version'], 'asset_sha256': meta['asset_sha256'],
              'motor_profile': meta['motor_profile'],
              'purpose': args.purpose,
              'training_control_dt': float(meta['control_dt']),
              'policy_control_dt': DEPLOYMENT_CONTROL_DT, 'motions': reports, 'hardware_tested': False}
    (args.output / 'bundle.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f'Validated bundle: {args.output}; copy the full ros_dex_deploy directory after simulation acceptance.')
