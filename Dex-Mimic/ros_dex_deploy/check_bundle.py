"""启动前只读核对部署包；不导入硬件接口，也不发送控制指令。"""
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src/mujoco_node'))
from mujoco_node.policy_contract import validate_identity, resolve_mimic_config, validate_deployment_timing

if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    bundle = Path(os.environ.get('MIMIC_BUNDLE_DIR', root / 'src/rl_control/policy/mimic/out'))
    model_dir = root / 'src/mujoco_node/resources/dex'
    manifest = json.loads((model_dir / 'model.json').read_text())
    if not (bundle / 'bundle.json').is_file():
        raise SystemExit(f'No prepared bundle at {bundle}. Run scripts/prepare_deploy.py first, or set MIMIC_BUNDLE_DIR.')
    report = json.loads((bundle / 'bundle.json').read_text())
    policy = bundle / 'policy_merged.onnx'
    if report['model_sha256'] != manifest['model_sha256']:
        raise ValueError('Deployment model differs from the prepared bundle')
    if hashlib.sha256(policy.read_bytes()).hexdigest() != report['onnx_sha256']:
        raise ValueError('Policy file has changed since preparation')
    session = ort.InferenceSession(str(policy), providers=['CPUExecutionProvider'])
    meta = session.get_modelmeta().custom_metadata_map
    validate_identity(meta, model_dir)
    if any(report.get(key) != meta.get(key) for key in ('contract_version', 'asset_sha256', 'motor_profile')):
        raise ValueError('Bundle dynamics contract differs from its ONNX policy')
    if meta.get('model_sha256') != report['model_sha256'] or not np.isclose(float(meta.get('control_dt', 0)), .02):
        raise ValueError('Policy identity or training timing mismatch')
    cfg = resolve_mimic_config(root / 'src/rl_control/policy/mimic/config/mimic.yaml')
    period = validate_deployment_timing(meta, cfg['physical_dt'], cfg['decimation'])
    if (not np.isclose(float(report.get('training_control_dt', 0)), float(meta['control_dt']))
            or not np.isclose(float(report.get('policy_control_dt', 0)), period)):
        raise ValueError('Bundle deployment timing differs from configuration; prepare the bundle for 100 Hz inference')
    expected = {x['file'] for x in report['motions']}
    if {p.name for p in (bundle / 'npz').glob('*.npz')} != expected:
        raise ValueError('Reference motion file list differs from bundle.json')
    for item in report['motions']:
        path = bundle / 'npz' / item['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f'Reference motion changed: {path.name}')
        with np.load(path, allow_pickle=False) as data:
            if str(data['model_sha256']) != report['model_sha256'] or float(data['fps']) != 50:
                raise ValueError(f'Wrong model or fps: {path.name}')
    print(f'Bundle OK: {len(expected)} reference clips; inference {1/period:g} Hz; training 50 Hz; model {report["model_sha256"]}')
