"""Subprocess boundary between the legacy GMR environment and Dex-Retarget."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def add_options(parser):
    parser.add_argument('--retarget-backend',choices=['gmr','dex-retarget'],default='gmr')
    parser.add_argument('--dex-retarget-root',type=Path,default=os.environ.get('DEX_RETARGET_ROOT'))
    parser.add_argument('--dex-retarget-python',default=os.environ.get('DEX_RETARGET_PYTHON'))
    parser.add_argument('--retarget-mode',choices=['auto','frax','hybrid','v8'],default='auto')
    parser.add_argument('--retarget-budget',type=float,default=86400.)
    parser.add_argument('--grounding',choices=['off','contact'],default='off')
    parser.add_argument('--smplx-python',default=os.environ.get('SMPLX_PYTHON',sys.executable))


def environment(args):
    if not args.dex_retarget_root:raise ValueError('Set --dex-retarget-root or DEX_RETARGET_ROOT')
    root=Path(args.dex_retarget_root).resolve()
    if not (root/'src/dex_frax/smplx_cli.py').is_file():raise ValueError('Dex-Retarget lacks the native SMPL-X adapter; install the matching version')
    python=str(args.dex_retarget_python or root/'.venv/bin/python')
    env=os.environ.copy();env['PYTHONNOUSERSITE']='1'
    env['PYTHONPATH']=str(root/'src')
    return root,python,env


def identity(args):
    root,python,env=environment(args)
    # All implementation, robot assets and runtime versions are pinned for resume.
    code='from dex_frax.cli import static_identity; import json; print(json.dumps(static_identity(),sort_keys=True))'
    raw=subprocess.check_output([python,'-c',code],env=env,text=True)
    ident=json.loads(raw)
    runtime_code='import importlib.metadata as m,json; print(json.dumps({k:m.version(k) for k in ("torch","smplx","numpy","scipy")}))'
    human_runtime=json.loads(subprocess.check_output([str(args.smplx_python),'-c',runtime_code],env=env,text=True))
    model=json.loads((ROOT/'assets/dex/model.json').read_text())
    if ident['assets'].get('mjcf/dex_29.xml')!=model['model_sha256']:
        raise ValueError('Dex-Retarget and Dex-GMR use different robot models')
    return dict(root=str(root),python=python,implementation=ident,mode=args.retarget_mode,
                budget=args.retarget_budget,grounding=args.grounding,smplx_python=str(args.smplx_python),human_runtime=human_runtime,
                bridge_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                quality_sha256=hashlib.sha256((Path(__file__).parent/'motion_quality.py').read_bytes()).hexdigest(),
                body_models_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted((Path(args.body_models)/'smplx').glob('SMPLX_*.npz'))})


def run(args,prediction,output,work,*,input_fps=30.,up_axis=None,units=None,max_frames=None):
    import numpy as np
    from motion_quality import sha,validate_payload,report
    root,python,env=environment(args);folder=Path(work)/'dex_retarget'
    command=[python,'-m','dex_frax','convert',str(prediction),str(folder),'--source','smplx',
             '--mode',args.retarget_mode,'--body-models',str(Path(args.body_models).resolve()),
             '--smplx-python',str(args.smplx_python),'--grounding',args.grounding,'--budget',str(args.retarget_budget)]
    for flag,value in [('--input-fps',input_fps),('--up-axis',up_axis),('--units',units),('--max-frames',max_frames)]:
        if value is not None:command.extend([flag,str(value)])
    if folder.exists():
        # Failed stages are evidence, not a reusable completed conversion.
        runfile=folder/'run.json'
        complete=json.loads(runfile.read_text()) if runfile.is_file() else {}
        if complete.get('status')!='complete':
            raise ValueError(f'Incomplete backend run preserved at {folder}; use a fresh work directory')
    else:
        subprocess.run(command,env=env,check=True)
    if args.retarget_mode=='auto':
        selection=json.loads((folder/'selection.json').read_text())
        selected=selection.get('selected_output') or selection.get('preview_output')
        if selected not in ('frax','hybrid','v8'):raise ValueError('No complete Dex-Retarget result; inspect backend logs')
        stage=folder/selected
    else:stage=folder
    record=json.loads((stage/'run.json').read_text())
    if record.get('status')!='complete':raise ValueError('Dex-Retarget stage is incomplete')
    for name in ('motion.npz','quality.json','identity.json'):
        if sha(stage/name)!=record['files'][name]:raise ValueError('Backend artifact hash mismatch: '+name)
    source_identity=json.loads((stage/'identity.json').read_text())
    if source_identity['input_sha256']!=sha(prediction):
        raise ValueError('Human prediction changed after the backend conversion')
    with np.load(stage/'motion.npz',allow_pickle=False) as data:
        payload={k:data[k] for k in data.files}
    for key in ('fps','quat_order','model_sha256','robot'):payload[key]=np.asarray(payload[key]).item()
    payload['joint_names']=payload['joint_names'].tolist()
    manifest=json.loads((ROOT/'assets/dex/model.json').read_text())
    validate_payload(payload,manifest)
    backend_quality=json.loads((stage/'quality.json').read_text())
    payload.update(retarget_backend='dex-retarget',quality_status=backend_quality['status'],
                   quality_flags=backend_quality['flags'],backend_fingerprint=record['fingerprint'])
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream:pickle.dump(payload,stream,protocol=4)
    quality=report(output,ROOT/'assets/dex/mjcf/dex_29.xml',manifest,'dex-retarget',backend_quality)
    output.with_suffix('.backend.json').write_text(json.dumps(dict(backend='dex-retarget',stage=str(stage),
            fingerprint=record['fingerprint'],quality_status=quality['status'],source_sha256=sha(prediction),
            output_sha256=sha(output),quality_sha256=sha(output.with_suffix('.quality.json'))),ensure_ascii=False,indent=2)+'\n')
    return output


def verify_export(output,prediction):
    from motion_quality import sha
    output=Path(output)
    receipt=json.loads(output.with_suffix('.backend.json').read_text())
    if receipt['source_sha256']!=sha(prediction) or receipt['output_sha256']!=sha(output):
        raise ValueError('Backend export source or PKL changed; use a fresh job')
    if receipt['quality_sha256']!=sha(output.with_suffix('.quality.json')):
        raise ValueError('Backend quality report changed; use a fresh job')
