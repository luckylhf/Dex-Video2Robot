"""MP4 → GVHMR → Dex PKL → matching training NPZ, with resumable stages."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--gvhmr-root', type=Path, default=os.environ.get('GVHMR_ROOT'))
    parser.add_argument('--gvhmr-python', default=os.environ.get('GVHMR_PYTHON'))
    parser.add_argument('--body-models', type=Path, default=os.environ.get('SMPLX_MODEL_DIR', ROOT / 'assets/body_models'))
    parser.add_argument('--static-camera', action='store_true', help='Use only with a fixed camera')
    parser.add_argument('--no-velocity-limit', action='store_true')
    parser.add_argument('--no-collision-avoidance', action='store_true')
    parser.add_argument('--ik-recovery', choices=['strict', 'table-retry'], default='strict')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--mimic-dir', type=Path)
    parser.add_argument('--mimic-python', default=os.environ.get('MIMIC_PYTHON', sys.executable))
    parser.add_argument('--mimic-output-dir', type=Path, help='Optional NPZ destination instead of Mimic dataset/')
    parser.add_argument('--work-dir', type=Path, help='Explicit job directory (ASCII path required by Hydra)')
    parser.add_argument('--resume', action='store_true', help='Resume the job selected by --work-dir')
    args = parser.parse_args()
    video = args.video.resolve()
    if not video.is_file() or video.suffix.lower() != '.mp4':
        parser.error('Provide an existing MP4 input')
    if not args.gvhmr_root or not args.gvhmr_python:
        parser.error('Set GVHMR_ROOT and GVHMR_PYTHON')
    gvhmr = args.gvhmr_root.resolve()
    if not (gvhmr / 'tools/demo/demo.py').is_file():
        parser.error(f'GVHMR demo.py missing in {gvhmr}')
    if not args.body_models.is_dir():
        parser.error('SMPL-X model directory does not exist')
    # Fail fast: GVHMR takes minutes, do not start it without the body models
    from check_body_models import require_body_models
    require_body_models(args.body_models, ('neutral',))
    output = args.output.resolve() if args.output else ROOT / 'data/outputs' / (video.stem + '.pkl')
    if args.resume and not args.work_dir:
        parser.error('--resume requires --work-dir')
    if args.work_dir:
        work = args.work_dir.resolve()
        if work.exists() and any(work.iterdir()) and not args.resume:
            parser.error('Job directory is not empty; use --resume or a fresh --work-dir')
        work.mkdir(parents=True, exist_ok=True)
    else:
        (ROOT / 'data/work').mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix='video_', dir=ROOT / 'data/work'))
    if not str(work).isascii():
        parser.error('GVHMR/Hydra requires an ASCII --work-dir, e.g. /tmp/dex_video_job')
    manifest = json.loads((ROOT / 'assets/dex/model.json').read_text())
    model_file = ROOT / 'assets/dex/mjcf/dex_29.xml'
    if hashlib.sha256(model_file.read_bytes()).hexdigest() != manifest['model_sha256']:
        parser.error('MJCF differs from model.json; rebuild the model first')
    identity = dict(video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                    model_sha256=manifest['model_sha256'], output=str(output),
                    static_camera=args.static_camera, no_velocity_limit=args.no_velocity_limit,
                    no_collision_avoidance=args.no_collision_avoidance,
                    ik_recovery=args.ik_recovery,
                    body_models=str(args.body_models.resolve()))
    state = work / 'pipeline.json'
    if args.resume:
        if not state.is_file() or json.loads(state.read_text()) != identity:
            parser.error('Resume identity differs (video, model, output or IK settings); use a fresh job')
    elif output.exists():
        parser.error(f'Output already exists: {output}; choose another --output')
    state.write_text(json.dumps(identity, ensure_ascii=False, indent=2)+'\n')
    print(f'[pipeline] Work directory: {work}', flush=True)
    env = os.environ.copy()
    env['PYTHONNOUSERSITE'] = '1'
    normalized = work / 'source_30fps.mp4'
    if not (args.resume and normalized.is_file()):
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(video), '-an', '-vf', 'fps=30',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(normalized)], check=True, env=env)
    prediction = work / 'human/source_30fps/hmr4d_results.pt'
    if not (args.resume and prediction.is_file()):
        command = [args.gvhmr_python, 'tools/demo/demo.py', '--video', str(normalized),
                   '--output_root', str(work / 'human')]
        if args.static_camera:
            command.append('-s')
        try:
            subprocess.run(command, cwd=gvhmr, check=True, env=env)
        except subprocess.CalledProcessError:
            if not prediction.is_file():
                raise
            print('[pipeline] GVHMR prediction exists; optional renderer failed. Continuing with the prediction.', flush=True)
    if not prediction.is_file():
        raise FileNotFoundError(f'GVHMR prediction missing: {prediction}')
    if not (args.resume and output.is_file()):
        retarget = [sys.executable, str(ROOT / 'scripts/retarget_motion.py'), str(prediction),
                    '--body-models', str(args.body_models.resolve()), '--output', str(output)]
        if args.no_velocity_limit:
            retarget.append('--no-velocity-limit')
        if args.no_collision_avoidance:
            retarget.append('--no-collision-avoidance')
        retarget.extend(['--ik-recovery', args.ik_recovery])
        subprocess.run(retarget, check=True, env=env)
    if args.mimic_dir:
        mimic = args.mimic_dir.resolve()
        destination = args.mimic_output_dir.resolve() if args.mimic_output_dir else mimic / 'dataset'
        subprocess.run([args.mimic_python, str(mimic / 'scripts/pkl_to_npz.py'), '--input_file', str(output),
                        '--robot', 'dex_evt', '--output_name', str(destination / 'npz_dex_raw' / output.stem),
                        '--output_dir', str(destination / 'npz_dex'), '--add_transition', '--recompute_velocities'],
                       cwd=mimic, check=True, env=env)
        subprocess.run([args.mimic_python, str(mimic / 'scripts/check_motion.py'),
                        str(destination / 'npz_dex' / (output.stem + '_with_transition.npz'))],
                       cwd=mimic, check=True, env=env)
    print(f'Video pipeline completed. PKL: {output}; job: {work}', flush=True)


if __name__ == '__main__':
    main()
