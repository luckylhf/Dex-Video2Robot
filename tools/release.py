#!/usr/bin/env python3
"""Build a source release containing the two packages and public documentation."""
import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'.git', '.venv', '__pycache__', '.pytest_cache', 'build', 'install', 'log',
            'logs', 'validation', 'dist', '.egg-info', '.DS_Store'}


def release_files(include_data=False):
    for name in ['.gitignore', 'README.md', 'docs', 'tools', 'Dex-GMR', 'Dex-Mimic']:
        entry = ROOT / name
        for path in ([entry] if entry.is_file() else sorted(entry.rglob('*'))):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(ROOT)
            if relative.parts[:2] == ('Dex-Mimic', 'outputs'):
                continue
            if any(p in EXCLUDED or p.endswith('.egg-info') for p in relative.parts):
                continue
            if path.suffix in {'.pyc', '.usd', '.usda', '.usdc', '.back'} or path.name == 'MUJOCO_LOG.TXT':
                continue
            if relative.parts[:2] == ('Dex-GMR', 'data') and not include_data:
                continue
            if relative.parts[:3] == ('Dex-GMR', 'assets', 'body_models') and path.suffix == '.npz':
                continue
            yield path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-data', action='store_true', help='Include preserved GMR user videos and job data')
    args = parser.parse_args()
    args.output = args.output.expanduser().resolve()
    if args.output.exists():
        parser.error('Output already exists; choose a fresh filename')
    subprocess.run([sys.executable, str(ROOT / 'tools/check_release.py')], check=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    entries = list(release_files(args.include_data))
    manifest = {}
    with tarfile.open(args.output, 'w:gz') as archive:
        for path in entries:
            relative = path.relative_to(ROOT).as_posix()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest[relative] = digest
            archive.add(path, arcname='dex-pipeline/' + relative, recursive=False)
    manifest_file = args.output.with_suffix(args.output.suffix + '.manifest.json')
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(f'{args.output}: {len(entries)} files; manifest: {manifest_file}')


if __name__ == '__main__':
    main()
