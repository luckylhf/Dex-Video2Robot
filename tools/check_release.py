#!/usr/bin/env python3
"""Read-only source, documentation and robot-asset checks; standard library only."""
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path

from release import ROOT, release_files


def main():
    files = list(release_files())
    errors = []
    for path in files:
        try:
            if path.suffix == '.py':
                ast.parse(path.read_text(), filename=str(path))
            elif path.suffix == '.sh':
                result = subprocess.run(['bash', '-n', str(path)], capture_output=True, text=True)
                if result.returncode:
                    errors.append(result.stderr.strip())
            elif path.suffix == '.md':
                for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)', path.read_text()):
                    if link.startswith(('https://', 'http://', '#', 'mailto:')):
                        continue
                    target = link.split('#')[0].split(' "')[0].strip('<>')
                    if target and not (path.parent / target).exists():
                        errors.append(f'{path.relative_to(ROOT)}: missing link {link}')
        except (SyntaxError, UnicodeError) as exc:
            errors.append(str(exc))
    manifests = []
    for directory in [ROOT/'Dex-GMR/assets/dex',
                      ROOT/'Dex-Mimic/source/whole_body_tracking/whole_body_tracking/assets/dex_evt',
                      ROOT/'Dex-Mimic/ros_dex_deploy/src/mujoco_node/resources/dex']:
        manifest = json.loads((directory/'model.json').read_text())
        actual = hashlib.sha256((directory/'mjcf/dex_29.xml').read_bytes()).hexdigest()
        if actual != manifest['model_sha256']:
            errors.append(f'{directory}: stale model.json')
        manifests.append(manifest)
    if not all(m == manifests[0] for m in manifests):
        errors.append('Model manifests differ across GMR / Isaac / ROS')
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'Release source checks passed: {len(files)} files; Python/shell syntax, local documentation links and model identity')


if __name__ == '__main__':
    main()
