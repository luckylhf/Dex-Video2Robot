#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONNOUSERSITE=1
exec "${PYTHON:-python}" "$SCRIPT_DIR/scripts/video_to_dex.py" --mimic-dir "$SCRIPT_DIR/../Dex-Mimic" "$@"
