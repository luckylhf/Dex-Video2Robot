#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
NPZ_DIR="${NPZ_DIR:-dataset/npz_dex}"
python scripts/check_motion.py "$NPZ_DIR"
exec python scripts/replay_npz_mujoco.py --npz-dir "$NPZ_DIR" "$@"
