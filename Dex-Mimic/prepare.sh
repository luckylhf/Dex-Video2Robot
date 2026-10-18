#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT="${1:-$SCRIPT_DIR/../Dex-GMR/data/outputs}"
cd "$SCRIPT_DIR"
python scripts/pkl_to_npz.py --robot dex_evt --batch_dir "$INPUT" \
    --raw_output_dir dataset/npz_dex_raw --output_dir dataset/npz_dex \
    --add_transition --recompute_velocities
python scripts/check_motion.py dataset/npz_dex
