#!/usr/bin/env bash
set -euo pipefail
DEX_MIMIC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DEX_MIMIC_ROOT"
exec bash scripts/isaac_python.sh "$DEX_MIMIC_ROOT/scripts/validate_standing_isaac.py" "$@"
