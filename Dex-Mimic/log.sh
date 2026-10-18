#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec tensorboard --logdir "$SCRIPT_DIR/logs/rsl_rl/dex_evt_fix" "$@"
