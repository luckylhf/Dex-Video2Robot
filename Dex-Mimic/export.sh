#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${STANDING_CHECKPOINT:?Set STANDING_CHECKPOINT explicitly; see README section 4}"
HEADLESS=1 MAX_STEPS=1 exec bash "$SCRIPT_DIR/play_dex.sh"
