#!/usr/bin/env bash
# One interpreter selector for training, playback and export; no machine paths.
set -euo pipefail
export PYTHONNOUSERSITE=1
export PYTHONEXE="${PYTHONEXE:-$(command -v python)}"
if [[ -n "${ISAAC_SIM_ROOT:-}" ]]; then
    [[ -f "$ISAAC_SIM_ROOT/python.sh" ]] || { echo "Missing $ISAAC_SIM_ROOT/python.sh" >&2; exit 1; }
    DEX_PYTHON_STDLIB="$("$PYTHONEXE" -c 'import sysconfig; print(sysconfig.get_path("stdlib"))')"
    export PYTHONPATH="$DEX_PYTHON_STDLIB${PYTHONPATH:+:$PYTHONPATH}"
    exec bash "$ISAAC_SIM_ROOT/python.sh" "$@"
elif [[ -n "${ISAACLAB_ROOT:-}" ]]; then
    [[ -f "$ISAACLAB_ROOT/isaaclab.sh" ]] || { echo "Missing $ISAACLAB_ROOT/isaaclab.sh" >&2; exit 1; }
    exec bash "$ISAACLAB_ROOT/isaaclab.sh" -p "$@"
else
    exec "$PYTHONEXE" "$@"
fi
