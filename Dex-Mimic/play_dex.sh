#!/usr/bin/env bash
# Isaac Lab 回放并导出；checkpoint 与动作必须由调用者明确指定。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
: "${MOTION_FILE:?Set MOTION_FILE to the matching NPZ file}"
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the tracking checkpoint}"
[[ -f "$CHECKPOINT_PATH" ]] || { echo "Checkpoint not found: $CHECKPOINT_PATH" >&2; exit 1; }
bash scripts/isaac_python.sh scripts/check_motion.py "$MOTION_FILE"
ARGS=(--task Tracking-Flat-DexEVT-Simple-v0 --motion_file "$MOTION_FILE"
      --checkpoint_path "$CHECKPOINT_PATH" --num_envs "${NUM_ENVS:-1}" --start_first_frame)
if [[ -n "${STANDING_CHECKPOINT:-}" ]]; then
    [[ -f "$STANDING_CHECKPOINT" ]] || { echo "Standing checkpoint not found" >&2; exit 1; }
    ARGS+=(--standing_checkpoint "$STANDING_CHECKPOINT")
fi
if [[ "${HEADLESS:-0}" == 1 ]]; then ARGS+=(--headless); fi
if [[ -n "${MAX_STEPS:-}" ]]; then ARGS+=(--max_steps "$MAX_STEPS"); fi
if [[ "${VIDEO:-0}" == 1 ]]; then ARGS+=(--video --video_length "${VIDEO_LENGTH:-500}"); fi
exec bash scripts/isaac_python.sh scripts/rsl_rl/play.py "${ARGS[@]}"
