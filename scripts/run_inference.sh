#!/usr/bin/env bash
# Zero-Shot Motion Tracking - MuJoCo inference runner
# Usage: ./scripts/run_inference.sh [simulator] [checkpoint] [motion_file]
#   simulator  : mujoco (default) | newton | isaacgym | isaaclab
#   checkpoint : path to .ckpt   (default: g1-bones-deploy/last.ckpt)
#   motion_file: path to .pt     (default: g1_bones_seed_mini.pt)

set -e
cd "$(dirname "$0")/.."  # always run from repo root

# ── WSLg display ────────────────────────────────────────────────────────────
export WAYLAND_DISPLAY=wayland-0
export XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir
export DISPLAY=:0

# ── venv ────────────────────────────────────────────────────────────────────
VENV="$HOME/venv_loco"
if [ ! -f "$VENV/bin/activate" ]; then
    echo "[ERROR] venv not found at $VENV" >&2
    exit 1
fi
source "$VENV/bin/activate"

# ── arguments ───────────────────────────────────────────────────────────────
SIMULATOR="${1:-mujoco}"
CHECKPOINT="${2:-data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt}"
MOTION_FILE="${3:-data/motion_for_trackers/g1_bones_seed_mini.pt}"
NUM_ENVS=1

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Simulator : $SIMULATOR"
echo " Checkpoint: $CHECKPOINT"
echo " Motion    : $MOTION_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 protomotions/inference_agent.py \
    --checkpoint  "$CHECKPOINT" \
    --motion-file "$MOTION_FILE" \
    --simulator   "$SIMULATOR" \
    --num-envs    "$NUM_ENVS"
