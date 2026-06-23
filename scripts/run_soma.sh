#!/usr/bin/env bash
# SOMA skeleton - bones_seed_mini motion set

set -e
cd "$(dirname "$0")/.."

export WAYLAND_DISPLAY=wayland-0
export XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir
export DISPLAY=:0

source "$HOME/venv_loco/bin/activate"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Robot     : SOMA"
echo " Motion    : soma23_bones_seed_mini"
echo " Simulator : mujoco"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 protomotions/inference_agent.py \
    --checkpoint  data/pretrained_models/motion_tracker/soma-bones/last.ckpt \
    --motion-file data/motion_for_trackers/soma23_bones_seed_mini.pt \
    --simulator   mujoco \
    --num-envs    1
