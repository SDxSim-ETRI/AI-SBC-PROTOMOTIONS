#!/usr/bin/env bash
# G1 robot - random_subset_tiny motion set (작은 모션 세트, 빠른 테스트용)

set -e
cd "$(dirname "$0")/.."

export WAYLAND_DISPLAY=wayland-0
export XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir
export DISPLAY=:0

source "$HOME/venv_loco/bin/activate"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Robot     : G1"
echo " Motion    : g1_random_subset_tiny"
echo " Simulator : mujoco"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 protomotions/inference_agent.py \
    --checkpoint  data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt \
    --motion-file data/motion_for_trackers/g1_random_subset_tiny.pt \
    --simulator   mujoco \
    --num-envs    1
