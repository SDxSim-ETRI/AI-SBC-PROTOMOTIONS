#!/usr/bin/env bash
# mimic_suit_active_cable_walk_23dof — IsaacLab flat walk 최초 학습 (Newton → IsaacLab sim-to-sim)
#
# warm_start: tasks/mimic_suit_active_cable_walk_23dof/output_newton_flat/score_based.ckpt
#
# 주의: output_isaaclab_flat/last.ckpt 가 없어야 warm_start 적용됨
#
# 실행:
#   cd /home/user/ProtoMotions
#   bash tasks/mimic_suit_active_cable_walk_23dof/train_isaaclab_initial.sh

set -euo pipefail

PYTHON=/home/user/venv_isaaclab/bin/python
cd "$(dirname "$(realpath "$0")")/../.."

$PYTHON protomotions/train_agent.py \
    --robot-name skeleton_torque_suit_active_cable \
    --simulator isaaclab \
    --experiment-path examples/experiments/mimic_isaaclab_active_cable/mlp.py \
    --experiment-name mimic_suit_active_cable_walk_23dof_isaaclab_flat \
    --save-dir tasks/mimic_suit_active_cable_walk_23dof/output_isaaclab_flat \
    --motion-file data/motion_for_trackers/skeleton_torque_suit_walk.pt \
    --num-envs 4096 \
    --batch-size 16384 \
    --checkpoint tasks/mimic_suit_active_cable_walk_23dof/output_newton_flat/score_based.ckpt
