#!/usr/bin/env bash
# mimic_suit_active_cable_walk_23dof — IsaacLab flat walk 학습 재개
#
# 학습 정보:
#   robot    : skeleton_torque_suit_active_cable (27-DOF, active cable)
#   simulator: IsaacLab, flat terrain
#   motions  : skeleton_torque_suit_walk.pt (1클립: walk)
#
# 재시작 (resume):
#   output_isaaclab_flat/last.ckpt 존재 시 자동 재개
#
# 실행:
#   cd /home/user/ProtoMotions
#   bash tasks/mimic_suit_active_cable_walk_23dof/train_isaaclab_resume.sh

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
    --checkpoint tasks/mimic_suit_active_cable_walk_23dof/output_isaaclab_flat/last.ckpt
