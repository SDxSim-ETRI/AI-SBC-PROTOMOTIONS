#!/usr/bin/env bash
# mimic_suit_active_cable_motions14_23dof — IsaacLab active cable 최초 학습 명령
#
# warm_start: output_newton/score_based.ckpt (Newton → IsaacLab sim-to-sim)
#
# 주의: output_isaaclab/last.ckpt 가 없어야 warm_start 적용됨
#
# 실행:
#   cd /home/user/ProtoMotions
#   bash tasks/mimic_suit_active_cable_motions14_23dof/train_isaaclab_initial.sh

set -euo pipefail

PYTHON=/home/user/miniforge3/envs/env_isaaclab/bin/python
cd "$(dirname "$(realpath "$0")")/../.."

$PYTHON protomotions/train_agent.py \
    --robot-name skeleton_torque_suit_active_cable \
    --simulator isaaclab \
    --experiment-path examples/experiments/mimic_isaaclab_active_cable/mlp.py \
    --experiment-name mimic_suit_active_cable_motions14_23dof_isaaclab \
    --save-dir tasks/mimic_suit_active_cable_motions14_23dof/output_isaaclab \
    --motion-file data/motion_for_trackers/skeleton_torque_suit_motions14.pt \
    --num-envs 4096 \
    --batch-size 16384 \
    --checkpoint tasks/mimic_suit_active_cable_motions14_23dof/output_newton/score_based.ckpt
