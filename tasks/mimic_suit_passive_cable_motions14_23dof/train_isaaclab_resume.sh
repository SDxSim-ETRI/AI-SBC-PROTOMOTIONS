#!/usr/bin/env bash
# mimic_suit_passive_cable_motions14_23dof — IsaacLab suit passive cable 학습 (14모션)
#
# 학습 정보:
#   robot    : skeleton_torque_suit_passive_cable (27-DOF, passive cable)
#   simulator: IsaacLab, flat terrain
#   motions  : skeleton_torque_suit_motions14.pt (14클립, constspeed 제외)
#              [0]highknee [1]onehopforward [2]onestepleft [3]onesteplong
#              [4]onestepright [5]run [6]squat [7]stepinplace1
#              [8]stepinplace2 [9]stepinplace3 [10]walk
#              [11]walk_koo [12]lunge_left_koo [13]lunge_right_koo
#   warm_start: output_isaaclab/last.ckpt (= v19, epoch 1600, Newton→IsaacLab sim-to-sim)
#
# 재시작 (resume):
#   output_isaaclab/last.ckpt 존재 시 자동 재개
#
# 실행:
#   cd /home/user/ProtoMotions
#   bash tasks/mimic_suit_passive_cable_motions14_23dof/train_isaaclab.sh

set -euo pipefail

PYTHON=/home/user/miniforge3/envs/env_isaaclab/bin/python
cd "$(dirname "$(realpath "$0")")/../.."

$PYTHON protomotions/train_agent.py \
    --robot-name skeleton_torque_suit_passive_cable \
    --simulator isaaclab \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name mimic_suit_passive_cable_motions14_23dof_isaaclab \
    --save-dir tasks/mimic_suit_passive_cable_motions14_23dof/output_isaaclab \
    --motion-file data/motion_for_trackers/skeleton_torque_suit_motions14.pt \
    --num-envs 4096 \
    --batch-size 16384 \
    --checkpoint tasks/mimic_suit_passive_cable_motions14_23dof/output_isaaclab/last.ckpt
