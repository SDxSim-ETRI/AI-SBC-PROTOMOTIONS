#!/usr/bin/env bash
# mimic_suit_passive_cable_motions14_23dof — Newton suit passive cable 학습 (14모션)
#
# 학습 정보:
#   robot    : skeleton_torque_suit_passive_cable (27-DOF, passive cable)
#   simulator: Newton, flat terrain
#   motions  : skeleton_torque_suit_motions14.pt (14클립, constspeed 제외)
#              [0]highknee [1]onehopforward [2]onestepleft [3]onesteplong
#              [4]onestepright [5]run [6]squat [7]stepinplace1
#              [8]stepinplace2 [9]stepinplace3 [10]walk
#              [11]walk_koo [12]lunge_left_koo [13]lunge_right_koo
#   warm_start: output_newton/last.ckpt (= v18_2, epoch 5000, 성공률 100%)
#
# 재시작 (resume):
#   output_newton/last.ckpt 존재 시 자동 재개 (warm_start → 학습 후 덮어씀)
#
# 실행:
#   cd /home/user/ProtoMotions
#   bash tasks/mimic_suit_passive_cable_motions14_23dof/train.sh

set -euo pipefail

PYTHON=/home/user/venv_newton/bin/python3
cd "$(dirname "$(realpath "$0")")/../.."

$PYTHON protomotions/train_agent.py \
    --robot-name skeleton_torque_suit_passive_cable \
    --simulator newton \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name mimic_suit_passive_cable_motions14_23dof \
    --save-dir tasks/mimic_suit_passive_cable_motions14_23dof/output_newton \
    --motion-file data/motion_for_trackers/skeleton_torque_suit_motions14.pt \
    --num-envs 4096 \
    --batch-size 16384 \
    --checkpoint tasks/mimic_suit_passive_cable_motions14_23dof/output_newton/last.ckpt
