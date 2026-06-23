#!/usr/bin/env bash
# mimic_suit_passive_cable_motions14_23dof — Newton 재학습 (failure_discount=0.3)
#
# 학습 정보:
#   robot    : skeleton_torque_suit_passive_cable (27-DOF, passive cable)
#   simulator: Newton, flat terrain
#   motions  : skeleton_torque_suit_motions14.pt (14클립)
#   warm_start: output_newton/last.ckpt (v18_2 기반 100% 성공, 무한 학습)
#   차이점   : failure_discount=0.3 (고난도 실패모션 지속 샘플링)
#
# 결과 저장: output_newton_fd03/ (실험 완료 후 output_newton으로 승격 가능)
#
# 실행:
#   cd /home/user/ProtoMotions
#   bash tasks/mimic_suit_passive_cable_motions14_23dof/train_newton_fd03.sh

set -euo pipefail

PYTHON=/home/user/venv_newton/bin/python3
cd "$(dirname "$(realpath "$0")")/../.."

$PYTHON protomotions/train_agent.py \
    --robot-name skeleton_torque_suit_passive_cable \
    --simulator newton \
    --experiment-path examples/experiments/mimic/mlp_motions14_suit_newton.py \
    --experiment-name mimic_suit_passive_cable_motions14_23dof \
    --save-dir tasks/mimic_suit_passive_cable_motions14_23dof/output_newton_fd03 \
    --motion-file data/motion_for_trackers/skeleton_torque_suit_motions14.pt \
    --num-envs 4096 \
    --batch-size 16384 \
    --checkpoint tasks/mimic_suit_passive_cable_motions14_23dof/output_newton/last.ckpt
