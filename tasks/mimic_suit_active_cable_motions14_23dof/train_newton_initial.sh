#!/usr/bin/env bash
# mimic_suit_active_cable_motions14_23dof — Newton active cable 최초 학습 명령 (v21_3)
#
# warm_start: checkpoints/v18_2_newton_suit_passive_cable/score_based.ckpt
#   (passive cable 평지 기반 → active cable 15모션 확장)
# 원본 스크립트: scripts/run_v21_3_newton_active_cable.sh
# 원본 experiment-name: mimic_newton_active_cable_v21_3
#
# 주의: output_newton/last.ckpt 가 없어야 warm_start 적용됨
#
# 실행:
#   cd /home/user/ProtoMotions
#   bash tasks/mimic_suit_active_cable_motions14_23dof/train_newton_initial.sh

set -euo pipefail

PYTHON=/home/user/venv_newton/bin/python3
cd "$(dirname "$(realpath "$0")")/../.."

$PYTHON protomotions/train_agent.py \
    --robot-name skeleton_torque_suit_active_cable \
    --simulator newton \
    --experiment-path examples/experiments/mimic_newton_active_cable/mlp.py \
    --experiment-name mimic_suit_active_cable_motions14_23dof \
    --save-dir tasks/mimic_suit_active_cable_motions14_23dof/output_newton \
    --motion-file data/motion_for_trackers/skeleton_torque_suit_motions_11+koo_4.pt \
    --num-envs 4096 \
    --batch-size 16384 \
    --checkpoint checkpoints/v18_2_newton_suit_passive_cable/score_based.ckpt
