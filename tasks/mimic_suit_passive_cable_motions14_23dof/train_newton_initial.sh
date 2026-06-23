#!/usr/bin/env bash
# mimic_suit_passive_cable_motions14_23dof — Newton passive cable 최초 학습 명령 (v18_2)
#
# warm_start: checkpoints/v18_newton_suit_passive_cable/last.ckpt  ← 미보관
#   (v18: Newton 1.2.1 평지 walk 단일모션 기반 → 15모션 확장)
#   ※ v18 체크포인트 미보관. 재현 필요 시 v18_2(output_newton)에서 이어서 학습 가능
# 원본 experiment-name: mimic_newton_suit_passive_cable_v18_2
#
# 주의: output_newton/last.ckpt 가 없어야 warm_start 적용됨
#
# 실행:
#   cd /home/user/ProtoMotions
#   bash tasks/mimic_suit_passive_cable_motions14_23dof/train_newton_initial.sh

set -euo pipefail

PYTHON=/home/user/venv_newton/bin/python3
cd "$(dirname "$(realpath "$0")")/../.."

$PYTHON protomotions/train_agent.py \
    --robot-name skeleton_torque_suit_passive_cable \
    --simulator newton \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name mimic_suit_passive_cable_motions14_23dof \
    --save-dir tasks/mimic_suit_passive_cable_motions14_23dof/output_newton \
    --motion-file data/motion_for_trackers/skeleton_torque_suit_motions_11+koo_4.pt \
    --num-envs 4096 \
    --batch-size 16384 \
    --checkpoint checkpoints/v18_newton_suit_passive_cable/last.ckpt
