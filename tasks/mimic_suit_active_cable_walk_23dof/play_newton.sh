#!/usr/bin/env bash
# mimic_suit_active_cable_walk_23dof — Newton mesh 시각화 인터랙티브 재생
#
# 체크포인트: output_newton/last.ckpt (기본) 또는 첫 인수로 경로 지정
#
# 실행:
#   cd /home/user/ProtoMotions
#   bash tasks/mimic_suit_active_cable_walk_23dof/play_newton.sh
#   bash tasks/mimic_suit_active_cable_walk_23dof/play_newton.sh output_newton/score_based.ckpt

set -euo pipefail

PYTHON=/home/user/venv_newton/bin/python3
cd "$(dirname "$(realpath "$0")")/../.."

CHECKPOINT="${1:-tasks/mimic_suit_active_cable_walk_23dof/output_newton_flat/last.ckpt}"

$PYTHON protomotions/inference_agent.py \
    --checkpoint "$CHECKPOINT" \
    --motion-file data/motion_for_trackers/skeleton_torque_suit_walk.pt \
    --simulator newton \
    --num-envs 1 \
    --overrides "robot.asset.asset_file_name=mjcf/skeleton_torque_suit_mesh.xml"
