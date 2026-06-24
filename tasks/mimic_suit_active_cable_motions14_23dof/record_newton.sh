#!/usr/bin/env bash
# mimic_suit_active_cable_motions14_23dof — Newton mesh 자동 녹화 (14모션 × 20초)
# 제외 모션: 02-constspeed_reduced_humanoid
#
# 녹화 설정:
#   record-steps 5600 = 280초 @ 20fps  (14모션 × 20초 = 280초)
#   ※ suit Newton: fps=120, decimation=6 → policy 20fps (dt=0.05s)
#   cycle-seconds 20  = 모션당 20초
#
# 출력: tasks/mimic_suit_active_cable_motions14_23dof/recordings/<ckpt>-<datetime>/
#
# 체크포인트: output_newton/score_based.ckpt (기본) 또는 첫 인수로 경로 지정
#
# 실행:
#   cd /home/user/ProtoMotions
#   bash tasks/mimic_suit_active_cable_motions14_23dof/record_newton.sh
#   bash tasks/mimic_suit_active_cable_motions14_23dof/record_newton.sh output_newton/last.ckpt

set -euo pipefail

PYTHON=/home/user/venv_newton/bin/python3
cd "$(dirname "$(realpath "$0")")/../.."

CHECKPOINT="${1:-tasks/mimic_suit_active_cable_motions14_23dof/output_newton/score_based.ckpt}"
RECORDINGS_DIR="tasks/mimic_suit_active_cable_motions14_23dof/recordings"

$PYTHON protomotions/inference_agent.py \
    --checkpoint "$CHECKPOINT" \
    --motion-file data/motion_for_trackers/skeleton_torque_suit_motions14.pt \
    --simulator newton \
    --num-envs 1 \
    --cycle-seconds 20 \
    --auto-record \
    --record-steps 5600 \
    --recording-path "$RECORDINGS_DIR" \
    --overrides "robot.asset.asset_file_name=mjcf/skeleton_torque_suit_mesh.xml"
