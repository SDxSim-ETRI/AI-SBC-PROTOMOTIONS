#!/usr/bin/env bash
# mimic_suit_active_cable_walk_23dof — Newton mesh walk 1분 녹화
#
# 녹화 설정:
#   record-steps 1200 = 60초 (1분 @ 20Hz: fps=120/decimation=6)
#   cycle-seconds 0   = 모션 사이클링 없음 (walk 1개)
#
# 출력: tasks/mimic_suit_active_cable_walk_23dof/recordings/<datetime>/
#
# 체크포인트: output_newton/score_based.ckpt (기본) 또는 첫 인수로 경로 지정
#
# 실행:
#   cd /home/user/ProtoMotions
#   bash tasks/mimic_suit_active_cable_walk_23dof/record_newton.sh
#   bash tasks/mimic_suit_active_cable_walk_23dof/record_newton.sh output_newton/last.ckpt

set -euo pipefail

PYTHON=/home/user/venv_newton/bin/python3
cd "$(dirname "$(realpath "$0")")/../.."

CHECKPOINT="${1:-tasks/mimic_suit_active_cable_walk_23dof/output_newton_flat/score_based.ckpt}"
RECORDINGS_DIR="tasks/mimic_suit_active_cable_walk_23dof/recordings"

$PYTHON protomotions/inference_agent.py \
    --checkpoint "$CHECKPOINT" \
    --motion-file data/motion_for_trackers/skeleton_torque_suit_walk.pt \
    --simulator newton \
    --headless \
    --num-envs 1 \
    --cycle-seconds 0 \
    --auto-record \
    --record-steps 1200 \
    --recording-path "$RECORDINGS_DIR" \
    --overrides "robot.asset.asset_file_name=mjcf/skeleton_torque_suit_mesh.xml"
