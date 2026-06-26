#!/usr/bin/env bash
# mimic_suit_active_cable_walk_23dof — IsaacLab walk 1분 녹화
#
# 녹화 설정:
#   record-steps 1200 = 60초 (1분 @ 20Hz: fps=120/decimation=6)
#   cycle-seconds 0   = 모션 사이클링 없음 (walk 1개)
#
# 출력: tasks/mimic_suit_active_cable_walk_23dof/recordings/<datetime>/
#
# 체크포인트: output_isaaclab/score_based.ckpt (기본) 또는 첫 인수로 경로 지정
#
# 실행:
#   cd /home/user/ProtoMotions
#   bash tasks/mimic_suit_active_cable_walk_23dof/record_isaaclab.sh
#   bash tasks/mimic_suit_active_cable_walk_23dof/record_isaaclab.sh output_isaaclab/last.ckpt

set -euo pipefail

PYTHON=/home/user/venv_isaaclab/bin/python
cd "$(dirname "$(realpath "$0")")/../.."

CHECKPOINT="${1:-tasks/mimic_suit_active_cable_walk_23dof/output_isaaclab_flat/score_based.ckpt}"
RECORDINGS_DIR="tasks/mimic_suit_active_cable_walk_23dof/recordings"

$PYTHON protomotions/inference_agent.py \
    --checkpoint "$CHECKPOINT" \
    --motion-file data/motion_for_trackers/skeleton_torque_suit_walk.pt \
    --simulator isaaclab \
    --headless \
    --num-envs 1 \
    --cycle-seconds 0 \
    --auto-record \
    --record-steps 1200 \
    --recording-path "$RECORDINGS_DIR" \
    --overrides \
        "robot.asset.usd_asset_file_name=usd/skeleton_torque_suit_mesh/skeleton_torque_suit_mesh.usda"
