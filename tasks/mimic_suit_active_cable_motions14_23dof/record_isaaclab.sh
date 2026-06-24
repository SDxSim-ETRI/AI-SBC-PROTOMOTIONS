#!/usr/bin/env bash
# mimic_suit_active_cable_motions14_23dof — IsaacLab USD mesh 자동 녹화 (14모션 × 20초)
# 제외 모션: 02-constspeed_reduced_humanoid
#
# 녹화 설정:
#   record-steps 5600 = 280초 (14모션 × 20초)
#   cycle-seconds 20  = 모션당 20초
#
# 출력: tasks/mimic_suit_active_cable_motions14_23dof/recordings/<ckpt>-<datetime>/
#
# 체크포인트: output_isaaclab/score_based.ckpt (기본) 또는 첫 인수로 경로 지정
#
# 실행:
#   cd /home/user/ProtoMotions
#   bash tasks/mimic_suit_active_cable_motions14_23dof/record_isaaclab.sh
#   bash tasks/mimic_suit_active_cable_motions14_23dof/record_isaaclab.sh output_isaaclab/last.ckpt

set -euo pipefail

PYTHON=/home/user/miniforge3/envs/env_isaaclab/bin/python
cd "$(dirname "$(realpath "$0")")/../.."

CHECKPOINT="${1:-tasks/mimic_suit_active_cable_motions14_23dof/output_isaaclab/score_based.ckpt}"
RECORDINGS_DIR="tasks/mimic_suit_active_cable_motions14_23dof/recordings"

$PYTHON protomotions/inference_agent.py \
    --checkpoint "$CHECKPOINT" \
    --motion-file data/motion_for_trackers/skeleton_torque_suit_motions14.pt \
    --simulator isaaclab \
    --headless \
    --num-envs 1 \
    --cycle-seconds 20 \
    --auto-record \
    --record-steps 5600 \
    --recording-path "$RECORDINGS_DIR" \
    --overrides "robot.asset.usd_asset_file_name=usd/skeleton_torque_suit_mesh/skeleton_torque_suit_mesh.usda"
