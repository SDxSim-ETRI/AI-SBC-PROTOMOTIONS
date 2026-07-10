#!/usr/bin/env bash
# mimic_suit_active_cable_walk_23dof — IsaacLab stairs_up_walk 1분 녹화
#
# 녹화 설정: record-steps 1200 = 60초 (20Hz), cycle-seconds 0 (walk 1개)
# 체크포인트: checkpoints/v30_isaaclab_active_cable_stairs_up_walk/score_based.ckpt
#             (기본) 또는 첫 인수로 경로 지정
# terrain: resolved_configs에 pyramid stairs curriculum 포함 — 계단 지형에서 재생
#
# 실행:
#   cd /home/user/ProtoMotions
#   bash tasks/mimic_suit_active_cable_walk_23dof/record_isaaclab_stairs_up_walk.sh

set -euo pipefail

PYTHON=/home/user/miniforge3/envs/env_isaaclab/bin/python
cd "$(dirname "$(realpath "$0")")/../.."

CHECKPOINT="${1:-checkpoints/v30_isaaclab_active_cable_stairs_up_walk/score_based.ckpt}"
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
        "robot.asset.usd_asset_file_name=usd/skeleton_torque_suit_mesh/skeleton_torque_suit_mesh.usda" \
        "robot.asset.asset_root=/home/user/ProtoMotions/protomotions/data/assets"
