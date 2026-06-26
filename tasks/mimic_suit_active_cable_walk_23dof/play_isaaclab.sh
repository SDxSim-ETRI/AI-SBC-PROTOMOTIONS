#!/usr/bin/env bash
# mimic_suit_active_cable_walk_23dof — IsaacLab USD mesh 시각화 인터랙티브 재생
#
# 체크포인트: output_isaaclab/last.ckpt (기본) 또는 첫 인수로 경로 지정
#
# 실행:
#   cd /home/user/ProtoMotions
#   bash tasks/mimic_suit_active_cable_walk_23dof/play_isaaclab.sh
#   bash tasks/mimic_suit_active_cable_walk_23dof/play_isaaclab.sh output_isaaclab/score_based.ckpt

set -euo pipefail

PYTHON=/home/user/venv_isaaclab/bin/python
cd "$(dirname "$(realpath "$0")")/../.."

CHECKPOINT="${1:-tasks/mimic_suit_active_cable_walk_23dof/output_isaaclab_flat/last.ckpt}"

$PYTHON protomotions/inference_agent.py \
    --checkpoint "$CHECKPOINT" \
    --motion-file data/motion_for_trackers/skeleton_torque_suit_walk.pt \
    --simulator isaaclab \
    --num-envs 1 \
    --overrides \
        "robot.asset.usd_asset_file_name=usd/skeleton_torque_suit_mesh/skeleton_torque_suit_mesh.usda" \
        "robot.asset.asset_root=/home/user/ProtoMotions/protomotions/data/assets"
