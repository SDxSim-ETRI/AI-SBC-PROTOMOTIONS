#!/usr/bin/env bash
# mimic_suit_passive_cable_motions14_23dof — Newton 라이브 시각화
#
# 실행:
#   cd /home/user/ProtoMotions
#   bash tasks/mimic_suit_passive_cable_motions14_23dof/live_viz_newton.sh
#   bash tasks/mimic_suit_passive_cable_motions14_23dof/live_viz_newton.sh output_newton/score_based.ckpt

set -euo pipefail

PYTHON=/home/bak/Projects/uv_env/protomotion/bin/python3
cd "$(dirname "$(realpath "$0")")/../.."

CHECKPOINT="${1:-tasks/mimic_suit_passive_cable_motions14_23dof/output_newton/score_based.ckpt}"

$PYTHON protomotions/infer_live_viz_suit.py \
    --checkpoint "$CHECKPOINT" \
    --motion-file data/motion_for_trackers/skeleton_torque_suit_motions_11+koo_4.pt \
    --simulator newton \
    --num-envs 1 \
    --overrides "robot.asset.asset_file_name=mjcf/skeleton_torque_suit_mesh.xml"
