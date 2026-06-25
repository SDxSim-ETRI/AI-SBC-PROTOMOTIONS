#!/usr/bin/env bash
# mimic_suit_passive_cable_motions14_23dof — Newton 데이터 수집
#
# 출력: data/collected/passive_cable_newton_<timestamp>.zarr
#
# 실행:
#   cd /home/user/ProtoMotions
#   bash tasks/mimic_suit_passive_cable_motions14_23dof/collect_newton.sh
#   bash tasks/mimic_suit_passive_cable_motions14_23dof/collect_newton.sh output_newton/score_based.ckpt

set -euo pipefail

PYTHON=/home/bak/Projects/uv_env/protomotion/bin/python3
cd "$(dirname "$(realpath "$0")")/../.."

CHECKPOINT="${1:-tasks/mimic_suit_passive_cable_motions14_23dof/output_newton/score_based.ckpt}"
TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
OUTPUT="data/collected/passive_cable_newton_${TIMESTAMP}.zarr"

mkdir -p data/collected

$PYTHON protomotions/collect_data_suit.py \
    --checkpoint "$CHECKPOINT" \
    --motion-file data/motion_for_trackers/skeleton_torque_suit_motions_11+koo_4.pt \
    --simulator newton \
    --num-envs 16 \
    --num-steps 5000 \
    --headless \
    --output "$OUTPUT"

echo "Saved to: $OUTPUT"
