#!/usr/bin/env bash
# Walk 에피소드 zarr 수집 스크립트
#
# 사용법:
#   bash collect_walk_zarr.sh [checkpoint] [terrain]
#
# terrain 옵션 (기본값: flat):
#   단일: flat, smooth_slope, rough_slope, stairs_up, stairs_down,
#         discrete, stepping, poles
#   복합: slope_discrete, slope_stairs, mixed
#
# 출력:
#   zarr_data/{terrain}/{terrain}-YYYY-MM-DD-HH-MM-SS.zarr
#
# 예시:
#   cd /home/user/ProtoMotions
#   bash tasks/mimic_suit_active_cable_walk_23dof/collect_walk_zarr.sh
#   bash tasks/mimic_suit_active_cable_walk_23dof/collect_walk_zarr.sh score_based.ckpt flat
#   bash tasks/mimic_suit_active_cable_walk_23dof/collect_walk_zarr.sh score_based.ckpt discrete
#   bash tasks/mimic_suit_active_cable_walk_23dof/collect_walk_zarr.sh score_based.ckpt slope_discrete

set -euo pipefail

PYTHON=/home/user/miniforge3/envs/env_isaaclab/bin/python

# zarr 미설치 시 자동 설치
$PYTHON -c "import zarr" 2>/dev/null || $PYTHON -m pip install "zarr>=2.18,<3" --quiet
cd "$(dirname "$(realpath "$0")")/../.."

TASK_DIR="tasks/mimic_suit_active_cable_walk_23dof"
CHECKPOINT="${1:-${TASK_DIR}/output_isaaclab/score_based.ckpt}"
TERRAIN="${2:-flat}"
NUM_ENVS="${3:-10}"
MOTION_FILE="data/motion_for_trackers/skeleton_torque_suit_walk.pt"

echo "=== zarr 수집 ==="
echo "  checkpoint : $CHECKPOINT"
echo "  terrain    : $TERRAIN"
echo "  num-envs   : $NUM_ENVS"
echo ""

$PYTHON "${TASK_DIR}/collect_walk_zarr.py" \
    --checkpoint "$CHECKPOINT" \
    --motion-file "$MOTION_FILE" \
    --terrain "$TERRAIN" \
    --num-envs "$NUM_ENVS" \
    --target-episodes 1000 \
    --episode-steps 1200 \
    --overrides \
        "robot.asset.usd_asset_file_name=usd/skeleton_torque_suit_mesh/skeleton_torque_suit_mesh.usda" \
        "robot.asset.asset_root=/home/user/ProtoMotions/protomotions/data/assets"
