#!/usr/bin/env bash
# Walk 에피소드 zarr 수집 스크립트 (ManiFlow hip-torque 학습 데이터)
#
# 기본 시뮬레이터는 Newton입니다 — RL 보행 체크포인트(output_newton_flat)가
# Newton에서 학습되었고 ManiFlow inference도 Newton에서 돌기 때문에 수집
# 도메인을 일치시킵니다. 비평지 terrain은 IsaacLab 전용(SIMULATOR=isaaclab).
#
# hip_torque 필드는 순수 hip 6개 DOF(공통 [0,1,2,5,6,7])를 기록합니다.
# 과거 수집본(slice(0,6) — 오른다리 전체+왼 hip flexion)과 attrs의
# action_dof_names / action_dof_indices로 구분됩니다.
#
# 사용법:
#   bash collect_walk_zarr.sh [checkpoint] [terrain] [num_envs] [target_episodes]
#
# 환경변수:
#   PYTHON     사용할 파이썬 (기본: sbc conda env)
#   SIMULATOR  newton(기본) / isaaclab
#
# 예시:
#   bash tasks/mimic_suit_active_cable_walk_23dof/collect_walk_zarr.sh
#   bash tasks/mimic_suit_active_cable_walk_23dof/collect_walk_zarr.sh \
#       tasks/mimic_suit_active_cable_walk_23dof/output_newton_flat/score_based.ckpt flat 64
#
# 출력:
#   zarr_data/{terrain}/{terrain}-{simulator}-YYYY-MM-DD-HH-MM-SS.zarr

set -euo pipefail

PYTHON=${PYTHON:-$HOME/miniconda3/envs/sbc/bin/python}
SIMULATOR=${SIMULATOR:-newton}
cd "$(dirname "$(realpath "$0")")/../.."

TASK_DIR="tasks/mimic_suit_active_cable_walk_23dof"
CHECKPOINT="${1:-${TASK_DIR}/output_newton_flat/score_based.ckpt}"
TERRAIN="${2:-flat}"
NUM_ENVS="${3:-10}"
TARGET_EPISODES="${4:-1000}"
MOTION_FILE="data/motion_for_trackers/skeleton_torque_suit_walk.pt"

echo "=== zarr 수집 ==="
echo "  checkpoint : $CHECKPOINT"
echo "  simulator  : $SIMULATOR"
echo "  terrain    : $TERRAIN"
echo "  num-envs   : $NUM_ENVS"
echo "  episodes   : $TARGET_EPISODES"
echo ""

$PYTHON "${TASK_DIR}/collect_walk_zarr.py" \
    --checkpoint "$CHECKPOINT" \
    --simulator "$SIMULATOR" \
    --motion-file "$MOTION_FILE" \
    --terrain "$TERRAIN" \
    --num-envs "$NUM_ENVS" \
    --target-episodes "$TARGET_EPISODES" \
    --episode-steps 1200
