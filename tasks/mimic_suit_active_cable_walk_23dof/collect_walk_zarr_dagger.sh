#!/usr/bin/env bash
# DAgger/DART식 walk 에피소드 zarr 수집 (ManiFlow hip-torque 재학습용, Newton)
#
# 기존 collect_walk_zarr.sh(순수 RL+PD)를 보완 — 매니폴드 이탈 상태 + full-PD
# 전문가 라벨 쌍을 수집합니다. 모드/라벨 규약은 collect_walk_zarr_dagger.py
# docstring 참고.
#
# 사용법:
#   bash collect_walk_zarr_dagger.sh <tag> <target_episodes> [추가 인자...]
#
# 예시:
#   # DART: full PD + hip ZOH 외란 (σ=0.3×기준 std)
#   bash tasks/mimic_suit_active_cable_walk_23dof/collect_walk_zarr_dagger.sh \
#       dart030 384 --perturb-scale 0.3
#   # DAgger on-policy: α=0.5 잔여 PD 블렌드 (+약한 외란)
#   bash tasks/mimic_suit_active_cable_walk_23dof/collect_walk_zarr_dagger.sh \
#       blend050 192 --residual-pd-scale 0.5 --perturb-scale 0.15
#
# 환경변수: PYTHON (기본: sbc conda env)
# 출력: zarr_data/flat/flat-newton-dagger-{tag}-{timestamp}.zarr

set -euo pipefail

PYTHON=${PYTHON:-$HOME/miniconda3/envs/sbc/bin/python}
cd "$(dirname "$(realpath "$0")")/../.."

TASK_DIR="tasks/mimic_suit_active_cable_walk_23dof"
TAG="${1:?tag이 필요합니다 (예: dart030, blend050)}"
TARGET="${2:?target_episodes가 필요합니다}"
shift 2

echo "=== DAgger zarr 수집 (tag=$TAG, episodes=$TARGET) ==="

$PYTHON "${TASK_DIR}/collect_walk_zarr_dagger.py" \
    --tag "$TAG" \
    --target-episodes "$TARGET" \
    --num-envs 64 \
    --episode-steps 1200 \
    "$@"
