#!/usr/bin/env bash
# mimic_suit_active_cable_walk_23dof — ManiFlow hip-torque 추정기 Newton inference
#
# RL walking policy로 Newton에서 보행하며 ManiFlow lowdim 정책의 hip torque
# 예측을 실제 적용 토크와 비교합니다. 결과는
# tasks/mimic_suit_active_cable_walk_23dof/maniflow_infer_results/<timestamp>/
#
# 환경:
#   PYTHON        사용할 파이썬 (기본: sbc conda env)
#   MANIFLOW_ROOT maniflow 패키지 경로 (기본: ~/Projects/ManiFlow_Policy/ManiFlow)
#
# 실행:
#   bash tasks/mimic_suit_active_cable_walk_23dof/infer_maniflow_newton.sh
#   bash tasks/mimic_suit_active_cable_walk_23dof/infer_maniflow_newton.sh \
#       --num-envs 4 --episode-steps 600
#   bash tasks/mimic_suit_active_cable_walk_23dof/infer_maniflow_newton.sh \
#       --maniflow-ckpt <path/to/epoch=NNNN-val_loss=*.ckpt>
#
# GUI / 동영상 (뷰어/녹화 시 skeleton mesh 에셋이 기본):
#   --viewer          실시간 뷰어 (UI에 hip_flexion pred/gt 라이브 플롯)
#   --record          시뮬 mp4 + 토크 플롯 합성 sim_with_torque.mp4 저장
#   --no-mesh         기본(캡슐) 에셋으로 시각화 (수집 조건 물리 재현용)

set -euo pipefail

PYTHON=${PYTHON:-$HOME/miniconda3/envs/sbc/bin/python}
cd "$(dirname "$(realpath "$0")")/../.."

$PYTHON tasks/mimic_suit_active_cable_walk_23dof/infer_maniflow_newton.py "$@"
