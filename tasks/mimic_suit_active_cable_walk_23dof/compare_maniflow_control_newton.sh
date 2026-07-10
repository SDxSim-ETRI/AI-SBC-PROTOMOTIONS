#!/usr/bin/env bash
# mimic_suit_active_cable_walk_23dof — ManiFlow 토크 제어 A/B 비교 (Newton)
#
# 같은 씬에 두 agent를 겹쳐 실행합니다:
#   Agent A (고스트 스켈레톤) : RL policy + built-in PD (순수 RL)
#   Agent B (메시)            : estimator 채널(공통 DOF 0-5)은 ManiFlow 토크,
#                               나머지 관절은 RL policy + built-in PD
# Newton은 env별 world가 분리되어 있어 두 agent는 물리적으로 간섭하지 않습니다.
#
# 결과: tasks/mimic_suit_active_cable_walk_23dof/maniflow_control_results/<timestamp>/
#
# 환경:
#   PYTHON        사용할 파이썬 (기본: sbc conda env)
#   MANIFLOW_ROOT maniflow 패키지 경로 (기본: ~/Projects/ManiFlow_Policy/ManiFlow)
#
# 실행:
#   bash tasks/mimic_suit_active_cable_walk_23dof/compare_maniflow_control_newton.sh
#   bash tasks/.../compare_maniflow_control_newton.sh --viewer            # 실시간 GUI
#   bash tasks/.../compare_maniflow_control_newton.sh --record --episode-steps 600
#   bash tasks/.../compare_maniflow_control_newton.sh --predict-mode every_step
#   bash tasks/.../compare_maniflow_control_newton.sh --torque-scale 0    # 무동력 sanity check

set -euo pipefail

PYTHON=${PYTHON:-$HOME/miniconda3/envs/sbc/bin/python}
cd "$(dirname "$(realpath "$0")")/../.."

$PYTHON tasks/mimic_suit_active_cable_walk_23dof/compare_maniflow_control_newton.py "$@"
