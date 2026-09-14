#!/usr/bin/env bash
# 착용자(mimic tracker) 재학습 — 관절각 추종 항 추가, 보행 계열(walk+run) 모션.
#
# 왜: 기존 tracker(v30/v31)는 몸체 위치만 맞추라고 배워서 좌우 고관절 가동범위가
# 최대 22° 벌어진 비대칭 보행을 한다(보조를 꺼도 그대로). mocap은 거의 대칭이다.
# 관절각 추종 보상(dof_pos_rew)을 넣어 이 걸음 자체를 바로잡는다.
#
# v31에서 warm start 하므로 처음부터 학습하는 것보다 빠르다.
# 진행: results/mimic_suit_gait_sym/{train.log,last.ckpt,lightning_logs/}
#   tensorboard: eval/success_rate, eval/gt_error/mean, env/raw_r/dof_pos_rew_mean
#
# 실행 (sbc env):
#   bash examples/experiments/assist_skeleton/train_tracker_gait_sym.sh
set -euo pipefail
cd "$(dirname "$(realpath "$0")")/../../.."
PYTHON=${PYTHON:-/home/bak/miniconda3/envs/sbc/bin/python}
EXP=${EXP:-mimic_suit_gait_sym}
WARM=${WARM:-checkpoints/v31_newton_suit_passive_motions15/last.ckpt}
MOTION=${MOTION:-data/motion_for_trackers/skeleton_torque_suit_gait.pt}
mkdir -p "results/$EXP"

nohup setsid "$PYTHON" protomotions/train_agent.py \
    --robot-name skeleton_torque_suit_passive_cable \
    --simulator newton \
    --experiment-path examples/experiments/assist_skeleton/tracker_gait_sym.py \
    --experiment-name "$EXP" \
    --motion-file "$MOTION" \
    --checkpoint "$WARM" \
    --num-envs 4096 --batch-size 16384 \
    >> "results/$EXP/train.log" 2>&1 < /dev/null &
echo "launched → results/$EXP/train.log"
