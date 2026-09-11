#!/usr/bin/env bash
# 착용자 대역(mimic tracker) 재학습 — walk 단일 모션, Newton, passive cable suit.
#
# 삭제된 v18_2 레시피(checkpoints/v18_2_newton_suit_passive_cable/resolved_configs.yaml)와
# env/agent 설정이 동일함을 2026-09-12에 --create-config-only로 확인했다
# (물리 120 Hz / decimation 6 = env 20 Hz, 4096 env, batch 16384). warm-start 계보
# (v17/v18)는 남아 있지 않아 처음부터 학습한다.
#
# 진행: results/mimic_suit_passive_walk/{train.log,last.ckpt,failed_motions/,lightning_logs/}
#   - failed_motions_epoch_N.txt 가 비어 있으면 그 평가에서 walk 성공
#   - tensorboard: eval/success_rate, eval/gt_error/mean
# 재개: 같은 명령을 다시 실행하면 last.ckpt에서 이어진다 (--overrides는 무시됨).
#
# 실행 (sbc env):
#   bash examples/experiments/assist_skeleton/train_tracker_walk.sh

set -euo pipefail
cd "$(dirname "$(realpath "$0")")/../../.."
PYTHON=${PYTHON:-/home/bak/miniconda3/envs/sbc/bin/python}
EXP=${EXP:-mimic_suit_passive_walk}
mkdir -p "results/$EXP"

nohup setsid "$PYTHON" protomotions/train_agent.py \
    --robot-name skeleton_torque_suit_passive_cable \
    --simulator newton \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name "$EXP" \
    --motion-file data/motion_for_trackers/skeleton_torque_suit_walk.pt \
    --num-envs 4096 --batch-size 16384 \
    > "results/$EXP/train.log" 2>&1 < /dev/null &
echo "launched pid $! → results/$EXP/train.log"
