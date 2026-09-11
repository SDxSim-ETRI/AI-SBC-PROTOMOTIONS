#!/usr/bin/env bash
# 착용자 대역(mimic tracker) — 15모션(11+koo_4)로 확장. walk tracker에서 warm start.
#
# v18_2와 같은 모션 파일. warm start 소스는 walk 학습 결과(last.ckpt); 네트워크
# 형상이 같으므로(27 DOF, 동일 obs) 그대로 이어받는다. 새 experiment-name을 쓰므로
# results/<EXP>/last.ckpt 가 없어야 warm start가 적용된다 (있으면 resume).
#
# 실행 (sbc env, walk tracker가 충분히 수렴한 뒤):
#   bash examples/experiments/assist_skeleton/train_tracker_motions15.sh

set -euo pipefail
cd "$(dirname "$(realpath "$0")")/../../.."
PYTHON=${PYTHON:-/home/bak/miniconda3/envs/sbc/bin/python}
EXP=${EXP:-mimic_suit_passive_motions15}
WARM=${WARM:-results/mimic_suit_passive_walk/last.ckpt}
mkdir -p "results/$EXP"

nohup setsid "$PYTHON" protomotions/train_agent.py \
    --robot-name skeleton_torque_suit_passive_cable \
    --simulator newton \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name "$EXP" \
    --motion-file "data/motion_for_trackers/skeleton_torque_suit_motions_11+koo_4.pt" \
    --num-envs 4096 --batch-size 16384 \
    --checkpoint "$WARM" \
    > "results/$EXP/train.log" 2>&1 < /dev/null &
echo "launched pid $! → results/$EXP/train.log"
