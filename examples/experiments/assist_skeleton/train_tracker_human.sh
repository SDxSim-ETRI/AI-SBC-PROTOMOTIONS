#!/usr/bin/env bash
# 착용자 v33 재학습 — 사람 범위 근력(hip 한계 200 N·m) + 관절각 추종 강화 + 파워·토크² 벌점.
# 왜·무엇: examples/experiments/assist_skeleton/tracker_human.py 머리말.
#
# systemd 사용자 유닛으로 띄운다(세그폴트 자동 재시작). v32에서 warm start.
#   진행: results/mimic_suit_human/{lightning_logs/,last.ckpt}, 로그 results/_pids/tracker_v33.log
#   tensorboard: eval/success_rate, eval/gt_error/mean, env/raw_r/{dof_pos_rew,pow_rew,torque_sq_rew}_mean
#   상태/중단: systemctl --user status|stop aisbc-tracker-v33
#
# 실행 (sbc env):
#   bash examples/experiments/assist_skeleton/train_tracker_human.sh
set -euo pipefail
cd "$(dirname "$(realpath "$0")")/../../.."
PYTHON=${PYTHON:-/home/bak/miniconda3/envs/sbc/bin/python}
EXP=${EXP:-mimic_suit_human}
UNIT=${UNIT:-aisbc-tracker-v33}
WARM=${WARM:-checkpoints/v32_newton_suit_gait_symmetric/last.ckpt}
MOTION=${MOTION:-data/motion_for_trackers/skeleton_torque_suit_gait.pt}
mkdir -p "results/$EXP" results/_pids

systemd-run --user --unit="$UNIT" --working-directory="$PWD" \
  --property=Restart=on-failure --property=RestartSec=15 --property=StartLimitIntervalSec=0 \
  --property=StandardOutput=append:"$PWD/results/_pids/tracker_v33.log" \
  --property=StandardError=append:"$PWD/results/_pids/tracker_v33.log" \
  --setenv=PYTHONFAULTHANDLER=1 \
  "$PYTHON" protomotions/train_agent.py \
    --robot-name skeleton_torque_suit_human \
    --simulator newton \
    --experiment-path examples/experiments/assist_skeleton/tracker_human.py \
    --experiment-name "$EXP" \
    --motion-file "$MOTION" \
    --checkpoint "$WARM" \
    --num-envs 4096 --batch-size 16384
echo "launched unit $UNIT → results/_pids/tracker_v33.log"
