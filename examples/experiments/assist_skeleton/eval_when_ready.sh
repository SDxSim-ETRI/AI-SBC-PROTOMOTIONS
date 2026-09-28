#!/usr/bin/env bash
# 학습 중인 LLP 체크포인트가 생기면 스켈레톤 걷기·달리기 A/B를 돌리고 비교표를 찍는다.
# 학습과 같은 GPU에서 동시에 돌아도 된다(40 s × 2 ≈ 5분).
#
# 사용 (sbc env):
#   bash examples/experiments/assist_skeleton/eval_when_ready.sh <ckpt> <tag> [unit] [poll_sec]
#   예: ... results/assist_pendulum_v5/epoch_2000.ckpt v5e2k aisbc-pend-v5
# unit을 주면 그 systemd 유닛이 3번 연속 비활성일 때 기다리기를 포기한다(학습이 죽은 경우).
set -euo pipefail
cd "$(dirname "$(realpath "$0")")/../../.."
CKPT=$1; TAG=$2; UNIT=${3:-}; POLL=${4:-120}
PY=${PYTHON:-$HOME/miniconda3/envs/sbc/bin/python}
dead=0
until [ -f "$CKPT" ]; do
  if [ -n "$UNIT" ] && ! systemctl --user is-active --quiet "$UNIT"; then
    dead=$((dead + 1)); [ "$dead" -ge 3 ] && { echo "unit $UNIT inactive — giving up on $CKPT"; exit 1; }
  else dead=0; fi
  sleep "$POLL"
done
sleep 30   # 파일 쓰기 완료 여유
for M in 0 1; do
  "$PY" examples/experiments/assist_skeleton/run_system.py --human tracker \
    --tracker-checkpoint checkpoints/v32_newton_suit_gait_symmetric/last.ckpt \
    --llp-checkpoint "$CKPT" --llp-dt 0.01 \
    --motion-file data/motion_for_trackers/skeleton_torque_suit_gait.pt --motion-id "$M" --seconds 40 \
    --calibration "5.70,2.63,-14.43,0.84" --out results/oracle --tag "${TAG}_m${M}" \
    > "results/_pids/eval_${TAG}_m${M}.log" 2>&1
done
echo "== 걷기 (motion 0) =="
"$PY" examples/experiments/assist_skeleton/compare_runs.py results/oracle/*_tracker_base results/oracle/*_tracker_beta05 \
  results/oracle/*_tracker_v4new6k_m0 results/oracle/*_tracker_"${TAG}"_m0
echo "== 달리기 (motion 1) =="
"$PY" examples/experiments/assist_skeleton/compare_runs.py results/oracle/2026-09-16_19-48-34_tracker_v10_m1 \
  results/oracle/*_tracker_v4new6k_m1 results/oracle/*_tracker_"${TAG}"_m1
