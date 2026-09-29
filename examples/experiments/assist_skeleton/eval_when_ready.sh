#!/usr/bin/env bash
# 학습 중인 LLP 체크포인트가 생기면 착용자 W1 위에서 걷기·달리기 A/B(= HLP+LLP 통합 평가)를 돌리고 비교표를 찍는다.
# 학습과 같은 GPU에서 동시에 돌아도 된다(40 s × 2 × 한계 수 ≈ 5분/한계).
#
# 사용 (sbc env):
#   bash examples/experiments/assist_skeleton/eval_when_ready.sh <ckpt> <tag> [unit] [poll_sec]
#   예: ... results/llp_L2_pendulum/epoch_2000.ckpt L2e2k aisbc-llp-L2
# 환경변수:
#   TRACKER  착용자 체크포인트 (기본 W1)          CALIB   HLP 영점 (기본 wearer = 착용자 스트림에서 측정)
#   LIMITS   평가할 보조 한계들 [N·m] (기본 "20 40") — L2부터 정책이 한계를 관측하지 않으므로 배포 한계마다 재 본다
#   REFS_M0 / REFS_M1  비교표에 같이 올릴 기준 실행 glob (기본: W1 위의 L1×0.5, β 오라클)
# unit을 주면 그 systemd 유닛이 3번 연속 비활성일 때 기다리기를 포기한다(학습이 죽은 경우).
set -euo pipefail
cd "$(dirname "$(realpath "$0")")/../../.."
CKPT=$1; TAG=$2; UNIT=${3:-}; POLL=${4:-120}
PY=${PYTHON:-$HOME/miniconda3/envs/sbc/bin/python}
TRACKER=${TRACKER:-checkpoints/W1_wearer_human/last.ckpt}
CALIB=${CALIB:-wearer}
LIMITS=${LIMITS:-"20 40"}
REFS_M0=${REFS_M0:-"results/oracle/*_tracker_v33e12218_v13s05_m0 results/oracle/*_tracker_v33e12218_beta05_m0"}
REFS_M1=${REFS_M1:-"results/oracle/*_tracker_v33e12218_v13s05_m1 results/oracle/*_tracker_v33e12218_beta05_m1"}
dead=0
until [ -f "$CKPT" ]; do
  if [ -n "$UNIT" ] && ! systemctl --user is-active --quiet "$UNIT"; then
    dead=$((dead + 1)); [ "$dead" -ge 3 ] && { echo "unit $UNIT inactive — giving up on $CKPT"; exit 1; }
  else dead=0; fi
  sleep "$POLL"
done
sleep 30   # 파일 쓰기 완료 여유
for L in $LIMITS; do
  for M in 0 1; do
    "$PY" examples/experiments/assist_skeleton/run_system.py --human tracker \
      --tracker-checkpoint "$TRACKER" \
      --llp-checkpoint "$CKPT" --llp-dt 0.01 --torque-limit "$L" \
      --motion-file data/motion_for_trackers/skeleton_torque_suit_gait.pt --motion-id "$M" --seconds 40 \
      --calibration "$CALIB" --out results/oracle --tag "${TAG}_lim${L}_m${M}" \
      > "results/_pids/eval_${TAG}_lim${L}_m${M}.log" 2>&1
  done
done
echo "== 걷기 (motion 0) =="
# shellcheck disable=SC2086
"$PY" examples/experiments/assist_skeleton/compare_runs.py $REFS_M0 results/oracle/*_tracker_"${TAG}"_lim*_m0
echo "== 달리기 (motion 1) =="
# shellcheck disable=SC2086
"$PY" examples/experiments/assist_skeleton/compare_runs.py $REFS_M1 results/oracle/*_tracker_"${TAG}"_lim*_m1
