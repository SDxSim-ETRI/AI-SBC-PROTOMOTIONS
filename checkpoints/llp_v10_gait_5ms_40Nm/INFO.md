# Checkpoint Info — LLP (보조 토크 정책), 5 ms 제어 + 50 ms 예측기

| 항목 | 내용 |
|------|------|
| 소스 | `results/llp_gait_v10` |
| 보관일 | 2026-09-15 |
| Epoch | 5500 (평가 파워 3600 epoch부터 188~195 W 평탄) |
| 착용자 | `checkpoints/v32_newton_suit_gait_symmetric` (관절각 추종 항으로 대칭 보행) |
| 예측기(HLP) | flexion100-h15 (100 fps, 과거 100 ms → 미래 0~50 ms 6점, denoise 1) |
| 제어 주기 | **5 ms** (물리 400 Hz / decimation 2). HLP 갱신 10 ms, tracker 50 ms |
| LLP 관측 | hip 2ch 각·각속도 + 히스토리 20(100 ms) + 미래 창 {0, 10, 20 ms} + 이전 action 10(50 ms) |
| 학습 모션 | 보행 계열 (walk 0.75 + run 0.25) |
| HLP 영점 | 착용자 스트림 기준 `5.70, 2.63, −14.43, 0.84` deg |
| 결과 (40 s, A/B) | walk **181 → 135 W (−25.4 %)**, run **362 → 309 W (−14.6 %)**, 낙상 0 |
| 같은 착용자에서 v6 | walk −16.7 %, run −19.3 % → 걷기는 개선, 달리기는 하락 |
| 토크 성격 | RMS 31/34 N·m, 변화율 RMS ≈1000 N·m/s (v6 662) — 제어 주기가 짧아 더 빨리 변한다 |

## 사용
```bash
# 평가 — 학습과 같은 5 ms 제어로 돌려야 한다
python examples/experiments/assist_skeleton/run_system.py --human tracker \
    --tracker-checkpoint checkpoints/v32_newton_suit_gait_symmetric/last.ckpt \
    --llp-checkpoint checkpoints/llp_v10_gait_5ms_40Nm/last.ckpt --llp-dt 0.005 \
    --motion-file data/motion_for_trackers/skeleton_torque_suit_gait.pt --motion-id 0 --seconds 40 \
    --calibration "5.70,2.63,-14.43,0.84"
# 구동기 대역이 정해지면: --torque-lpf-hz <대역>
```

## 학습 재현
```bash
python protomotions/train_agent.py --robot-name skeleton_torque_suit_passive_cable --simulator newton \
    --experiment-path examples/experiments/assist_skeleton/llp_walk.py --experiment-name llp_gait_v10 \
    --motion-file data/motion_for_trackers/skeleton_torque_suit_gait.pt --num-envs 4096 --batch-size 16384 \
    --tracker-checkpoint checkpoints/v32_newton_suit_gait_symmetric/last.ckpt --assist-limit 40 \
    --decimation 2 --hlp-calibration "5.70,2.63,-14.43,0.84"
```
학습 중 파이썬 세그폴트로 1회 죽었다가 systemd 자동 재시작으로 이어졌다 —
장시간 학습은 `Restart=on-failure`로 띄울 것.
