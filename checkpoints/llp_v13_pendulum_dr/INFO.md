# Checkpoint Info — LLP v13 (랜덤화 진자에서 학습, 스켈레톤은 시험장)

| 항목 | 내용 |
|------|------|
| 소스 | `results/assist_pendulum_v4/epoch_6000.ckpt` (run `assist_pendulum_v4`, unit `aisbc-pend-v4`) |
| 보관일 | 2026-09-28 (학습은 2026-09-23 시작, 09-28 epoch 670,000에서 중단) |
| Epoch | **6,000** — 스켈레톤 전이 성능의 정점. 10,000·14,000은 물러남(아래), 그 뒤 평탄 |
| 학습 플랜트 | 고정 골반 hip 진자 (`hip_pendulum`), 실험 파일 `examples/experiments/assist_pendulum/mlp_v4.py` (기본값 그대로) |
| 학습 분포 | 관성 ×1~25(hip 축 0.43~3.5 kg·m², 질량·중력 고정) · 착용자 PD ω_n 12~24 rad/s, ζ 0.3~0.8 · 입각 사건(외부 하중 0~2 N·m/kg × 90 kg, 상한 0.8 × effort_limit 150 = 120 N·m; 굳힘 Kp×2~8, Kd×1.5~4; 0.08~0.45 s, 간격 0.4~1.6 s, 부호 무작위) · 작은 섭동 ≤0.3 N·m/kg · θ_g = sine + 2차 고조파 + hold 30 %, 블렌드 0.25~2.5 s, 진폭을 착용자 능력에 묶음 |
| 정책이 못 보는 것 | 위 분포의 값 전부 — 관성·게인·하중을 관측하지 않고 각도·각속도 히스토리와 예측 창으로 효과만 본다 |
| 예측기(HLP) 형식 | **배포 HLP flexion100-h15와 동일**: 10 ms × 6점(0~50 ms), 10 ms 갱신. 학습 때는 에뮬레이션(θ_g 미래 + 실측 오차 통계) |
| 제어 주기 | **10 ms** (물리 400 Hz / decimation 4). 평가·배포 때 `--llp-dt 0.01` |
| LLP 관측 | hip 2ch 각·각속도 + 히스토리 25 step(250 ms) + 미래 창 {0, 10, 20 ms} + 이전 action 10(100 ms) |
| 보상 | 추종 exp(−40·err²) w 1 + \|τ·ω\| w −0.0016 + τ² w −3e-5 + assist_effort −2e-5 + action_rate −0.005 |
| 토크 한계 | 40 N·m (tanh 스케일 = clamp) |
| 학습 데이터 | 없음 — 스켈레톤·mocap·걷기·달리기를 한 번도 본 적 없다 |

## 스켈레톤 시험 결과 (40 s A/B, 착용자 v32, 낙상 0) — `docs/llp_v13_report.html`

| | τ² (근활성) | \|τ·ω\| 파워 | 몸체 오차 | \|ω\| | 보조 RMS | 한계 도달 | 착용자 토크와 상관 |
|---|---|---|---|---|---|---|---|
| 걷기 (motion 0) | **−30.5 %** | −23.2 % | +0.9 % | −12.5 % | 33.1 N·m | 39 % | 0.66 |
| 달리기 (motion 1) | **−25.5 %** | −5.6 % | +10.2 % (0.064 → 0.070 m) | −3 % | 36.0 N·m | 64 % | 0.72 |
| β 오라클 (걷기 상한) | −32.3 % | −20.3 % | −1.1 % | −0.9 % | 22.2 N·m | 16 % | 0.86 |

걷기 위상별 τ² 기여 (절대 (N·m)²): 착지 0~20 % **−2,114**, 유각 30~70 % −1,000, 합계 −3,441
(β 오라클 −2,331 / −991 / −3,641; v10은 착지 +2,188로 까먹었다). **달리기에서 착용자 근토크를
줄인 첫 버전** — 스켈레톤에서 직접 학습한 v10·v11·v12는 +21 / +230 / +9.3 %였다.

같은 run의 뒤 체크포인트: 10,000 → 걷기 −26.8 / 달리기 −18.6 %, 14,000 → −28.0 / −23.3 %,
보조 RMS 33 → 26, 한계 도달 39 → 11 %. 진자 안 착용자 τ²는 5,000 이후 제자리라 후기 학습은 보상의
토크 절약 항이 이끈다 → 하중 3배인 스켈레톤에서 덜 돕는다. 그래서 6,000을 보관.

주의: 이 착용자(tracker v32)는 사람보다 세고(주기 평균 피크 1.7 N·m/kg, 스트라이드 3.7) hip 스윙이
mocap보다 8° 크다. \|ω\| −12 %는 정책과 무관한 착용자 모델의 산물(v10·v12도 같음).

**⚠️ 사람 범위 착용자 v33(`checkpoints/v33_newton_suit_human`, hip 한계 200)에서는 (2026-09-28 저녁):**
40 N·m 그대로 걷기 τ² −0.1 %(무효), 달리기 **4.4 s 낙상**. `--assist-scale 0.5`로 줄이면 걷기 **−19.3 %**,
달리기 **−23.9 %**(β 오라클 상한 −35 / −41). 위 −30.5 / −25.5는 v32의 과한 근력이 과보조·오방향을 받아 준
결과다. **v33·실기기 후보로는 반드시 ×0.5(≈20 N·m)로 쓸 것.** 상세 `docs/NEXT_SESSION.md` §2-H.

## 사용
```bash
# 평가 — 학습과 같은 10 ms 제어 (--llp-dt 0.01). motion-id 0 = 걷기, 1 = 달리기
python examples/experiments/assist_skeleton/run_system.py --human tracker \
    --tracker-checkpoint checkpoints/v32_newton_suit_gait_symmetric/last.ckpt \
    --llp-checkpoint checkpoints/llp_v13_pendulum_dr/last.ckpt --llp-dt 0.01 \
    --motion-file data/motion_for_trackers/skeleton_torque_suit_gait.pt --motion-id 0 --seconds 40 \
    --calibration "5.70,2.63,-14.43,0.84"
# 여러 실행을 한 표에:  compare_runs.py / report_html.py ;  이 체크포인트 하나를 깊게: report_checkpoint_html.py
```

## 학습 재현
```bash
systemd-run --user --unit=aisbc-pend-v4 --working-directory="$PWD" \
  --property=Restart=on-failure --property=RestartSec=15 --property=StartLimitIntervalSec=0 \
  --property=StandardOutput=append:$PWD/results/_pids/pend_v4.log \
  --property=StandardError=append:$PWD/results/_pids/pend_v4.log \
  --setenv=PYTHONFAULTHANDLER=1 \
  ~/miniconda3/envs/sbc/bin/python protomotions/train_agent.py --robot-name hip_pendulum --simulator newton \
    --experiment-path examples/experiments/assist_pendulum/mlp_v4.py --experiment-name assist_pendulum_v4 \
    --motion-file none --num-envs 4096 --batch-size 16384
```
epoch 3,144에서 세그폴트(매 step 게인 통지가 MuJoCo 상수를 재계산) → 408f0b6으로 고친 뒤 systemd가
이어받았다. 4096 env 기준 epoch당 약 6 s. 6,000까지 약 10시간.
