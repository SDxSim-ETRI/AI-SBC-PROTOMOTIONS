# HLP + LLP 통합 시스템 — 스켈레톤 착용자 (Newton)

ManiFlow flexion 예측기(HLP)와 assist torque RL 정책(LLP)을 **하나의 제어기**로 묶어
인체 스켈레톤(`skeleton_torque_suit`, 슈트 관절 4개는 토크 0)의 고관절에 보조
토크를 전달하는 폐루프 시스템. 2026-09-12 구축.

```
      스켈레톤 상태 (Newton, 물리 400 Hz)
          │ 10 ms마다 (서브스텝 훅)
          ▼
   hip_flexion_r/l 각·각속도 + trunk pitch/roll  ──▶  AssistController  ──▶  τ_assist (2ch)
                                                      ├ HLP: 링버퍼→25 ms 재표집→ManiFlow→chunk(7 knot, 0~150 ms)
                                                      └ LLP: 히스토리·미래 창·이전 action → actor → tanh×25 N·m
          ▲                                                                        │
          └──────────── qfrc_applied 주입 (JointTorqueOverride, hip flexion) ◀──────┘
   착용자 대역: mocap PD(골반 고정) 또는 mimic tracker(보행), env 20 Hz
```

## 구성 파일

| 파일 | 역할 |
|------|------|
| `protomotions/assist/controller.py` | **`AssistController`** — env 비의존 제어기. 입력 관절각·각속도·몸통자세 스트림(10 ms), 출력 assist 토크. 실기기/시뮬 공용 |
| `protomotions/assist/llp_policy.py` | `LLPPolicy` — `assist_pendulum` 체크포인트에서 actor + 관측 레이아웃만 복원 (env/Fabric 없이) |
| `protomotions/assist/skeleton_features.py` | 스켈레톤 상태 → HLP 관측 4ch (학습 데이터 추출기와 동일 규약) + 영점 보정 |
| `protomotions/assist/verify_controller.py` | 진자 학습 env와의 step 단위 등가성 검증 (`python -m protomotions.assist.verify_controller`) |
| `protomotions/simulator/newton/substep_hook.py` | `SubstepHookMixin` — env step(50 ms) 안의 서브스텝 루프에서 10 ms마다 콜백 (LLP 실행 + qfrc 갱신 + 창 단위 PD 토크 평균) |
| `run_system.py` | 폐루프 러너 (A: 보조 없음 / B: 보조, paired) → traces·metrics·report·plots·mp4 |
| `analysis.py` | 지표·그림·리포트 |
| `viz.py` | 뷰어 오버레이 (A 고스트 라인, B 메시 + assist 화살표) |
| `train_tracker_walk.sh`, `train_tracker_motions15.sh` | 착용자 대역(mimic tracker) 재학습 |

## 실행 (sbc env)

```bash
# 1) 독립 제어기 ↔ 진자 env 등가성 (한 번 확인) — PASS 기대
python -m protomotions.assist.verify_controller --checkpoint results/assist_pendulum_v3/last.ckpt

# 2) 골반 고정 대역 (tracker 불필요). 기본: 착용자 PD를 진자 학습 플랜트와 같은 동역학 영역으로 (×12.4/×6.2)
python examples/experiments/assist_skeleton/run_system.py --human mocap_pd --seconds 30
#    강체 다리(무릎·발목 잠금) 진단 / assist 배율 스윕
python examples/experiments/assist_skeleton/run_system.py --human mocap_pd --lock-knee-ankle --assist-scale 3

# 3) 보행 착용자 (tracker 체크포인트 필요; train_tracker_walk.sh)
python examples/experiments/assist_skeleton/run_system.py --human tracker \
    --tracker-checkpoint results/mimic_suit_passive_walk/last.ckpt --seconds 30

# 4) 뷰어 / 녹화 (DISPLAY 필요; A=고스트 라인, B=메시, 초록 화살표=assist)
DISPLAY=:1 python examples/experiments/assist_skeleton/run_system.py --human mocap_pd --record
```

출력: `results/assist_skeleton/<timestamp>_<human>[_tag]/{report.md, metrics.json, plots.png, traces.npz, meta.json, sim.mp4}`.

## 설계 결정

- **시간 구조**: tracker는 env dt 50 ms(120 Hz/6)로 학습됐고 관측(미래 레퍼런스 프레임)이
  dt에 묶여 있어 env 주기를 못 바꾼다. LLP는 10 ms. 그래서 물리 400 Hz / decimation 20 /
  hook_every 4 로 두고 LLP를 서브스텝 루프 안에서 돌린다. 진자 학습도 물리 400 Hz였다.
- **부호 규약**: 데이터/HLP는 +θ = 굴곡, LLP(진자)는 +θ = 신전. `AssistController(joint_sign)`이
  양방향 변환 (skeleton +1, hip_pendulum −1). 등가성 검증에서 진자 env와 bit-exact 일치 확인.
- **HLP 영점 보정**: 재생 클립의 4ch 평균을 SEED 학습 데이터 평균(`SEED_CHANNEL_MEAN_DEG`)에
  맞추는 오프셋을 관측에 가산 (eval_abc `--align-ref-zarr`와 같은 방식). walk 클립 기준
  flexion +12.3°/+10.1°.
- **골반 고정 대역의 게인**: 로봇 config 게인(hip Kp 200/Kd 20)은 다리 관성(hip 기준
  3.53 kg·m²)에 비해 물러 ζ≈0.38로 12° 오버슈트 → HLP 분포 밖. 기본값은 진자 플랜트
  (I=0.43, Kp 300/Kd 15 → ω_n 26.5 rad/s, ζ 0.66)와 같은 영역이 되도록 Kp ×12.4, Kd ×6.2
  (추종 오차 0.78°).
- **CUDA graph 끔**: 콜백이 서브스텝 사이에 끼므로. 소수 env 평가용.

## 결과 (2026-09-12, walk 클립 30 s, LLP v3 그대로)

착용자 hip flexion 파워 |τ·ω| 합 (A 대비 B), `assist_vs_agentA_corr` = assist 토크와 착용자
단독 토크의 상관 (방향·타이밍 일치도).

| 착용자 대역 | assist 배율 | 착용자 파워 A→B [W] | 절감 | corr |
|---|---|---|---|---|
| 골반 고정, 관절 다리 (무릎·발목 mocap 추종) | 1 | 117.6 → 112.3 | **−4.5 %** | 0.19 |
| 〃 | 3 | 116.6 → 111.9 | −4.0 % (토크 RMS +3 %) | 0.19 |
| 골반 고정, **강체 다리** (무릎·발목 잠금) | 1 | 169.3 → 157.6 | −7.0 % | **0.80** |
| 〃 | 2 | 169.3 → 146.5 | −13.5 % | 0.79 |
| 〃 | 3 | 169.3 → 136.5 | −19.4 % | 0.79 |
| 〃 | 4 | 169.3 → 127.4 | −24.8 % | 0.78 |
| 보행 tracker (학습 중, epoch ≈1300, 30 s) | 1 | 167.1 → 173.3 | +3.7 % | −0.07 |

해석:

1. **시스템은 닫힌다** — HLP 예측(150 ms MAE 6.3°, suit14 오프라인 5.7°와 같은 수준)이 chunk로
   들어가고 LLP가 10 ms마다 토크를 내며, 추론 시간은 HLP 7.9 ms + LLP 0.4 ms (batch 2).
2. **진자 LLP는 "강체 진자" 플랜트에는 모양이 맞는다** (corr 0.80, 배율에 따라 −7 → −25 %
   단조 절감). 크기만 부족하다: 스켈레톤 다리는 진자보다 관성이 8.3배라 같은 궤적에 8배
   토크가 필요한데 LLP 출력 한계는 25 N·m.
3. **무릎이 움직이면 맞지 않는다** (corr 0.19). 스윙 중 무릎 굴곡(≤65°)이 다리 관성을 바꾸고
   무릎 PD 반력이 hip 토크에 ±100 N·m 스파이크를 만든다 — 진자에는 없던 성분. 보행
   tracker에서는 지면 반력까지 더해져 corr ≈ 0.
   → LLP를 실제 플랜트(관절 다리 + 지면 접촉, 실제 토크 스케일)에서 재학습해야 한다는 뜻.
   관측 계약(hip-local)은 그대로 두고 플랜트·사람 모델만 바꾸면 된다 — 이 러너가 그 학습
   환경의 평가 하네스로 그대로 쓰인다.

tracker 항목은 학습 중간 체크포인트 기준(자체 평가 성공률 1.0, 위치 오차 8.6 cm, hip 추종
10.8°, HLP 150 ms MAE 9.5° — tracker의 hip 궤적이 mocap보다 덜 사람 같아 HLP도 나빠진다)이며
400 Hz 물리(학습 120 Hz)에서 30 s 낙상 없이 걸었다. 학습이 끝나면 같은 명령으로 갱신할 것.
