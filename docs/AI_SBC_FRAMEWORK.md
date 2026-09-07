# AI-SBC 프레임워크: HLP + LLP 고관절 보조 제어

> 강체(rigid-link) 고관절 assist 착용로봇의 딥러닝 기반 제어기 연구 (2026).
> 이 문서는 프레임워크 전체 구조, 코드 지도, 실험 이력, 로드맵의 단일 참조점이다.
> 최종 갱신: 2026-09-01

---

## 1. 개요

제어기는 두 계층으로 구성된다.

```
                        ┌──────────────────────────────┐
   관절각/몸통자세      │  HLP (High-Level Policy)     │   미래 목표각 chunk
   히스토리 (40 Hz) ──▶ │  ManiFlow flexion 예측 모델  │ ──▶ θ_d[t..t+K]
                        │  (BONES-SEED mocap 학습)     │   (저주파, ~10 Hz 갱신)
                        └──────────────────────────────┘
                                                            │
                        ┌──────────────────────────────┐    ▼
   관절 상태 + action   │  LLP (Low-Level Policy)      │   assist 토크
   히스토리 (100 Hz) ──▶│  PPO assist torque 정책      │ ──▶ τ_assist (고주파)
                        └──────────────────────────────┘
```

- **HLP**: 사용자의 flexion 방향 고관절 각도를 예측한다. 관절각 4채널
  (hip_flexion_r/l + trunk pitch/roll) 히스토리를 입력받아 미래 flexion
  각도 chunk를 출력하는 ManiFlow 모델. 이 예측을 **pseudo GT 목표**로 삼는다.
- **LLP**: HLP가 준 목표 각도로의 움직임을 돕는 **assist torque**를 출력하는
  RL(PPO) 정책. 착용자의 에너지(기계적 파워)를 최소화하되 궤적 추종을
  깨뜨리지 않도록 학습된다.

물리 모델 (모델 설명 자료 10p 진자 모델):

```
τ_total = τ_agent + τ_assist
  τ_agent  = 착용자(사람) 몫 — 시뮬레이션에서는 PD 제어기 또는 mimic tracker가 대역
  τ_assist = 로봇 보조 몫  — LLP 출력, additive 주입
```

LLP의 목적함수: 사람 기계적 파워 E = ∫|τ_agent·θ̇|dt 최소화 + 추종 유지.

---

## 2. 핵심 개념

### 2.1 세 가지 각도 신호 — 반드시 구분할 것

| 신호 | 의미 | 생성 주체 | 소비 주체 |
|------|------|-----------|-----------|
| **θ_g** | 의도 궤적 (GT) | 궤적 생성기 (합성 sine 또는 mocap 데이터) | "사람"(PD/tracker)이 추종; tracking 보상·평가 기준 |
| **θ_d** | HLP 예측 chunk | 학습 시: θ_g + 노이즈/지연 에뮬레이션 / 통합 시: 실제 ManiFlow | LLP의 **관측** (미래 목표 창) |
| **θ** | 실제 관절각 | 시뮬레이션 물리 | 모두의 피드백; HLP의 입력 히스토리 |

핵심: HLP는 θ_g를 볼 수 없다 — **실제 관절각 θ의 히스토리만 보고** 미래를
예측한다. 사람은 자기 의도(θ_g)를 정확히 알고 추종하지만, 로봇(LLP)은
HLP의 불완전한 예측(θ_d)만 받는다. 이 정보 비대칭이 프레임워크의 본질이다.

### 2.2 용어 혼동 주의: mimic tracker ≠ ManiFlow

| | mimic tracker | ManiFlow flexion 모델 (HLP) |
|---|---|---|
| 정체 | ProtoMotions Mimic(PPO) 전신 모션 추종 RL 정책 | BONES-SEED mocap 지도학습 예측 모델 |
| 역할 | 시뮬레이션 속 **"착용자" 대역** (skeleton을 걷게 함) | **사용자 의도 예측** (미래 flexion 출력) |
| 실존 체크포인트 | `checkpoints/v18_2_newton_suit_passive_cable` | `ManiFlow_Policy/.../locomotion-flexion40-run01` |

### 2.3 Chunk 인터페이스 (HLP ↔ LLP 결합점)

`AssistTargetControl`이 관리하는 chunk 버퍼가 두 정책의 유일한 접점이다:

- **버퍼**: knot K개 × DOF, knot 간격 `hlp_dt`, 기준시각 `t0`
- **채움** (`_refresh_chunk`): 저주파(chunk_refresh_steps)마다 재생성.
  현재는 θ_g에서 미래 knot을 떠서 노이즈(bias+white)와 지연을 주입하는
  **에뮬레이션**. → 실제 HLP 통합 = 이 함수의 knot 소스를 ManiFlow 예측으로 교체
- **소비** (`_interp_chunk` → `populate_context`): policy step마다 knot 사이를
  선형 interpolation한 미래 목표 창(`theta_d_future`, 0~0.4 s)을 LLP 관측으로 제공

시간 스케일 (진자 기준): 물리 400 Hz → policy 100 Hz (decimation 4) →
chunk 갱신 10 Hz. ManiFlow 실모델의 native 스케일: 40 Hz(25 ms) 프레임,
horizon 16(= 400 ms), n_obs 10(= 250 ms), n_act 4. 통합 시 `hlp_dt=0.025`,
`chunk_knots=16`으로 맞추면 interpolation 코드는 무변경.

### 2.4 사람 모델 (시뮬레이션 근사)

- **v1**: Newton BUILT_IN_PD — τ_agent = Kp(θ_g−θ) − Kd·θ̇ (substep implicit)
- **v2 (현행)**: v1 + 속도 피드포워드 τ_ff = Kd·θ̇_g를 qfrc로 가산
  → τ_agent = Kp(θ_g−θ) + Kd(θ̇_g−θ̇). 사람 단독 추종 0.072→0.019 rad로 개선,
  assist 효과를 "순수 에너지 절감"으로 분리 (`human_velocity_feedforward=True`)
- **게인 DR**: env별 ke/kd 배율 0.7~1.3 랜덤 (착용자마다 다른 근력/임피던스),
  에피소드 간 고정, critic만 privileged로 관측

### 2.5 Assist 주입·계측 경로 (Newton 전용)

- **주입**: `JointTorqueOverride.engage(1.0)` + `set_torques(τ_assist)` —
  `control.joint_f` → `qfrc_applied`로 매 substep 가산. 게인 무변경(additive).
  qfrc 경로는 effort limit 클램프를 받지 않으므로 env 쪽에서
  `assist_torque_limit`으로 직접 제한
- **계측**: `get_substep_mean_dof_forces()` — `qfrc_actuator` substep 평균 =
  사람 PD 몫만 분리 계측 (assist는 qfrc_applied 경로라 미포함). 점 샘플이
  아닌 substep 평균인 이유: PD 스파이크를 계통 과소평가하지 않기 위함
  (torque 트랙 라벨 v2 실험에서 확립)

### 2.6 레거시 torque 트랙이 남긴 설계 근거

2026-07 ManiFlow **hip torque 직접 예측** 트랙의 결론이 현 LLP 설계를 정당화한다:

1. **순수 피드포워드 토크는 예측이 완벽해도 보행 불가** — substep 피드백
   (임피던스) 부재가 병목 (handover 실험: 예측오차 0.38 N·m에도 1.3 s 낙상).
   → 그래서 현 구조는 사람 PD(임피던스)를 유지하고 assist를 **additive**로 얹는다.
2. **가산 보조(β 모드)는 낙상 위험 없이 착용자 토크 절감** — β=0.6에서 −31%,
   전 구간 60 s 완주. → skeleton 위에서 additive assist가 작동함을 이미 실증.
3. 토크 예측 모델은 sim 수집 데이터 의존 + 피드포워드 한계 → **각도 예측
   (레퍼런스 생성기)으로 방향 전환**: mocap만으로 학습 가능하고, 각도는
   임피던스 제어(PD)가 소비하는 안전한 인터페이스.

---

## 3. 구성요소 현황 (2026-08-20)

### 3.1 HLP — ManiFlow flexion 예측 모델 ✅ 오프라인 검증 완료

| 항목 | 내용 |
|------|------|
| 학습 데이터 | HF `bones-studio/seed` locomotion 68,960클립 / 20.7M프레임 @ **40 fps** (120 fps ÷3) |
| 관측 | 각도 4ch: hip_flexion_r/l + trunk pitch/roll (rad, skeleton hinge 규약) |
| 출력 | flexion 2ch × horizon 16 (미래 400 ms) |
| 구조 | ManiFlow lowdim, n_obs 10 / n_act 4 |
| run | `hip_flexion_seed-maniflow_lowdim_policy_walking-locomotion-flexion40-run01_seed42` |
| best ckpt | `checkpoints/epoch=0021-val_loss=0.002854.ckpt` |
| 성능 (suit14 교차평가, 캘리브레이션 후) | **MAE 0.95° / R² 0.991 / corr 0.996** (denoise 3) — 전 14모션 R²≥0.96 |
| lead별 오차 | 25 ms 0.45° / 50 ms 1.18° / 75 ms 2.06° |
| 추론 지연 (batch 1, RTX 5090) | denoise 3 = **5.95 ms** (권장), 10 = 19.5 ms — 과제 목표 ≤20 ms 충족 |

- **캘리브레이션 (영점 오프셋)**: SEED와 ETRI suit는 중립자세 각도 규약이
  채널별 상수만큼 다름 — 실측: obs flexion +13.9°/+12.0°, trunk_pitch −10.2°,
  출력 +13.9°/+12.2°. 오프라인 평가는 데이터 전체 평균 정렬로 처리했고,
  온라인에서는 상수 고정(시뮬 통합) 또는 시작 시 중립자세 영점조정(실기기,
  미결정) 필요.
- suit14 평가 모션(ETRI 자체 수집 14종): walk, run, squat, highknee,
  stepinplace×3, onestepleft/right/long, onehopforward, walk_koo,
  lunge_left/right_koo.

### 3.2 LLP — hip pendulum assist RL ✅ 시뮬 검증 완료 (1단계)

2-DOF 진자(고정 골반 + 좌/우 대퇴 hinge, Kp=300/Kd=15/effort 150 N·m)에서
PPO로 학습. 비대칭 actor-critic: actor는 실측 가능값만(관절+히스토리, action
히스토리, chunk 창), critic은 +privileged(τ_agent, 깨끗한 θ_g, gain_scale).

**v2 결과** (paired 8env×10 s, θ_g bit-exact 동일): 사람 파워 **76.6% 절감**
(4.04 vs 17.27 W), 추종 오차 0.005 vs 0.015 rad, assist RMS 6.4 N·m,
전 env 67–83% 일관 절감. 체크포인트: `results/assist_pendulum_v2/`
(v1 순수 위치 PD: `results/assist_pendulum/`).

보상: tracking(exp, w=1.0) + human_power(|τ·θ̇|, w=−0.01) +
assist_effort(Στ², w=−2e-5) + action_rate(w=−0.005).
종료: 추종 오차 >1 rad 또는 각속도 >25 rad/s.

### 3.3 레거시 — ManiFlow hip torque 트랙 ⏸ 방향 전환으로 종료

`tasks/mimic_suit_active_cable_walk_23dof/`에 수집(collect_walk_zarr*)·
추론(infer_maniflow_newton)·A/B 비교(compare_maniflow_control_newton)·분석
문서(MANIFLOW_CONTROL_ANALYSIS.md, MANIFLOW_INFERENCE.md)가 남아 있다.
run02(재수집)→run03(DAgger)→run04(substep 평균 라벨)까지 진행 후 §2.6의
결론과 함께 종료. **단, 여기서 만든 인프라는 현역**: `JointTorqueOverride`,
`get_substep_mean_dof_forces()`, ManiFlow loader — 모두 LLP가 사용 중.

### 3.4 mimic tracker (skeleton 단계의 "착용자" 대역)

| 체크포인트 | 내용 |
|------------|------|
| `checkpoints/v18_2_newton_suit_passive_cable/` | suit(passive cable) 15모션(11+koo_4), Newton, epoch 5000, 성공률 100% — **git 추적됨(공유)** |
| `checkpoints/v18_newton_suit_passive_cable/` | v18_2의 이전 버전 |
| `tasks/mimic_suit_active_cable_walk_23dof/output_newton_flat/score_based.ckpt` | walk 전용 (active cable) — β 실험에 사용 |

⚠️ 비착용(plain) skeleton 로봇은 현재 **에셋이 레포에 없다**:
`skeleton.py`/`skeleton_torque.py`가 참조하는 `mjcf/skeleton_torque.xml`(및
`mjcf/31dof/skeleton_torque.xml`)이 존재하지 않음 (suit 계열 XML만 존재).
skeleton 단계는 suit 로봇(어차피 착용 상태가 물리적으로 맞음) 기준으로 진행.

---

## 4. 파일 지도

### 4.1 AI-SBC-PROTOMOTIONS (이 레포)

**LLP 스택** (모두 hip_pendulum 1단계에서 신설):

| 파일 | 역할 |
|------|------|
| `protomotions/envs/base_env/assist_env.py` | `AssistEnv` — step 흐름: action→τ_assist clamp→qfrc 주입→사람 PD step→τ_agent 계측. `_lazy_init_assist`(override 생성+게인 DR) |
| `protomotions/envs/control/assist_target_control.py` | `AssistTargetControl` — θ_g 생성(합성 sine, 세그먼트 재샘플+블렌드) + **chunk 에뮬레이션**(`_refresh_chunk`) + interpolation + 계측 버퍼. **HLP 통합의 결합점** |
| `protomotions/envs/context_views.py` | `AssistContext` (theta_d/theta_d_future/chunk_age/theta_g/tau_agent/tau_assist/gain_scale) |
| `protomotions/envs/obs/assist_obs.py` | proprio / target(chunk 창) / privileged 관측 커널 |
| `protomotions/envs/rewards/assist.py` | human_power / tracking / assist_effort 보상 + 평가 지표 커널 |
| `protomotions/envs/terminations/assist.py` | 추종 실패 / 과속 종료 |
| `protomotions/envs/action/action_functions.py` | `normalized_torque_action` (tanh × torque_scale) |
| `protomotions/robot_configs/hip_pendulum.py` | 로봇 정의 (factory.py에 등록) + `protomotions/data/assets/mjcf/hip_pendulum.xml` |
| `protomotions/simulator/newton/fixed_base.py` | `FixedBaseNewtonSimulator` — 고정 베이스용 root write/velocity 무해화 |
| `examples/experiments/assist_pendulum/mlp.py` | 실험 config (chunk 파라미터, 보상 weight, PPO 구조 전부 여기) |
| `examples/experiments/assist_pendulum/{play,compare_rollout,runtime_utils}.py` | 시각화(θ_g 고스트/토크 화살표) / paired 비교 / ckpt 재구성 유틸 |

**ManiFlow 통합 레이어** (`protomotions/maniflow/` — 시뮬 쪽에서 maniflow를
직접 import하지 않고 반드시 이 패키지를 경유):

| 파일 | 역할 | 재사용성 |
|------|------|----------|
| `loader.py` | maniflow 패키지 경로 해석(MANIFLOW_ROOT) + workspace ckpt→추론 정책 로드 | **범용** — flexion 모델도 그대로 사용 |
| `torque_estimator.py` | 레거시 torque 모델용 온라인 래퍼 (88차원 obs 전용) | 히스토리/priming 패턴만 재사용 |
| `hybrid_control.py` | `JointTorqueOverride` — per-env/DOF qfrc 주입 | **현역** (LLP가 사용) |
| `channels.py` | hip DOF 채널 계약 (`hip_dof_indices()`) | skeleton 단계에서 재사용 |
| `angle_estimator.py` | `ManiFlowAngleEstimator` — flexion 모델용 온라인 래퍼 (4ch 각도 obs → 2ch chunk, 영점 캘리브레이션·denoise 오버라이드 내장) | **HLP 실행기** (Phase C에서 결합) |
| `verify_angle_estimator.py` | Phase B 등가성 검증 스크립트 (`python -m protomotions.maniflow.verify_angle_estimator`, sbc env) — 2026-08-24 **PASS** (suit14 스트리밍 MAE 0.940°/0.961°, 기준과 Δ≤0.004°) | 래퍼 회귀 테스트로 상시 사용 가능 |

**데이터 파이프라인**: `data/scripts/extract_soma23_hip_imu_zarr.py` —
BONES-SEED .motion → flexion zarr 추출기 (`--features flexion_trunk`,
`--resample-fps`, soma23↔skeleton 규약 변환 내장).
`data/scripts/extract_seed_locomotion_bvh.py` — 45 GB 아카이브에서 locomotion
선별 추출.

### 4.2 ManiFlow_Policy (sibling 레포, `~/Projects/ManiFlow_Policy`)

| 경로 | 내용 |
|------|------|
| `ManiFlow/maniflow/config/walking_task/hip_flexion_seed.yaml` | flexion 태스크 config |
| `ManiFlow/data/hip-flexion-seed-locomotion-40fps.zarr` | 학습 데이터 (LLP θ_g 풀로도 재사용 예정) |
| `ManiFlow/data/hip-flexion-etri-suit14-40fps.zarr` | 교차평가 데이터 (분포 밖 검증용) |
| `ManiFlow/data/outputs/hip_flexion_seed-...-flexion40-run01_seed42/` | 학습 run + best ckpt |
| `scripts/compare_hip_imu_reference.py` | 오프라인 교차평가 (receding-horizon, 캘리브레이션, `--denoise_steps`, `--latency_bench`) — Phase B 등가성 검증의 기준 |
| `scripts/train_walking.sh`, `scripts/eval_walking_lowdim.py` | 학습/오프라인 평가 진입점 |

### 4.3 실행 환경 구분

| conda env | 용도 | 주의 |
|-----------|------|------|
| `sbc` | ProtoMotions 시뮬(Newton), 데이터 수집, **ManiFlow 온라인 추론** (runtime deps 설치됨: einops, timm, dill, hydra-core, omegaconf, zarr 3) | zarr 파일을 v2 포맷으로 쓸 때 `zarr.config.set({'default_zarr_format':2})` 필요 |
| `maniflow` | ManiFlow 학습·오프라인 평가 (zarr 2.12) | 학습 재개 시 latest.ckpt 대신 topk(epoch=*-val_loss=*.ckpt) 사용 |

---

## 5. 재현 명령어

```bash
# ── LLP 학습 (sbc env, 물리 400 Hz / policy 100 Hz) ──
python protomotions/train_agent.py --robot-name hip_pendulum --simulator newton \
    --experiment-path examples/experiments/assist_pendulum/mlp.py \
    --experiment-name assist_pendulum --motion-file none \
    --num-envs 4096 --batch-size 16384

# ── LLP 평가 (베이스라인 = assist 차단) ──
python protomotions/inference_agent.py --checkpoint results/assist_pendulum_v2/last.ckpt \
    --simulator newton --num-envs 16 --headless --full-eval
#   baseline: --overrides "env.assist_torque_limit=0"

# ── LLP 시각화 / paired 비교 ──
python examples/experiments/assist_pendulum/play.py --checkpoint results/assist_pendulum_v2/last.ckpt
python examples/experiments/assist_pendulum/compare_rollout.py --checkpoint results/assist_pendulum_v2/last.ckpt

# ── HLP 오프라인 교차평가 (maniflow env, ManiFlow_Policy repo에서) ──
python scripts/compare_hip_imu_reference.py \
    --zarr_path ManiFlow/data/hip-flexion-etri-suit14-40fps.zarr \
    --denoise_steps 3
```

---

## 6. 로드맵 (2026 하반기)

| Phase | 내용 | 상태 |
|-------|------|------|
| **A** | 프레임워크 문서화(본 문서) + CLAUDE.md 현행화 | ✅ 2026-08-20 |
| **B** | `ManiFlowAngleEstimator` 온라인 래퍼 신설 + **오프라인 등가성 검증** (suit14에서 compare 스크립트 수치 MAE 0.95° 재현) — 스트리밍 검증 PASS: MAE 0.940°/0.961°, R² 0.991/0.992, lead별 MAE까지 기준과 Δ≤0.004° (`results/angle_estimator_verify/2026-08-24_16-04-52/`) | ✅ 2026-08-24 |
| **C** | 진자에서 **실 HLP 연결**: ① SEED flexion GT 궤적 풀 로더 (40 fps zarr 재사용, trunk 채널 동시 스트리밍) ② `AssistTargetControl` 의도 소스 옵션화(synthetic/motion_data) ③ SEED 궤적 환경에서 LLP 재학습(chunk는 에뮬레이션 유지) ④ A/B/C 평가: assist off / 에뮬 chunk / **실 ManiFlow chunk** — 평가 궤적은 HLP 미학습분(SEED val split 3,449클립 + suit14) | 예정 |
| **D** | **skeleton(suit) 확장**: mimic tracker(v18_2)=착용자, LLP assist(hip flexion 2ch qfrc), HLP 온라인(trunk는 torso 자세에서 계산). 균형·낙상 개입 상태의 보상/종료 재설계. β 가산 실험(−31%)이 주입 인프라의 실증 전례 | 올해 목표 |
| 증강 (선택) | Kimodo 등 모션 생성 모델 → HLP 재학습 데이터 + LLP θ_g 풀 양쪽에 추가. skeleton 단계에서 특정 동작 예측 약점 확인 시 착수 | 보류 |
| 정리 (병행) | git zarr 18.9 GB 완전 제거(히스토리 재작성 포함, §7) — 대용량 데이터 HF 전환은 계속 | ✅ 2026-08-31 |

**Phase C의 검증 논리**: 에뮬레이션 chunk로 학습된 LLP가 실 HLP chunk에서도
성능을 유지하면 노이즈/지연 에뮬레이션 설계가 옳았다는 검증. 실패하면 실
HLP 오차 통계에 맞춰 에뮬레이션 파라미터 조정 또는 HLP-in-the-loop 재학습.
사람 의도(θ_g)를 SEED GT로 바꾸는 것이 전제 조건 — HLP는 locomotion으로
학습됐으므로 합성 sine 궤적에서는 예측이 성립하지 않는다 (§2.1).

---

## 7. 레포 위생 현황

- **zarr 18.9 GB 완전 제거 완료** (2026-08-31, 업로더인 협업 연구원님 승인):
  untrack + 로컬 사본 삭제 + **git 히스토리 재작성**(filter-repo,
  `tasks/mimic_suit_active_cable_walk_23dof/zarr_data/` 전 히스토리 제거).
  pack 17.7 GiB → 311 MiB, `.git` 22 GB → 4.0 GB(그중 3.6 GB는 LFS =
  체크포인트, 보존됨). 재작성으로 **전 커밋 해시가 변경**됨 — force push
  후 협업자는 재클론 필요. 재작성 전 백업:
  `~/Projects/AI-SBC-PROTOMOTIONS-backup-20260831.git` (안정화 후 삭제 가능).
- 대용량 데이터는 향후 HF(또는 사내 스토리지)로: repo에는 다운로드
  스크립트/경로만 커밋.
- `data/seed/` 260 GB(BVH 원본+변환본), `tasks/` 24 GB — 로컬 전용(.gitignore).

## 8. 함정 모음 (디버깅 시 먼저 볼 것)

- **paired 비교의 RNG 함정**: `_lazy_init_assist`의 게인 랜덤화가 첫 step에서
  RNG를 소비 → 같은 seed여도 두 조건이 다른 궤적을 받음. rollout 전에 lazy
  init 강제 (`runtime_utils.rollout_recorded` 참조). seed 기반 비교 시 항상 의심.
- **MdpComponent metadata 키**: static_params의 `threshold`/`weight`/`min_value`
  등은 metadata로 걸러져 compute 함수에 전달되지 않음 → 종료 조건 파라미터는
  `max_err`/`max_vel` 같은 다른 이름 사용.
- **Newton TORQUE/PROPORTIONAL 모드 배선 끊김** (`_update_torques` 미호출)
  → BUILT_IN_PD + qfrc 경로를 사용하는 것이 현 구조의 이유 중 하나.
- **Newton 마커**: `_update_simulator_markers`가 no-op — viewer의
  `log_lines/log_arrows` + `_render_hook`으로 직접 그려야 함 (play.py 참조).
- **fps 정합**: SEED 변환본·모델·suit 데이터 모두 **40 fps(25 ms)** 통일
  (2026-08-13). 30 fps 혼용 시 시간축 4/3 왜곡 — 과거 평가 일부가 이 상태였음.
- **zarr 버전**: 수집(sbc, zarr 3)은 v2 포맷 강제 기록, 학습(maniflow, 2.12)은
  v2만 읽음. 추출기의 `zarr_format` 설정을 건드리지 말 것.
- **`resolved_configs.pt`는 pickle** — `weights_only=False`로 로드. resume은
  실험 파일을 재실행하지 않고 pickle을 읽으므로 CLI `--overrides` 무시.
- **`apply_inference_overrides`는 학습 시점에 실행된다** — `train_agent.py:853`이
  config를 deepcopy해 적용한 뒤 `resolved_configs_inference.pt`로 저장한다.
  `inference_agent.py`는 실험 파일을 아예 로드하지 않고
  `apply_backward_compatibility_fixes`(:273)만 부른다. 따라서 **학습이 끝난
  뒤 `apply_inference_overrides`를 고쳐도 기존 체크포인트엔 반영 안 됨** →
  `train_agent.py --create-config-only`로 config만 다시 굽고 `--checkpoint`로
  가중치를 이어받아야 한다.
- **`examples/experiments/format.py`의 시그니처를 믿지 말 것** — upstream v3.1
  (`d6bd922bb`)에서 호출부가 8개 인자(+terrain/motion_lib/scene_lib)로 확장됐는데
  템플릿 `format.py`의 `apply_inference_overrides`만 5개로 방치됐다(upstream 버그,
  fork 이전부터). 호출부는 `try/except Exception`
  (`inference_utils.py:66-74`) 안이라 시그니처가 틀리면 `TypeError`가
  **`log.warning` 한 줄로 삼켜지고 학습은 그대로 진행** → eval 설정이 조용히
  누락된 채 `resolved_configs_inference.pt`가 구워진다.
  → **새 실험 파일은 `format.py`가 아니라 동작 검증된
  `examples/experiments/assist_pendulum/mlp.py`를 복사할 것.**
  (upstream 순정 파일이므로 고치지 않는다 — rebase 비용 회피)
- **실험 파일 안에서 class/함수/lambda를 정의해 config에 담지 말 것** —
  실험 파일은 `importlib` `exec_module`로 로드되어 모듈명이 항상
  `"experiment_module"`이고 `sys.modules`에 등록되지 않는다. pickle은 클래스를
  "모듈명+이름"으로만 저장하므로 이 이름표는 복원 불가 → `save_configs`에서
  `PicklingError`로 죽는다. 새 보상/obs 함수나 config 타입은 `protomotions/`
  아래에 정의하고 실험 파일에서 import만 할 것(값·상수는 안전).
- **감시/대기 루프의 pgrep -f 자기매칭** — 파이프라인은 알림 체인 대신 한
  스크립트로 체인할 것.

---

## 9. 코드 이해 가이드 (임시 — 완료한 항목은 지워나갈 것)

> 목적: exosuit 개발 관점에서 이 레포를 읽기 — 학습/평가가 어떻게 돌아가고
> 시뮬레이터가 어떻게 쓰이는지 파악해 앞으로 올바른 개발 지시를 내리는 것.
> LLP(assist_pendulum) 실험을 앵커로 삼는다.
>
> **읽는 방법: §9.1로 깊이를 정하고, §9.2의 실행 트레이스를 A→F 순서대로
> 따라간다.** 트레이스는 명령 실행 순서 그대로이므로 위에서 아래로 읽으면 된다.
> HLP(§9.4)는 LLP와 별도 학습이라 접점이 없다. 파악이 끝나면 이 섹션 삭제.

### 9.0 큰 그림 — 한 화면 요약

```
[학습]  train_agent.py
          config 조립 (factory → mlp.py → --overrides) → resolved_configs.pt 저장
          → Env(내부에 Simulator 보유) / Agent / 모델 생성        ← 여기까지 1회
          → agent.fit() 루프:
              rollout 32스텝(no_grad) → GAE → minibatch PPO → 주기적 eval/ckpt
                └ env.step() 안에서만 물리가 돈다 (400 Hz × decimation 4 = 100 Hz)
          상세 트레이스 = §9.2

[추론]  inference_agent.py
          resolved_configs_inference.pt 로드 (mlp.py 재실행 안 함)
          → 같은 Env/모델 재구성, fit() 없이 rollout만
          상세 = §9.3
```

시뮬레이터와의 경계는 **`Simulator` 추상 클래스 + `RobotState`(common
ordering)** 하나뿐이다. env 이하 모든 코드(obs/reward/모델/학습 루프)는 이
인터페이스만 보므로, `--simulator`를 바꿔도(sim2sim) 정책이 그대로 돈다.
예외가 §2.5의 Newton 전용 경로(qfrc 주입·계측) — IsaacLab 이전 시 재구현
대상이 바로 이것.

**IsaacLab 이전을 고려한 깊이 조절 원칙**: 추상화 경계 위의 코드(§9.2의 A~D·F
전부, E의 env 쪽)는 이전 후에도 100% 그대로이므로 깊게. Newton 내부 메커니즘
(MuJoCo Warp, qfrc, per-world 게인)은 "무슨 기능을 하는지"만 알면 됨 — 어차피
IsaacLab에서 같은 기능을 다시 만들 목록이다 (그 목록 = §9.1의 3줄).

### 9.1 깊이 배분 — "앞으로 고칠 코드인가"로 가른다

| 층 | 파일 | 어디까지 |
|----|------|----------|
| **A. 연구 자산** — 계속 고칠 곳, 시뮬레이터 교체 후에도 그대로 | `envs/base_env/assist_env.py`, `envs/control/assist_target_control.py`, `envs/rewards/assist.py`, `envs/obs/assist_obs.py`, `envs/terminations/assist.py`, `envs/context_views.py`(AssistContext), `examples/experiments/assist_pendulum/mlp.py`, `robot_configs/hip_pendulum.py` | **설계 의도까지** |
| **B. 계약** — 안 고침 | `envs/base_env/env.py`의 `post_physics_step()` 호출 순서, `envs/mdp_component.py` 규약, `simulator/base_simulator/simulator.py` 추상 + `RobotState`(common ordering), `agents/base_agent/agent.py` `fit()` + `agents/ppo/*` | "언제 무엇이 어떤 인자로 불리는가"만 |
| **C. 소모품** — Newton 전용, 버릴 것 | `simulator/newton/simulator.py` 내부(Warp 커널·CUDA graph·MuJoCo qfrc), `maniflow/hybrid_control.py`의 `robot_view` 접근 | **기능 이름만** (코드 안 읽어도 됨) |

C층이 A층에 침투한 지점은 딱 3곳 — 이것이 IsaacLab 이전 작업 목록 전부다:

| `assist_env.py` | 기능 | IsaacLab에서 대체 필요한 것 |
|-----------------|------|------------------------------|
| `_torque_override.set_torques(inject)` :178 | 관절 토크 **가산** 주입 | PD 위에 외부 토크 더하는 수단 |
| `simulator.get_substep_mean_dof_forces()` :186 | 사람 PD 토크 substep 평균 계측 | actuator 토크 읽기 (IsaacLab은 built-in PD에서도 applied torque를 보고하므로 오히려 쉬움) |
| `_randomize_human_gains()` :114 | env별 PD 게인 배율 | per-env PD 게인 설정 수단 |

### 9.2 LLP 실행 트레이스 (체크리스트)

명령 한 줄이 실행되는 순서 그대로. **경계는 `train_agent.py:891`의
`agent.fit()`** — 그 이전은 객체를 *만드는* 1회성 구간(물리도 gradient도 없음),
그 이후는 만든 객체를 *굴리는* 반복 구간.

```bash
python protomotions/train_agent.py --robot-name hip_pendulum --simulator newton \
    --experiment-path examples/experiments/assist_pendulum/mlp.py \
    --experiment-name assist_pendulum --motion-file none \
    --num-envs 4096 --batch-size 16384
```

#### 1부 — 준비 (fit() 이전, 1회)

- [ ] **A. 명령 → config 7덩어리**

```
train_agent.py:493  main()
├ :509  detect_checkpoint_mode()                        → mode = "fresh"
├ :567  load_experiment_module(experiment_path)
│   └ :366  importlib exec_module                       → mlp.py 본문 실행
├ :577  getattr(module, "env_config") 등 슬롯 함수 8개 꺼냄
├ :588  build_standard_configs(슬롯 8개, args)          → config_builder.py:18
│   ├ :50  robot_config("hip_pendulum") → robot_configs/hip_pendulum.py
│   │        └ base.py:235 __post_init__ → MJCF 파싱 → number_of_actions = 2
│   ├ :53  simulator_config("newton", …)                → NewtonSimulatorConfig
│   ├ mlp.py:51  configure_robot_and_simulator()
│   │              → sim_cfg._target_ 를 FixedBaseNewtonSimulator 로 교체
│   ├ mlp.py:61/66/70  terrain / scene_lib / motion_lib  (motion_file=None)
│   ├ mlp.py:75  env_config(robot_cfg, args)            → AssistEnvConfig
│   │              obs 4 / reward 4 / termination 2 /
│   │              control_components{"assist": AssistTargetControlConfig} /
│   │              action_config / assist_torque_limit=25  ← 전부 여기서 dict 조립
│   └ mlp.py:223 agent_config(robot_cfg, env_cfg, args) → PPOAgentConfig
│                  actor_in_keys 3 / critic_in_keys 4 / MLP 512·256·128
├ :608  CLI --overrides 적용
└ :688  Fabric(...)                                     → fabric.device
```

  - 확인: **resume(:515)은 이 A 구간을 통째로 건너뛰고** pickle만 읽는다 → `--overrides` 무시 (§8)
  - 확인: `ASSIST_TORQUE_LIMIT`(mlp.py:48) 하나가 `action_config.torque_scale`과 `assist_torque_limit` **두 곳**으로 복사된다

- [ ] **B. config → 객체** (`_target_` 문자열이 클래스가 되는 지점)

```
├ :767  build_all_components(..., device=fabric.device)
│   └ component_builder.py:102 build_simulator_from_config
│       └ :125  get_class(sim_cfg._target_)          ★ FixedBaseNewtonSimulator 객체
├ :788  EnvClass = get_class(env_cfg._target_)       ★ AssistEnv
├ :789  env = AssistEnv(config, robot_config, device, terrain, scene_lib,
│                       motion_lib, simulator)
│   └ env.py:143  BaseEnv.__init__
│       ├ :220  initialize_simulator()               → Newton model 빌드·기동
│       ├ :277  ControlManager(config.control_components, self)
│       │         └ AssistTargetControl 객체
│       │            (:138~152 버퍼: _chunk(n,8,2) / _tau_agent(n,2) / _gain_scale(n,2))
│       └ :285  ComponentManager(device)             → MdpComponent 등록
├ :802  AgentClass = get_class(agent_cfg._target_)   ★ PPO
├ :803  agent = PPO(config, env, fabric)             → base_agent/agent.py:86
└ :805  agent.setup()                                → agent.py:173
    ├ :175  create_model() → ppo/agent.py:114 → PPOModel → ppo/model.py:116
    │         ├ :121  PPOActor  → :60 MuClass = MLPWithConcat (512·256·128, num_out=2)
    │         │                 → :55 logstd = Parameter(ones(2) * −2.0)
    │         └ :124  critic = ModuleContainer
    ├ :193  dummy_obs = env.get_obs()
    ├ :196  self.model(dummy_obs_td)        ★ LazyLinear 형상 확정 (75→512 / 83→512)
    └ :201  create_optimizers()             → actor lr 1e-4 / critic lr 5e-4
```

  - 확인: `get_class` 3곳이 설정→객체 다리 전부. 문자열은 mlp.py와 robot config가 씀
  - 확인: :196의 dummy forward가 네트워크를 **실제로 만든다** → obs 구성을 바꾸면 기존 ckpt와 가중치가 안 맞음

#### 2부 — 루프 (fit() 이후, 반복)

- [ ] **C. 루프 골격**

```
train_agent.py:891  agent.fit()                     → base_agent/agent.py:385
├ :404  ExperienceBuffer(num_envs=4096, num_steps=32)
├ :409  env.reset() → obs / :414 obs 키 등록 / :421 1회 forward로 out_keys 등록
└ :451  while epoch < max_epochs:                            ← ⑤ 복귀
    ├ :455  self.eval()                             (normalizer 갱신 정지)
    ├ :459  for step in range(32):                           ← ④ 복귀
    │   ├ :464  env.reset(done_indices)              (끝난 env만)
    │   ├ :470  obs → 버퍼
    │   ├ :472  collect_rollout_step()               → D
    │   ├ :475  env.step(action)                     → E
    │   └ :490  record_rollout_step()                (rewards / dones 저장)
    ├ :503  normalize_rewards_in_buffer()
    ├ :514  optimize_model()                         → F
    └ :521 ckpt 저장 / :544 evaluator.evaluate() (주기적)
```

  - 확인: rollout 32스텝은 전부 `no_grad`(:456). gradient는 F에서만 흐른다
  - 확인: epoch당 4096 × 32 = 131,072 transition → batch_size 16384 → 8 미니배치
  - `num_steps`는 `batch_size`와 무관한 독립 config (기본 32)

- [ ] **D. 모델 forward — obs → action**

```
agent.py:472  collect_rollout_step(obs_td, step)
└ :376  self.model(obs_td)                          → ppo/model.py:141 PPOModel.forward
    ├ :157  self._actor(td)                         → model.py:69 PPOActor.forward
    │   ├ :79  self.mu(td)                          → common/mlp.py:121
    │   │        torch.cat([assist_proprio 44, assist_target 21,
    │   │                   previous_actions 10]) = 75
    │   │        → normalize → Linear 512→256→128→2
    │   │        → td["actor_trunk_out"] = mu  (4096, 2)
    │   ├ :81  std = exp(logstd) ≈ 0.135            (actor_logstd = −2.0, 고정)
    │   ├ :87  action = Normal(mu, std).sample()     (4096, 2)   ← 범위 제한 없음
    │   └ :90  neglogp = −log_prob.sum(−1)
    └ :160  self._critic(td)
             cat(위 3개 + assist_privileged 8) = 83 → value (4096, 1)
→ :381  action / mean_action / neglogp / value 를 버퍼에 저장
```

  - 확인: actor 75 vs critic 83 — 차이는 `assist_privileged`(τ_agent, τ_assist, θ_g−θ, gain_scale) 뿐. mlp.py의 `actor_in_keys` / `critic_in_keys`에서 갈라진다
  - 확인: 모델 출력은 **범위 없는 실수**. N·m로 만드는 건 E의 `tanh × 25`

- [ ] **E. env.step 한 번 — action → 물리 → 보상** ← 가장 중요

```
agent.py:475  env.step(actor_output["action"])       ← (4096, 2) 원시 실수
└ assist_env.py:148  AssistEnv.step(action)
    ├ :149  _lazy_init_assist()                     (첫 호출만)
    │   ├ hybrid_control.py:57  JointTorqueOverride(simulator, …)
    │   ├ :100  engage(gain_scale=1.0)
    │   └ :101  _randomize_human_gains()            → env별 ke/kd × 0.7~1.3
    ├ :157  _process_action(action, context)         → env.py:423
    │   └ action_functions.py:458  normalized_torque_action
    │       :480 tanh(action) → :484 × 25.0          → (4096, 2) [N·m]
    ├ :158  clamp(±assist_torque_limit)              → tau_assist
    ├ :170  comp.get_pd_targets_and_vel()            → assist_target_control.py:232
    │   └ :209  _target_at(t)  해석적 사인 궤적      → theta_g, target_vel
    ├ :171  tau_ff = _human_kd * target_vel                    ← 사람 몫 피드포워드
    ├ :178  _torque_override.set_torques(tau_assist + tau_ff)
    │   └ hybrid_control.py:184  nan_to_num → effort limit clamp
    │       └ :205  sim.robot_view.set_dof_forces(sim.control, wp)
    │                          ★ 정책 출력이 물리에 들어가는 유일한 지점 (qfrc 경로)
    ├ :181  simulator.step(theta_g, markers_callback)  ★ 인자가 정책 출력이 아니다
    │   └ base_simulator/simulator.py:648
    │       └ :676  _physics_step()                  → newton/simulator.py:1059
    │           ├ :1061  _apply_control() → base:1158  BUILT_IN_PD
    │           │         targets[:, dof_convert_to_sim]
    │           │         → :1360 _apply_simulator_pd_targets(theta_g)
    │           └ _simulate()  :1007
    │               └ :1012  for _ in range(decimation=4):
    │                     solver.step(state_0, state_1, control, contacts, 1/400)
    │                     state_0, state_1 = state_1, state_0
    │                     :1026  accumulate_qfrc_kernel → qfrc_actuator 누적
    ├ :186  tau_agent = simulator.get_substep_mean_dof_forces()  → :1301 누적/4
    ├ :188  tau_agent += tau_ff
    ├ :189  comp._tau_agent[:] = tau_agent
    ├ :191  post_physics_step()                      → env.py:688
    │   ├ :694  progress_buf += 1
    │   ├ :697  state_history.rotate_and_update()    (히스토리 10칸)
    │   ├ :749  control_manager.step()               → assist_target_control.py:382
    │   │         10 step마다 :308 _refresh_chunk()
    │   │           θ_g를 8 knot 떠서 delay+bias+noise → _chunk(n, 8, 2)
    │   │                                    ★ Phase C = 이 함수만 ManiFlow로 교체
    │   ├ :763  _build_global_context()
    │   │   └ :933  control_manager.populate_context(ctx) → :456 AssistContext 채움
    │   │           theta_d_future / theta_g / tau_agent / tau_assist /
    │   │           gain_scale / chunk_age
    │   ├ :765  compute_observations(context)
    │   │         MdpComponent가 dynamic_vars 경로로 텐서를 뽑아
    │   │         → assist_obs.py:31 / :62 / :89 호출
    │   ├ :766  compute_reward(context)              → rewards/assist.py 4개
    │   │         tracking(θ_g 기준!) · human_power · assist_effort · action_rate
    │   │         → weight 곱해 합 → rew_buf
    │   └ :767  check_resets_and_terminations        → terminations/assist.py
    └ :196  return obs, rew_buf, reset_buf, terminate_buf, extras
```

  - 확인: **힘이 두 갈래 병렬** — PD 경로(:181, 사람) / qfrc 경로(:178, 정책).
    `qfrc_actuator` 계측에 사람 토크만 잡히는 이유 (§2.5)
  - 확인: **:749가 :765보다 먼저** — chunk 갱신 후 obs 계산. control → context →
    obs → reward 이 순서가 `post_physics_step` 전체를 지배
  - 확인: tracking 보상은 정책이 못 보는 **θ_g로 채점**한다 (θ_d 아님) → 노이즈 주입의 목적
  - 함께 볼 것: `envs/mdp_component.py` (compute_func + dynamic_vars + static_params 규약),
    `simulator/base_simulator/simulator_state.py` (RobotState / common ordering / 쿼터니언 xyzw)

- [ ] **F. 최적화 — 버퍼 → GAE → PPO 업데이트**

```
agent.py:514  optimize_model()                       → agent.py:709
├ :710  pre_process_dataset()                        → ppo/agent.py:672
│   └ :643  compute_advantages()  GAE(gamma, tau)    → advantages / returns 버퍼에
├ :711  process_dataset(buffer.make_dict())          → DictDataset(batch_size=16384)
└ :715  for batch_idx in range(max_num_batches()):
    └ :733  perform_optimization_step(batch)         → ppo/agent.py:308
        ├ :321  actor_step(batch)                    → :396
        │     actor forward → mu
        │     current_neglogp = −Normal(mu, std).log_prob(batch["action"])
        │     ratio = exp(batch["neglogp"] − current_neglogp)
        │     clipped surrogate (e_clip) → actor_loss
        │     :325  adaptive_lr → KL 기반 lr 조정
        ├ critic_step(batch)                         → :536
        │     critic forward → value, MSE vs batch["returns"]
        └ fabric.backward → grad clip 50.0 → optimizer.step()
```

  - 확인: rollout 때 저장한 `neglogp`가 old policy 값 → ratio의 분자/분모가 여기서 만난다
  - 확인: eval 직후 1 epoch은 정책 업데이트를 건너뛴다 (`_skip_next_policy_update`, agent.py:559)

### 9.3 추론·평가 경로

- [ ] `examples/experiments/assist_pendulum/runtime_utils.py` **먼저**
      (inference_agent.py의 60줄 축소판)
- [ ] `protomotions/inference_agent.py` — `resolved_configs_inference.pt` 로드(:233)
      → `apply_backward_compatibility_fixes`(:273) → CLI overrides.
      **`apply_inference_overrides`는 여기서 호출되지 않는다** (§8)
      → 위 1부 A를 건너뛰고 B부터 재구성, `fit()` 없이 rollout만
- [ ] `examples/experiments/assist_pendulum/compare_rollout.py` —
      paired 평가 구성 + RNG pairing 함정 (§8)
- 베이스라인 트릭: `--overrides env.assist_torque_limit=0` → E의 :158 clamp가 0
      → 정책은 계속 forward하지만 주입값 전부 0 = 사람 PD 단독
- IsaacLab 관점: 경로 동일, `--simulator isaaclab`만 바뀜. 학습 Newton → 추론
      IsaacLab 같은 **sim2sim이 왜 가능한지**(모델이 common ordering 관측 계약에만
      의존)를 여기서 납득하는 것이 이전 준비의 핵심.

### 9.4 HLP — ManiFlow (LLP와 완전히 별도 학습)

위 트레이스와 **접점이 없다.** HLP는 `maniflow` env에서 mocap 지도학습으로
따로 훈련되고, LLP 학습 중에는 `_refresh_chunk`의 에뮬레이션이 그 자리를
대신한다 (Phase C에서 실제 결합). 시뮬레이터와 무관.

- [ ] `protomotions/maniflow/angle_estimator.py` — 온라인 래퍼.
      `observe(obs)`(:167) → `predict()`(:191), 내부는 `policy.predict_action()` 한 줄
- [ ] ManiFlow_Policy repo의 lowdim policy — `predict_action()` = normalizer →
      노이즈에서 denoise 반복 → action. obs_encoder 소형 MLP,
      DiTX transformer 6블록 / hidden 256 / 10.5M 파라미터
- [ ] walking workspace + dataset이 zarr를 (n_obs 10, horizon 16) 창으로 자르는 방식
- 확인: 래퍼의 히스토리 프라이밍(첫 관측을 10칸 복제)은 학습의 무엇과 대응인가
