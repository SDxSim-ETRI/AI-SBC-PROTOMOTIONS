# AI-SBC 프레임워크: HLP + LLP 고관절 보조 제어

> 강체(rigid-link) 고관절 assist 착용로봇의 딥러닝 기반 제어기 연구 (2026).
> 이 문서는 프레임워크 전체 구조, 코드 지도, 실험 이력, 로드맵의 단일 참조점이다.
> 최종 갱신: 2026-08-20

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
| 정리 (병행) | git zarr 18.9 GB untrack(연구원님과 협의 후, §7) + 대용량 데이터 HF 전환 | 협의 중 |

**Phase C의 검증 논리**: 에뮬레이션 chunk로 학습된 LLP가 실 HLP chunk에서도
성능을 유지하면 노이즈/지연 에뮬레이션 설계가 옳았다는 검증. 실패하면 실
HLP 오차 통계에 맞춰 에뮬레이션 파라미터 조정 또는 HLP-in-the-loop 재학습.
사람 의도(θ_g)를 SEED GT로 바꾸는 것이 전제 조건 — HLP는 locomotion으로
학습됐으므로 합성 sine 궤적에서는 예측이 성립하지 않는다 (§2.1).

---

## 7. 레포 위생 현황

- **zarr 18.9 GB / 88,318파일 untrack 완료** (2026-08-31, 커밋 `77ec55edb`,
  업로더인 협업 연구원님 승인). 로컬 사본(22 GB)도 삭제 — 데이터는 git
  히스토리에서 복구 가능. ⚠️ 이 커밋을 pull하면 협업자 로컬의
  `tasks/mimic_suit_active_cable_walk_23dof/zarr_data/`가 삭제됨.
  `.git`은 여전히 22 GB(히스토리 보존) — 완전한 용량 회수는
  히스토리 재작성(filter-repo, 전원 재클론 필요)로만 가능, 별도 협의.
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
- **감시/대기 루프의 pgrep -f 자기매칭** — 파이프라인은 알림 체인 대신 한
  스크립트로 체인할 것.
