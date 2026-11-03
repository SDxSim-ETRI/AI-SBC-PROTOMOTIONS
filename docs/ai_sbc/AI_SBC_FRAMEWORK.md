# AI-SBC 프레임워크: 설계와 코드 지도

> 2026-10-13 문서 정리. 현재 상태는 [HANDOFF.md](HANDOFF.md), 결과는 [실험 요약](experiment_summary.html),
> 모델 경로는 [AI_SBC_VERSIONS.md](AI_SBC_VERSIONS.md)를 읽는다.
> 이 문서는 설계·코드·디버깅 참조다. 과거 실험 수치와 세션별 진행 상황은 덧붙이지 않는다.
> 초기 구성과 이전 실험 기록은 [정리 전 원문](archive/20261013/AI_SBC_FRAMEWORK.md)에 보존했다.

---

## 1. 개요

각도 예측기(HLP, ManiFlow)가 실제 관절 움직임으로 미래 고관절 각도를 예측하고,
보조 제어기(LLP, PPO 정책)가 그 예측과 고관절 상태를 받아 좌우 보조 토크를 낸다.
가상 착용자(W1)는 시뮬레이션에서 사람 대신 원본 전신 동작을 추종한다.
실제 장치에 사용하는 제어 모델은 HLP와 LLP이며 W1은 시뮬레이션용이다.

```text
실제 고관절 각도·몸통 자세 → H1 각도 예측 → 현재~50 ms 미래각
실제 고관절 상태·히스토리 + 예측 → LLP → 좌우 보조 토크
착용자 토크 + 보조 토크 → suit 전신 물리 → 실제 상태로 피드백
```

물리는 400 Hz, W1은 20 Hz, 최근 보조 제어 실험은 100 Hz다.
목표는 원래 움직임을 방해하지 않으면서 착용자 부담을 줄이는 것이다.
고관절 토크 제곱 평균, 동작 추종, 낙상, 전신 관절 부담을 함께 평가한다.
기계적 파워 `|τ·ω|`만으로 성공을 판정하거나 토크 지표를 대사량 실측으로 해석하지 않는다.

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
| 현재 체크포인트 | `checkpoints/W1_wearer_human/last.ckpt` | H1 flexion100-h15-swap (버전표의 경로) |

### 2.3 Chunk 인터페이스 (HLP ↔ LLP 결합점)

`AssistTargetControl`이 관리하는 chunk 버퍼가 두 정책의 유일한 접점이다:

- **버퍼**: knot K개 × DOF, knot 간격 `hlp_dt`, 기준시각 `t0`
- **채움** (`_refresh_chunk`): 저주파(chunk_refresh_steps)마다 재생성.
  θ_g에서 미래 knot을 떠서 노이즈(bias+white+lead 비례 drift)와 지연을
  주입하는 **에뮬레이션**(학습용). 실 HLP 결합은 이 함수를 override해
  ManiFlow 예측으로 교체한다 (`MotionAssistTargetControl`, §4.1).
- **소비** (`_interp_chunk` → `populate_context`): policy step마다 knot 사이를
  선형 interpolation한 미래 목표 창(`theta_d_future`)을 LLP 관측으로 제공.
  v2 = 0~0.4 s (5×0.1), v3 = **0~0.15 s** (4×0.05, `obs_future_*`).

초기 H0 실험의 시간 스케일 (현재 H1은 §3.1): 물리 400 Hz → policy 100 Hz (decimation 4) →
chunk 갱신 v2 10 Hz / v3 50 Hz. ManiFlow 실모델의 native 스케일: 40 Hz(25 ms)
프레임, n_obs 10(= 250 ms), n_act 4, horizon 16.

⚠️ **horizon 16 ≠ 미래 400 ms** (2026-09-08 정정). horizon은 **관측 창과
정렬**돼 있어서 `action_pred` index k ↔ 상대 프레임 k−(n_obs−1) = k−9다.
즉 k=0..8은 과거(−225…−25 ms), k=9가 현재, k=10..15가 미래 → **미래는
6프레임 = 150 ms**뿐이다. 따라서 통합 시 `hlp_dt=0.025`,
`chunk_knots = horizon − n_obs + 1 = 7`(현재 + 미래 6). knot 수·간격은
`ManiFlowChunkSource`가 모델에서 유도하므로 더 긴 horizon으로 재학습해도
코드는 무변경이고 `obs_future_*`만 늘리면 된다.

### 2.4 사람 모델 (시뮬레이션 근사)

- **v1**: Newton BUILT_IN_PD — τ_agent = Kp(θ_g−θ) − Kd·θ̇ (substep implicit)
- **진자 v2 이후의 한 구성**: v1 + 속도 피드포워드 τ_ff = Kd·θ̇_g를 qfrc로 가산
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

## 3. 현재 구성과 실행 경로

### 3.1 각도 예측기

H1 = ManiFlow flexion100-h15-swap (2026-10-29부터). DiT-X에서 관측 토큰은 self-attention, 노이즈 섞인 action 토큰열은
cross-attention에 둔다(ManiFlow `obs_routing=swap`). 관측을 cross-attention에 두던 옛 H1(flexion100-h15-run01)은 폐기했고,
2026-10-29 이전 결과는 옛 H1로 측정했다. 고관절 좌우 각도와 몸통 pitch/roll의 4채널을 입력한다.
입력 간격 10 ms, 현재부터 50 ms까지 6점을 예측한다. LLP는 0/10/20 ms 미래 창을 소비한다.
체크포인트 위치는 버전표에 고정했다. 시뮬레이션 코드는 `protomotions/maniflow/`를 통해 접근한다.
40 Hz·150 ms의 H0는 초기 비교 모델이며 H1과 시간 계약을 혼용하지 않는다.

### 3.2 보조 제어기

기본 비교 모델 L1은 진자에서 학습했다. W1 평가 시 출력 0.5배와 10 ms 제어 주기를 사용한다.
L9는 전신 학습의 시작 모델이고 L13R 400회·L14 200회는 기존 걷기의 연구 후보다.
후보의 관측은 고관절 상태·히스토리·예측이며, 학습용 무보조 정답을 실행 입력으로 쓰지 않는다.
L12는 코드 준비 후 학습하지 않았다. 성능과 적용 한계는 실험 요약·버전표를 참조한다.

### 3.3 가상 착용자

W1 = `checkpoints/W1_wearer_human/last.ckpt`. suit 전신 동작 추종 모델이며 보조 실험에서는 동결한다.
현재 기준은 W1이고 W0 및 이전 착용자는 과거 비교용이다. 새 실제 보행의 무보조 추종은 아직 확보하지 못했다.

### 3.4 데이터와 변환

기존 gait 파일은 걷기(motion 0)와 달리기(motion 1)다. 재생 속도 변형은 같은 동작의 시간축을 바꾼다.
새 실제 보행은 SOMA23 원본을 suit로 변환한 후보이며, 원본 시간·속도를 유지한다.
`actor_uid`의 Axxx는 동작 수행자 코드다. 새 체형이나 새 W 모델을 뜻하지 않는다.

### 3.5 통합 평가

`run_system.py` → W1/모션 로드 → 영점 보정 → `AssistController` → 서브스텝 보조 주입 → A/B 토크 계측.
무보조 A와 보조 B는 같은 초기 조건으로 시작한다. 토크 창 정렬과 물리 실행 산포를 검사한다.
새 모션의 착용자만 검사할 때는 `evaluate.py native-wearer`를 사용한다. 짧은 클립을 이어 붙이지 않는다.

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
| `examples/experiments/assist_pendulum/mlp.py` | 진자 LLP 레시피 하나. `--recipe L1`(랜덤화 진자 40 N·m), `R0`(토크 경로 수정), `motion_leg`(R0 + 20 N·m tanh + SEED 동작 의도·단일 관절). `Recipe` 사전이 chunk 파라미터·보상 weight·PPO 구조를 겹쳐 쌓고, `protomotions/tests/test_pendulum_recipes.py`가 L1/R0/A0D0 `resolved_configs.pt`와 필드 단위로 일치하는지 검사 |
| `protomotions/envs/control/assist_disturbance_control.py` | `DisturbedAssistTargetControl` — 랜덤화 하중·수동 감쇠·구조화 사건·위상 사건·PD/FF 목표 유지(옛 pendulum_research/general control을 흡수, 새 항목은 기본 꺼짐). `protomotions/tests/test_pendulum_disturbance_events.py`가 병합 전 결과와 일치하는지 검사 |
| `examples/experiments/assist_pendulum/runtime.py` | `PendulumRuntime` — ckpt/실험 파일에서 env+agent 재구성, 기록 rollout(lazy init 강제), paired 비교, 시각화(θ_g 고스트/토크 화살표), env 점검, A/B/C 평가(`abc`: 보조 없음 / 에뮬 chunk / 실 ManiFlow chunk) |
| `examples/experiments/assist_pendulum/evaluate.py` | `PairedPendulumEvaluation`(고정 코호트 쌍 평가·실제 H1 예측), `ChunkDiagnosis`(학습 때 예측 신호 vs 실제 H1), `TrainingDistributionAudit`, `EtriEvalData`(ETRI 13동작 평가 데이터 준비) |
| `protomotions/agents/common/shared_joint_mlp.py` | 외부 관측 형식은 그대로 두고 다리별 입력과 출력을 분리하는 공유 가중치 actor (`motion_leg` 단일 관절 학습에 사용) |
| `examples/experiments/assist_pendulum/study.py` | `MotionLegStudy` — 20 N·m 단일 관절 연구의 준비·학습 lane·평가·진단·검증 순차 실행(소스 해시 고정, 재시도). 닫힌 연구 폴더가 기본값이므로 새 실행은 `--study results/llp20_motion_<date>` |
| `examples/experiments/assist_pendulum/report.py` | `MotionLegReport` — 위 연구의 HTML 보고서 (스켈레톤 `analysis.py`/`report.py`를 importlib로 불러 씀) |
| `results/<study>/scripts/` | 닫힌 진자 연구(L2–L12, 23 N·m 일반화, 적응, 한계 제어, F1/F2, R1)의 당시 실행 스크립트 보관. `SNAPSHOT.md`가 통합 전 스냅샷 커밋 `8058f3c`을 가리킨다 — 재현은 그 커밋의 worktree에서 |

**Phase C 결합 계층** (2026-09-08 신설 — 위 파일들은 **무변경**, 전부 상속·신규):

| 파일 | 역할 |
|------|------|
| `protomotions/envs/control/assist_motion_control.py` | `MotionAssistTargetControl` — `AssistTargetControl`을 상속해 훅 4개(`_target_at`/`_refresh_chunk`/`reset`/`step`)만 override. 의도 소스와 chunk 소스를 각각 교체 가능 |
| `protomotions/envs/control/motion_intent.py` | `MotionIntentSource` — mocap zarr → θ_g. 클립 필터(HLP val 분할 재현·동작종류·관절범위), 40 fps 선형보간, 클립 연결 crossfade, **전용 난수 generator** |
| `protomotions/maniflow/chunk_source.py` | `ManiFlowChunkSource` — 자체 링 버퍼(policy 10 ms) → 25 ms 격자 재표집 → `predict_horizon()` → chunk knot |
| `protomotions/maniflow/verify_chunk_source.py` | chunk 소스 등가성 검증 (suit14 스트리밍, 모드 A 25 ms / B 10 ms) |
| `protomotions/tests/test_motion_intent.py` | 의도 소스 단위 테스트 15개 |
| `examples/experiments/assist_pendulum/mlp.py` (`_env_hlp_chunk_format`) | HLP chunk 형식·관측창·control 클래스 교체 단계 (옛 mlp_v3 실험 config를 레시피 체인 한 단계로 흡수) |
| `examples/experiments/assist_pendulum/runtime.py abc` | A/B/C paired 평가 (assist off / 에뮬 chunk / 실 ManiFlow chunk) |

**통합 제어기 + 스켈레톤 시스템** (2026-09-12, §3.5):

| 파일 | 역할 |
|------|------|
| `protomotions/assist/controller.py` | `AssistController` — HLP+LLP 결합 독립 제어기 (env 비의존, 실기기 후보). `reset/observe/act`; `joint_sign`으로 관절 규약 변환 |
| `protomotions/assist/llp_policy.py` | `LLPPolicy` — assist_pendulum ckpt에서 actor + `LLPObsLayout`(히스토리·미래창·스케일·토크 한계) 복원 |
| `protomotions/assist/skeleton_features.py` | 스켈레톤 상태 → HLP 4ch (trunk = pelvis × lumbar FK), `calibration_offset_from_motion` |
| `protomotions/assist/verify_controller.py` | 진자 env ↔ 제어기 등가성 검증 (chunk 주입 / 링버퍼 / RNG 페어링) |
| `protomotions/simulator/newton/substep_hook.py` | `SubstepHookMixin` + `SubstepHook{,FixedBase}NewtonSimulator` — 서브스텝 콜백, 창 단위 PD 토크 평균, fixed-base 루트 z 오프셋 |
| `examples/experiments/assist_skeleton/run_system.py` | 폐루프 러너 (A 무보조 / B 보조 paired; `--human mocap_pd|tracker`) |
| `examples/experiments/assist_skeleton/{analysis,report,evaluate,plots,viz}.py` | 지표 클래스(`ObjectiveComparison`, `WalkingPhaseDigest`, `SwingGateBins` 등) / HTML 보고서(`report.py comparison|checkpoint|torque|swing-gate`) / 모션 일괄·착용자 평가(`evaluate.py motions|native-wearer|wearer`) / 그림(`plots.py`) / 뷰어 고스트·토크 화살표 |
| `examples/experiments/assist_skeleton/tracker_human.py`, `train_tracker_human.sh` | W1 착용자 tracker 학습 (옛 v30–v32 tracker 실행기는 `results/mimic_*/scripts/`에 보관) |
| `protomotions/tests/test_assist_controller.py` | 특징량·interpolation 단위 테스트 (CPU) |
| `protomotions/envs/base_env/skeleton_assist_env.py` | `SkeletonAssistEnv` — 보행 착용자(동결 tracker) 위 LLP 학습 env (action 2ch, τ_agent hip 계측) |
| `protomotions/envs/control/skeleton_assist_control.py` | `TrackerPolicy`/`TrackerMimicControl`/`SkeletonAssistControl` — 내장 tracker, 실 HLP chunk, hip 부분집합 컨텍스트 |
| `examples/experiments/assist_skeleton/llp_walk.py` | LLP 재학습 실험 파일 (학습 env 점검 `check_train_env.py`는 `results/llp_L13_fullbody_walk_20261011/scripts/`에 보관) |

**L13/R/L14 전신 걷기 추가학습 (2026-10-11)** — 결과와 현재 판정은
`HANDOFF.md`, `experiment_summary.html`을 우선한다. 정상 걷기 개발 통과와
다른 속도·구간의 일반화 판정을 구별한다.

| 파일 | 역할 |
|------|------|
| `protomotions/envs/base_env/paired_skeleton_assist_env.py` | 독립 무보조 A / 보조 B 전신 시뮬레이션. 같은 초기 상태·모션·난수, paired 부담·동작 비악화 보상. A 정보는 actor에 넣지 않음 |
| `protomotions/agents/ppo/assist_finetune.py` | actor만 전이, critic/optimizer 새 초기화. 진자 −굴곡 → 전신 +굴곡 좌표를 정규화·첫층·출력층까지 등가 변환. 전신 소스 +굴곡은 그대로 복사 |
| `protomotions/agents/fpo/` (2026-11-02) | FPO++(flow policy) agent. `config.py`(`FPOAgentConfig`·`FlowActorConfig`·`TemporalTransformerConfig`), `model.py`(`FlowActor`: context 한 번 → Euler N step 속도 MLP, `cfm_loss`; `TemporalTransformer`: 평탄 관측 137을 시간별 토큰 29개로 되돌려 self-attention 뒤 현재 토큰 읽기), `agent.py`(`FPO(PPO)`: actor 손실 = CFM 손실 비율 exp(L_old − L_new)·ASPO clip 0.05, actor EMA를 checkpoint actor에 구워 넣고 원 가중치는 `fpo_actor_raw`). `llp_walk.py --policy mlp_fpo|tfm_fpo --actor-init random`으로 쓰고, `LLPPolicy`는 `act_deterministic`(초기 잡음 0)으로 평가. `protomotions/tests/test_fpo_agent.py` |
| `results/llp_L13_fullbody_walk_20261011/scripts/` (보관) | `llp_walk_finetune.py`(L13: W1/실제 HLP 동결, 137차원 actor, 100 Hz, gain 1 / 최종 ±100 N·m), `llp_walk_finetune_relative.py`(L13R 상대 국소 비용), `llp_walk_tempo.py`(L14 .90/1/1.10 시간배율), `verify_walk_finetune.py`, `run_walk_{finetune,relative,tempo}_sequence.py`, `validate_walk_finetune.py`. 2026-10-26 닫힌 연구로 보관; 재현은 `SNAPSHOT.md`의 스냅샷 커밋에서 |

현재 paired 전신 env는 headless / flat / scene 없음만 지원한다. 두 env가 contact
label을 각각 smoothing하므로 motion library는 독립 사본이어야 한다. 재개 시 내부
평가를 비활성화한 recipe는 강제 내부 평가도 건너뛰고 외부 `run_system.py` 평가를 쓴다.
단일 GPU NCCL 통신이 정지한 이 환경에서는 `train_agent.py --distributed-backend gloo`로
실제 학습을 검증했다. 기본 backend 선택은 변경하지 않았다.

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

레포 루트, sbc 환경에서 기존 기준 모델을 평가하는 예다. 새 학습 시작 명령이 아니다.

```bash
/home/bak/miniconda3/envs/sbc/bin/python examples/experiments/assist_skeleton/run_system.py \
    --human tracker --tracker-checkpoint checkpoints/W1_wearer_human/last.ckpt \
    --llp-checkpoint checkpoints/L1_llp_pendulum/last.ckpt --llp-dt 0.01 --assist-scale 0.5 \
    --motion-file data/motion_for_trackers/skeleton_torque_suit_gait.pt --motion-id 0 \
    --seconds 40 --calibration wearer --seed 42
```

새 보행 무보조 평가의 저장 소스·설정·요약은 `results/native_walk_validation_20261013/`에 있다.
학습 재현은 각 run의 resolved config와 소스 스냅샷을 사용한다. 옛 학습 명령을 현재 계획으로 실행하지 않는다.

---

## 6. 다음 검증의 순서

현재 실행할 과제는 HANDOFF에서 관리한다. 연구 순서는 다음과 같다.

1. [진자 사전학습 개선안](pendulum_pretrain_discussion.md)을 먼저 실행한다. 진자 환경의 20 N·m 대조군, 외란, 관절별 정책을 비교한다.
2. 진자에서 선택한 정책을 기존 W1에 연결해 전이를 평가한다. 전신 보행 학습은 이번 단계에 포함하지 않는다.
3. 새 실제 보행의 무보조 추종 확보와 전신 추가학습은 후속 과제로 둔다. 새 자료의 보조 평가는 무보조 검증 뒤에 한다.

기존 걷기 한 패턴의 개선을 다양한 사람·계단·험지·실기기 검증으로 읽지 않는다.

---

## 7. 문서와 산출물 관리

- HANDOFF에는 현재 결론과 다음 작업만 남긴다. 실험 결론은 `experiment_summary.html`에 갱신한다.
- 모델 이름과 경로는 `AI_SBC_VERSIONS.md`, 실행 근거는 `results/<실험>/`에 저장한다.
- 보관 모델은 `checkpoints/<버전>/`와 INFO.md로 관리하며 가중치는 LFS 대상이다.
- 과거 문서는 `docs/ai_sbc/archive/20261013/`에 보존했다. 그 안의 중간 상태는 현재 지시가 아니다.
- 대용량 원본, 가중치, 실패 기록은 문서가 오래됐다는 이유만으로 삭제하지 않는다.

---

## 8. 함정 모음 (디버깅 시 먼저 볼 것)

- **이전 GPU 오류를 현재 접근 제한으로 단정하지 않는다 (2026-10-13).**
  현재 세션의 권한 설정부터 확인하고 `nvidia-smi` → sbc 절대경로 Python의 실제 CUDA 계산 →
  평가 명령으로 확인한다. Full access에서는 보통 실행으로 진행하고 승인 대기를 만들어내지 않는다.
  실행이 실패하면 그 명령·인터프리터·오류를 기록한다. 오래된 sandbox/Fabric/NVML 오류는
  현재 GPU 상태의 증거가 아니다. 10월 13일 Full access에서 실제 Newton 평가 32조건을 완료했다.
  HANDOFF 최신 인계와 과거 일지(archive/20261013/NEXT_SESSION.md) §2-AA 참조.

- **모션 재생 속도 변경을 실제 속도별 보행으로 해석하지 않는다 (2026-10-12).**
  L14의 .85–1.15배 데이터는 전신 위치·관절각·접촉 배열을 그대로 두고 시간 간격과 속도만 바꿨다.
  몸체 이동도 느려지지만 참조 보폭·발 높이·접촉 비율은 유지한다. 실제로 속도에 맞춰 보폭과
  자세를 바꾼 보행의 성공/실패를 입증하지 않는다. 기존 수치와 실패 판정은 보존하되,
  이 인공 조건의 통과를 다른 실제 동작 평가의 절대 선행 조건으로 삼지 않는다.
  실제 데이터 점검은 진자 학습 개선 이후의 과제다. 로컬 SEED 원본 후보가 있지만 SOMA23→suit
  전신 변환·착용자 무보조 추종·클립/인물 분할 검증이 필요하다. HLP 원본 목록에 포함된 후보를
  전체 시스템 미사용 평가라고 부르지 않는다. [데이터 점검 보고서](walking_data_audit.html),
  과거 일지(archive/20261013/NEXT_SESSION.md) §2-Y 참조.

- **paired 비교의 RNG 함정**: `_lazy_init_assist`의 게인 랜덤화가 첫 step에서
  RNG를 소비 → 같은 seed여도 두 조건이 다른 궤적을 받음. rollout 전에 lazy
  init 강제 (`runtime.PendulumRuntime.rollout` 참조). seed 기반 비교 시 항상 의심.
- **MdpComponent metadata 키**: static_params의 `threshold`/`weight`/`min_value`
  등은 metadata로 걸러져 compute 함수에 전달되지 않음 → 종료 조건 파라미터는
  `max_err`/`max_vel` 같은 다른 이름 사용.
- **Newton TORQUE/PROPORTIONAL 모드 배선 끊김** (`_update_torques` 미호출)
  → BUILT_IN_PD + qfrc 경로를 사용하는 것이 현 구조의 이유 중 하나.
- **Newton 마커**: `_update_simulator_markers`가 no-op — viewer의
  `log_lines/log_arrows` + `_render_hook`으로 직접 그려야 함 (`runtime.py play` 참조).
- **fps 정합**: 원본/변환 모션은 40 Hz, 현재 H1 입력은 100 Hz다. 저장된 시간 간격과
  모델의 입력 계약을 각각 읽고 명시적으로 재표집한다. 과거 H0의 40 Hz 계약을 H1에 적용하지 않는다.
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
- **Newton은 실행 간 bit-reproducible이 아니다** — 같은 seed로 두 번 돌려도
  `dof_pos` 최대 7.9e-3 rad, `tau_agent` 5.2 N·m 차이(2026-09-08 실측, 16env×10 s).
  반면 `theta_g`/`theta_d`는 완전히 동일. → **회귀 판정은 RNG 유래 신호의
  bit-identical + 집계 지표가 재실행 산포 안인지**로 한다. 기준값은 편집 전에
  같은 seed로 2회 기록해 산포를 먼저 측정할 것.
- **ManiFlow denoise가 전역 torch RNG를 소비한다** — paired 비교에서 의도
  궤적이 조건마다 어긋난다. `MotionIntentSource`는 **전용 `torch.Generator`**로
  클립을 샘플링한다 (`motion_seed`).
- **새 config 필드는 plain default만** — `resolved_configs*.pt`(pickle)는
  `__init__`을 거치지 않고 클래스 속성으로 해석되므로
  `field(default_factory=...)`는 기존 체크포인트에서 복원되지 않는다.
- **오프라인 기준과 lead별 MAE를 비교할 때 바이어스 추정 범위를 맞출 것** —
  기준 스크립트는 `action` 4프레임만으로 출력 바이어스를 추정한다. 7프레임
  전체로 추정하면 긴 lead의 잔차가 섞여 lead 0 MAE가 0.10°→0.31°로 부풀려진다.
- **감시/대기 루프의 pgrep -f 자기매칭** — 파이프라인은 알림 체인 대신 한
  스크립트로 체인할 것.
- **tracker의 env dt(50 ms)는 바꾸면 안 된다** — mimic 관측(미래 레퍼런스 프레임)이
  dt 간격이라 주기를 바꾸면 분포가 어긋난다. LLP(10 ms)는 `SubstepHookMixin`으로
  서브스텝 루프 안에서 돌린다 (물리 400 Hz / decimation 20 / hook_every 4).
- **서브스텝 훅 시뮬레이터는 CUDA graph를 아예 시도하지 않는다** — 캡처 중
  파이썬 콜백·torch 할당이 끼면 다른 프로세스가 GPU를 쓰는 동안 간헐적으로
  프로세스째 죽었다(학습 병행 시 2/14회). `wp.is_mempool_enabled`를 잠시 False로.
- **골반 고정 대역의 PD 게인은 로봇 config 값(hip 200/20)을 그대로 쓰면 안 된다** —
  다리 관성(3.53 kg·m²)에 비해 물러 ζ≈0.38, 12° 오버슈트 → HLP 분포 밖(150 ms
  MAE 8.8°). 진자 플랜트와 같은 ω_n/ζ(×12.4/×6.2)로 두면 추종 0.78°, MAE 6.3°.
- **`train_agent.py`에서 `import maniflow`가 통합 레이어를 잡는다** — 스크립트
  디렉터리 `protomotions/`가 sys.path[0]이고 ManiFlow의 `maniflow`는 namespace
  package라, 일반 패키지 `protomotions/maniflow`가 순서와 무관하게 이긴다.
  `loader.ensure_maniflow_importable`이 진짜 패키지를 `sys.modules`에 명시 등록한다.
- **mimic tracker의 PD 목표는 "의도"가 아니다** — 실제 각도와 15~20° 차이 나는 토크
  명령이다. 보행 착용자에서 LLP의 θ_g는 mocap hip flexion(평가용)으로 두고, 보상은
  hip 추종 대신 착용자 몸체 추종 항으로.
- **진자 LLP의 토크 스케일** — 출력 한계 25 N·m는 진자(I=0.43 kg·m²) 기준.
  스켈레톤 다리(3.53)는 8배 토크가 필요하다. `--assist-scale`은 진단용 배율일 뿐
  해법이 아니다(관절 다리에서는 배율을 올려도 corr 0.19로 절감이 늘지 않는다).

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
│       └ :218  initialize_simulator()               (정의는 :220)
│           ├ :277  ControlManager(config.control_components, self)
│           │         └ AssistTargetControl 객체
│           │            (:138~152 버퍼: _chunk(n,8,2) / _tau_agent(n,2) / _gain_scale(n,2))
│           ├ :282  simulator._initialize_with_markers(markers)
│           │   └ base_simulator/simulator.py:214  _create_simulation()
│           │       └ newton/simulator.py:218      ★ Newton model 빌드·기동
│           ├ :285  ComponentManager(device)         → MdpComponent 등록
│           └ :292  _initialize_observations()
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

- [ ] `examples/experiments/assist_pendulum/runtime.py` `PendulumRuntime.from_checkpoint` **먼저**
      (inference_agent.py의 축소판)
- [ ] `protomotions/inference_agent.py` — `resolved_configs_inference.pt` 로드(:233)
      → `apply_backward_compatibility_fixes`(:273) → CLI overrides.
      **`apply_inference_overrides`는 여기서 호출되지 않는다** (§8)
      → 위 1부 A를 건너뛰고 B부터 재구성, `fit()` 없이 rollout만
- [ ] `examples/experiments/assist_pendulum/runtime.py compare` (`PendulumRuntime.compare`) —
      paired 평가 구성 + RNG pairing 함정 (§8)
- 베이스라인 트릭: `--overrides env.assist_torque_limit=0` → E의 :158 clamp가 0
      → 정책은 계속 forward하지만 주입값 전부 0 = 사람 PD 단독
- IsaacLab 관점: 경로 동일, `--simulator isaaclab`만 바뀜. 학습 Newton → 추론
      IsaacLab 같은 **sim2sim이 왜 가능한지**(모델이 common ordering 관측 계약에만
      의존)를 여기서 납득하는 것이 이전 준비의 핵심.

### 9.4 HLP — ManiFlow (LLP와 완전히 별도 학습)

HLP는 `maniflow` env에서 mocap 지도학습으로 따로 훈련되고, **LLP 학습
중에는** `_refresh_chunk`의 에뮬레이션이 그 자리를 대신한다 (학습 트레이스와
접점 없음). **평가 시에는** `MotionAssistTargetControl`이 `_refresh_chunk`를
override해 `ManiFlowChunkSource`(§4.1)로 실제 예측을 넣는다.

- [ ] `protomotions/maniflow/angle_estimator.py` — 온라인 래퍼.
      `observe(obs)` → `predict()`, 내부는 `policy.predict_action()` 한 줄.
      미래 전체(150 ms)는 `predict_horizon()`, 외부 격자 주입은 `set_history()`
- [ ] ManiFlow_Policy repo의 lowdim policy — `predict_action()` = normalizer →
      노이즈에서 denoise 반복 → action. obs_encoder 소형 MLP,
      DiTX transformer 6블록 / hidden 256 / 10.5M 파라미터
- [ ] walking workspace + dataset이 zarr를 (n_obs 10, horizon 16) 창으로 자르는 방식
- 확인: 래퍼의 히스토리 프라이밍(첫 관측을 10칸 복제)은 학습의 무엇과 대응인가

## 10. 사용자용 보고서 작성 원칙

사용자 지시(2026-10-11): 사용자는 대화 중간의 설명을 읽지 않고 보고서만 볼 수 있다.
**보고서는 대화와 다른 문서를 읽지 않아도 이해할 수 있어야 한다.**
ASD-STE100의 명확한 문장과 일관된 용어 원칙을 참고한다.
참고: [ASD-STE100 공식 설명](https://www.asd-ste100.org/STE_faq.html).
한국어 보고서를 영어 표준 전체에 적합하다고 표현하지 않는다.

- 한 문장에는 한 가지 핵심 내용을 쓴다. 주체와 동작을 직접 쓴다.
- 같은 대상에는 같은 이름을 사용한다. 대화에서 설명했어도 보고서에서 다시 정의한다.
- 약어는 처음 사용할 때 전체 이름과 역할을 설명한다. 불필요한 약어는 쓰지 않는다.
- 표 제목, 행 이름, 그래프 축과 범례에도 독자가 이해할 설명 이름을 사용한다.
  예: `L13R_e300` → **부담 비율 반영 · 학습 300회**.
  `R`은 Relative(상대값)다. 무보조 토크 크기에 대한 부담 증가 비율을 비용에 반영한 실험이다.
  `e300`은 이 실험의 학습 반복 300회 시점이다. 반복 1회의 데이터 양도 설명한다.
- “독립 비악화 검증” 같은 압축된 표현 대신 실제 절차를 쓴다.
  예: **학습을 멈춘 모델로 별도 시뮬레이션을 실행했다. 보조 전후의 토크와 걷기 동작을 비교했다.**
  별도 실행인지, 학습에 쓰지 않은 데이터인지도 구분한다.
- 각 결과에 비교 대상, 평가 조건, 단위, 값의 부호와 계산 방법을 설명한다.
  변화량과 감소량의 부호가 반대이면 해당 표와 그래프에서 각각 설명한다.
- “통과”에는 적용한 기준을 쓴다. 일부 조건 통과를 모든 조건의 개선으로 표현하지 않는다.
- 측정 결과, 원인에 대한 해석, 아직 확인하지 못한 내용을 구분한다.
- 원본 실험 이름, 경로, 해시는 재현용 부록에 보존한다. 설명 이름을 바꿔도 원본 결과는 바꾸지 않는다.
- HTML 그림은 파일 안에 넣는다. 생성 뒤 긴 이름의 잘림, 숫자의 보존, 외부 이미지 의존성을 확인한다.

이 원칙을 적용한 보고서: [`walking_finetune_report.html`](walking_finetune_report.html).

HTML 보고서의 입구는 [reports.html](reports.html)이다. 최근 결과·비교 기준·원인 진단·과거 참고 자료를 구분한다.
종료된 연구의 상세 보고서는 `docs/ai_sbc/archive/reports/`에 두고 이전 주소에는 이동 안내를 남긴다.
보고서를 다시 생성한 뒤 `python docs/ai_sbc/organize_reports.py`를 실행하면 공통 탐색·결론·펼침 구성을 적용한다.
표·SVG·이미지·원본 JSON·스크립트·선택/저장 버튼의 원문을 보존하는지 함께 검사한다.
