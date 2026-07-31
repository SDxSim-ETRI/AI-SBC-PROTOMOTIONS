# ManiFlow 예측·제어 괴리 분석 메모 (2026-07-10)

> **질문**: passive 관찰에서는 hip 토크 예측이 corr 1.000으로 완벽한데, 왜 그
> 토크를 실제로 인가(A/B 비교)하면 넘어지는가? 그것도 처음부터? ManiFlow의
> 입력으로 가야 할 것이 안 가고 있는 것 아닌가?

## 결론 3줄 요약

1. **입력 배선은 정상** — 4가지 독립 근거로 검증됨 (아래 "입력 배선 검증").
2. 리셋 직후 **첫 chunk(스텝 0–2) 오예측은 실재**했고 원인도 규명됨(리셋 상태
   s₀가 학습 데이터에 없음). `--handover-steps`로 우회 가능.
3. 우회해도 **6/6 낙상** → 진짜 병목은 예측 품질이 아니라 **substep 피드백
   (관절 임피던스)의 부재**. 예측이 문자 그대로 완벽해도 개루프(피드포워드)
   토크 재생으로는 보행이 유지되지 않는다 — 피드백 결합이 필수.

## 셋업 확인

| | 제어 | 비고 |
|---|------|------|
| **Agent A** (env 0, 고스트) | 전 관절 RL policy + built-in PD | 순수 RL |
| **Agent B** (env 1, 메시) | **hip 6채널만 ManiFlow 토크**(게인 0 + `qfrc_applied` 주입), 나머지 21 DOF는 RL+PD | ManiFlow는 hip만 담당 |

두 agent는 매 에피소드 **같은 모션(id 0)·t=0·같은 스폰 좌표·reset noise 없이
함께** 시작한다 (워밍업 구간 hip 관절각 차 7e-5 rad로 실측 확인). "따로
시작"하는 일은 없다.

## 관찰 1 — 영상의 빨강 vs 검정은 예측오차 지표가 아니다

- 검정(`tau_a_applied`) = **A의 상태**에서 A의 PD가 낸 토크,
  빨강(`tau_b_cmd`) = **B의 상태**를 보고 ManiFlow가 낸 토크.
- 상태가 갈라진 뒤에는 완벽한 모델이라도 두 곡선은 달라야 정상.
  metrics의 채널 corr(≈0.02–0.25)도 같은 이유로 낮게 나오는 것 — 모델 품질
  지표로 읽으면 안 됨.
- **진짜 예측 품질 지표**는 같은 chunk에서 함께 기록되는 주황 점선
  `pred_a_passive`(A 상태 기준 예측) vs 검정(A 실측).

## 관찰 2 — "처음부터 나쁘다"의 실체: 리셋 직후 첫 chunk만 진짜 실패

`maniflow_control_results/2026-07-10_09-58-38`(handover 없음, run02 모델)
traces.npz를 에피소드-상대 시간으로 분해:

| 에피소드 내 스텝 | \|predA−A\| (같은 상태 기준) | B−A hip 관절각 차 | B root z |
|---|---|---|---|
| 0–3 | **62.7 N·m** (t=0 실측 +160.6 vs 예측 −85.5, 부호 반대) | 0.39 rad | 0.97 m |
| 3–6 | 0.72 N·m | 0.60 rad | 0.98 m |
| 10–20 | 0.33 N·m | 0.90 rad | 0.51 m (낙상 중) |
| 40–160 | 0.28 N·m | — | 0.07 m (쓰러짐) |

- 즉 **t≥3부터 모델은 이 run 안에서도 완벽**(corr 0.98–0.99). 첫 3스텝의
  ±250 N·m급 오토크가 게인 0(무저항)이 된 hip을 0.9–1.3 rad 밀어버리고,
  이후의 "정확한" 예측은 이탈한 B 상태 기준이라 복원력이 없어 0.65–0.9 s에
  6/6 낙상.

### 첫 chunk가 실패하는 원인 (코드로 확정)

- **수집기(`collect_walk_zarr.py`)는 `env.reset()` 후 `env.step()`을 먼저 하고
  기록** → 학습 데이터의 에피소드 첫 프레임은 물리 1스텝 후의 **s₁**.
  리셋 직후 상태 **s₀는 학습 데이터에 아예 없다** (contact 플래그 미갱신 —
  물리 스텝 전이라 stale, 속도는 solver 출력이 아닌 기구학 세팅값).
- 반면 inference(estimator)는 리셋 직후 s₀를 관측해 즉시 첫 chunk를 예측.
- 히스토리 복제 padding(`pad_before=1`) 샘플은 에피소드당 1개 = 학습 샘플의
  **~0.08%** → 시작 케이스 자체가 사실상 학습 안 됨.
- 리셋 직후 실제 토크는 +160→−327→−251 N·m의 정착(transient) 스파이크인데
  모델은 분포 밖 입력에 −85 근처의 "무난한 보행 토크"를 출력.

### passive가 완벽해 보였던 이유

passive run(600스텝)은 termination이 없고 RL이 안 넘어져 **리셋이 처음 1회뿐**
→ 첫 3스텝 오염이 지표에 묻힘 (보고된 MAE 0.383 N·m의 대부분이 그 3스텝 몫:
3/600×60≈0.3). A/B는 매 에피소드(grace 5 s=100스텝)마다 리셋하므로 매번 첫
chunk 실패가 반복 재생된 것.

## 관찰 3 — warm handover 실험: 첫 chunk를 고쳐도 넘어진다 (지배 요인 확정)

`--handover-steps K`(compare 스크립트에 구현): 에피소드 첫 K스텝은 B도 순수
PD로 보행(estimator에는 실제 관측 히스토리 축적)한 뒤 ManiFlow 제어로 전환.

| 조건 | 전환 시점 예측오차 | ManiFlow 제어 후 생존 | 결과 |
|---|---|---|---|
| 리셋 직후 전환 (기존) | 62.7 N·m (부호 반대) | 13–18스텝 (0.7–0.9 s) | 6/6 낙상 |
| **handover 40스텝** | **0.38 N·m (완벽)** | **~26스텝 (1.3 s ≈ 한 보행주기)** | 6/6 낙상 (65–67) |
| handover + every_step 재예측 | 동일 | ~23스텝 | 6/6 낙상 (62–64) |

- 전환 순간: B 상태는 A와 7e-5 rad 차이, 첫 인가 토크는 A 실측과 0.33 N·m
  차이 — **완벽한 상태·히스토리·예측에서 출발**.
- 그래도 root z 0.96→0.68 단조 발산, 6/6이 거의 같은 스텝에서 낙상. 그동안
  A-상태 기준 예측오차는 0.26–0.38 N·m로 **계속 완벽** — 예측이 나빠져서
  넘어지는 게 아니라, 넘어지는 동안에도 예측은 완벽.
- **every_step(재예측 지연 150ms→50ms)이 생존 시간을 전혀 늘리지 못함** →
  예측 신선도도 병목이 아님.

## 입력 배선 검증 — "입력이 안 간다" 가설 배제 근거 4가지

1. **입력 경로는 A·B 공유**: `estimator.observe()`가 두 env를 한 배치(2행)로
   받아 한 번의 `predict()`로 두 행을 함께 출력. B만 입력이 끊길 분기가 없고,
   같은 run에서 A행 예측이 전 구간 완벽 = 파이프라인 상시 건강검진.
2. **행 스왑(A↔B) 배제**: B가 A의 관측을 받고 있었다면 상태가 갈라진 구간에서
   B 명령이 A 실측과 corr≈1이어야 하는데 실제 0.03–0.25(무상관) → B는 B 자신의
   상태로 예측 중. 역으로 A행 예측은 B가 쓰러진 동안에도 완벽.
3. **전환 순간 직접 증거**: 실제 히스토리 2프레임 + 상태 일치(7e-5 rad)에서
   첫 인가 토크가 A 실측과 0.33 N·m 차이.
4. **출력측 검증**: override 채널 qfrc_actuator 잔여 0.000 (PD 완전 차단,
   주입은 `qfrc_applied` 경유로 의도한 채널에만).

(참고) "참조 모션이 입력에 없다"는 지적이라면 — 맞음, 설계상 고유수용성
센서(dof_pos/vel, root pos/vel, contacts = 88차원)만 입력. 다만 전환 직후
예측이 완벽했으므로 이것도 낙상 원인은 아님.

## 왜 "완벽한 예측 ↔ 낙상"이 모순이 아닌가

- passive가 보여준 것: **(상태 → 그때 RL+PD가 낸 토크) 함수 근사가 완벽**하다.
- 제어가 요구하는 것: **그 토크 시퀀스를 개루프로 재생해도 같은 운동이 나올
  것** — 전혀 다른 성질(개루프 안정성)이고, 보행(도립진자, 시정수 ~0.3 s)에서는
  성립하지 않는다.
- 토크가 만들어지는 방식의 차이가 본질:
  - **A의 hip**: 매 물리 substep마다 현재 상태로 재계산되는 피드백 법칙
    (kp=200 N·m/rad → 0.1 rad 오차에 같은 substep 안에서 즉시 +20 N·m 보정).
  - **B의 hip**: 게인 0 + ManiFlow 출력 하나를 50 ms 내내 상수(ZOH) 유지.
    학습 GT 자체가 **마지막 substep의 점 샘플**(`state_0.mujoco.qfrc_actuator`
    readback)이라, 값이 정확해도 스텝 내부 토크 프로파일은 PD와 다름.
- 비유: 손바닥 위 빗자루 세우기. 성공한 손의 힘을 완벽히 기록해 눈 감고
  재생하면 1초 안에 떨어진다 — 기록이 틀려서가 아니라 균형이 매 순간의 미세
  보정으로 유지되기 때문. ManiFlow는 법칙의 **출력값**을 20 Hz에서 복제하지만
  법칙 **자체**(연속 피드백)를 대체하지 못한다.

## 잔여 PD 결합 실험 (2026-07-14) — 위 결론의 실증 + 해법

위 "다음 단계 1순위"를 구현·실행했다. Agent B의 hip 토크를

    τ_hip = α·PD(q*, s)  +  (1-α)·τ_ManiFlow      (convex 블렌드)

로 구성: `JointTorqueOverride.engage(gain_scale=α)`가 게인을 0 대신 α배로
남기고(substep 임피던스 α 비율 복원), ManiFlow 주입 토크에 (1-α)를 곱한다.
정상 매니폴드에서는 τ_MF ≈ PD 적용 토크이므로 총토크가 A와 같게 유지된다
(명목 보행 보존 — 게인만 낮추고 MF를 전량 더하면 과구동으로 걸음 자체가
바뀜). 실행: `compare_maniflow_control_newton.sh --residual-pd-scale α`.

결과 (run02 ckpt, 1200스텝=60 s, receding, chunk_offset 1, seed 42):

| α (PD 몫) | handover | 생존 | B 낙상 시점(전환 후) | dof err6 (A→B) | corr(A, B총토크) |
|---|---|---|---|---|---|
| 0 (7/10 기준) | 40 | 6/6 낙상 | ~26스텝 (1.3 s) | — | — |
| 0.1 | 40 | 12회 낙상 | 27–32스텝 | 0.136→0.337 | — |
| 0.25 | 40 | 6회 낙상 | 114–218스텝 (5.7–10.9 s) | 0.135→0.216 | 0.34–0.49 |
| 0.375 | 40 | **60 s 완주** | — | 0.128→0.137 | ~0.90 |
| 0.5 | 40 | **60 s 완주** | — | 0.129→**0.130** | **0.91–0.95** |
| 0.5 | **0** | **60 s 완주** | — | 0.128→0.131 | 0.94–0.95 |
| 0.5 + **MF 끔** | 40 | 60 s 완주 | — | 0.128→**0.157** | 0.47–0.63 |
| **0.375 + MF 끔** † | 40 | **6/6 낙상** | **38–40스텝 (~2.0 s)** | 0.136→0.308 | ≈0 (−0.18~0.16) |

† 2026-07-16 녹화런(run04 로드, 600스텝)에서 측정 — MF 기여가 0이라 모델
무관, 순수 0.375·PD 단독 조건.

읽는 법:

- **생존 문턱은 α ∈ (0.25, 0.375]**. α=0.1은 α=0과 사실상 동일(임피던스
  10%로는 무의미), α=0.25는 생존 시간을 5–8배 늘리지만 결국 낙상.
- **α=0.5는 A와 구분 불가능한 보행**: 트래킹 오차 +1%, reward -0.7%, A↔B
  root 발산 평균 1.1 cm. ManiFlow 주입분 std ≈ A 토크 std의 절반 — 블렌드
  산술 그대로 hip 토크의 50%를 ManiFlow가 담당한다.
- **α=0.5는 handover도 불필요**: 리셋 직후 첫 chunk 오예측(관찰 2)까지 PD가
  흡수 — h0에서도 완주. 첫 chunk 문제는 실용상 해소.
- **ablation(α=0.5, MF 끔)이 중요**: 반게인 PD 단독도 생존은 한다(RL policy가
  20 Hz로 목표각을 보정하므로). 즉 α=0.5에서 "생존"은 ManiFlow의 공이 아니다.
  ManiFlow의 기여는 **품질 복원**: MF를 끄면 dof err6 +22%, reward -11%,
  corr(A,B총) 0.47–0.63으로 걸음이 A에서 이탈하는 것을, MF 피드포워드가
  A 수준(+1%, 0.95)으로 되돌린다. "PD가 안정화를, ManiFlow가 보행 토크
  본체를" 분담이 실증됨.
- **α=0.375에서는 MF가 생존 필수 (2026-07-16 확인)**: 같은 MF 끔 조건을
  α=0.375로 내리면 **매 에피소드 보조 차단 ~2 s 뒤 낙상**(6/6, 재현성 극도로
  높음 — 38–40스텝), corrT ≈0으로 보행 와해. 즉 **PD 단독 생존 문턱은
  α ∈ (0.375, 0.5]** — α=0.375 착용자(잔존 근력 37.5%)에게 슈트 보조는
  품질 복원이 아니라 **낙상 방지 그 자체**다. 반면 MF 켜면 같은 α로 30 s
  완주(err6 +7.6%, corrT 0.82–0.88, root 발산 1.6 cm). α=0.5 ablation과
  합치면: 보조 몫 50%는 품질 개선, 62.5%는 생존 조건.
- **응용 관점 (2026-07-14 추가)**: 실제 적용 대상은 완전한 하반신 마비자가
  아니라 **보조력(assist) 제공**이 목표이므로, α<0.375에서 낙상하는 것이
  치명적 한계는 아니다. 시뮬레이션의 잔여 PD 항은 실사용에서 **착용자의 잔존
  근력·반사(자체 임피던스)** 가 맡는 역할에 대응한다: α ≈ 착용자 잔존 기여,
  (1-α) ≈ 슈트 보조 몫으로 읽으면 α 스윕은 곧 **보조 비율 스윕**이고,
  α=0.5는 "hip 토크의 50%를 슈트가 보조해도 보행이 A와 동등"이라는
  응용상 유의미한 결과다.
- 그래도 α를 더 내리려면(슈트 보조 몫 확대 방향) estimator-in-the-loop
  (DAgger류) 재학습이 다음 수단 — off-manifold 교정 토크를 배우게 하는 것.

### 시연/녹화 명령 (권장 설정: α=0.5, run02)

```bash
bash tasks/mimic_suit_active_cable_walk_23dof/compare_maniflow_control_newton.sh --viewer --record \
    --maniflow-run-dir ~/Projects/ManiFlow_Policy/ManiFlow/data/outputs/walking_flat-maniflow_lowdim_policy_walking-newton-hips-run02_seed42 \
    --residual-pd-scale 0.5
```

A=반투명 고스트(순수 RL), B=메시(0.5·PD + 0.5·ManiFlow). `--record`는 시뮬
mp4와 토크 패널 합성 `sim_with_torque.mp4`를 결과 디렉토리에 저장한다.
2026-07-14부터 `--maniflow-run-dir` 기본값이 run02라 생략해도 되지만, 재현
기록을 위해 명시를 권장.

**보조 ON/OFF 대비 영상 (α=0.375, 2026-07-16 제작)** — "보조 없으면 낙상,
보조하면 보행" 시연용 페어. `--record`는 뷰어를 자동 활성화하므로 headless
셸에서도 그대로 실행 가능(DISPLAY 필요):

```bash
# 보조 OFF: 착용자 잔존 근력 37.5%만 — 차단 ~2 s 뒤 매번 낙상
bash tasks/mimic_suit_active_cable_walk_23dof/compare_maniflow_control_newton.sh --record \
    --maniflow-run-dir <run04> --residual-pd-scale 0.375 --torque-scale 0 \
    --handover-steps 40 --episode-steps 600

# 보조 ON: 같은 조건 + ManiFlow 62.5% 보조 — 30 s 완주
bash tasks/mimic_suit_active_cable_walk_23dof/compare_maniflow_control_newton.sh --record \
    --maniflow-run-dir <run04> --residual-pd-scale 0.375 \
    --handover-steps 40 --episode-steps 600
```

두 영상 모두 매 에피소드 첫 2 s(40스텝)는 full-PD 워밍업이라 정상 보행으로
시작하고, 전환 시점부터 조건이 갈린다 — OFF 영상은 "정상 → 차단 → 2 s 낙상"
이 6회 반복되는 구조.

## DAgger 재학습 결과 (run03, 2026-07-15) — 문턱 α≤0.25 달성 (every_step 조건)

수집 832 eps(위 "다음 단계 2" 설계) + 기존 2000 eps 병합
(`walking-flat-newton-hips-dagger.zarr`, 2832 eps)로 run03 학습 완료
(200 epochs ≈ 16.5 h, best topk `epoch=0190-val_loss=0.012492.ckpt` —
외란이 미래 chunk를 확률화해 val_loss 절대값은 run02보다 높은 게 정상).

| 평가 (run03, 1200스텝) | 결과 | dof err6 (A→B) | corr(A,B총) |
|---|---|---|---|
| passive (on-manifold 예측 품질) | R² 0.997–0.999 / corr ≈1.000 / MAE 0.7–1.7 N·m | — | — |
| α=0.5, h0, receding | **완주** (회귀 없음) | 0.128→0.130 (+1.6%) | 0.90–0.94 |
| α=0.375, h40, receding | **완주** | 0.128→0.140 | 0.74–0.83 |
| α=0.25, h40, receding | 7/8 낙상 82–220스텝 (**run02와 동일 — 개선 없음**) | 0.135→0.225 | 0.27–0.42 |
| **α=0.25, h40, every_step** | **60 s 완주** (root 발산 4.6 cm) | 0.128→0.153 (+19%) | 0.53–0.72 |
| α=0.1, h40, every_step | 11/12 낙상, 전환 후 24–77스텝 (run02 receding 27–32보다 꼬리만 연장) | 0.136→0.285 | 0.01–0.21 |
| α=0, h40, every_step | 12/12 낙상, 전환 후 19–30스텝 (**run02 기준선과 동일**) | 0.136→0.431 | ≈0 |

**핵심 해석 — 교정 게인 × 지연의 안정성 한계**:
- run02 시절(α=0, 순수 FF) every_step은 생존 시간을 전혀 못 늘렸다(62–64 vs
  65–67스텝) — 모델에 교정 응답 자체가 없었기 때문.
- run03에서는 **receding(예측 지연 최대 150 ms)이면 낙상, every_step(50 ms)
  이면 완주** — DAgger 데이터로 상태 의존 교정 응답을 실제로 배웠고, 그
  교정(사실상 피드백 게인)은 신선도 50 ms에서만 안정하다는 뜻. 고전 제어의
  "루프 게인 × 지연 상한"과 정확히 일치. receding에서 run02보다 약간 빨리
  넘어지는 것(86–136 vs 114–218 @epoch100 프로브)도 같은 논리: 배운 고게인
  교정이 stale하게 재생되면 오히려 과진동을 유발.
- 반대급부: α=0.25 es의 트래킹은 +19%로 α=0.5(+1.6%)보다 거칠고, rotation
  채널 B std가 과대(교정 노이즈). 배포 관점에서 every_step은 매 스텝
  디퓨전 추론(50 ms 주기 내 완료 필요)이라 연산 예산도 3배.
- **정리 (경계 확정, 2026-07-15)**: 생존 문턱 **receding α ∈ (0.25, 0.375]
  → every_step α ∈ (0.1, 0.25]**. α=0.1 es는 낙상 꼬리만 2–3배 연장,
  **α=0(순수 ManiFlow) es는 run02 기준선과 동일한 19–30스텝 낙상** — 교정을
  배웠고 신선해도, substep 임피던스가 전혀 없으면 50 ms ZOH 교정만으로는
  도립진자를 세울 수 없음이 확정. "완전 대체 불가, 문턱 인하"라는 예상과
  일치.
- 추가 인하 수단(v2 후보): 라벨을 마지막-substep 점 샘플 대신 **substep 평균
  토크**로(20 Hz 일관 피드포워드), DART 외란 강도 완화(고게인 교정 학습
  억제), chunk 내 시간 감쇠 블렌딩.

## Denoising 스텝 축소 (2026-07-15) — 재학습 없이 3~6배 연산 절감

**코드 확인**: 추론 스텝 수는 `num_inference_steps`(기본 10, 학습용 t-grid인
`denoise_timesteps`와 별개)이며, `sample_ode()`가 학습된 flow를 Euler로 N번
적분한다. 학습 배치의 25%(consistency 분기)가 **dt ~ Uniform(0,1)의 임의
크기 점프**를 `target_t`(=dt) 조건으로 명시해 "그 점프의 평균 속도"를
배우므로(shortcut/mean-velocity 방식), **N 축소는 재학습 없이 지원되는 추론
파라미터**다. infer/compare 스크립트에 `--denoise-steps` 플래그 추가
(`policy.num_inference_steps` 오버라이드, metrics에 기록).

오프라인 (run03, 실측 obs/GT 1200쌍, batch 2, RTX 5090):

| N | MAE [N·m] | corr | predict 지연 |
|---|---|---|---|
| 10 (기본) | 1.40 | 0.9997 | 18.9 ms |
| 5 | 0.38 | 0.9999 | 9.7 ms |
| **3** | **0.32** | 0.9999 | **5.8 ms** |
| 2 | 0.37 | 0.9999 | 4.0 ms |
| 1 | 0.48 | 0.9999 | 2.1 ms |

**적은 스텝이 오히려 더 정확** — consistency 증류가 직선 점프를 배워 두어
잘게 쪼갠 Euler(N=10)의 미소 편향 누적이 사라진다 (ManiFlow 논문의 "쉬운
task는 2-3 스텝이 최고" 결과와 동일 패턴).

폐루프 (run03): α=0.5 receding **N=3 완주** (err6 0.131 — N=10과 동일);
α=0.25 es N=1 완주(0.153), N=3 seed42 1회 780스텝 낙상 / seed43 완주(0.156),
N=10 seed42/43 완주 — **경계 작동점(α=0.25)의 run 간 편차이며 N에 따른
계통 차이 없음**.

**권장·기본값**: 2026-07-15부터 infer/compare 스크립트의 `--denoise-steps`
**기본값이 3** (이전 동작 재현은 `--denoise-steps 10`). 극한 절감이 필요하면
N=1(2.1 ms)도 품질 유지.

### N=3에서 receding vs every_step 추론시간 (RTX 5090 실측, 200회 중앙값)

predict **1회의 지연은 두 모드가 동일**(같은 모델 호출)하고, 다른 것은 **호출
빈도**다: receding은 chunk(사용분 3스텝)당 1회 = 150 ms마다, every_step은
매 스텝 = 50 ms마다.

| | predict 1회 | 호출 빈도 | 스텝당 평균 부하 (50 ms 예산 대비) | 60 s 총 추론시간 |
|---|---|---|---|---|
| receding, N=3 | 5.9 ms | 150 ms마다 (400회/60 s) | 2.0 ms (**3.9%**) | 2.4 s |
| every_step, N=3 | 5.9 ms | 50 ms마다 (1200회/60 s) | 5.9 ms (**11.8%**) | 7.1 s |
| (참고) every_step, N=10 | 19.3 ms | 50 ms마다 | 19.3 ms (38.6%) | 23.2 s |

- batch 1(실기 단일 로봇)과 batch 2(A/B 시뮬)의 지연이 동일 — 이 모델
  크기(10.5M)에선 GPU가 batch를 공짜로 흡수.
- 실시간 관점의 최악 스텝 지연(=predict가 실행되는 스텝)은 두 모드 모두
  5.9 ms로 동일 — 차이는 평균 연산량 3배(every_step 7.1 s vs receding
  2.4 s / 60 s)뿐. N=10 시절 every_step의 부담(38.6%)이 N=3에서 11.8%로
  내려가 SBC 이식 마진이 확보됨.

## 라벨 v2: substep 평균 토크 (2026-07-15 착수 — run04)

### 왜 라벨을 바꾸나 (receding 문턱 인하의 다음 수단)

기존 라벨(원본 수집기·run02·run03 공통)은 `hip_torque` = **한 control
step(50 ms = substep 6개)의 마지막 substep 순간의 qfrc_actuator 점 샘플**이다.
그런데 배포에서 ManiFlow 토크는 50 ms 동안 상수(ZOH)로 재생되므로, 물리적으로
일관된 피드포워드 라벨은 점 샘플이 아니라 **스텝 내 6개 substep의 평균 토크**
(= 50 ms 동안 상수로 재생했을 때 임펄스/충격량이 실제와 같아지는 값)다.

### 실측: 점 샘플은 스텝 임펄스를 계통적으로 과소평가한다 (스모크, 200스텝)

같은 궤적에서 두 readback을 동시 기록한 비교 (외란 0.3σ 포함):

| 채널 | last(점) std | **mean std** | corr | \|차이\| 평균 / 최대 |
|---|---|---|---|---|
| hip_flexion_r | 72.2 | **84.2** | 0.968 | 18.1 / 71.7 N·m |
| hip_adduction_r | 50.6 | 55.8 | 0.957 | 12.8 / 53.9 |
| hip_rotation_r | 17.6 | 20.5 | 0.869 | 8.1 / 44.0 |
| hip_flexion_l | 58.2 | 68.1 | 0.946 | 17.5 / 61.6 |
| hip_adduction_l | 52.5 | 58.9 | 0.963 | 13.2 / 46.5 |
| hip_rotation_l | 17.8 | 20.0 | 0.899 | 7.3 / 24.4 |

예상과 방향이 반대였다 — 평균이 점 샘플보다 **크다**. 메커니즘: RL이 20 Hz로
PD 목표각을 점프시키면 implicit PD 토크는 **스텝 초반 substep에서 스파이크
후 정착**하는 프로파일을 그리는데, 마지막 substep은 "정착된 꼬리"라 그 스텝이
실제 전달한 임펄스보다 작다. 즉 **기존 라벨로 배운 모델은 ZOH 재생 시 매
스텝 임펄스가 계통적으로 부족**했다(채널 std 기준 ~15-17%) — 잔여 PD 없이
재생하면 root z가 단조 하강하며 주저앉던 관찰(관찰 3)과 부합하는 결손이고,
DART 교정 응답(고주파)에서는 점 샘플의 왜곡이 더 크다(corr 0.87-0.97).

### 구현

- `NewtonSimulator`: `accumulate_qfrc_kernel`(warp)로 `_simulate()`의
  decimation 루프에서 substep마다 qfrc를 누적(CUDA graph 캡처에 포함),
  `get_substep_mean_dof_forces()`가 평균을 COMMON ordering으로 반환.
- `collect_walk_zarr_dagger.py`: `--label-mode {mean,last}` (**기본 mean**,
  last=기존 재현용), zarr attrs `label_source` 기록. `--denoise-steps`(기본 3,
  블렌드 모드 estimator를 배포 표준과 일치). α=1·외란 0·mean 조합(순수 보행
  재수집)도 허용 — 원본 2000 eps를 mean 라벨로 재수집하는 용도.

### v2 데이터셋·학습 (run04)

라벨 정의가 다른 데이터를 섞지 않기 위해 **전량 mean 라벨로 재수집** (2026-07-15):
purev2 2000(순수 보행) + dart030v2 384 + dart060v2 128 + blend050av2 192
(run03 estimator, N=3, 외란 0.15σ) + blend050bv2 128 → 총 2832 eps →
`walking-flat-newton-hips-dagger-v2.zarr` → 학습 tag
`newton-hips-dagger-v2-run04` (200 epochs ≈ 16.5 h).

### v2 결과 (2026-07-16, run04 best=epoch180 val 0.01560) — receding 문턱 못 넘음

학습 중 segfault 5회(8-worker fork+CUDA 산발 네이티브 크래시, run02/03은 운
좋게 무사고) → latest.ckpt 자동 재개 루프로 완주. epoch 180 이후 val 플래토
확인 후 210에서 정지.

| 평가 (run04, 1200스텝, N=3) | 결과 | dof err6 (A→B) |
|---|---|---|
| α=0.5, h0, receding | 완주 (run03과 동일, +2%) | 0.128→0.131 |
| α=0.375, h40, receding | 완주 | 0.128→0.139 |
| **α=0.25, h40, receding** | **여전히 낙상 105–275스텝** (run02 114–218 / run03 82–220 — 꼬리만 소폭 연장) | 0.134→0.214 |
| α=0.25, h40, every_step | 1124스텝(56 s) 낙상 1회 — run03과 동급(경계 편차 범위) | 0.129→0.194 |
| α=0.1, h40, every_step | 낙상 75–113 (run03 64–117과 동일) | 0.136→0.323 |
| passive (vs 점 샘플 GT) | MAE 12.6 N·m / corr 0.94 — **모델 열화 아님**: 점 샘플 GT와 mean 라벨의 정의 차(스모크 실측 \|mean−last\| 13–18 N·m, corr 0.87–0.97)가 그대로 지표에 나타난 것 | — |

**판정**: 임펄스 결손 보정(라벨 v2)은 방향은 맞지만 **receding 문턱
(0.25, 0.375]을 넘기기엔 불충분** — corr은 소폭 개선(0.21→0.31-0.44)됐으나
낙상 자체는 유지, 저α 과교정(B std 과대, 특히 rotation)도 잔존. 즉 **문턱을
고정하는 지배 요인은 라벨 품질이 아니라 피드백 신선도(stale ZOH 교정)**임이
v1(run03)·v2(run04) 두 번의 재학습으로 확정됨.

**실용 결론**: 배포 작동점은 (a) **receding α≥0.375**(연산 최소, 착용자
기여 37.5%↑) 또는 (b) **every_step α=0.25**(N=3, 예산 11.8%, 착용자 기여
25%) — 두 모델(run03/run04) 모두 이 작동점에서 동등하게 동작. 그 아래
α는 50 ms ZOH 교정의 대역폭 한계로 어느 라벨/데이터로도 불가(α=0 es 재확인).
남은 아이디어(우선순위 낮음): chunk 내 시간 감쇠 블렌딩, DART 강도 완화,
제어 주기 상향(50→100 Hz 재수집·재학습 — 시스템 전반 변경 필요).

## 결과물 경로

| 내용 | 경로 |
|---|---|
| 낙상 원인 트레이스 분해 원본 (handover 없음) | `maniflow_control_results/2026-07-10_09-58-38` |
| handover 40 (receding) | `maniflow_control_results/2026-07-10_14-11-54` |
| handover 40 + every_step | `maniflow_control_results/2026-07-10_14-13-01` |
| **handover 40 녹화** (고스트 A + 메시 B + 토크 패널, 600f 2640×1080) | `maniflow_control_results/2026-07-10_14-44-53/sim_with_torque.mp4` |
| passive 완벽 추종 녹화 (참고) | `maniflow_infer_results/2026-07-10_09-46-41/sim_with_torque.mp4` |
| 잔여 PD α 스윕 (0.1/0.25/0.375/0.5, h40) | `maniflow_control_results/2026-07-14_residual_a{010,025,0375,050}_h40` |
| 잔여 PD α=0.5, handover 0 | `maniflow_control_results/2026-07-14_residual_a050_h0` |
| ablation: α=0.5 + MF 끔 (torque_scale 0) | `maniflow_control_results/2026-07-14_residual_a050_h40_mf0` |
| DAgger 수집 원본 (832 eps, 4종) | `zarr_data/flat/flat-newton-dagger-{dart030,dart060,blend050a,blend050b}-20260714.zarr` |
| run03 평가: passive / α=0.5 h0 / α=0.375 h40 / α=0.25 h40 / **α=0.25 h40 every_step(완주)** | `maniflow_infer_results/2026-07-15_run03_passive`, `maniflow_control_results/2026-07-15_run03_{a050_h0,a0375_h40,a025_h40,a025_h40es}` |
| run03 중간(epoch100) 프로브 | `maniflow_control_results/2026-07-15_probe_run03e100_a025{,_h40}` |
| denoise 스텝 축소 폐루프 (N=3/1, seed 42/43) | `maniflow_control_results/2026-07-15_run03_{a025_h40es_n3,a050_h0_n3,a025_h40es_n1,a025_h40es_n3_s43,a025_h40es_n10_s43}` |
| v2 수집 원본 (mean 라벨, 2832 eps 5종) | `zarr_data/flat/flat-newton-dagger-{purev2,dart030v2,dart060v2,blend050av2,blend050bv2}-20260715.zarr` |
| run04(v2) 평가: passive / receding α=0.5·0.375·0.25 / es α=0.25·0.1 | `maniflow_infer_results/2026-07-16_run04_passive`, `maniflow_control_results/2026-07-16_run04_{a050_h0,a0375_h40,a025_h40,a025_h40es,a010_h40es}` |
| run04 중간(epoch100) receding 프로브 | `maniflow_control_results/2026-07-15_probe_run04e100_a025_h40` |
| **α=0.375 보조 OFF 녹화** (torque_scale 0, 6/6 낙상 — `sim_with_torque.mp4` 포함) | `maniflow_control_results/2026-07-16_run04_a0375_h40_mf0_video` |
| **α=0.375 보조 ON 녹화** (30 s 완주 — `sim_with_torque.mp4` 포함) | `maniflow_control_results/2026-07-16_run04_a0375_h40_on_video` |

영상 읽는 법(handover 녹화): 매 에피소드 첫 2초는 워밍업(빨간 선 없음, B가
A와 겹쳐 보행) → 전환 직후 빨간 선이 검정과 겹치며 시작 → ~1.3 s 뒤 B만
낙상, 이후 빨강-검정 분리(상태가 다르니 당연).

## 다음 단계

1. ~~**잔여 PD 결합 (1순위)**~~ — **완료 (2026-07-14, 위 섹션)**: α=0.5에서
   A 동등 보행, 생존 문턱 α ∈ (0.25, 0.375]. `--residual-pd-scale`로 사용.
2. ~~**estimator-in-the-loop 학습**(DAgger류)~~ — **완료 (2026-07-15,
   newton-hips-dagger-run03 — 결과는 위 "DAgger 재학습 결과" 섹션)**: 문턱
   α ∈ (0.25, 0.375] → **α ≤ 0.25 (every_step 조건)** 인하. 수집기
   `collect_walk_zarr_dagger.{py,sh}` 신규 작성:
   - **DART식 외란 주입**(α=1 유지 + hip에 ZOH 랜덤 토크 외란): 배포 시
     ManiFlow ZOH 토크 오차와 같은 구조의 교란 아래에서 PD의 매 substep
     교정 응답을 라벨로 수집 (`--perturb-scale`σ배율/`--perturb-hold`).
   - **잔여 PD 블렌드 on-policy**(α=0.5 배포 동일 하이브리드로 수집):
     estimator 자신이 만든 상태 분포(리셋 첫 chunk 과도 포함)를 데이터화.
   - **라벨 규약**: hip_torque = qfrc_actuator readback / α (effort limit
     클램프) = "이 상태에서 full-gain PD(전문가)가 냈을 토크"의 DAgger
     expert query. 주입 외란/MF 토크는 라벨에 불포함.
   - 데이터: 기존 2000 eps + 신규 832 eps (dart030 384 / dart060 128 /
     blend050a 192(+외란 0.15) / blend050b 128) → 병합
     `walking-flat-newton-hips-dagger.zarr`, 학습 tag
     `newton-hips-dagger-run03` (200 epochs ≈ 16 h 예상).
   - 완료 후 평가: `compare_maniflow_control_newton.sh --maniflow-run-dir
     <run03>`로 α ∈ {0.25, 0.1, 0} 스윕 재실행 → 문턱 이동 확인.
   - 한계 예상: ManiFlow 보정도 결국 50 ms ZOH라 120 Hz PD의 대역폭은 못
     따라감 — α→0 완전 대체가 아니라 문턱을 낮추는 것이 목표. (응용
     관점에서는 위 "응용 관점" 항목처럼 α = 착용자 잔존 기여로 해석.)
3. ~~every_step 재예측~~ — 실험으로 기각(생존 시간 개선 없음).
4. (참고) 수집기를 s₀부터 기록하게 바꾸면 리셋 직후 chunk 문제는 완화되지만,
   handover 실험이 보여주듯 그것만으로는 보행 불가. α≥0.375에서는 잔여 PD가
   흡수하므로 실용상 불필요.
