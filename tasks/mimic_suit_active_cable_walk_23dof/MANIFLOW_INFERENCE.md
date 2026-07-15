# ManiFlow Hip-Torque 추정기 — Newton Inference 가이드

ManiFlow_Policy에서 학습한 **센서 상태 전용(vision-free) hip torque 추정기**를
ProtoMotions 시뮬레이터(기본 Newton) 안에서 폐루프로 돌리고, 예측 토크를 실제
적용 토크와 비교하는 파이프라인 문서입니다.

---

## TL;DR — 다시 시작할 때 먼저 읽을 것

1. **스크립트 두 개, 목적이 다름**:
   - `infer_maniflow_newton.{py,sh}` — RL이 걷고 ManiFlow는 **관찰만**(예측 vs
     실제 토크 비교). 로봇 동작에 영향 없음.
   - `compare_maniflow_control_newton.{py,sh}` — ManiFlow 예측 토크를 **실제로
     로봇에 인가**하는 A/B 비교. Agent A(순수 RL, 고스트) vs Agent B(estimator
     채널만 ManiFlow 토크 + 나머지는 RL, 메시)를 같은 씬에 겹쳐 실행.
     자세한 구조는 아래 "폐루프 제어 A/B 비교" 섹션.
2. **⚠️ 채널 계약 (2026-07-09 전면 정정)**: 파이프라인 전체(수집→학습→추론)가
   이제 **순수 hip 6개 DOF = 공통 [0,1,2,5,6,7]** =
   `[hip_flexion_r, hip_adduction_r, hip_rotation_r, hip_flexion_l,
   hip_adduction_l, hip_rotation_l]`을 사용합니다 (이름 기반 파생 —
   `protomotions.maniflow.channels`). 수집 버그(공통 DOF 0-5 = 오른다리
   전체 + 왼쪽 hip flexion)로 학습됐던 legacy 모델(run01_seed42)은
   **2026-07-14 삭제**했고, 두 스크립트의 `--action-dofs` 옵션과
   `--maniflow-run-dir` 기본값도 run02(hips 계약)로 정리했습니다.
   ManiFlow_Policy repo 쪽(process/eval/train/dataset)도 같은 채널 계약 —
   "관측 계약" 섹션 참고.
3. **현재 결과 (2026-07-10, newton-hips-run02 모델)**:
   - **관찰(passive) 예측**: 오프라인 val split R²≈1.000 · corr 1.000 ·
     MAE 0.16–0.68 N·m, **Newton 폐루프도 R²≥0.998 · corr 1.000 · MAE ≤0.8
     N·m** (`maniflow_infer_results/2026-07-10_08-14-01`). legacy corr≈0.1
     문제 완전 해소.
   - **제어(active) 인가**: `compare_maniflow_control_newton.sh` 기본 설정에선
     여전히 Agent B가 초반에 낙상 (`maniflow_control_results/2026-07-10_08-15-01`).
     예측이 완벽해도 **상태 피드백 없는 피드포워드 토크 재생**이라 미세 오차가
     상태 드리프트로 누적되기 때문 — 이 로봇은 explicit PD조차 발산하는
     특성(아래 참고)이라 예상된 결과. 제어 활용은 잔여 PD 결합·torque-scale
     블렌딩·estimator-in-the-loop 등 별도 과제.
   - legacy(run01) 모델이 나빴던 원인 (a) 채널 오배선(무릎/발목 포함),
     (b) IsaacLab↔Newton 도메인 갭, (c) 학습 데이터 낙상 오염(1000개 중
     172개 에피소드 root z −4000 m대, termination 부재로 done 필터 무력)은
     모두 해결됨: hip 채널 정정 + Newton 재수집(자체 낙상 필터) + 재학습.
4. **아직 커밋 안 됨**: 이번 통합 작업 전체가 uncommitted 상태입니다
   (`git status`로 확인). 커밋 전에 `pre-commit run --files <목록>`으로 lint
   확인 권장(이 머신엔 pre-commit이 안 깔려 있어 `py_compile`로만 문법 확인함 —
   실제 커밋 전에는 ruff가 있는 환경에서 한 번 더 검증 필요).
5. **막히면**: 이 문서의 "알아두면 좋은 것들" 섹션에 Newton PD 이슈, 녹화
   인프라, obs 정합성 등 이번에 부딪혔던 문제와 해결책이 정리돼 있음.

---

## 핵심 구조: 보행은 RL policy가, ManiFlow는 관찰자

**ManiFlow의 예측 토크는 로봇에 적용되지 않습니다.** 로봇을 걷게 하는 것은
보행 학습된 RL 체크포인트(mimic tracker)이고, ManiFlow는 옆에서 센서 상태만
보고 hip torque를 추정하는 **관찰자(estimator)** 입니다:

```
RL walking policy ──PD 위치 타겟──▶ 시뮬레이터 (BUILT_IN_PD가 토크 생성·적용)
                                        │
                                        ├─▶ robot state (dof_pos/vel, root, contacts)
                                        │       │
                                        │       ├─▶ RL policy의 다음 obs
                                        │       └─▶ ManiFlow obs history (2 steps)
                                        │               └─▶ 예측 hip torque (6)  ─┐
                                        │                                        비교
                                        └─▶ 실제 적용 토크 dof_forces[:, 0:6]  ─┘
                                            (MuJoCo qfrc_actuator readback)
```

- 예측 토크로 로봇을 **직접 구동하는 폐루프 A/B 비교**는
  `compare_maniflow_control_newton.{py,sh}`로 별도 구현되어 있습니다 —
  아래 "폐루프 제어 A/B 비교" 섹션 참고.
- 데이터 수집(`collect_walk_zarr.py`, 기본 Newton)과 동일한 관계입니다 —
  수집에서도 RL policy가 걷고 estimator 채널(hip 6개) 토크는 기록만 하며,
  ManiFlow는 그 기록으로 학습됩니다.

---

## 빠른 실행

```bash
# 기본: headless, 지표만 (Newton, 2 envs × 1200 steps = 60초 보행)
bash tasks/mimic_suit_active_cable_walk_23dof/infer_maniflow_newton.sh

# 실시간 GUI — skeleton 근골격 모델 + 뷰어 UI에 hip_flexion pred/gt 라이브 플롯
bash tasks/mimic_suit_active_cable_walk_23dof/infer_maniflow_newton.sh --viewer

# 동영상 저장 — 시뮬 mp4 + 토크 플롯 합성 sim_with_torque.mp4
bash tasks/mimic_suit_active_cable_walk_23dof/infer_maniflow_newton.sh \
    --record --episode-steps 600

# 다른 run/체크포인트 지정 (기본값은 run02 디렉토리에서 best topk 자동 선택)
bash tasks/mimic_suit_active_cable_walk_23dof/infer_maniflow_newton.sh \
    --maniflow-run-dir ~/Projects/ManiFlow_Policy/ManiFlow/data/outputs/walking_flat-maniflow_lowdim_policy_walking-newton-hips-run02_seed42
```

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--rl-checkpoint` | `output_newton_flat/score_based.ckpt` | 보행 RL policy (로봇을 실제로 걷게 하는 주체) |
| `--maniflow-ckpt` | newton-hips-run02_seed42에서 best topk 자동 | ManiFlow 추정기 체크포인트 |
| `--simulator` | `newton` | `isaaclab`도 가능 (학습 도메인 교차검증용) |
| `--num-envs` / `--episode-steps` | 2 / 1200 | 1200 steps = 60 s @ 20 Hz |
| `--viewer` | off | 실시간 GUI (skeleton mesh 기본) |
| `--record` | off | mp4 + 토크 합성 비디오 (뷰어 자동 활성화) |
| `--no-mesh` | off | 시각화를 캡슐 에셋으로 (수집 조건 물리 완전 재현) |
| `--predict-mode` | `receding` | 4스텝 청크 예측(오프라인 eval과 동일) / `every_step` |
| `--denoise-steps` | `3` | 추론 ODE(Euler) 스텝 수. consistency 학습 덕분에 재학습 없이 축소 가능 — N=3이 ckpt 설정(10)보다 정확·3배 빠름 (MANIFLOW_CONTROL_ANALYSIS.md "Denoising 스텝 축소") |
| `--control-mode` | `config` | 체크포인트 설정 그대로(=BUILT_IN_PD). `proportional`은 이 로봇에서 발산 — 금지 |
| `--overrides` | — | ProtoMotions config override 패스스루 |

### 뷰어 조작키 (`--viewer` / `--record`)

| 키 | 동작 | 키 | 동작 |
|----|------|----|------|
| `Q` | 종료 | `[` / `]` | 카메라 좌/우 회전 (30°) |
| `R` | 전체 리셋 | `B` / `N` | 후방 / 정면 뷰 |
| `O` | 카메라 타겟 전환 | `C` | 접촉 시각화 토글 |
| `L` | 수동 녹화 시작/종료 | `M` | 마커 토글 |
| `J` | 외란(프로젝타일) 발사 | `;` | 녹화 취소 |

뷰어 창을 닫으면 rollout이 그 시점에서 조기 종료되고, 거기까지 수집된
데이터로 지표·영상이 만들어집니다.

### 출력 (`maniflow_infer_results/<timestamp>/`)

| 파일 | 내용 |
|------|------|
| `metrics.{json,txt}` | per-joint MAE / RMSE / R² / corr (낙상 없이 완주한 env만) |
| `traces.npz` | `pred`/`gt` hip torque (N,T,6), `obs` (N,T,88), `env_failed` |
| `env*_{full,zoom}.png` | 관절별 예측 vs 실제 trace 플롯 |
| `sim-<ts>/sim-<ts>.mp4` | (`--record`) 시뮬 녹화 (+ `.motion`/`.markers.pt` 부산물) |
| `sim_with_torque.mp4` | (`--record`) 영상 + 6관절 토크 플롯 합성 (프레임↔스텝 1:1) |

---

## 폐루프 제어 A/B 비교 (`compare_maniflow_control_newton.{py,sh}`)

ManiFlow 예측 토크를 **실제 제어에 사용**했을 때를 순수 RL과 나란히 확인하는
스크립트입니다. Newton 멀티월드(2 envs)에 두 agent를 **같은 위치에 겹쳐**
스폰합니다 — env별 world가 분리되어 있어 물리적 상호 간섭은 없습니다:

| | 제어 | 뷰어 표현 |
|---|------|-----------|
| **Agent A** (env 0) | RL policy + built-in PD (순수 RL) | 반투명 라인 스켈레톤 (고스트) |
| **Agent B** (env 1) | estimator 채널(기본 hip 6개 = 공통 [0,1,2,5,6,7]) = ManiFlow 토크, 나머지 = RL policy + built-in PD | 일반 메시 |

```bash
# headless 지표만
bash tasks/mimic_suit_active_cable_walk_23dof/compare_maniflow_control_newton.sh

# 실시간 GUI (고스트 A + 메시 B 중첩, 라이브 플롯)
bash tasks/.../compare_maniflow_control_newton.sh --viewer

# 동영상 (sim mp4 + 토크 패널 합성 sim_with_torque.mp4)
bash tasks/.../compare_maniflow_control_newton.sh --record --episode-steps 600

# 최종 학습 모델로
bash tasks/.../compare_maniflow_control_newton.sh \
    --maniflow-ckpt ~/Projects/ManiFlow_Policy/.../checkpoints/latest.ckpt
```

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--rl-checkpoint` | `output_newton_flat/score_based.ckpt` | 두 agent 모두가 쓰는 보행 RL policy |
| `--maniflow-ckpt` / `--maniflow-run-dir` | newton-hips-run02_seed42 best topk 자동 | ManiFlow 체크포인트 |
| `--maniflow-root` | `$MANIFLOW_ROOT` 또는 관례 경로 | maniflow 패키지 위치 |
| `--episode-steps` | 1200 | 총 rollout 스텝 (에피소드 끝나면 이어서 진행) |
| `--predict-mode` | `receding` | 4스텝 청크 예측 / `every_step`(매 스텝 재예측) |
| `--denoise-steps` | `3` | 추론 ODE 스텝 수 (재학습 불필요; N=3이 10보다 정확·3배 빠름) |
| `--chunk-offset` | 1 | 청크 시작 인덱스. 1=다음 전이용(권장), 0=한 스텝 지연 사후추정 |
| `--torque-scale` | 1.0 | Agent B 인가 토크 배율. 0 = 해당 채널 무동력 sanity check |
| `--residual-pd-scale` | 0.0 | estimator 채널에 남길 PD 게인 비율 α — hip 토크 = α·PD + (1-α)·ManiFlow ("잔여 PD 결합" — MANIFLOW_CONTROL_ANALYSIS.md 참고) |
| `--handover-steps` | 0 | 에피소드 첫 K스텝은 B도 순수 PD로 보행(워밍업) 후 ManiFlow 제어로 전환. 리셋 직후 관측(s₀)이 학습 분포 밖이라 첫 chunk가 어긋나는 문제를 우회 |
| `--fall-z` / `--fall-hold` | 0.5 / 10 | 넘어짐 판정: root 높이[m] × 연속 유지 스텝 수 |
| `--divergence-reset` | 5.0 | A↔B root XY 거리[m] 초과 시 에피소드 종료 (<=0 비활성) |
| `--min-episode-seconds` | 5.0 | 넘어짐/발산 감지돼도 이 시간까진 리셋 안 함(그대로 시뮬 지속) — grace period |
| `--viewer` / `--record` | off / off | 실시간 GUI / mp4+토크 합성 영상 저장 (record는 viewer 자동 활성화) |
| `--no-mesh` | off | 캡슐 에셋(기본은 skeleton mesh) |
| `--ghost-alpha` / `--ghost-line-width` | 0.5 / 3.5 | Agent A 고스트 라인 투명도 / 두께[px] |
| `--seed` | 42 | RNG seed |
| `--output` | `maniflow_control_results/<timestamp>/` | 결과 디렉토리 |

뷰어 조작키는 `infer_maniflow_newton.sh`와 동일 (같은 NewtonSimulator 뷰어 — 위
"뷰어 조작키" 표 참고).

동작 방식:
- **하이브리드 제어**: `protomotions.maniflow.JointTorqueOverride`가 env 1의
  estimator 채널만 PD 게인을 0으로 만들고(`notify_model_changed`, per-world 게인)
  ManiFlow 토크를 `control.joint_f`(→`qfrc_applied`)로 주입. effort limit으로
  클램프, decimation 구간 동안 상수 유지(표준 토크 제어). Agent B의 해당 채널
  qfrc_actuator 잔여가 0인지 런타임 자체 검증.
- **동기화 에피소드**: 두 env는 항상 같은 모션(id 0)·t=0·같은 스폰 위치에서
  시작하고, 한쪽이 done/낙상/과대발산하면 **둘 다 함께 리셋**. inference
  config에는 termination이 없으므로 스크립트가 자체 감지:
  `--fall-z 0.5`(root 높이) × `--fall-hold 10`(연속 스텝), `--divergence-reset
  5.0`(A↔B 거리 m).
- **최소 에피소드 길이**: `--min-episode-seconds 5.0`(기본) 동안은 넘어짐/발산이
  감지돼도 리셋하지 않고 **쓰러진 채로 시뮬레이션을 계속**합니다 — 매
  에피소드가 최소 5초는 관찰 가능하도록 보장하는 grace period. env 자체의 진짜
  termination(현재는 거의 발생 안 함)은 게이트 없이 즉시 반영됩니다. 실제
  control step 주파수(`1/env.simulator.dt`)로 스텝 수를 계산하므로 fps/decimation
  설정이 달라져도 초 단위 의미가 유지됩니다.
- **정렬**: 수집 관례상 chunk[0]은 직전 전이의 사후 추정치라 제어에는
  `--chunk-offset 1`(기본)부터 사용. Agent A에 대해서는 같은 chunk 원소를 수동
  예측으로 기록해 기존 passive 비교도 함께 얻습니다.
- `--torque-scale 0`으로 "해당 채널 무동력" sanity check 가능.
- **워밍업 핸드오버(`--handover-steps K`)**: 에피소드 시작 후 K스텝 동안 B도
  순수 RL+PD로 보행(override 해제, `tau_b_cmd`/`pred_a_passive`는 NaN 기록 →
  지표에서 자동 제외)하고 estimator에 실제 관측 히스토리를 쌓은 뒤, K스텝째에
  게인을 다시 0으로 만들고 ManiFlow 토크 제어로 전환. **왜 필요한가**: 수집
  데이터의 에피소드 첫 프레임은 물리 1스텝 후의 s₁이라(수집기가 `env.step()`
  후 기록) 리셋 직후 상태 s₀(contacts 미갱신, 기구학 리셋 속도)는 학습 분포
  밖 → 첫 chunk가 크게 어긋남(실측 +160 N·m vs 예측 -86 N·m 수준). 핸드오버는
  이를 우회해 on-distribution 상태에서 피드포워드 제어의 순수 생존 시간을
  측정.

출력(`maniflow_control_results/<timestamp>/`): `metrics.{json,txt}`(에피소드
통계·종료 원인·트래킹 오차 A vs B·채널별 토크 통계), `traces.npz`,
`torque_channels_{full,zoom}.png`, `tracking_{full,zoom}.png`,
(`--record`) `sim-<ts>.mp4` + `sim_with_torque.mp4`.

첫 실행 결과 (legacy epoch30 ckpt, first6 채널, Newton, 기본 설정): **Agent B는 낙상 자체는 매
에피소드 약 1초 내에 감지**되지만(fall-hold 0.5s 포함), min-episode-seconds
grace period로 실제 리셋은 5초까지 지연되어 **쓰러진 채로 나머지 구간을
시뮬레이션**합니다(물리적으로 안정적으로 확인됨 — NaN/발산 없음). 6채널
트래킹 오차 A 0.23→B 0.94 rad, 보상 A 0.84→B -0.05. 현재 sim2sim 예측 품질(corr≈0.1)로는
예상된 결과이며, 아래 "현재 Newton 예측 품질이 낮은 이유"의 개선 루프(재수집→
재학습→본 스크립트로 재평가)를 위한 기준선입니다.

---

## 체크포인트

- **RL (보행)**: `output_newton_flat/{score_based,last}.ckpt` — Newton에서 학습된
  tracker. 수집·추론 모두 이걸 사용 (도메인 일치).
- **ManiFlow (신규, hip 6채널 — 기본 사용)**: Newton 재수집 데이터
  (`walking-flat-newton-hips.zarr`, 2000 eps)로 2026-07-09~10 학습 완료 —
  `.../outputs/walking_flat-maniflow_lowdim_policy_walking-newton-hips-run02_seed42`,
  best topk `epoch=0190-val_loss=0.000716.ckpt` (자동 선택 정상).
  infer/compare 스크립트의 `--maniflow-run-dir`을 이 run으로 지정하고
  `--action-dofs hips`(기본값) 사용. Newton 폐루프 passive 예측
  R²≥0.998/corr 1.000.
- **ManiFlow (legacy, first6 채널 — 2026-07-14 삭제됨)**: run01_seed42,
  2026-07-08 완주였으나 **채널이 오른다리 전체+왼 hip flexion**(수집 버그)
  이고 학습 데이터에 낙상 오염이 있어 디스크에서 삭제. 스크립트의
  `--action-dofs first6` 옵션도 함께 제거됨. 두 스크립트의
  `--maniflow-run-dir` 기본값은 이제 run02.
- 학습 진행 중인 run에서 로드할 때는 latest.ckpt가 주기적으로 재작성되므로
  topk(`epoch=*-val_loss=*.ckpt`) 파일을 쓸 것.

## 환경

- **Python**: `sbc` conda env (`~/miniconda3/envs/sbc`, `.sh`가 기본 사용).
  newton 1.2.0.dev0, torch 2.7.0+cu128. ManiFlow inference용으로 zarr, timm,
  lightning, dm_control, typer, easydict 추가 설치됨 (torch/mujoco 미변경).
- **maniflow 패키지 위치**: 기본 `~/Projects/ManiFlow_Policy/ManiFlow`.
  다른 위치면 `MANIFLOW_ROOT` 환경변수 또는 `--maniflow-root`.
  통합 레이어 상세는 `protomotions/maniflow/README.md` 참고 — 시뮬 코드는
  `maniflow.*`를 직접 import하지 않고 반드시 그 패키지를 거침 (추후 vendoring 시
  `loader.py`만 수정하면 되는 구조).
- **ManiFlow_Policy 쪽 수정 1건**: `maniflow/model/lowdim/lowdim_obs_encoder.py`의
  `create_mlp`을 인라인해 `vision_3d`(→ pytorch3d) import를 제거했습니다 — sbc
  env에 pytorch3d 없이 inference가 가능한 이유. 동작·state_dict는 동일하며,
  ManiFlow repo를 리셋/재클론하면 이 수정을 유지해야 합니다.

---

## 알아두면 좋은 것들

### 1. Newton에서 GT 토크를 읽는 방법 (BUILT_IN_PD + qfrc_actuator)
Newton의 built-in PD는 implicit(MuJoCo 솔버 내부)이라 적용 토크가
`control.joint_f`에 기록되지 않아 원래 `dof_forces`가 전부 0이었습니다.
`protomotions/simulator/newton/simulator.py`가 Newton extended state
`mujoco:qfrc_actuator`를 요청해 두었고, BUILT_IN_PD일 때 이 readback(솔버가 매
substep 채움)에서 토크를 읽습니다 → **학습과 동일한 actuation을 유지하면서 GT
확보**. explicit PD(`--control-mode proportional`)로 우회하는 방법은 이 로봇
(active cable)에서 시뮬레이션이 발산하므로 쓰지 마세요.

### 2. 관측 계약 (수집·학습·inference 모두 동일해야 함)
```
obs(88) = dof_pos(27) + dof_vel(27) + root_pos(3) + root_vel(3) + contacts(28)
action(6) = 순수 hip 6개 DOF 적용 토크 (공통 DOF [0,1,2,5,6,7])
          = [hip_flexion_r, hip_adduction_r, hip_rotation_r,
             hip_flexion_l, hip_adduction_l, hip_rotation_l]
```
채널 인덱스는 하드코딩하지 말고 `protomotions.maniflow.channels`의
`hip_dof_indices(dof_names)`로 이름 기반 파생하세요. 공통 DOF 순서는
kinematic tree 순서(오른다리 체인→왼다리 체인→…)라서 앞 6개를 자르면(`[:6]`)
DOF 3, 4에 오른쪽 무릎/발목이 섞입니다 — 초기 수집 스크립트가 바로 이
가정(hips=0-5)을 잘못해서 legacy 모델이 오른다리 전체+왼 hip flexion을
학습했습니다.

⚠️ **Legacy 구분법**: 새 수집본은 zarr attrs에 `action_dof_names`/
`action_dof_indices`가 있습니다(자기술). 없는 zarr(예: `flat-2026-06-26-*`)은
legacy — `hip_torque` 필드가 실제로는 공통 DOF 0-5이고,
`process_data_walking.py`가 `--allow_legacy_channels` 없이는 거부합니다.
(legacy 채널로 학습된 run01 모델과 스크립트의 `--action-dofs` 옵션은
2026-07-14 삭제·제거됨.)

`Simulator.get_robot_state()`가 모든 필드를 common ordering으로 변환하므로
시뮬레이터가 달라도 레이아웃이 유지됩니다. root_pos/vel = pelvis(body 0) 월드
위치/선속도, contacts = binary flag.

### 3. 구(legacy) 모델의 Newton 예측 품질이 낮았던 이유 — 전부 해결됨
- **배선 검증**: 학습 zarr 데이터를 estimator 경로로 흘리면 R²≈0.98–0.997 —
  obs 조립/정렬/정규화 자체는 정상이었음.
- **Newton 폐루프**: corr ~0.15–0.18. 원인 진단과 해결:
  1. (도메인 갭) Newton에서 실현되는 보행 자세가 학습 데이터(IsaacLab)보다
     레퍼런스에서 크게 이탈 (ankle ~1 rad 수준까지)
     → **해결**: 수집을 Newton에서 직접 수행 (`collect_walk_zarr.py`가
     `--simulator newton` 기본, RL ckpt도 Newton 학습본 사용).
  2. (도메인 갭) 발뒤꿈치 contact 플래그(contacts[4]/[11])가 IsaacLab 수집
     데이터에선 **항상 1**인데 Newton에선 0.16~0.22로 토글 — 모델이 본 적
     없는 입력 → **해결**: Newton 수집 데이터도 동일 분포(스모크 수집에서
     0.16~0.22 확인).
  3. (데이터 오염) 구 수집본 1000개 중 172개 에피소드에 낙상 구간이 통째로
     기록됨(root z 최저 -4098 m — inference config에 termination이 없어
     done 필터가 무력) → normalizer가 높이 정보를 사실상 소거
     → **해결**: 수집기 자체 낙상 감지(`--fall-z 0.5` × `--fall-hold 10`,
     비유한값 감지 포함)로 해당 에피소드 폐기.
  4. (채널 오배선) 애초에 action 6채널에 오른쪽 무릎/발목이 섞여 있었음
     → **해결**: hip 6개([0,1,2,5,6,7])로 재수집·재학습.

### 4. 영상/플롯의 동기
`simulator.step()`이 물리 스텝 직후 `render()`를 1회 호출 → **mp4 프레임 k =
rollout 스텝 k** 가 보장됩니다. `sim_with_torque.mp4`의 파란 커서가 가리키는
시점이 왼쪽 영상의 현재 프레임입니다.

### 5. Newton 녹화 인프라 (이 머신)
`NewtonSimulator._write_viewport_to_file`이 원래 빈 stub이어서 Newton 녹화가
전부 불가였음 → `ViewerGL.get_frame()`(GPU PBO readback)으로 구현되어
`record_newton.sh`, 뷰어 'L' 키 녹화도 동작합니다. 프레임 캡처에는 GL 컨텍스트
(뷰어 창)가 필요 — headless 녹화가 필요해지면 `ViewerGL(headless=True)`
오프스크린 모드를 추가할 것.

### 6. 고스트(반투명) 렌더링 구현 — 깨지면 여기를 볼 것
Newton의 메시 셰이더(PBR)는 인스턴스별 alpha를 지원하지 않아 "반투명 메시"는
불가능. 대신 `GhostSkeletonRenderer`(`compare_maniflow_control_newton.py`)가
Agent A를 **반투명 라인 스켈레톤**으로 그림:
- `viewer.set_visible_worlds([MANIFLOW_ENV])`로 world 0(A)의 메시 렌더링을 끔
- 매 프레임 `_render_hook`에서 kinematic tree의 parent→child 본을
  `viewer.log_lines()`로 그림 (라인 파이프라인만 블렌딩을 지원)
- `RendererGL._wireframe_shader.update_frame`을 몽키패치해 `alpha` 유니폼을
  강제로 덮어써서 반투명 적용 (newton 내부 구조가 바뀌면 `_patch_line_alpha`의
  `AttributeError` 폴백으로 불투명 라인이 되며 경고 로그 출력)
- `viewer.set_world_offsets((0,0,0))`로 Newton이 world별로 자동으로 벌려 그리는
  간격을 0으로 고정 (두 world가 실제 물리 좌표에서 겹쳐야 하므로)

### 7. walk 모션 파일
`data/motion_for_trackers/skeleton_torque_suit_walk.pt`(walk 1클립)가 이 머신에
없어서 `extract_walk_motion.py`로 학습용 멀티모션 파일
(`skeleton_torque_suit_motions_11+koo_4.pt`)의 walk 클립(idx 10)에서 추출·복원해
두었습니다. play/record/collect 스크립트 전부가 이 파일을 참조합니다.

### 8. 관련 파일 맵 (ManiFlow 통합 전체)

`git status --short`로 최신 미커밋 목록을 직접 확인하세요 — 아래 "상태"는 이
문서 작성 시점 기준입니다.

| 위치 | 상태 | 내용 |
|------|------|------|
| `protomotions/maniflow/channels.py` | 신규, uncommitted | **채널 계약 단일 소스** — `HIP_DOF_NAMES`, `hip_dof_indices()` |
| `protomotions/maniflow/hybrid_control.py` | 신규, uncommitted | `JointTorqueOverride` — per-env/per-DOF 토크 오버라이드 (잔여 PD: `engage(gain_scale)`) |
| `protomotions/maniflow/{README,__init__,torque_estimator}.py` | 수정, uncommitted | channels 등록 + hip 채널 계약으로 문서/주석 정정 |
| `tasks/.../infer_maniflow_newton.py` | 수정, uncommitted | 수동(passive) 비교 — hip 채널 고정 (legacy `--action-dofs` 제거) |
| `tasks/.../compare_maniflow_control_newton.{py,sh}` | 신규, uncommitted | 폐루프 제어 A/B 비교 — `--residual-pd-scale` 포함 |
| `tasks/.../collect_walk_zarr.{py,sh}` | 수정, uncommitted | 수집기 — hip 6채널([0,1,2,5,6,7]) 기록, `--simulator newton` 기본, 자체 낙상 필터, zarr v2 포맷 고정, attrs 자기술 |
| `tasks/.../collect_walk_zarr_dagger.{py,sh}` | 신규, uncommitted | DAgger/DART 수집기 — hip 외란 주입(DART) + 잔여 PD 블렌드 on-policy 수집, 라벨 = full-PD 전문가 질의(readback/α). run03 재학습용 |
| `tasks/.../MANIFLOW_INFERENCE.md` | 수정, uncommitted | 이 문서 |
| `.gitignore` | 수정, uncommitted | `tasks/*/maniflow_{infer,control}_results/` 제외 |
| `protomotions/simulator/newton/simulator.py` | 이전 세션에 커밋됨 | qfrc_actuator readback(GT 토크) + `_write_viewport_to_file` 프레임 캡처 구현 |
| `tasks/.../infer_maniflow_newton.sh` | 이전 세션에 커밋됨 | 수동(passive) 비교 스크립트 실행 wrapper |
| `tasks/.../extract_walk_motion.py` | 이전 세션에 커밋됨 | walk 모션 클립 추출 유틸 |
| `data/motion_for_trackers/skeleton_torque_suit_walk.pt` | 이전 세션에 커밋됨 | walk 1클립 모션 (extract_walk_motion.py 산출물) |
| `~/ManiFlow_Policy/scripts/process_data_walking.py` | ManiFlow_Policy repo, 수정 | 채널 계약 검증(legacy 거부, `--allow_legacy_channels`), attrs 전파, 기본 출력 `walking-flat-newton-hips.zarr` |
| `~/ManiFlow_Policy/scripts/eval_walking_lowdim.py` | ManiFlow_Policy repo, 수정 | 채널 라벨을 데이터셋 attrs에서 파생 (hip_j 하드코딩 제거) |
| `~/ManiFlow_Policy/scripts/train_walking.sh` | ManiFlow_Policy repo, 수정 | zarr_path 6번째 인자화, 기본값 신규 hips 데이터셋 |
| `~/ManiFlow_Policy/.../config/walking_task/walking_flat.yaml` | ManiFlow_Policy repo, 수정 | 기본 zarr_path 신규 데이터셋 + 채널 주석 |
| `~/ManiFlow_Policy/.../dataset/walking_dataset.py` | ManiFlow_Policy repo, 수정 | 로드 시 action 채널 계약 출력/legacy 경고 |
| `~/ManiFlow_Policy/.../lowdim_obs_encoder.py` | ManiFlow_Policy repo, 이전 세션에 수정 | create_mlp 인라인 (pytorch3d 의존 제거) |

---

## 재수집·재학습 파이프라인 (2026-07-09 — hip 6채널)

위 "다음 단계" 후보였던 항목들이 실행되었습니다:

1. **재수집 (완료)**: Newton에서 hip 6채널로 재수집.
   ```bash
   bash tasks/mimic_suit_active_cable_walk_23dof/collect_walk_zarr.sh \
       tasks/mimic_suit_active_cable_walk_23dof/output_newton_flat/score_based.ckpt flat 64 2000
   # → zarr_data/flat/flat-newton-<ts>.zarr (attrs: action_dof_indices=[0,1,2,5,6,7])
   ```
2. **변환 (완료)**: ManiFlow flat 레이아웃으로.
   ```bash
   cd ~/Projects/ManiFlow_Policy
   python scripts/process_data_walking.py \
       --zarr_paths <ProtoMotions>/tasks/.../zarr_data/flat/flat-newton-<ts>.zarr \
       --save_path ManiFlow/data/walking-flat-newton-hips.zarr
   ```
3. **재학습 (완료, 2026-07-09~10)**: maniflow conda env에서.
   ```bash
   conda activate maniflow && cd ~/Projects/ManiFlow_Policy
   bash scripts/train_walking.sh maniflow_lowdim_policy_walking walking_flat \
       newton-hips-run02 42 0
   # zarr_path 기본값이 walking-flat-newton-hips.zarr (6번째 인자로 변경 가능)
   ```
   200 epochs ≈ 11시간 @ RTX 5090, best topk epoch=0190-val_loss=0.000716.
4. **평가 (완료)**:
   - 오프라인 val split: R²≈1.000 / corr 1.000 / MAE 0.16–0.68 N·m.
   - Newton 폐루프 passive (`infer_maniflow_newton.sh --maniflow-run-dir
     <run02>`): R²≥0.998 / corr 1.000 / MAE ≤0.8 N·m — legacy corr≈0.1
     완전 해소 (`maniflow_infer_results/2026-07-10_08-14-01`).
   - 폐루프 제어 A/B (`compare_maniflow_control_newton.sh`): 예측 토크
     피드포워드 인가만으로는 여전히 B 낙상 — 피드백 부재로 인한 오차 누적
     (`maniflow_control_results/2026-07-10_08-15-01`). 제어 활용은 아래
     탐색 옵션 참고.

### A/B 낙상 원인 분석 + 핸드오버 실험 (2026-07-10)

> 전체 분석(입력 배선 검증 4근거, 에피소드-상대 분해 표, 영상 읽는 법 포함)은
> **`MANIFLOW_CONTROL_ANALYSIS.md`** 참고. 아래는 요약.

"passive는 완벽한데 왜 A/B는 처음부터 나쁜가"를 traces.npz로 분해한 결과:

1. **영상의 빨강(B cmd) vs 검정(A applied)은 예측오차 지표가 아님** — 상태가
   갈라진 두 로봇의 토크라 완벽한 모델이어도 달라짐. 진짜 예측 품질은 같은
   chunk에서 함께 기록되는 주황 점선(`pred_a_passive`, A 상태 기준) vs 검정.
2. **리셋 직후 첫 chunk(스텝 0–2)만 진짜 실패** (`2026-07-10_09-58-38` 기준
   |predA−A| 62.7 N·m, 부호 반대; t≥3부터는 0.3 N·m로 완벽). 원인: 수집기는
   물리 1스텝 후 s₁부터 기록하므로 **리셋 상태 s₀가 학습 데이터에 없음**
   (contacts 미갱신 + 기구학 리셋 속도) + `pad_before=1` 복제 padding 샘플은
   에피소드당 1개(~0.08%)뿐. 그 3스텝의 잘못된 토크가 게인 0(무저항)인 hip을
   0.9 rad 밀어버려 0.7초 낙상.
3. **워밍업 핸드오버 실험** (`--handover-steps 40`, run02 모델,
   `2026-07-10_14-11-54`): 워밍업 40스텝 동안 B는 A와 완전 동일 보행
   (|qB−qA| 7e-5 rad), 전환 직후 첫 chunk 예측오차 0.38 N·m(완벽)에서 시작
   → 그래도 6/6 에피소드가 전환 후 **~26스텝(1.3초 ≈ 한 보행주기)** 만에
   낙상 (root z 0.96→0.68 단조 발산). every_step 재예측(`2026-07-10_14-13-01`)
   도 62–64스텝 낙상으로 **차이 없음**.
4. **결론**: 병목은 예측 품질/신선도가 아니라 **substep 피드백(관절 임피던스)
   부재**. A의 hip은 kp=200 N·m/rad PD가 물리 substep마다 상태 오차를 즉시
   보정하지만, B는 50ms ZOH 상수 토크만 받음 — 도립진자 시정수(~0.3 s)보다
   느린 보정이라 구조적으로 발산. 예측이 문자 그대로 완벽해도 순수
   피드포워드로는 보행 불가.

남은 탐색 옵션 (제어 활용 — 피드백 결합이 필수라는 위 결론 반영):
- ~~**잔여 PD 결합**~~ (**완료, 2026-07-14**): hip 토크 = α·PD + (1-α)·ManiFlow
  convex 블렌드 (`--residual-pd-scale α`, `JointTorqueOverride.engage(gain_scale)`).
  **α=0.5에서 A 동등 보행 60 s 완주**(err6 +1%, corr(A,B총) 0.95, handover
  불필요), 생존 문턱 α∈(0.25, 0.375]. ablation(MF 끔)으로 ManiFlow 기여 =
  품질 복원(hip 토크 50% 담당) 실증. 상세: `MANIFLOW_CONTROL_ANALYSIS.md`
  "잔여 PD 결합 실험".
- ~~**estimator-in-the-loop 학습**(DAgger류)~~ (**완료 2026-07-15,
  newton-hips-dagger-run03**): 외란 주입(DART식) + 잔여 PD 블렌드 on-policy
  데이터(832 eps)를 섞어 재학습. 결과: **α=0.25 + every_step에서 60 s 완주**
  (receding은 여전히 낙상 — 배운 교정 응답이 신선도 50 ms에서만 안정),
  문턱 α∈(0.25,0.375] → α≤0.25(every_step) 인하. passive 품질 유지
  (corr ≈1.000). 상세: `MANIFLOW_CONTROL_ANALYSIS.md` "DAgger 재학습 결과".
- ~~`--predict-mode every_step`~~ (기각): 실험 결과 생존 시간 개선 없음 —
  신선도가 병목이 아님을 확인.
- **obs 정제**(선택): 절대 x,y 제거 등 — Newton 재수집으로 도메인 갭이
  해소되어 우선순위 낮음.
- (참고) 수집기를 s₀부터 기록하게 바꾸면 리셋 직후 chunk 문제 자체는 완화
  가능하나, 핸드오버 실험이 보여주듯 그것만으로는 보행이 안 됨 (α≥0.375
  잔여 PD가 리셋 첫 chunk 오예측을 흡수하므로 실용상 해소).
