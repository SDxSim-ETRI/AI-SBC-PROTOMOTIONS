# ManiFlow Hip-Torque 추정기 — Newton Inference 가이드

ManiFlow_Policy에서 학습한 **센서 상태 전용(vision-free) hip torque 추정기**를
ProtoMotions 시뮬레이터(기본 Newton) 안에서 폐루프로 돌리고, 예측 토크를 실제
적용 토크와 비교하는 파이프라인 문서입니다.

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

- 예측 토크로 로봇을 직접 구동하는 모드는 **아직 없습니다**. 만들려면 hip 6개
  DOF만 TORQUE 제어 + 나머지는 PD로 두는 하이브리드 제어가 필요합니다 (미구현).
- 데이터 수집(IsaacLab, `collect_walk_zarr.py`)과 동일한 관계입니다 — 그때도
  RL policy가 걷고 hip torque는 기록만 했으며, ManiFlow는 그 기록으로 학습됨.

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

# 최종 학습 모델 지정 (아래 '체크포인트' 참고 — 현재 자동 선택은 epoch30을 집음)
bash tasks/mimic_suit_active_cable_walk_23dof/infer_maniflow_newton.sh \
    --maniflow-ckpt ~/Projects/ManiFlow_Policy/ManiFlow/data/outputs/walking_flat-maniflow_lowdim_policy_walking-run01_seed42/checkpoints/latest.ckpt
```

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--rl-checkpoint` | `output_newton_flat/score_based.ckpt` | 보행 RL policy (로봇을 실제로 걷게 하는 주체) |
| `--maniflow-ckpt` | run01_seed42에서 best topk 자동 | ManiFlow 추정기 체크포인트 |
| `--simulator` | `newton` | `isaaclab`도 가능 (학습 도메인 교차검증용) |
| `--num-envs` / `--episode-steps` | 2 / 1200 | 1200 steps = 60 s @ 20 Hz |
| `--viewer` | off | 실시간 GUI (skeleton mesh 기본) |
| `--record` | off | mp4 + 토크 합성 비디오 (뷰어 자동 활성화) |
| `--no-mesh` | off | 시각화를 캡슐 에셋으로 (수집 조건 물리 완전 재현) |
| `--predict-mode` | `receding` | 4스텝 청크 예측(오프라인 eval과 동일) / `every_step` |
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

## 체크포인트

- **RL (보행)**: `output_newton_flat/{score_based,last}.ckpt` — Newton에서 학습된
  tracker. `output_isaaclab_flat/score_based.ckpt`(수집 데이터를 만든 policy)를
  Newton에서 sim2sim으로 돌릴 수도 있음 (자세가 더 무너짐 — 아래 도메인 갭 참고).
- **ManiFlow**: 학습 run이 2026-07-08 완주 (epoch 225, 최종 eval loss ~0.0003).
  ⚠️ **topk 파일이 `epoch=0030`에서 갱신되지 않은 이슈**가 있어 자동 선택이 epoch30을
  집습니다. 최종 모델을 쓰려면 `--maniflow-ckpt .../latest.ckpt` 명시.
  (학습이 끝났으므로 latest.ckpt를 읽어도 안전. 학습 재개 시에는 latest.ckpt가
  주기적으로 재작성되므로 topk 파일을 쓸 것.)

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
action(6) = hip torque  [hip_flexion_r, hip_adduction_r, hip_rotation_r,
                         hip_flexion_l, hip_adduction_l, hip_rotation_l]
```
`Simulator.get_robot_state()`가 모든 필드를 common ordering으로 변환하므로
시뮬레이터가 달라도 레이아웃이 유지됩니다. root_pos/vel = pelvis(body 0) 월드
위치/선속도, contacts = binary flag.

### 3. 현재 Newton 예측 품질이 낮은 이유 (배선 문제 아님 — 도메인 갭)
- **배선 검증**: 학습 zarr 데이터를 estimator 경로로 흘리면 R²≈0.98–0.997 —
  obs 조립/정렬/정규화 정상.
- **Newton 폐루프**: corr ~0.15–0.18. 원인 진단 결과:
  1. Newton에서 실현되는 보행 자세가 학습 데이터(IsaacLab, 모션 추종 타이트)보다
     레퍼런스에서 크게 이탈 (ankle ~1 rad 수준까지) — newton-policy든
     isaaclab-policy sim2sim이든 마찬가지.
  2. 발뒤꿈치 contact 플래그(contacts[4]/[11])가 학습 데이터에선 **항상 1**인데
     Newton에선 0.16~0.22로 토글 — 모델이 본 적 없는 입력.
  3. (학습 데이터 자체 이슈) root_pos z에 낙하 쓰레기 값(-4032 m)이 섞여 있어
     normalizer가 높이 정보를 사실상 소거함.
- **개선 방향**: Newton에서 데이터 재수집 후 재학습/파인튜닝 (이 스크립트가
  `traces.npz`에 obs까지 저장하므로 수집기로 재사용 가능), 또는 obs 정제
  (절대 x,y 제거, contact 정합화) 후 재학습.

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

### 6. walk 모션 파일
`data/motion_for_trackers/skeleton_torque_suit_walk.pt`(walk 1클립)가 이 머신에
없어서 `extract_walk_motion.py`로 학습용 멀티모션 파일
(`skeleton_torque_suit_motions_11+koo_4.pt`)의 walk 클립(idx 10)에서 추출·복원해
두었습니다. play/record/collect 스크립트 전부가 이 파일을 참조합니다.

### 7. 관련 파일 맵 (이번 통합에서 만든/수정한 파일 포함 — 커밋 시 참고)

| 위치 | 상태 | 내용 |
|------|------|------|
| `protomotions/maniflow/` | 신규 | 통합 레이어 (loader + ManiFlowTorqueEstimator + README) |
| `protomotions/simulator/newton/simulator.py` | 수정 | qfrc_actuator readback(GT 토크) + `_write_viewport_to_file` 프레임 캡처 구현 |
| `tasks/.../infer_maniflow_newton.{py,sh}` | 신규 | 이 문서의 메인 스크립트 |
| `tasks/.../extract_walk_motion.py` | 신규 | walk 모션 클립 추출 유틸 |
| `tasks/.../MANIFLOW_INFERENCE.md` | 신규 | 이 문서 |
| `data/motion_for_trackers/skeleton_torque_suit_walk.pt` | 복원 | walk 1클립 모션 (extract_walk_motion.py 산출물) |
| `.gitignore` | 수정 | `tasks/*/maniflow_infer_results/` 제외 추가 |
| `tasks/.../collect_walk_zarr.py` | 기존 | (IsaacLab) 학습 데이터 수집기 — obs 레이아웃의 원본 정의 |
| `~/Projects/ManiFlow_Policy/scripts/eval_walking_lowdim.py` | 기존 | 오프라인 eval (val split 전체) |
| `~/Projects/ManiFlow_Policy/.../lowdim_obs_encoder.py` | 수정 | create_mlp 인라인 (pytorch3d 의존 제거, ManiFlow repo 쪽) |
