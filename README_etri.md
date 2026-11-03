# ProtoMotions — ETRI 적용 가이드

기반 프로젝트: [NVlabs/ProtoMotions](https://github.com/NVlabs/ProtoMotions)

본 저장소는 ETRI SDxSim 연구를 위해 ProtoMotions를 수정·확장한 버전입니다.  
suit(수트) 로봇 모델(skeleton_torque_suit, passive/active cable 23-DOF)과 Newton 시뮬레이터 기반 학습·추론·녹화 파이프라인을 포함합니다.

---

## 주요 추가·변경 사항

### 추가된 로봇 모델
- `protomotions/robot_configs/skeleton_torque_suit.py` — 수트 기본 모델 (torque 제어, 23-DOF)
- `protomotions/robot_configs/skeleton_torque_suit_passive_cable.py` — 수동 케이블 수트
- `protomotions/robot_configs/skeleton_torque_suit_active_cable.py` — 능동 케이블 수트
- `protomotions/robot_configs/skeleton_torque_suit_muscle.py` — 근육 기반 수트
- `protomotions/robot_configs/skeleton.py`, `skeleton_torque.py`, `skeleton_torque_31dof.py`
- `protomotions/robot_configs/etrisuit.py`, `etrisuit_active_cable.py`

### 추가된 에셋
- `protomotions/data/assets/mjcf/skeleton_torque_suit*.xml` — 수트 MJCF (기본/mesh/muscle 변형)
- `protomotions/data/assets/mesh/skeleton/` — 수트 골격 메쉬 (.stl)
- `data/motion_for_trackers/skeleton_torque_suit_motions14.pt` — 14개 모션 데이터

### 추가된 태스크
- `tasks/mimic_suit_passive_cable_motions14_23dof/` — 14개 모션 학습·추론·녹화 스크립트

### 수정된 파일
- `protomotions/inference_agent.py` — `--auto-record`, `--record-steps`, `--cycle-seconds`, `--recording-path`, `--use-skin` 인수 추가
- `protomotions/simulator/newton/simulator.py` — Newton 1.3.0 호환성 업데이트; 카메라 방위각 회전(`[`/`]`), contact 시각화 토글(`C`), UI 메뉴 기본 숨김
- `protomotions/simulator/isaaclab/simulator.py` — headless 모드 키보드 guard 추가
- `protomotions/simulator/base_simulator/record.py` — 녹화 기능 개선

---

## 환경 설정

### 요구사항
- Python 3.10+ (3.11+ 권장)
- NVIDIA GPU (compute capability ≥ 5.0)
- NVIDIA Driver 545+, CUDA 12.4

### 가상환경 생성 및 패키지 설치

```bash
# 가상환경 생성
python -m venv venv_newton
source venv_newton/bin/activate

# PyTorch (CUDA 12.4)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Newton (반드시 아래 버전 사용)
pip install "newton[examples]==1.3.0"

# mujoco (반드시 아래 버전 사용)
pip install "mujoco==3.8.1"

# ProtoMotions 및 의존성
pip install -e /path/to/ProtoMotions
pip install -r /path/to/ProtoMotions/requirements_newton.txt
```

---

## 업그레이드된 라이브러리

기본 ProtoMotions 설치 대비 아래 버전으로 업그레이드가 필요합니다.

| 패키지 | 기본 버전 | ETRI 적용 버전 | 비고 |
|--------|-----------|----------------|------|
| `newton` | 1.2.1 | **1.3.0** | body label 포맷 변경 |
| `warp-lang` | 1.13.0 | **1.14.0** | newton 1.3.0 의존성 |
| `mujoco` | 3.5.0 | **3.8.1** | `mujoco_warp` 호환성 |

```bash
pip install "newton==1.3.0" "warp-lang==1.14.0" "mujoco==3.8.1"
```

---

## 필수 패치

### Newton MJCF rgba 색상 패치

Newton 1.2.x / 1.3.x에서 MJCF의 `rgba` 속성이 box, cylinder, sphere 등  
primitive shape에 전달되지 않는 버그가 있습니다.  
아래 패치를 **환경 설정 후 반드시** 적용해야 시각화 색상이 올바르게 표시됩니다.

```bash
cd /path/to/ProtoMotions

# 패치 적용
python scripts/patch_newton_mjcf_rgba.py

# 적용 여부 확인
python scripts/patch_newton_mjcf_rgba.py --check

# 패치 되돌리기 (필요 시)
python scripts/patch_newton_mjcf_rgba.py --revert
```

> **주의**: `pip install --upgrade newton` 등으로 newton을 재설치하면 패치가 초기화됩니다.  
> 재설치 후 반드시 패치를 다시 적용하세요.

---

## 실행 방법

### Newton 시각화 (interactive play)

```bash
cd /path/to/ProtoMotions

# last.ckpt 사용 (기본)
bash tasks/mimic_suit_passive_cable_motions14_23dof/play_newton.sh

# 특정 체크포인트 지정
bash tasks/mimic_suit_passive_cable_motions14_23dof/play_newton.sh \
    tasks/mimic_suit_passive_cable_motions14_23dof/output_newton/score_based.ckpt
```

키보드 조작:
- `R` — 다음 모션으로 전환
- `L` — 녹화 시작/종료 (수동)
- `O` — 카메라 추적 대상 전환
- `C` — contact 시각화 ON/OFF
- `[` — 카메라 왼쪽 30도 회전
- `]` — 카메라 오른쪽 30도 회전
- `B` — 후방 뷰 (180°)
- `N` — 전방 뷰 (0°)
- `Q` — 종료

> UI 메뉴는 기본적으로 숨겨진 상태로 시작합니다.

### Newton 자동 녹화 (14모션 × 20초)

```bash
bash tasks/mimic_suit_passive_cable_motions14_23dof/record_newton.sh
```

출력 위치: `tasks/mimic_suit_passive_cable_motions14_23dof/recordings/<ckpt>-<datetime>/`

### IsaacLab 시각화 (interactive play)

```bash
bash tasks/mimic_suit_passive_cable_motions14_23dof/play_isaaclab.sh

# 특정 체크포인트 지정
bash tasks/mimic_suit_passive_cable_motions14_23dof/play_isaaclab.sh \
    tasks/mimic_suit_passive_cable_motions14_23dof/output_isaaclab/score_based.ckpt
```

### IsaacLab 자동 녹화 (14모션 × 20초, headless)

```bash
bash tasks/mimic_suit_passive_cable_motions14_23dof/record_isaaclab.sh
```

출력 위치: `tasks/mimic_suit_passive_cable_motions14_23dof/recordings/<ckpt>-<datetime>/`

---

## 전체 설치 요약 (빠른 시작)

```bash
# 1. 가상환경
python -m venv venv_newton && source venv_newton/bin/activate

# 2. 패키지
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install "newton[examples]==1.3.0" "mujoco==3.8.1"
pip install -e . && pip install -r requirements_newton.txt

# 3. rgba 색상 패치
python scripts/patch_newton_mjcf_rgba.py

# 4. 동작 확인
bash tasks/mimic_suit_passive_cable_motions14_23dof/play_newton.sh
```

---

## 설치·실행 중 겪은 문제와 수정 기록 (2026-06, 협업 연구원 작성)

원래 `NOTES.md`에 있던 개인 기록을 2026-11-03에 이 문서로 합쳤다. 작성 환경은 Windows 11 + WSL2 Ubuntu 22.04,
RTX 4080, CUDA 11.8, Newton 1.3.0이며 upstream 데모 모델(G1 `g1-bones-deploy`, SOMA `soma-bones`)로 확인한 내용이다.
2026-11-03 기준으로 MuJoCo gear 보상·실시간 재생 조절, 뷰어의 모션 이름 표시, Newton `include_joints=list(...)`·`sensing_indices`
처리는 현재 코드에 있고, SOMA 리셋 바닥 보정(아래 마지막 항목)은 현재 코드에 없다. 이 저장소의 `sbc` 환경은 Newton 1.2.0.dev0이다.

### SOMA 하체가 땅 속으로 꺼짐
원인: `init_start_prob=1.0`이라 바닥 크롤링 클립(root_z ≈ 0.09 m)의 임의 프레임에서 시작.
해결: `data/pretrained_models/motion_tracker/soma-bones/resolved_configs_inference.pt`의
`env_config.motion_manager.init_start_prob`를 0.0으로 바꿔 항상 클립 첫 프레임에서 시작.

### SOMA T-포즈 고정·물리 폭발 (MuJoCo)
원인: SOMA MJCF 액추에이터의 `gear=200~250`이 implicit PD force에 곱해져 실효 kp = 500 × 250 = 125,000 N·m/rad
(증상 `max_dof_vel=2243 rad/s`, `root_vel=77 m/s`, 한 스텝 뒤 root_z → 0.002 m).
해결: `protomotions/simulator/mujoco/simulator.py`의 `_configure_actuators_for_pd()`가 kp/kd/effort를 gear로 나눠 설정한다.

```python
gear = float(self.model.actuator_gear[act_idx, 0]) or 1.0
self.model.actuator_gainprm[act_idx, 0] = kp / gear
self.model.actuator_biasprm[act_idx, 1] = -kp / gear
self.model.actuator_biasprm[act_idx, 2] = -kd / gear
self.model.actuator_forcerange[act_idx] = [-effort / gear, effort / gear]
```

### MuJoCo 뷰어가 6배 빠르게 재생
원인: `viewer.sync()` 뒤에 대기가 없음. 해결: `_physics_step()`에 실시간 조절(정책 주기 − 경과 시간만큼 `time.sleep`)을 넣었다.

### 화면 상단 모션 이름 표시
`protomotions/inference_agent.py`가 뷰어 위쪽 가운데에 재생 중인 클립 이름을 표시한다.

### Newton 1.3.0 API 차이 (당시 수정 기록)
- `ArticulationView(include_joints=...)`가 `dict_keys`를 못 받아 `list(...)`로 넘긴다.
- `SensorContact`: 인자 `sensing_obj_bodies` → `sensing_bodies`, force 속성 `net_force` → `total_force`(`hasattr`로 양쪽 지원),
  매칭 수는 `sensing_indices`로 센다.
- body label이 전체 경로(`humanoid/worldbody/Hips/LeftLeg/LeftShin/LeftFoot`)라 패턴은 `*/LeftFoot` 대신 `*LeftFoot`.
- 첫 실행의 CUDA graph 컴파일은 2~5분 걸린다(정상, 이후 캐시).

### SOMA가 Newton에서 쓰러짐 (sim2sim 차이)
원인 1(당시 수정, 현재 코드에는 없음): 클립 0번 프레임의 발이 바닥보다 약 7 cm 위라 자유낙하 뒤 튕김 →
`_set_simulator_env_state()`에서 리셋 때 가장 낮은 body z를 0으로 내리는 보정을 넣었었다.
원인 2(미해결): 모델이 IsaacGym/PhysX(단단한 접촉)에서 학습되어 Newton의 부드러운 접촉·다른 마찰에서 균형을 못 잡는다
(보정 뒤 step 1 접촉 423 N, step 19 root_z 0.78 m). Newton에서 추가 학습이 필요하다.
