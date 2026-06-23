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
