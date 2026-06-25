# 데이터 수집 및 라이브 시각화 가이드

수트 착용 인체모델(skeleton\_torque\_suit, 27-DOF)의 추론 중 로봇 상태 데이터를 수집하고
실시간으로 시각화하기 위한 스크립트 및 사용법을 정리합니다.

---

## 생성 파일 목록

| 파일 | 위치 | 용도 |
|------|------|------|
| `collect_data_suit.py` | `protomotions/` | 추론 데이터 → Zarr 저장 |
| `infer_live_viz_suit.py` | `protomotions/` | 추론 중 실시간 matplotlib 시각화 |
| `collect_newton.sh` | `tasks/mimic_suit_active_cable_motions14_23dof/` | Active cable 데이터 수집 |
| `live_viz_newton.sh` | `tasks/mimic_suit_active_cable_motions14_23dof/` | Active cable 라이브 시각화 |
| `collect_newton.sh` | `tasks/mimic_suit_passive_cable_motions14_23dof/` | Passive cable 데이터 수집 |
| `live_viz_newton.sh` | `tasks/mimic_suit_passive_cable_motions14_23dof/` | Passive cable 라이브 시각화 |

---

## 1. 데이터 수집 (`collect_data_suit.py`)

[ProtoMotions의 `collect_data.py`](../protomotions/collect_data.py) 기반.
`--active-cable` 플래그 추가 시 케이블 전용 필드를 추가 저장합니다.

### Zarr 출력 구조

```
dataset.zarr/
├── data/
│   ├── dof_pos          [N, 27]        관절 위치 (rad)
│   ├── dof_vel          [N, 27]        관절 속도 (rad/s)
│   ├── dof_forces       [N, 27]        관절 토크 (N·m)
│   ├── body_pos         [N, n_body, 3] 링크 위치 (m)
│   ├── body_rot         [N, n_body, 4] 링크 회전 (xyzw 쿼터니언)
│   ├── body_vel         [N, n_body, 3] 링크 선속도 (m/s)
│   ├── body_ang_vel     [N, n_body, 3] 링크 각속도 (rad/s)
│   ├── contacts         [N, n_body]    접촉 이진 플래그
│   ├── contact_forces   [N, n_body, 3] 접촉력 (N)
│   ├── actions          [N, 27]        PPO 네트워크 출력 액션
│   │
│   │   ── 아래는 --active-cable 전용 ──
│   ├── cable_pos        [N, 4]         slide1-4 실제 변위 (m)
│   ├── cable_forces     [N, 4]         slide1-4 토크 (N·m)
│   ├── hip_angles       [N, 2]         [hip_r, hip_l] 굴곡각 (rad)
│   ├── dofc_targets     [N, 2]         [slide2_tgt, slide4_tgt] DOFC 목표 (m)
│   └── dofc_balance     [N, 1]         y = sin(−hip_r) − sin(−hip_l)
└── meta/
    ├── episode_ends     [n_episodes]   에피소드 끝 인덱스
    └── episode_env_ids  [n_episodes]   소스 env 번호
```

루트 속성: `fps`, `dt`, `body_names`, `dof_names`, `num_envs`, `total_timesteps`,
`num_episodes`, `checkpoint`, `active_cable`
— Active cable 시 추가: `cable_dof_names`, `dofc_params`

### 실행 예시

```bash
cd /home/user/ProtoMotions

# Active cable 데이터 수집 (16 env × 5000 스텝 → ~1,500 에피소드)
python protomotions/collect_data_suit.py \
    --checkpoint tasks/mimic_suit_active_cable_motions14_23dof/output_newton/score_based.ckpt \
    --motion-file data/motion_for_trackers/skeleton_torque_suit_motions14.pt \
    --simulator newton --num-envs 16 --num-steps 5000 --active-cable --headless \
    --output data/collected/active_cable_newton.zarr

# Passive cable (케이블 필드 없음)
python protomotions/collect_data_suit.py \
    --checkpoint tasks/mimic_suit_passive_cable_motions14_23dof/output_newton/score_based.ckpt \
    --motion-file data/motion_for_trackers/skeleton_torque_suit_motions_11+koo_4.pt \
    --simulator newton --num-envs 16 --num-steps 5000 --headless \
    --output data/collected/passive_cable_newton.zarr
```

또는 셸 스크립트로 바로 실행 (타임스탬프 자동 부여):

```bash
bash tasks/mimic_suit_active_cable_motions14_23dof/collect_newton.sh
bash tasks/mimic_suit_passive_cable_motions14_23dof/collect_newton.sh
```

### 수집 데이터 불러오기 예시 (Python)

```python
import zarr, numpy as np

store = zarr.open("data/collected/active_cable_newton.zarr", mode="r")

print(store.attrs.asdict())          # fps, dof_names, dofc_params, …
dof_pos = store["data/dof_pos"][:]   # [N, 27]
cable_pos  = store["data/cable_pos"][:]    # [N, 4]  — slide1-4 실제 변위
dofc_tgts  = store["data/dofc_targets"][:] # [N, 2]  — [slide2, slide4] 목표
hip_angles = store["data/hip_angles"][:]   # [N, 2]  — [hip_r, hip_l]
balance_y  = store["data/dofc_balance"][:] # [N, 1]

episode_ends = store["meta/episode_ends"][:] # 에피소드 구분 인덱스
```

---

## 2. 라이브 시각화 (`infer_live_viz_suit.py`)

[ProtoMotions의 `infer_live_viz.py`](../protomotions/infer_live_viz.py) 기반.
`--active-cable` 플래그 추가 시 케이블 전용 패널 2개가 추가됩니다.

### 화면 구성

| 행 | 왼쪽 패널 | 오른쪽 패널 |
|----|----------|------------|
| 0 | **Root Height — 전체 Env 개요** (가로 전체) | |
| 1 | Root Height + Z속도 | Contact Forces (N) |
| 2 | Joint Torques (N·m) | Joint Positions (rad) |
| 3 | Actions | Root Linear Velocity (m/s) |
| 4★ | **Cable Displacement** | **Hip Angles & DOFC Balance** |

★ `--active-cable` 일 때만 표시

#### Cable Displacement 패널 (행 4 왼쪽)

- **slide2 actual** (파란 실선): DOF 24 실제 변위
- **slide2 target** (파란 점선): DOFC A버전이 계산한 slide2 목표
- **slide4 actual** (주황 실선): DOF 26 실제 변위
- **slide4 target** (주황 점선): slide4 목표

실선과 점선의 간격 = PD 제어기 추종 오차

#### Hip Angles & DOFC Balance 패널 (행 4 오른쪽)

- **hip_r** (빨간 실선): 오른쪽 엉덩이 굴곡각 (DOF 0)
- **hip_l** (파란 실선): 왼쪽 엉덩이 굴곡각 (DOF 5)
- **y balance** (초록 점선): `y = sin(−hip_r) − sin(−hip_l)` — DOFC 입력 신호

### 키보드 단축키

| 키 | 동작 |
|----|------|
| `←` / `→` | 이전/다음 env 선택 |
| `0` ~ `9` | env 번호로 바로 이동 |
| `q` | 종료 |

오른쪽 CheckButton 패널로 Torques / Positions 그룹별 표시 토글 가능.

### 실행 예시

```bash
cd /home/user/ProtoMotions

# Active cable 라이브 시각화 (케이블 패널 포함)
python protomotions/infer_live_viz_suit.py \
    --checkpoint tasks/mimic_suit_active_cable_motions14_23dof/output_newton/score_based.ckpt \
    --motion-file data/motion_for_trackers/skeleton_torque_suit_motions14.pt \
    --simulator newton --num-envs 1 --active-cable --headless

# Passive cable (케이블 패널 없음)
python protomotions/infer_live_viz_suit.py \
    --checkpoint tasks/mimic_suit_passive_cable_motions14_23dof/output_newton/score_based.ckpt \
    --motion-file data/motion_for_trackers/skeleton_torque_suit_motions_11+koo_4.pt \
    --simulator newton --num-envs 1 --headless

# 여러 env + 특정 관절만 표시
python protomotions/infer_live_viz_suit.py \
    --checkpoint ... --simulator newton --num-envs 4 --active-cable \
    --torque-dofs hip_flexion knee --pos-dofs hip_flexion knee --headless
```

또는 셸 스크립트:

```bash
bash tasks/mimic_suit_active_cable_motions14_23dof/live_viz_newton.sh
```

---

## 3. DOFC A버전 수식 참고

`active_cable_env.py`에 정의된 파라미터와 동일 값을 수집/시각화 스크립트에서 재계산합니다.

```
y       = sin(−hip_r) − sin(−hip_l)
τ_right = −κ · y
τ_left  =  κ · y

target_slide2 = clamp(  τ_right · G / r / k ,  0,  0.51 )
target_slide4 = clamp( −τ_left  · G / r / k ,  0,  0.51 )
```

| 파라미터 | 값 | 의미 |
|---------|-----|------|
| κ (kappa) | 1.1 | 제어 게인 |
| G (ext_gain) | 0.8 | 익스텐더 비율 |
| r (pulley_radius) | 0.042 m | 풀리 반경 |
| k (stiffness) | 50.0 N/m | slide joint 강성 |

> **Note**: 수집/시각화에서 계산하는 `dofc_targets`는 **현재 스텝의 post-physics 상태**에서
> 다음 스텝 목표를 예측한 값입니다. 실제 제어 시점은 pre-physics 상태 기준으로
> 한 스텝 앞서므로 약 10 ms(1 control step) 차이가 있습니다.

---

## 4. 참조 스크립트

| 스크립트 | 위치 | 역할 |
|---------|------|------|
| `collect_data.py` | ProtoMotions `protomotions/` | 범용 데이터 수집 원본 |
| `infer_live_viz.py` | ProtoMotions `protomotions/` | 범용 라이브 시각화 원본 |
| `active_cable_env.py` | `protomotions/envs/base_env/` | DOFC A버전 구현 |
