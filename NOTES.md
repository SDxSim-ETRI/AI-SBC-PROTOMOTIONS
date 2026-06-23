# ProtoMotions 설치 및 실행 노트

## 환경

- OS: Windows 11 + WSL2 Ubuntu 22.04
- GPU: NVIDIA GeForce RTX 4080 (16GB)
- CUDA: 11.8 (torch 2.7.1+cu118)
- Python: 3.10.12
- 가상환경: `~/venv_loco`

---

## 실행 스크립트

```bash
cd ~/ProtoMotions

# G1 로봇 (MuJoCo)
./scripts/run_g1_tiny.sh

# SOMA 인체 모델 (MuJoCo)
./scripts/run_soma.sh

# SOMA 인체 모델 (Newton GPU - 권장)
./scripts/run_soma_newton.sh
```

### 키보드 단축키 (실행 중)
| 키 | 기능 |
|---|---|
| `R` | 환경 리셋 |
| `J` | 랜덤 외력 적용 (강인성 테스트) |
| `O` | 카메라 뷰 전환 |
| `L` | 영상 녹화 시작/정지 |
| `Q` | 종료 |

---

## 발생했던 문제 및 해결

### 1. SOMA 하체가 땅 속으로 꺼짐

**원인**: `init_start_prob=1.0` → 바닥 크롤링 클립(root_z≈0.09m)의 랜덤 프레임에서 시작  
**해결**: `resolved_configs_inference.pt`에서 `init_start_prob` 0.0으로 변경 (항상 클립 첫 프레임부터 시작)

```python
# 수정 위치: data/pretrained_models/motion_tracker/soma-bones/resolved_configs_inference.pt
env_config.motion_manager.init_start_prob = 0.0
```

---

### 2. SOMA T-포즈 고정 / 물리 폭발 (MuJoCo)

**원인**: SOMA MJCF 액추에이터의 `gear=200~250`이 implicit PD force에 곱해져  
실효 kp = 500×250 = 125,000 Nm/rad → 관절 토크 75,000 Nm → 물리 폭발

**증상**: `max_dof_vel=2243 rad/s`, `root_vel=77 m/s`, 1스텝 후 root_z→0.002m

**해결**: `protomotions/simulator/mujoco/simulator.py`의 `_configure_actuators_for_pd()` 수정  
kp/kd/effort를 gear 값으로 나눠서 설정 (gear 보상)

```python
# protomotions/simulator/mujoco/simulator.py: _configure_actuators_for_pd()
gear = float(self.model.actuator_gear[act_idx, 0])
if gear <= 0:
    gear = 1.0
kp_scaled = kp / gear
kd_scaled = kd / gear
effort_scaled = effort / gear
self.model.actuator_gainprm[act_idx, 0] = kp_scaled
self.model.actuator_biasprm[act_idx, 1] = -kp_scaled
self.model.actuator_biasprm[act_idx, 2] = -kd_scaled
self.model.actuator_forcerange[act_idx] = [-effort_scaled, effort_scaled]
```

---

### 3. MuJoCo 뷰어 실시간 동기화 없음 (6배 빠른 재생)

**원인**: `viewer.sync()` 호출 후 sleep 없음 → 6x fast-forward 재생

**해결**: `protomotions/simulator/mujoco/simulator.py`의 `_physics_step()`에 real-time throttle 추가

```python
policy_dt = self.decimation / self.config.sim.fps
elapsed = time.perf_counter() - self._step_wall_time
remaining = policy_dt - elapsed
if remaining > 0.001:
    time.sleep(remaining)
self._step_wall_time = time.perf_counter()
self.viewer.sync()
```

---

### 4. 화면 상단 모션 이름 HUD

**구현 위치**: `protomotions/inference_agent.py` (line 367~378)  
MuJoCo 뷰어 상단에 현재 재생 중인 모션 클립 이름 표시

---

### 5. Newton 시뮬레이터 설치

**필요 시스템 패키지**:
```bash
sudo apt-get install -y libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libxext-dev libopengl0
```

**Newton 설치**:
```bash
source ~/venv_loco/bin/activate
pip install "newton[examples]"
```

---

### 6. Newton API 호환성 문제 (Newton 1.3.0 기준)

ProtoMotions 코드가 Newton 1.0 기준으로 작성되어 1.3.0에서 수정 필요.

#### 6-1. `ArticulationView` — `dict_keys` 에러
**파일**: `protomotions/simulator/newton/simulator.py`  
```python
# 수정 전
include_joints=self._newton_dof_names.keys()
# 수정 후
include_joints=list(self._newton_dof_names.keys())
```

#### 6-2. `SensorContact` — 파라미터명 변경
**파일**: `protomotions/simulator/newton/simulator.py`  
```python
# 수정 전 (Newton 1.0)
SensorContact(model, sensing_obj_bodies=patterns, ...)
# 수정 후 (Newton 1.3)
SensorContact(model, sensing_bodies=patterns, ...)
```

#### 6-3. `SensorContact` — body 패턴 매칭
Newton의 body label이 전체 경로 형식 (`humanoid/worldbody/Hips/LeftLeg/LeftShin/LeftFoot`)이므로  
`*/LeftFoot` 대신 `*LeftFoot` 사용 (`*`는 `/` 포함 매칭)

```python
# 수정 전
return [body_name, f"*/{body_name}"]
# 수정 후
return [f"*{body_name}"]
```

#### 6-4. `SensorContact` — force 속성명 변경
```python
# 수정 전 (Newton 1.0): net_force
# 수정 후 (Newton 1.3): total_force
force_attr = "total_force" if hasattr(sensor, "total_force") else "net_force"
```

#### 6-5. Newton contact sensor 매칭 카운트
```python
# 수정 전: getattr(sensor, "sensing_objs", [])
# 수정 후: sensor.sensing_indices
idx = getattr(sensor, "sensing_indices", None)
if idx is not None:
    return len(idx)
```

---

### 7. Newton CUDA graph 최초 컴파일

첫 실행 시 CUDA graph 컴파일로 **2~5분 소요** — 정상입니다.  
이후 실행부터는 캐시되어 빠르게 시작합니다.

---

### 8. SOMA Newton sim2sim 갭 — 캐릭터 쓰러짐

**현상**: 실행 시 즉시 쓰러짐

**원인 1 (초기화 버그, 수정됨)**: 모션 클립 0번 프레임에서 발이 Newton 바닥면(z=0)으로부터 ~7cm 위에 위치 → 수 스텝 자유낙하 후 강한 바운스  
**해결**: `_set_simulator_env_state()`에 그라운딩 보정 추가 — 리셋 시 최저 body z를 0으로 내림

```python
# protomotions/simulator/newton/simulator.py: _set_simulator_env_state()
gap = min_body_z.clamp(min=0.0)   # ~0.07m
root_z -= gap                      # 발이 바닥에 닿도록 보정
```

**원인 2 (sim2sim 갭, 파인튜닝 필요)**: 모델이 IsaacGym/PhysX 환경에서 훈련됨  
- PhysX: rigid contact (높은 강성), 즉각적 지면 반력  
- Newton/MuJoCo: soft constraint contact (더 유연), 다른 마찰 모델  
- 정책이 PhysX 다이나믹스에 최적화되어 있어 Newton에서 균형 유지 불가

**진단 데이터** (그라운딩 보정 후):
```
step 0: root_z=0.956m, 접촉=0N (그라운딩 보정 완료)
step 1: root_z=0.955m, RightToeBase=423N (즉시 접촉 성공)
step 7: root_z=1.077m (정책이 보행 시도)
step 19: root_z=0.778m (Newton 물리에서 균형 실패)
```

**권장 해결책**: Newton에서 파인튜닝 훈련 필요  
- 기존 체크포인트에서 시작 (`--checkpoint ...last.ckpt`)
- `--simulator newton`으로 추가 학습
- 수천 스텝의 fine-tuning으로 Newton 물리에 적응

---

## 모델 정보

| 모델 | 체크포인트 | DOFs | Bodies | 훈련 환경 |
|---|---|---|---|---|
| G1 | `data/pretrained_models/motion_tracker/g1-bones-deploy/` | 29 | 33 | IsaacGym |
| SOMA | `data/pretrained_models/motion_tracker/soma-bones/` | 66 | 23 | IsaacGym (mujoco로 기재되어 있으나 실제 훈련 시 isaacgym 사용) |

## 모션 데이터

| 파일 | 클립 수 | FPS | 특징 |
|---|---|---|---|
| `g1_bones_seed_mini.pt` | 58 | 30 | locomotion 위주 (걷기/달리기/점프) |
| `soma23_bones_seed_mini.pt` | 61 | 30 | 전신 인체 모션 (idle/크롤/댄스 등 다양) |
