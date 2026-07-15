# protomotions.maniflow — ManiFlow 정책 통합 레이어

외부에서 학습된 ManiFlow 정책(현재: 센서 상태만으로 hip torque를 추정하는
lowdim inverse-dynamics 모델)을 ProtoMotions 시뮬레이션 루프 안에서 돌리기
위한 통합 패키지입니다.

**설계 원칙**: 시뮬레이션 쪽 코드는 `maniflow.*`를 직접 import하지 않고 반드시
이 패키지를 통해서만 접근합니다. ManiFlow 코드베이스와의 결합점을 이 패키지
하나로 좁혀 두었기 때문에, 추후 ManiFlow 모델 코드를 이 repo에 포함(vendoring)
하는 개편을 하더라도 이 패키지 내부(`loader.py`의 경로 해석)만 바꾸면 됩니다.

## 구성

| 파일 | 역할 |
|------|------|
| `channels.py` | action 채널 계약의 단일 소스 — `HIP_DOF_NAMES`, `hip_dof_indices()`(이름 기반 공통 인덱스 파생). 수집·학습·inference가 모두 이걸 사용 |
| `loader.py` | `maniflow` 패키지 위치 해석 + 학습 workspace 체크포인트를 inference 전용 정책으로 로드 (workspace/dataset/wandb 등 학습 의존성 우회) |
| `torque_estimator.py` | `ManiFlowTorqueEstimator` — `RobotState`(common ordering) → 관측 벡터 구성, obs history 관리, receding-horizon 토크 예측 |
| `hybrid_control.py` | `JointTorqueOverride` — Newton BUILT_IN_PD 위에서 특정 env의 특정 DOF만 직접 토크 구동 (per-world PD 게인 zero-out + `control.joint_f`→`qfrc_applied` 주입). ManiFlow 예측 토크를 실제 제어에 사용할 때 씀 |

## maniflow 패키지 해석 순서 (`ensure_maniflow_importable`)

1. 이미 import된 `maniflow` 재사용
2. 명시적 `maniflow_root=` 인자 → `$MANIFLOW_ROOT` 환경변수
   (둘 다 `maniflow` 패키지를 담은 디렉토리, 예: `.../ManiFlow_Policy/ManiFlow`)
3. 설치/vendoring되어 경로 조작 없이 import 가능한 패키지 ← **추후 개편 시 여기로 수렴**
4. 관례 경로 `~/Projects/ManiFlow_Policy/ManiFlow`

## 관측 계약 (walking hip-torque 태스크)

`collect_walk_zarr.py`가 기록하고 ManiFlow가 학습한 레이아웃과 동일해야 합니다:

```
obs(88) = dof_pos(27) + dof_vel(27) + root_pos(3) + root_vel(3) + contacts(28)
action(6) = 순수 hip 6개 DOF 적용 토크 (공통 DOF [0,1,2,5,6,7])
          = [hip_flexion_r, hip_adduction_r, hip_rotation_r,
             hip_flexion_l, hip_adduction_l, hip_rotation_l]
```

채널 인덱스는 반드시 `channels.hip_dof_indices(dof_names)`로 이름 기반
파생하세요 — 공통 DOF 순서는 kinematic tree 순서(오른다리 체인 → 왼다리
체인 → …)라서 앞 6개를 자르면(`[:6]`) knee_angle_r(3)/ankle_angle_r(4)가
섞입니다.

⚠️ **Legacy 이력 (2026-07-09 이전 수집·학습)**: 초기 수집 스크립트가
hips=0-5로 잘못 가정해, 그때 학습된 모델(`walking_flat-...-run01_seed42`)은
공통 DOF 0-5 = **오른다리 전체 + 왼쪽 hip flexion**을 예측했습니다. 해당
모델은 2026-07-14 삭제되었고 task 스크립트의 `--action-dofs first6` 옵션도
함께 제거되었습니다. 신규 수집본은 zarr attrs의
`action_dof_names`/`action_dof_indices`로 채널 계약을
자기술(self-describe)합니다.

- `root_pos`/`root_vel` = pelvis(body 0) 월드 위치/선속도
- `contacts` = binary rigid-body contact flag (`get_binary_body_contacts`)
- `Simulator.get_robot_state()`가 모든 필드를 common ordering으로 변환해
  주므로 Newton/IsaacLab/IsaacGym 어디서든 동일 레이아웃이 보장됩니다.

## 사용 예

```python
from protomotions.maniflow import ManiFlowTorqueEstimator, discover_best_checkpoint

ckpt = discover_best_checkpoint(run_dir)  # 또는 특정 .ckpt 경로
estimator = ManiFlowTorqueEstimator.from_checkpoint(
    ckpt, num_envs=env.num_envs, device=fabric.device)

obs, _ = env.reset(); estimator.reset()
for t in range(T):
    ...  # RL policy → env.step(action)
    robot_state = env.simulator.get_robot_state()
    estimator.observe(robot_state)
    if t % estimator.n_action_steps == 0:
        torque_chunk = estimator.predict()  # (N, n_action_steps, 6), raw N·m
```

엔트리 포인트:
- `tasks/mimic_suit_active_cable_walk_23dof/infer_maniflow_newton.{py,sh}` —
  수동(passive) 평가: RL이 보행하고 ManiFlow는 예측만, 적용 토크와 비교
- `tasks/mimic_suit_active_cable_walk_23dof/compare_maniflow_control_newton.{py,sh}` —
  폐루프 제어 A/B: Agent A(순수 RL, 고스트)와 Agent B(estimator 채널을
  ManiFlow 토크로 구동, 나머지는 RL+built-in PD)를 같은 씬에 겹쳐 비교

## 폐루프 제어 (JointTorqueOverride)

```python
from protomotions.maniflow import JointTorqueOverride, hip_dof_indices

hips = hip_dof_indices(env.robot_config.kinematic_info.dof_names)  # [0,1,2,5,6,7]
override = JointTorqueOverride(env.simulator, env_ids=[1],
                               common_dof_indices=hips)
override.engage()                      # env 1의 hip 채널 PD 게인 0 + notify
...
tau = estimator.predict()[1, k]        # (6,) raw N·m
override.set_torques(tau[None])        # effort limit 클램프 후 qfrc 주입
```

동작 원리: MuJoCo Warp는 actuator 게인을 world(env)별로 저장하므로
`joint_target_ke/kd`를 해당 env만 0으로 만들고
`notify_model_changed(JOINT_DOF_PROPERTIES)`를 호출하면 그 env의 해당 DOF만
implicit PD가 꺼집니다. 토크는 `ArticulationView.set_dof_forces`(→
`control.joint_f` → `qfrc_applied`)로 주입하며, 이는 actuator 힘과 독립적으로
매 substep 가산됩니다(decimation 구간 동안 상수 유지 = 표준 토크 제어).
in-place scatter라 CUDA graph와도 안전하게 공존합니다.

## GUI 확인 / 동영상 저장

```bash
# 실시간 GUI: Newton 뷰어 + 뷰어 UI에 hip_flexion 좌/우 pred/gt 라이브 플롯,
# 타이틀바에 러닝 MAE 표시. 뷰어/녹화 모드에서는 skeleton mesh 에셋이 기본
# (play/record_newton.sh와 동일; 캡슐 에셋으로 돌리려면 --no-mesh)
bash tasks/.../infer_maniflow_newton.sh --viewer

# 동영상 저장: 시뮬 녹화 mp4 + 예측/실제 토크 플롯을 옆에 붙인
# sim_with_torque.mp4 생성 (뷰어 자동 활성화; 프레임↔스텝 1:1 동기)
bash tasks/.../infer_maniflow_newton.sh --record --episode-steps 600
```

Newton 프레임 캡처는 `NewtonSimulator._write_viewport_to_file`이
`ViewerGL.get_frame()`으로 구현되어 있어(이 repo에서 추가) `record_newton.sh` 등
기존 녹화 경로도 이 머신에서 동작합니다. 캡처에는 GL 컨텍스트(뷰어 창)가
필요하므로 headless에서는 녹화가 되지 않습니다 — 필요해지면
`ViewerGL(headless=True)` 오프스크린 모드 지원을 추가할 것.

## 런타임 의존성

ManiFlow lowdim 정책 import 체인에 필요한 패키지 (sbc env 기준 설치 완료):
`torch, einops, timm, termcolor, zarr, dill, hydra-core, omegaconf`

pytorch3d는 필요 **없습니다** — lowdim 모달리티가 vision_3d를 import하지 않도록
ManiFlow 쪽 `lowdim_obs_encoder.py`에서 `create_mlp`을 인라인해 두었습니다.

## 주의사항

- **Newton BUILT_IN_PD의 GT 토크**: Newton의 built-in PD는 implicit(solver
  내부)이라 적용 토크가 `control.joint_f`에 기록되지 않습니다. 이를 위해
  `protomotions/simulator/newton/simulator.py`가 Newton의 extended state
  attribute `mujoco:qfrc_actuator`를 요청해 두었고, `_get_simulator_dof_forces`
  가 BUILT_IN_PD일 때 이 readback(솔버가 매 substep 채움)에서 토크를
  읽습니다 — 학습과 동일한 actuation을 유지하면서 GT 확보. PROPORTIONAL
  (explicit PD)로 바꾸는 방법도 있으나 이 로봇(active cable)에서는 시뮬레이션이
  발산하므로 사용하지 마세요. IsaacLab은 built-in PD에서도 applied torque를
  보고하므로 해당 없음 (수집 데이터도 IsaacLab built-in PD 기준).
- **학습 중인 run에서 로드할 때**: `latest.ckpt`는 주기적으로 재작성되므로
  `epoch=*-val_loss=*.ckpt`(topk) 파일을 사용하세요. `discover_best_checkpoint`가
  자동으로 topk를 우선합니다.
- **root_pos의 절대 x,y**: 학습 데이터의 root_pos는 IsaacLab env origin 기준
  월드 좌표입니다. 시뮬레이터/env 배치가 다르면 x,y 절대값 분포가 달라질 수
  있습니다(z와 상대적 변화가 주 정보라 큰 영향은 없을 것으로 예상되나, sim2sim
  갭 분석 시 참고).
