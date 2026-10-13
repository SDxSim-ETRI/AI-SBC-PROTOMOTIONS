# protomotions.maniflow — 각도 예측기 연결

현재 AI-SBC에서 ManiFlow는 **최근 관절 움직임으로 미래 고관절 각도를 예측하는 HLP**다.
보조 토크는 별도로 학습한 LLP가 만든다. 시뮬레이션 코드는 이 패키지를 통해 접근하며
외부 `maniflow.*`를 직접 import하지 않는다.

현재 모델 경로는 [버전표](../../docs/AI_SBC_VERSIONS.md), 시스템 설계는
[프레임워크](../../docs/AI_SBC_FRAMEWORK.md), 실행은
[통합 평가 README](../../examples/experiments/assist_skeleton/README.md)를 참조한다.

## 입력·출력과 시간

| 항목 | 현재 H1 flexion100-h15 |
|---|---|
| 입력 | 오른쪽·왼쪽 고관절 굴곡, 몸통 pitch·roll의 4채널, rad |
| 출력 | 오른쪽·왼쪽 고관절 굴곡의 2채널, rad |
| 입력 간격 | 10 ms (100 Hz) |
| 예측 범위 | 현재부터 50 ms까지, 10 ms 간격의 6점 |
| LLP가 소비하는 창 | 현재·10 ms·20 ms |
| 통합 평가의 denoise | 1회 |

`predict_horizon()`의 전체 배열은 관측 창과 정렬돼 있다.
현재 프레임은 `n_obs_steps - 1`이며, 미래를 포함한 부분은
`prediction[:, n_obs_steps - 1:]`이다. 점 수는 `horizon - n_obs_steps + 1`로 정해진다.
H1은 6점이고, 이전 H0(40 Hz)는 7점·150 ms였다. 두 모델의 시간 설정을 혼용하지 않는다.

`ManiFlowAngleEstimator.from_checkpoint()`는 학습 zarr의 `frame_dt`를 읽는다.
학습 데이터 경로를 읽을 수 없으면 기존 25 ms 기본값으로 돌아가므로,
H1을 별도 환경에서 로드할 때는 `frame_dt=0.01`을 명시하고 모델 형식과 대조한다.
`ManiFlowChunkSource`는 이 간격으로 관측을 재표집하고 예측점을 만든다.

## 구성 파일

| 파일 | 역할 |
|---|---|
| `loader.py` | 외부 패키지 위치 확인, 체크포인트에서 추론 정책 로드 |
| `angle_estimator.py` | 4채널 관측 히스토리, 영점 보정, 각도 예측 |
| `chunk_source.py` | 제어 주기 관측을 모델 시간 격자에 맞추고 LLP용 예측 생성 |
| `hybrid_control.py` | Newton의 PD 토크와 독립적인 추가 토크 주입 |
| `channels.py` | 관절 이름에서 고관절 DOF 인덱스 도출 |
| `verify_angle_estimator.py`, `verify_chunk_source.py` | 스트리밍 예측과 오프라인 기준 비교 |
| `torque_estimator.py` | 이전 토크 직접 예측 연구용 래퍼. 과거 task 스크립트가 사용 |

검증 스크립트의 기본 자료·기준값은 **이전 H0/40 Hz 실험**이다.
인자 없이 실행한 결과를 H1 검증으로 보고하지 않는다. H1 검증에는 체크포인트와
100 Hz 평가 자료·시간 설정·기준 지표를 함께 지정해야 한다.

## 각도 예측기 사용

```python
from protomotions.maniflow import ManiFlowAngleEstimator

estimator = ManiFlowAngleEstimator.from_checkpoint(
    h1_checkpoint, num_envs=num_envs, device=device,
    denoise_steps=1, frame_dt=0.01,
)
# 10 ms마다 관측: (num_envs, 4), rad, 위 표의 채널 순서
estimator.observe(observation_rad)
prediction = estimator.predict_horizon()
current_and_future = prediction[:, estimator.n_obs_steps - 1:]
```

영점 보정은 `obs_offset`을 관측에 더하고 `output_bias`를 예측에서 뺀다.
통합 러너는 첫 warmup 구간의 무보조 착용자 스트림으로 보정하고 히스토리를 다시 초기화한다.
`set_history()`는 외부에서 재표집한 관측 창 전체를 넣을 때 사용한다.

## 외부 패키지와 토크 주입

패키지 탐색 순서: 이미 로드한 `maniflow` → 명시적 `maniflow_root` 또는
`MANIFLOW_ROOT` → 설치된 패키지 → `~/Projects/ManiFlow_Policy/ManiFlow`.
런타임은 `sbc`, 모델 학습은 별도 `maniflow` 환경을 사용한다.

현재 보조는 `JointTorqueOverride.engage(gain_scale=1.0)`으로 착용자의 PD 게인을 유지하고
`qfrc_applied`에 토크를 더한다. `engage()`의 기본 인자를 그대로 쓰는 과거 직접 구동 예제와 다르다.
토크 한계는 호출부와 `set_torques()`의 인자를 확인한다. 착용자 토크는
`qfrc_actuator`의 서브스텝 평균에서 읽으며 추가 보조 토크와 구분한다.

## 과거 토크 직접 예측 연구

88채널 상태 → 고관절 6채널 토크를 예측하던 모델은 현재 HLP와 별개다.
채널 순서·이전 제어 예제는 [이전 README](../../docs/archive/20261013/maniflow_README.md)에 보존했다.
관련 `tasks/mimic_suit_active_cable_walk_23dof/` 스크립트가 래퍼를 사용하므로
`torque_estimator.py`와 공개 import를 임의로 삭제하지 않는다.
