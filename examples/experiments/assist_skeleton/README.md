# AI-SBC 전신 보조 평가와 실행

현재 작업은 [HANDOFF](../../../docs/HANDOFF.md), 결과는 [보고서 목록](../../../docs/reports.html),
모델 경로는 [버전표](../../../docs/AI_SBC_VERSIONS.md)를 기준으로 한다.
이 문서는 실행법과 파일 역할만 다룬다. 초기 실험 일지는 [보관본](../../../docs/archive/20261013/assist_skeleton_README.md)에 있다.

## 기본 비교 기준

`run_system.py`는 같은 초기 상태의 무보조 A와 보조 B를 Newton에서 비교한다.
기본값은 아래와 같다. 모든 명령은 저장소 루트의 `sbc` 환경에서 실행한다.

| 항목 | 기본값 |
|---|---|
| 가상 착용자 | W1, `checkpoints/W1_wearer_human/last.ckpt` |
| 각도 예측기 | H1 flexion100-h15, 버전표에 고정한 epoch 6 파일 |
| 보조 제어기 | L1, `checkpoints/L1_llp_pendulum/last.ckpt`, 출력 0.5배 |
| 동작 | `data/motion_for_trackers/skeleton_torque_suit_gait.pt`, 걷기 0번 (달리기 1번) |
| 시간 | 물리 400 Hz, 착용자 20 Hz, 보조 100 Hz, 평가 40초 |
| 예측 | 입력 100 Hz, 현재부터 50 ms까지 6점, LLP가 쓰는 미래 창은 0/10/20 ms |
| 영점 | 첫 1초의 무보조 착용자 관측 평균 (`--calibration wearer`) |

L1 출력 0.5배는 학습 토크 한계 ±40 N·m에 곱해 약 ±20 N·m를 주입하는 조건이다.
`--torque-limit`은 배율 적용 전 제어기 한계다. 두 값을 혼동하지 않는다.
**다른 경로의 LLP 정책과 오라클 진단은 배율을 생략하면 1을 쓴다.**
L1을 다른 위치로 복사해 평가할 때는 `--assist-scale 0.5`를 명시한다.
명시한 `--assist-scale`은 자동 선택보다 우선한다.

## 실행

```bash
# 기본 W1 + H1 + L1 × 0.5, 기존 걷기 40초. 기본은 headless.
python examples/experiments/assist_skeleton/run_system.py

# 같은 기준의 달리기
python examples/experiments/assist_skeleton/run_system.py --motion-id 1

# 화면 / 영상 (그래픽 디스플레이 필요)
python examples/experiments/assist_skeleton/run_system.py --viewer --record

# 새 보행의 착용자 추종만 검사: 짧은 클립을 이어 붙이지 않는 전용 평가기
python examples/experiments/assist_skeleton/evaluate_native_wearer.py \
    --motion-file results/real_walk_retarget_20261012_224957/native_pairs_v1/suit_native_walk.pt \
    --starts 0 0.5 --max-seconds 8 --out results/native_walk_check
```

전용 평가기의 출력 폴더는 실행마다 새 경로를 사용한다. 기존 결과를 덮어쓰지 않는다.
새 실제 보행은 W1의 무보조 추종부터 확인한다. 이 단계를 통과하기 전에는 보조 성능을 판정하지 않는다.

연구 후보를 비교할 때는 체크포인트·배율·한계를 명시한다. 예:

```bash
python examples/experiments/assist_skeleton/run_system.py \
    --llp-checkpoint results/llp_L13R_walk_continue/epoch_400.ckpt \
    --assist-scale 1 --torque-limit 100 --motion-id 0 --seconds 40
```

이 명령은 기존 걷기의 연구 후보 평가다. 범용 보행이나 실기기 성능을 입증하지 않는다.
골반 고정 진단은 `--human mocap_pd`로 선택한다. 지면을 딛고 걷는 W1 평가와 별도 조건이다.
과거 결과를 재현하려면 당시 `meta.json`의 모델·모션·영점·배율·시간 옵션을 모두 명시한다.
이전 기본값(v3, 초기 착용자, 단일 walk 파일, motion 영점, 30초)에 의존한 명령은 그대로 재현되지 않는다.

## 출력과 해석

`results/assist_skeleton/<시각>_<착용자>[_태그]/`에 `meta.json`, `traces.npz`,
`metrics.json`, `report.md`, `plots.png`를 저장한다. `--record`는 영상을 추가한다.
`meta.json`에 실제 사용한 모델·시간·배율·토크 한계·실행 명령이 남는다.

토크 제곱 평균, 몸체 추종 오차, 전진 속도, 낙상, 다른 관절의 부담을 함께 본다.
토크 지표는 시뮬레이션 부담이며 실제 근활성·대사량이 아니다.
HLP 계산 시간은 별도 기록한다. `--hlp-latency-ms 0`은 계산 지연을 물리 시간에 추가하지 않는 설정이다.
`--oracle`과 `--mask-window`는 참조·착용자 토크를 사용하는 원인 진단용이다.

## 파일 역할

| 파일 | 역할 |
|---|---|
| `run_system.py`, `analysis.py`, `viz.py` | 통합 비교 실행, 지표·그림, 시각화 |
| `evaluate_native_wearer.py` | 새 보행의 무보조 착용자 평가 |
| `diagnose_walking.py`, `fullbody_diagnostic.py`, `phase_screen.py` | 구간별 방해와 전신 물리 진단 |
| `report_*.py` | 저장한 결과의 보고서 생성. 생성 뒤 `python docs/organize_reports.py` 실행 |
| `llp_walk_finetune*.py`, `llp_walk_tempo.py` | 전신 추가학습 실험 구성 |
| `run_walk_*_sequence.py`, `validate_walk_finetune.py` | 해당 실험의 실행·평가 절차 재현 |
| `tracker_human.py`, `train_tracker_human.sh` | W1 착용자 학습 구성·실행 |
| `llp_walk.py`, 초기 tracker 스크립트 | 과거 실험 재현 및 일부 공통 구성 제공 |

공통 제어기는 `protomotions/assist/`, ManiFlow 연결은 `protomotions/maniflow/`에 있다.
구체적인 코드 흐름과 주의점은 [프레임워크 문서](../../../docs/AI_SBC_FRAMEWORK.md)를 참조한다.
학습 스크립트가 남아 있다는 이유로 종료된 실험을 다시 시작하지 않는다.
