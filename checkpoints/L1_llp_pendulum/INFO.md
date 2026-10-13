# L1 — 진자에서 학습한 기본 보조 모델

현재 역할은 W1 착용자 위에서 비교하는 **시뮬레이션 기준 모델**이다.
옛 이름은 `llp_v13`이며 전신 추가학습 실험 `L13`과 다르다.
모델 목록은 [버전표](../../docs/AI_SBC_VERSIONS.md), 현재 결론은
[실험 요약](../../docs/experiment_summary.html)을 참조한다.

## 보관 정보

| 항목 | 내용 |
|---|---|
| 가중치 | `last.ckpt`, 학습 6,000회 시점 |
| 원본 실행 | `results/assist_pendulum_v4/epoch_6000.ckpt` |
| 보관일 | 2026-09-28 |
| 학습 구성 | `experiment_config.py`, `resolved_configs*.pt/yaml`, `config.yaml` |
| 학습 환경 | 고정 골반 진자. 관성·착용자 게인·외부 하중·목표 궤적을 무작위화 |
| 관측 | 고관절 각·각속도, 250 ms 히스토리, 0/10/20 ms 예측 창, 100 ms 이전 action |
| 예측 형식 | 학습 중 HLP 오차 모사. H1과 같은 10 ms 간격·현재부터 50 ms까지 6점 |
| 학습 토크 | `tanh(action) × 40 N·m` |
| 제어 주기 | 10 ms (물리 400 Hz, decimation 4) |

## 현재 평가 조건과 한계

W1 + H1에서 **출력 0.5배**를 비교 기준으로 쓴다. 주입 한계는 약 ±20 N·m다.
2026-10-01의 기존 걷기·달리기 40초 비교에서 고관절 토크 제곱 평균은
각각 19.3%, 23.7% 감소했다. 조건과 원본 수치는 [기준 보고서](../../docs/l1_w1_report.html)에 있다.
이 값은 시뮬레이션 부담 지표이며 실제 근활성·대사량 측정값이 아니다.

초기 W1 평가에서 출력 1배는 달리기 4.4초 낙상을 보였다.
출력 0.5배의 기존 동작 결과도 새로운 실제 보행이나 실기기 성능을 보증하지 않는다.
이전 착용자 W0의 결과와 당시 학습 분포·선정 근거는 [보관 기록](../../docs/archive/20261013/L1_INFO.md)에 있다.

## 실행

저장소 루트의 `sbc` 환경에서 실행한다. H1은 러너에 고정된 기본 파일을 사용한다.

```bash
python examples/experiments/assist_skeleton/run_system.py \
    --human tracker --tracker-checkpoint checkpoints/W1_wearer_human/last.ckpt \
    --llp-checkpoint checkpoints/L1_llp_pendulum/last.ckpt \
    --llp-dt 0.01 --assist-scale 0.5 --calibration wearer \
    --motion-file data/motion_for_trackers/skeleton_torque_suit_gait.pt \
    --motion-id 0 --seconds 40
```

달리기는 `--motion-id 1`로 바꾼다. 상세 옵션은 [실행 README](../../examples/experiments/assist_skeleton/README.md)에 있다.
재학습 구성은 보관한 `experiment_config.py`와 현재 소스를 대조한다. 이 문서는 학습 재개 지시가 아니다.
