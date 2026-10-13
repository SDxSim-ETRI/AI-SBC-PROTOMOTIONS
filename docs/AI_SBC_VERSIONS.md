# AI-SBC 모델·데이터 이름과 경로

2026-10-13 기준. 현재 작업은 [HANDOFF.md](HANDOFF.md), 실험 결론은 [실험 결과 요약](experiment_summary.html).
이 문서의 모델은 시뮬레이션 연구용이다. 일반 보행이나 실제 장치의 성능을 보증하지 않는다.

## 이름 읽는 법

| 표기 | 뜻 |
|---|---|
| H1 | 미래 고관절 각도 예측기. HLP(High-Level Predictor), ManiFlow 모델 |
| L1, L9, L13R, L14 등 | 보조 토크 제어기의 실험 버전. LLP(Low-Level Policy). 큰 번호가 항상 더 좋은 모델은 아님 |
| W1 | 시뮬레이션에서 사람 역할을 하는 가상 착용자. 보조 제어기와 별도 모델 |
| A443 같은 Axxx | BONES-SEED 원본 메타데이터의 `actor_uid`, 촬영한 사람을 구분하는 코드. 실험·모델 번호가 아님 |
| 무보조 A / 보조 B | 같은 조건에서 보조 유무를 비교하는 두 시뮬레이션. 위의 동작 수행자 코드와 무관 |
| epoch_400 | 해당 학습 실행에서 400회 갱신한 저장 모델. 실행마다 수집 데이터 양이 달라 번호만으로 학습량을 비교하지 않음 |

과거 이름 대응: `llp_v13`→L1, `tracker v33`→W1, `tracker v32`→W0.
`L13`은 전신 추가학습 실험이며 과거 `llp_v13`과 다른 모델이다.

## 지금 보존하고 비교할 모델

| 설명 이름 | 원래 이름 | 경로 | 역할과 한계 |
|---|---|---|---|
| 각도 예측기 | H1 | 아래 H1 경로 | 최근 실험에서 동결. 100 Hz 입력, 현재부터 50 ms 예측 |
| 가상 착용자 | W1 | `checkpoints/W1_wearer_human/last.ckpt` | 기존 걷기·달리기 평가 기준. 새 실제 보행은 추종 실패 |
| 진자 학습 기준 모델 | L1, 출력 0.5배 | `checkpoints/L1_llp_pendulum/last.ckpt` | 기존 걷기·달리기 비교 기준. 적용 한계 약 ±20 N·m |
| 전신 추가학습의 시작 모델 | L9, 2,000회 | `results/llp_L9_opposition/epoch_2000.ckpt` | 비교·학습 계보용. 구간별 방해가 남음 |
| 부담 비율 반영 모델 | L13R, 400회 | `results/llp_L13R_walk_continue/epoch_400.ckpt` | 기본 걷기 연구 후보. 느린 재생 조건 실패 |
| 세 재생 속도 혼합 모델 | L14, 200회 | `results/llp_L14_walk_tempo/epoch_200.ckpt` | 기본 걷기에서 개선. 느린 재생 조건과 허리·무릎 부담 미해결 |

L9·L13R·L14 평가는 배율 1, 최종 토크 한계 ±100 N·m를 사용했다.
L1과는 출력 한계와 학습 조건도 다르므로 성능 차이를 보상식 하나의 효과로 해석하지 않는다.

H1의 검증된 저장 경로:
`/home/bak/Projects/ManiFlow_Policy/ManiFlow/data/outputs/hip_flexion_seed-maniflow_lowdim_policy_walking-flexion100-h15-run01_seed42/checkpoints/epoch=0006-val_loss=0.000762.ckpt`

## 현재 데이터

| 자료 | 위치 | 용도 |
|---|---|---|
| 기존 suit 걷기·달리기 | `data/motion_for_trackers/skeleton_torque_suit_gait.pt` | 기존 결과 재현. motion 0=걷기, 1=달리기 |
| 기존 걷기 재생 속도 변형 | `results/llp_walk_curriculum_20261011/walk_tempo_variants.pt` | 같은 동작을 0.85–1.15배로 재생. 실제 속도별 촬영 자료가 아님 |
| 실제 보통·느린 보행 변환 후보 | `results/real_walk_retarget_20261012_224957/native_pairs_v1/suit_native_walk.pt` | 4명 표기의 각 보통/느림, 총 8클립. 무보조 추종 미확보 |
| 빠른 보행 변환 후보 | `results/real_walk_retarget_20261012_224957/fast_windows_v1/suit_native_walk.pt` | 다른 3명 표기의 4초 구간. 걷기/달리기 분류 미확정 |

새 후보의 원본은 H1 원본 자료 목록과 겹친다. 시스템 전체가 보지 않은 평가 자료라고 부르지 않는다.
동작 수행자가 달라도 시뮬레이션에서는 같은 suit 체형과 같은 W1을 사용했다. 여러 체형의 착용자를 검증한 것이 아니다.

## 과거 버전

L2–L11의 중간 결과와 더 오래된 버전은 [요약의 실험 이력](experiment_summary.html#history)과
[보관한 버전표](archive/20261013/AI_SBC_VERSIONS.md)에 있다. L12는 코드 준비 후 학습하지 않았다.
종료된 실험을 자동 재개하지 않는다. 가중치·원본 수치·실패 로그는 기존 위치에 보존한다.
