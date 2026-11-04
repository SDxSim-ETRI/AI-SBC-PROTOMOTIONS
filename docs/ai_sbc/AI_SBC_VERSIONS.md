# AI-SBC 모델·데이터 이름과 경로

2026-11-04 기준. 현재 작업은 [HANDOFF.md](HANDOFF.md), 실험 결론은 [실험 결과 요약](experiment_summary.html).
이 문서의 모델은 시뮬레이션 연구용이다. 일반 보행이나 실제 장치의 성능을 보증하지 않는다.

## 이름 읽는 법

| 표기 | 뜻 |
|---|---|
| H1 | 미래 고관절 각도 예측기. HLP(High-Level Predictor), ManiFlow 모델 |
| L15, L16, L17, L18 등 | 보조 토크 제어기의 실험 버전. LLP(Low-Level Policy). 큰 번호가 항상 더 좋은 모델은 아님 |
| W1 | 시뮬레이션에서 사람 역할을 하는 가상 착용자. 보조 제어기와 별도 모델 |
| 폴더 이름 | `<이름>_<구조>_<알고리즘>_<학습>`: `mlp`·`tf`(Temporal Transformer), `ppo`·`fpo`(FPO++), `pendulum`(진자 학습)·`ft`(진자 모델 이어서 학습)·`scratch`(처음부터 학습), `walk`·`run`·`walkrun`(학습 동작). L15_ft는 L15를 동작별로 이어서 학습한 모델의 이름이라 `L15_ft_mlp_ppo_walk`처럼 쓴다 |
| A443 같은 Axxx | BONES-SEED 원본 메타데이터의 `actor_uid`, 촬영한 사람을 구분하는 코드. 실험·모델 번호가 아님 |
| 무보조 A / 보조 B | 같은 조건에서 보조 유무를 비교하는 두 시뮬레이션. 위의 동작 수행자 코드와 무관 |
| epoch_400 | 해당 학습 실행에서 400회 갱신한 저장 모델. 실행마다 수집 데이터 양이 달라 번호만으로 학습량을 비교하지 않음 |

과거 이름 대응: `tracker v33`→W1, `tracker v32`→W0.

## 보관한 모델

모두 저장소의 `checkpoints/`에 Git LFS로 둔다. 각 폴더의 `INFO.md`에 학습 조건·평가·로드 방법이 있다.
LLP는 모두 20·tanh 출력, 한계 ±20 N·m, 출력 배율 1, 10 ms 제어다.
평가 열은 W1 + H1, 기존 걷기·달리기 기록 40 s, 평가 seed 42·7 평균의 착용자 고관절 τ² 변화다.

| 이름 | 경로 | 내용 | 걷기 / 달리기 τ² |
|---|---|---|---|
| H1 | `checkpoints/H1_maniflow_swap/epoch_0008.ckpt` | 각도 예측기. 관측 토큰은 self-attention, 노이즈 섞인 미래 action 토큰은 cross-attention(ManiFlow `obs_routing=swap`). 원본 169 MB에서 EMA 가중치만 남긴 사본(42.5 MB, 원본과 가중치·예측 같음) | — |
| W1 | `checkpoints/W1_wearer_human/last.ckpt` | 가상 착용자 하나가 걷기·달리기를 모두 따라간다 | — |
| **L18** (기본) | `checkpoints/L18_tf_fpo_scratch_walkrun/epoch_1200.ckpt` | Temporal Transformer + FPO++, 걷기·달리기 반반, 진자 가중치 없이 전신 시뮬레이션에서 처음부터 학습. 보조 토크 실효값 10–12 N·m | −30.1 / −29.3 % |
| L17 | `checkpoints/L17_mlp_ppo_scratch_walkrun/epoch_200.ckpt` | MLP+PPO, L18과 같은 조건으로 처음부터 학습 | −24.0 / −35.4 % |
| L16 | `checkpoints/L16_mlp_ppo_ft_walkrun/epoch_{800,1200}.ckpt` | L15에서 걷기·달리기 반반으로 이어서 학습 | 800: −22.7 / −27.1 %, 1,200: −19.9 / −27.4 % |
| L15_ft 걷기 | `checkpoints/L15_ft_mlp_ppo_walk/epoch_1200.ckpt` | L15에서 걷기만으로 이어서 학습. L15 계보 걷기 best | −26.6 / (−13.3) % |
| L15_ft 달리기 | `checkpoints/L15_ft_mlp_ppo_run/epoch_400.ckpt` | L15에서 달리기만으로 이어서 학습 | — / −25.5 % |
| L15 | `checkpoints/L15_mlp_ppo_pendulum/epoch_6000.ckpt` | 진자에서만 학습(`mlp.py --recipe L15`). 위 L15_ft·L16의 출발 모델 | −18.0 / −21.8 % |

괄호는 학습하지 않은 동작에 쓴 값이다. 옛 착용자 `checkpoints/v30_*`, `v31_*`, `v32_*`(W0)와 설정만 남은 `v18_*`,
그리고 착용자 학습 run `results/mimic_suit_*`는 그대로 보존한다.

## 2026-11-04 가중치 정리

- 위 표 밖의 LLP 가중치를 지웠다: `results/` 아래 903개(4.4 GB)와 git의 `checkpoints/L1_llp_pendulum`, `checkpoints/llp_v{3,4,5,6,10}_*`.
  L1·L2–L14·G0–G4·F1/F2·R0/R1·20 N·m 다리별 4종·진자 FPO++ 모델과, 남긴 run의 다른 epoch·`last.ckpt`가 포함된다.
  각 run 폴더의 학습 설정(`resolved_configs*.pt`, `config.yaml`)·로그·평가 기록은 남아 있다.
- ManiFlow 체크포인트는 H1 원본 하나만 남기고 21개(8.8 GB)를 지웠다: 옛 H1(관측 cross-attention, flexion100-h15-run01), 40 fps 판(flexion40),
  초기 hips·IMU 모델, 2026-05 이미지 정책. 학습 자료(zarr)는 그대로다.
- L1은 더 이상 기준 모델이 아니다. 코드 기본값과 진자 레시피에서 빼고, L1 계보의 보고서는 [보관 자료](reports.html#past)로 옮겼다.
- 남긴 run의 원래 위치(`results/llp_R0lim20_s0_20261029`, `llp_L15ft_{walk,run}_20261029`, `llp_L16_20261029`,
  `llp_L16scratch_20261102`, `llp_fpo_tfm_20261102`)에는 이제 checkpoint 없이 설정·로그만 있다.

## 현재 계보의 연구 원본

| 실험 | 원본 | 보고서 |
|---|---|---|
| L15 진자 학습, 동작별 추가학습, H1 토큰 자리 맞바꿈 채택 (2026-10-29) | `results/llp_lim20_finetune_20261029/`, `results/hlp_routing_swap_20261029/`, `results/h1_swap_adoption_20261029/` | [결과](261029_lim20_finetune/report.html) |
| L16 대 동작별 추가학습, 진자 사전학습 없는 처음부터 학습(L17) (2026-10-29 ~ 11-02) | `results/llp_L16_study_20261029/` (`summary.json`, `train_curve.json`, `cross_summary.json`, `scratch_summary.json`, 학습·평가 스크립트, 영상 녹화 `video/`) | [결과](261029_L16_joint_finetune/report.html) |
| Transformer+FPO++(L18) 대 MLP+PPO, 전신 학습 (2026-11-02 ~ 03) | `results/llp_fpo_study_20261102/` (`summary.json`, `latency_bench.py`, 학습·평가 스크립트) | [결과](261102_fpo_transformer/report.html) |
| 같은 세 정책의 진자 학습 (2026-11-03 ~ 04) | `results/llp_fpo_pendulum_20261103/` (`summary.json`, `latency.json`, `best_setting.json`) | [결과](261103_fpo_pendulum/report.html) |

## 현재 데이터

| 자료 | 위치 | 용도 |
|---|---|---|
| 기존 suit 걷기·달리기 | `data/motion_for_trackers/skeleton_torque_suit_gait.pt` | 학습·평가 기본 자료. motion 0=걷기, 1=달리기 |
| H1 학습 자료 | `~/Projects/ManiFlow_Policy/ManiFlow/data/hip-flexion-seed-locomotion-100fps.zarr` | SEED 보행 계열 100 fps |
| 기존 걷기 재생 속도 변형 | `results/llp_walk_curriculum_20261011/walk_tempo_variants.pt` | 같은 동작을 0.85–1.15배로 재생. 실제 속도별 촬영 자료가 아니라 일반화 평가에서 제외 |
| 실제 보통·느린 보행 변환 후보 | `results/real_walk_retarget_20261012_224957/native_pairs_v1/suit_native_walk.pt` | 4명 표기의 각 보통/느림, 총 8클립. 무보조 추종 미확보 |
| 빠른 보행 변환 후보 | `results/real_walk_retarget_20261012_224957/fast_windows_v1/suit_native_walk.pt` | 다른 3명 표기의 4초 구간. 걷기/달리기 분류 미확정 |

새 후보의 원본은 H1 원본 자료 목록과 겹친다. 시스템 전체가 보지 않은 평가 자료라고 부르지 않는다.
동작 수행자가 달라도 시뮬레이션에서는 같은 suit 체형과 같은 W1을 사용했다. 여러 체형의 착용자를 검증한 것이 아니다.

## 지난 계보

2026-09 ~ 10-22의 L1 계보(L1–L14, G0–G4, F1/F2, R0/R1, 20 N·m 다리별 4종)는 가중치를 지웠고 기록만 남았다.
결과는 [요약의 실험 이력](experiment_summary.html#history), 보고서는 [보관 자료](reports.html#past),
정리 직전의 경로표는 [2026-11-04 보관 버전표](archive/20261104/AI_SBC_VERSIONS.md),
그보다 오래된 표는 [2026-10-13 보관 버전표](archive/20261013/AI_SBC_VERSIONS.md)에 있다.
종료된 실험을 자동 재개하지 않는다.
