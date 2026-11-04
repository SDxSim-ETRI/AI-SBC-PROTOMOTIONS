# AI-SBC — 현재 상태와 다음 작업

2026-11-04 가중치를 정리하고 기본 LLP를 L18(Transformer+FPO++)로 정했다. L15 이전 계보는 더 쓰지 않는다. 같은 날 Transformer+FPO++ 등 세 정책을 진자 학습에 적용한 비교도 끝냈다(진자에서는 L15를 넘지 못함). 이 파일에는 현재 판단만 쓴다. 지난 세션의 진행 상황을 덧붙이지 않는다. 정리 전 판본은 [archive/20261104/HANDOFF.md](archive/20261104/HANDOFF.md)에 있다.

## 2026-11-04 정리: 남긴 모델, 기본 조합, 인수인계

**기본 조합은 W1 + H1 + L18이다.** `run_system.py`를 인자 없이 실행하면 `checkpoints/`의 H1·L18을 출력 배율 1(±20 N·m)로 쓴다. 그렇게 돌린 걷기·달리기(seed 42)의 보조 토크 실효값은 10.14 / 11.62 N·m로 학습 연구의 기록(10.13 / 11.62)과 같았다. 무보조 착용자 토크 실효값 차이(걷기 38.58 vs 38.83 N·m)는 Newton 재실행 산포다.

| 모델 | 경로 (`checkpoints/`, Git LFS) | W1 + H1 걷기 / 달리기 τ² |
|---|---|---|
| H1 | `H1_maniflow_swap/epoch_0008.ckpt` — 원본에서 EMA 가중치만 남긴 사본(42.5 MB). 가중치 164개와 같은 입력의 예측이 원본과 같음을 확인 | — |
| W1 | `W1_wearer_human/last.ckpt` — 착용자 하나가 걷기·달리기를 모두 따라감 | — |
| **L18** (기본) | `L18_tf_fpo_scratch_walkrun/epoch_1200.ckpt` — Transformer+FPO++, 처음부터 | −30.1 / −29.3 % |
| L17 | `L17_mlp_ppo_scratch_walkrun/epoch_200.ckpt` — MLP+PPO, 처음부터 | −24.0 / −35.4 % |
| L16 | `L16_mlp_ppo_ft_walkrun/epoch_{800,1200}.ckpt` — L15에서 걷기·달리기 동시 추가학습 | −22.7 / −27.1 %, −19.9 / −27.4 % |
| L15_ft | `L15_ft_mlp_ppo_walk/epoch_1200.ckpt`, `L15_ft_mlp_ppo_run/epoch_400.ckpt` — 동작별 추가학습 | −26.6 / —, — / −25.5 % |
| L15 | `L15_mlp_ppo_pendulum/epoch_6000.ckpt` — 진자 학습 20 N·m | −18.0 / −21.8 % |

- 지운 것: `results/`의 나머지 LLP checkpoint 903개(4.4 GB; L15 이전 계보 전부, 진자 FPO++, 남긴 run의 다른 epoch·`last.ckpt`)와 git의 옛 LLP 폴더 6개, H1 외 ManiFlow checkpoint 21개(8.8 GB; 옛 H1 포함). 각 run의 설정·로그·평가 기록과 학습 자료(zarr)는 남겼다. 착용자 모델은 모두 남겼다.
- 기본값: `run_system.py` 기본 LLP·H1과 출력 배율(항상 1), `verify_controller` 기본 모델(L15), 진자 레시피 `mlp.py --recipe L15`(유일한 프리셋), H1 기본 경로는 `protomotions.maniflow.loader.DEFAULT_H1_CHECKPOINT` 하나. `test_pendulum_recipes.py`는 L15 설정을 고정한다. 제어기 일치 검증(L15 + H1 사본)은 세 부분 모두 정확히 일치했고, ManiFlow 검증 2개도 인자 없이 H1을 검증한다(PASS).
- 옛 계보에만 쓰던 코드를 뺐다: 한쪽 다리 모델을 양다리에 따로 돌리는 공유 LLP(`share_legs`·`SingleJointMLP`·`motion_leg` 레시피·진자 연구 실행기와 보고서), 제어기의 유각 브레이크 게이트, L1 출력 절반 기준 비교 보고서(`report.py torque|swing-gate`, `PhaseScreen` 등), 7월 고관절 토크 예측기(`torque_estimator.py`, `tasks/`의 추론·제어 비교·DAgger 스크립트). 원본은 각 연구의 `results/<연구>/scripts/retired_20261104/`(스냅샷 커밋 `e74e83e`)에 있다. 지금 LLP는 모두 actor 하나가 양다리 관측을 함께 받아 오른쪽·왼쪽 토크를 한 번에 낸다.
- L15 이전 계보의 보고서 17개(2026-10-01 ~ 10-22)는 `docs/ai_sbc/archive/reports/`로 옮겼고 옛 주소에는 안내 페이지가 남는다. 요약·버전표·이 파일의 정리 전 판본은 `docs/ai_sbc/archive/20261104/`에 있다.
- 인수인계: H1·L15–L18을 `checkpoints/`에 Git LFS로 올렸다(2026-11-04 push, LFS 파일 20개). 받는 쪽에는 ManiFlow 코드(`d9e536e` 이후)가 필요하다. 각 폴더의 `INFO.md`에 입출력·학습 조건·로드 방법이 있다.

## 진자 학습에 적용한 Temporal Transformer + FPO++ 대 MLP+PPO(L15) (2026-11-03 ~ 04, 새 H1)

**전신 학습에서 비교한 세 정책(MLP+PPO, MLP+FPO++, Transformer+FPO++)을 L15와 같은 진자 학습 조건(L15 레시피, ±20 N·m, 4,096 환경 × 32 step, seed 0, 1,200 epoch 예산)에 적용해 W1 걷기·달리기로 전이 평가했다. 전신 학습과 같은 기본 설정(flow 출력 배율 0.25, actor lr 1e-4)의 FPO++ 두 모델은 진자에서 거의 학습되지 않았다(1,200 epoch 동안 W1 보조 RMS 1~2 N·m, τ² −0.6 ~ −1.4 %; 진자 환경 자체 평가도 같음). 학습률 3e-4·1e-3은 효과가 없고 출력 배율만 효과가 있어(200 epoch 진단, 배율 1·2·4·8) 배율 4.0으로 두 모델을 1,200 epoch 다시 학습했다. 평가 동작별 best checkpoint(1,200 epoch 이하)의 착용자 고관절 τ² 변화: 걷기 L15 −24.1 %(1,000 epoch) · Transformer+FPO++ 배율 4 −14.6 %(400) · MLP+FPO++ 배율 4 −11.0 %(1,000); 달리기 L15 −22.1 %(1,000) · Transformer+FPO++ 배율 4 −16.4 %(600) · MLP+FPO++ 배율 4 −11.3 %(1,200). 진자 학습에서는 FPO++가 L15에 못 미친다(전신 학습에서는 걷기에서 PPO보다 좋았음). 낙상 0.** [보고서: 설정 진단·그래프·영상·지연](261103_fpo_pendulum/report.html). 학습 seed 하나, 평가 seed 둘(설정 진단은 하나).

- 학습·평가: `examples/experiments/assist_pendulum/mlp.py --recipe L15 --policy mlp_fpo|tfm_fpo [--fpo-actor-scale 4.0]`(FPO 설정은 전신 `llp_walk.py`와 공용 `protomotions/agents/fpo/recipe.py`), checkpoint 200 epoch마다, W1 + H1 40 s seed 42·7 배율 1. 기준선 L15는 2026-10-29 run의 500~6,000 epoch checkpoint를 같은 조건으로 평가했다(비교는 1,200 이하의 500·1,000 중 best; 전체로는 달리기 1,500 epoch −24.2 %가 best). run `results/llp_fpo_pend_{mlp,tfm}_20261103/`(기본 설정), `results/llp_fpo_pend_{mlp,tfm}_scale4_20261103/`(배율 4; 이 연구의 checkpoint는 2026-11-04에 모두 지움), 진단 `results/llp_fpo_pend_mlp_diag_*_20261103/`·`results/llp_fpo_pend_tfm_diag_scale4_20261103/`, 평가·스크립트 `results/llp_fpo_pendulum_20261103/`(`summary.json`, `latency.json`, `best_setting.json`).
- 기본 설정이 학습되지 않은 원인: 학습 rollout의 보조 토크(RMS ≈ 5 N·m)는 거의 탐색 잡음(초기 잡음 × 배율 0.25 → 20·tanh)이고 결정적 출력(초기 잡음 0에서 적분)은 0 근처에 머문다. TensorBoard에서 진자 FPO++ run은 actor gradient norm 0.15~0.8(clip 1.0에 거의 안 걸림)·비율 clip(0.05)에 걸린 비율 1 % 안팎, 전신 run은 gradient가 항상 clip되고 clip 비율 4~18 % → 신뢰영역이 아니라 작은 기울기가 병목. 진자 최적은 20 N·m 한계 근처(tanh 입력 ≥ 2~3)인데 배율 0.25로는 flow가 x₀를 8~12만큼 옮겨야 하고, 전신 학습의 최적(10 N·m 수준)은 가까웠다. 기본 설정 Transformer run은 1,000 epoch 뒤 gradient norm이 10²~10⁷로 튀었으나(clip이 막음) 결과는 그대로였다.
- 설정 진단(MLP+FPO++ 200 epoch, 진자 paired 평가 legacy 128 env·W1 seed 42): lr 3e-4·1e-3 → W1 보조 0.3~0.8 N·m(효과 없음); 출력 배율 1.0 걷기/달리기 −4.0/−2.5 %, 2.0 −7.4/−2.6, 4.0 −7.8/−6.8(진자 −15.8 %, L15 500 epoch −14.7 수준), 8.0 진자 −18.4 %지만 W1 +2.1/+0.6(전이 깨짐; 거의 ±20 bang-bang). 배율 s는 20·tanh(s·x₀)의 s라 초기 토크 분포가 tanh 선형 구간(|토크| < 19)에 남는 비율이 97/73/42/22 %(s = 1/2/4/8)이고, 결정적 출력이 ±19에 닿는 데 필요한 x₀ 이동은 2.2/s다. Transformer+FPO++ 배율 4의 200 epoch 확인 run은 걷기 −13.8 %·달리기 −9.2 %(보조 8.3/13.2 N·m)로 같은 설정이 Transformer에도 듣는다.
- 배율 4 본 학습의 추세(걷기 / 달리기, τ² %): Transformer+FPO++ 200 → 400 → 600 → 800 → 1,000 → 1,200 epoch −9.8 → −14.6 → −13.0 → −11.7 → −7.3 → −10.6 / −8.2 → −14.8 → −16.4 → −15.6 → −11.1 → −9.8; MLP+FPO++ −6.8 → −6.1 → −4.7 → −10.6 → −11.0 → −10.8 / −0.0 → −3.2 → −3.1 → −8.3 → −9.6 → −11.3. 두 모델 모두 보조 RMS 9 → 18~20 N·m, 최대출력 95% 이상 사용 시간비율 0 → 65~92 %로 출력이 한계 쪽으로 커지며(L15도 500 → 6,000 epoch에 0 → 77 %) Transformer는 정점 뒤 전이가 나빠진다. best의 최대출력 95% 이상 사용 시간비율: Transformer 배율 4 걷기 8 %·달리기 68 %, MLP 배율 4 77 / 92 %, L15 32 / 45 %.
- 구간별(best, 착용자 토크 실효값 감소량 N·m): 걷기 70–80 / 80–90 % 방해 L15 −3.3 / −6.5, Transformer 배율 4 −10.5 / −4.1, MLP 배율 4 −10.6 / −8.2; 달리기 80–90 % 방해 L15 −10.0, Transformer 배율 4 −19.0, MLP 배율 4 −13.5. 기본 설정 두 모델은 모든 구간이 ±1 N·m 안이다.
- 추론 지연(진자 checkpoint, CPU batch 1 / GPU batch 1): MLP+PPO 0.06 / 0.13 ms, MLP+FPO++ 0.63 / 1.30 ms, Transformer+FPO++ 0.97 / 1.79 ms(전신 모델과 같은 구조라 같은 수준). actor 파라미터 235,140 / 240,258 / 281,410.
- 영상: Transformer+FPO++ 배율 4 걷기 400·달리기 600 epoch을 방위각 315°에서 녹화해 `docs/ai_sbc/261103_fpo_pendulum/videos/`(`Walk_진자_Transformer+FPO++_배율4_400epoch_diag.mp4`, `Run_진자_Transformer+FPO++_배율4_600epoch_diag.mp4`)에 두었다. 녹화 실행의 착용자 고관절 토크 RMS 걷기 38.7 → 35.7, 달리기 56.0 → 51.1 N·m.
- 남은 검증: 학습 seed 1개; FPO++ 64회 대 PPO 8회(rollout당 gradient step)의 갱신 횟수 차이; 배율 4 출력이 학습 중 한계로 커지는 원인(보조·변화율 절약 항과 FPO++ 갱신의 상호작용)은 확인하지 않았다; L15의 결정적 정책을 flow actor에 모방 학습시킨 뒤 FPO++로 이어 학습하는 실험은 하지 않았다.

진자 학습에서는 FPO++가 L15를 넘지 못했으므로 진자 학습 정책을 바꿀 근거는 없다. 다음은 전신 학습 FPO++ 절의 남은 검증(학습 seed 반복, PPO 기준선을 같은 갱신 횟수로 재학습)이 먼저다. epoch을 더 늘리지 않는다. 새 학습을 자동 재개하지 않는다.

## Temporal Transformer + FPO++ LLP 대 MLP+PPO 기준선 (2026-11-02 ~ 03, 새 H1)

**FPO++(flow 정책, conditional flow matching 손실의 비율로 PPO 비율을 대신함)와 관측 이력·HLP 예측을 시간별 토큰으로 읽는 소형 Temporal Transformer를 ProtoMotions LLP에 구현해, 같은 조건(W1 + H1, 관측 137, ±20 N·m, 걷기·달리기 반반, 진자 사전학습 없음, 256 환경 × 32 step × 1,200 epoch, seed 13)에서 MLP+PPO(2026-11-02 처음부터 학습)와 비교했다. 평가 동작별 best checkpoint의 착용자 고관절 τ² 변화: 걷기 Transformer+FPO++ −30.1 % · MLP+FPO++ −29.8 % · MLP+PPO −24.4 %, 달리기 MLP+PPO −35.4 %(200 epoch) · Transformer+FPO++ −29.3 % · MLP+FPO++ −29.0 %. FPO++ 두 모델은 보조 토크 실효값 10–12 N·m(PPO 17–18)·최대출력 95% 이상 사용 시간비율 0 %(PPO 39–57 %)로 작은 토크를 쓰고, 걷기에서 PPO의 80–90 % 방해(−8.9 N·m)가 없으며(10 구간 모두 감소), 1,200 epoch에서도 τ² 감소가 매 checkpoint 커지는 추세(수렴 전)다. 낙상 0, 몸체 오차 변화 −1.5 ~ −2.2 %, 전진 속도 변화 0.1 % 안팎.** [보고서: 구현 다이어그램·그래프·호환성·남은 검증](261102_fpo_transformer/report.html). 학습 seed 하나, W1 하나, 걷기·달리기 기록 하나씩.

- 구현: `protomotions/agents/fpo/`(`FPO(PPO)` agent, `FlowActor`, `TemporalTransformer`, 설정), `examples/experiments/assist_skeleton/llp_walk.py --policy mlp_fpo|tfm_fpo --actor-init random`(env·관측·보상·토크 한계는 처음부터 학습 레시피 그대로), `protomotions/assist/llp_policy.py`가 flow actor를 `act_deterministic`(초기 잡음 0에서 Euler 16 step)으로 평가. 단위 시험 `protomotions/tests/test_fpo_agent.py` 9개(토큰 슬롯, transformer는 action당 1회 실행, 비율 1 불변량, 합성 bandit 수렴, EMA checkpoint, 레시피). 학습 run `results/llp_fpo_mlp_20261102`·`results/llp_fpo_tfm_20261102`(동시 실행 1시간 31분; checkpoint는 Transformer+FPO++ 1,200 epoch만 L18로 `checkpoints/L18_tf_fpo_scratch_walkrun/`에 남김), 평가·스크립트·로그 `results/llp_fpo_study_20261102/`(`common.sh`, `short_runs.sh`, `full_run_{mlp,tfm}.sh`, `summarize.py`, `latency_bench.py`, `build_report.py`).
- 영상(2026-11-03): Transformer+FPO++ 1,200 epoch(걷기·달리기 모두 best checkpoint)을 2026-10-30 영상과 같은 조건(40 s, seed 42, ±20 N·m)·같은 상자 레이아웃으로 오른쪽 앞 대각선(방위각 315°)에서 녹화해 `docs/ai_sbc/261102_fpo_transformer/videos/`(`Walk_Transformer+FPO++_1200epoch_diag.mp4`, `Run_Transformer+FPO++_1200epoch_diag.mp4`; 보고서 7절 끝)에 두었다. 녹화 실행의 착용자 고관절 토크 RMS는 걷기 38.7 → 32.4 N·m(16.1 % 감소), 달리기 55.9 → 47.1 N·m(15.8 % 감소). 만드는 법은 `results/llp_fpo_study_20261102/video/record_diag.sh` → `make_videos.sh`(상자는 L16 연구의 `overlay.py`를 그대로 씀) → `build_report.py` → `organize_reports.py`.
- 알고리즘 설정 차이(같은 환경 step 예산): FPO++는 rollout당 gradient step 32회(8 epoch × 4 minibatch 2,048), 비율 clip 0.05 ASPO, AdamW 1e-4·weight decay 1e-4·grad clip 1, actor EMA 0.95(200 epoch 뒤, checkpoint의 actor = EMA), Euler 16 step, actor_scale 0.25(초기 탐색 폭 raw 0.25). 기준선 PPO는 1회(진자 학습부터 써 온 기존 설정), clip 0.2, σ = e⁻², Adam 1e-4, grad clip 50. critic·보상 정규화·advantage 정규화는 공통.
- epoch 추세(걷기 / 달리기, τ² %): MLP+FPO++ 200 → 400 → 600 → 800 → 1,000 → 1,200: −14.0 → −19.9 → −24.0 → −27.0 → −29.4 → −29.8 / −12.0 → −18.4 → −23.3 → −26.0 → −28.3 → −29.0; Transformer+FPO++ −13.8 → −21.1 → −24.9 → −26.9 → −28.7 → −30.1 / −12.7 → −19.9 → −24.3 → −26.7 → −28.4 → −29.3. 보조 토크 실효값은 4 → 10–12 N·m로 같이 커진다. PPO는 200 epoch에 이미 −24.0 / −35.4. 학습 끝 step당 보상 MLP+FPO++ 0.30·Transformer+FPO++ 0.33·PPO 0.21.
- 구간별(best, N·m): 걷기 FPO++ 두 모델은 10 구간 모두 감소(최악 구간이 −1.6 ~ −1.9 감소 = 증가 없음), PPO는 80–90 % −8.9. 달리기는 PPO 200 epoch이 10–30 %·50–70 %에서 +15 ~ +20으로 가장 크고, FPO++는 같은 구간 +12 ~ +15, 80–90 % 거의 0.
- 추론 지연(CPU batch 1 단일 스레드 / 평가 중 GPU batch 2, eager): MLP+PPO 0.06 / 0.22 ms, MLP+FPO++ 0.62 / 1.7 ms, Transformer+FPO++ 0.98 / 2.2 ms — 모두 10 ms 예산 안. flow는 Euler step 수에 비례(step당 속도 MLP 1회 ≈ PPO MLP 1회), transformer context는 step당 1회(CPU 0.42 ms)라 flow step 수와 무관.
- 호환성 처리: rollout 중 모델이 eval 모드라 FPO++의 `self.training` 분기(잡음/0 초기값)를 못 씀 → `forward`(표본)와 `act_deterministic`(평가) 분리; PPO의 `actor_clip_frac_threshold`(좁은 ε에서 항상 걸림)와 critic clip 끔; `load_parameters`가 `logstd`를 전제해 따로 둠; EMA 가중치를 checkpoint actor에 구워 넣고 원 가중치는 `fpo_actor_raw`로 보존(resume용); 진자 LLP 가중치 이어받기는 flow actor에 적용 불가(random 초기화만); FPO++의 `torch.compile` 적분은 쓰지 않음.
- 남은 검증: 학습 seed 1개·평가 seed 2개; PPO 기준선을 같은 32 step/rollout로 다시 학습해 알고리즘 효과와 갱신 횟수 효과를 분리; Euler step 수(16)·actor_scale(0.25)·clip(0.05)·EMA 시작 시점은 첫 값; transformer 읽기 방식(현재 토큰 + 현재 특징)·블록 수(2) 한 가지만 시험; 실기기 CPU 지연 재측정; FPO++는 1,200 epoch에서 수렴 전.

## 진자 사전학습의 효과: 걷기+달리기 추가학습(L16) 대 처음부터 학습 (2026-11-02, 새 H1)

**진자 사전학습 없이 L16과 같은 전신 학습 조건에서 actor를 무작위 초기화해 걷기·달리기를 같이 처음부터 학습한 모델(1,200 epoch, 학습 40분)이 L16보다 두 동작 모두에서 좋다. W1 걷기 τ² −24.4 %(1,200 epoch; L16 best −22.7 %), 달리기 −35.4 %(200 epoch; L16 best −27.4 %). 200~1,200 epoch의 모든 checkpoint가 달리기에서 L16 best보다 좋고 걷기도 1,000 epoch(−20.9 %) 하나만 못 미친다. 보조 토크 실효값 16~19 N·m(L16 19.5), 최대출력 95% 이상 사용 시간비율 16~71 %(L16 88~95 %)로 더 작은 토크로 더 많이 줄인다. 달리기 80–90 % 방해(진자모델만 학습 −16.4, L16 −6.7 N·m)와 걷기 20–30 % 방해(−4.6 / −3.8)가 없어졌고 걷기 80–90 % 방해(−8.9 N·m)는 남는다. 출력이 거의 항상 최대인 진자 정책에서 출발한 추가학습은 그 근처를 벗어나지 못한 것으로 보인다.** [그래프·표](261029_L16_joint_finetune/report.html)의 '진자 사전학습의 효과' 절. 학습 seed 하나, W1 하나, 걷기·달리기 기록 하나씩.

- 처음부터 학습 = `llp_walk.py --initial-actor-checkpoint <L15 ckpt> --actor-init random --finetune-actor-lr 1e-4 --finetune-assist-limit 20 --motion-weights 0.5,0.5 --hlp-calibration "0:…;1:…"`: 출발 checkpoint는 관측 구성·action 변환·PPO 설정의 틀로만 쓰고 가중치는 쓰지 않는다(`AssistFinetunePPO.setup`이 빈 경로면 이어받기를 건너뜀). L16과 다른 점은 actor 초기값과 actor 학습률(1e-4 = 진자 학습값; 추가학습 3e-5)뿐이다. 걷기·달리기 따로 처음부터 학습한 모델은 만들지 않았다(사용자 지시).
- epoch 효과(걷기 / 달리기): 200 → 400 → 600 → 800 → 1,000 → 1,200 epoch 걷기 −24.0 → −23.3 → −22.9 → −23.7 → −20.9 → −24.4 %, 달리기 −35.4 → −31.6 → −32.3 → −34.3 → −29.1 → −32.9 %. 200 epoch에서 이미 L16 best를 넘고 그 뒤 오르내린다(1,000 epoch이 두 동작 모두 가장 나쁨). 학습 보상(에피소드 보상 합; 둘 다 걷기·달리기 반반이라 절대값 비교 가능): 처음부터 학습 200 epoch 222 → 1,200 epoch 183, L16 800 epoch 104 → 1,200 epoch 102.
- 구간별(best checkpoint, N·m): 달리기 처음부터 학습 200 epoch은 1 N·m 넘게 방해하는 구간이 없고(최악 −0.2) 10–30 %에서 +16.0 / +14.8(L16 +7.9 / +9.9); 걷기 1,200 epoch은 80–90 % −8.9만 남는다. 낙상 0, 몸체 오차 변화 걷기 −2.2 %·달리기 −0.2 %.
- 원본·코드: `results/llp_L16scratch_20261102/`(설정·로그; checkpoint는 200 epoch만 L17로 `checkpoints/L17_mlp_ppo_scratch_walkrun/`에 남김), 평가 24건 `results/llp_L16_study_20261029/L16scratch/`(달리기 400 epoch seed 7은 세그폴트로 재실행), `scratch_train_and_eval.sh`, `scratch_section.py`(보고서 절 + 그림 `figures/학습보상곡선_사전학습유무`, `{걷기,달리기}_사전학습유무_tau2변화_epoch`, `{걷기,달리기}_사전학습유무_구간별감소량_{막대,꺾은선}`), `scratch_summary.json`; `summarize.py`·`build_report.sh`에 포함. 보고서 본문의 네 모델 비교 그래프는 그대로다. `Svg.legend_auto`에 가려도 되는 보조선(기준 점선) 인자 `soft` 추가(네 모서리가 다 막힐 때 데이터 점을 가리던 문제).

다음은 (1) 처음부터 학습과 L16을 학습 seed 하나 더로 반복해 차이가 seed 때문이 아닌지 확인, (2) 추가학습의 actor 학습률을 1e-4로 올려 차이가 초기값 때문인지 학습률 때문인지 분리, (3) 처음부터 학습 모델의 걷기 80–90 % 방해 원인 확인. 기본 LLP는 2026-11-04에 L18로 정했다(맨 위 절). epoch을 더 늘리지 않는다. 새 학습을 자동 재개하지 않는다.

## L15 계보: 걷기·달리기 동시 추가학습 L16 대 동작별 추가학습, 1,200 epoch까지 (2026-10-29 ~ 30, 새 H1)

**L15(보조 한계 20 N·m로 진자에서 학습한 정책)의 전신 추가학습은 1,200 epoch까지 이어 학습해도 더 줄지 않았다. 걷기 전용 모델은 600 epoch부터 W1 걷기 τ² −25 ~ −27 %(1,200 epoch −26.6 %), 달리기 전용 모델은 400 epoch(−25.5 %)이 가장 좋았고 그 뒤 물러났다가 1,200 epoch −24.9 %로 돌아왔다. 두 동작을 동시에 학습한 단일 모델 L16은 달리기 −27.4 %(1,200 epoch)로 수렴했지만 걷기는 800 epoch −22.7 %가 가장 좋았고 1,000·1,200 epoch은 −19.6 / −19.9 %이며, 걷기 80–90 % 구간의 방해(−8.2 ~ −8.5 N·m)는 epoch을 늘려도 남는다. L15 계보에서 걷기는 걷기 전용 1,200 epoch, 달리기는 L16 1,200 epoch이 가장 좋고, 한 모델로 두 동작을 맡기면 L16 800 epoch이 가장 균형 잡혀 있다.** [그래프·표·학습 보상 곡선](261029_L16_joint_finetune/report.html). 학습 seed 하나, W1 하나, 걷기·달리기 기록 하나씩.

- 추가학습 = 진자에서 학습한 정책을 가상 착용자 W1과 예측기 H1을 연결한 전신 시뮬레이션에서 이어서 학습(관측 구성·20·tanh 출력 유지, actor 가중치만 이어받고 critic·optimizer 새로, 같은 난수로 나란히 걷는 무보조 착용자 대비 고관절 토크 감소가 보상, 기록 앞 20 s에서 8 s씩, 256 환경·seed 13). 동작별 모델은 한 동작만, L16은 두 동작을 반반 섞어(`--motion-weights 0.5,0.5`) 학습하므로 같은 epoch에서 한 동작의 표본 수는 전용 모델의 절반이다(epoch은 항상 총 epoch으로 적는다). 착용자 영점은 동작마다 측정값이고 L16은 리셋 때 그 동작의 영점을 예측기에 넣는다(`--hlp-calibration "0:…;1:…"`, `ManiFlowAngleEstimator.set_calibration_envs`).
- 1,200 epoch까지는 400(동작별)·800(L16) epoch에서 끝난 run을 같은 디렉터리에서 그대로 이어받았다(last.ckpt의 optimizer·보상 정규화 포함). 이어서 학습은 저장된 설정의 학습 길이만 보므로 `resolved_configs.pt`의 `training_max_steps`를 1,200 epoch으로 바꿨고 원본은 `resolved_configs.pt.ep400` / `.ep800`에 있다(`results/llp_L16_study_20261029/extend_1200{,_run}.sh`).
- epoch 효과(걷기 / 달리기, 각자 동작): 걷기 전용 200 → 400 → 600 → 800 → 1,000 → 1,200 epoch −21.3 → −24.9 → −26.6 → −25.6 → −24.8 → −26.6 %. 달리기 전용 −24.9 → −25.5 → −23.1 → −21.5 → −22.7 → −24.9 %. L16 400 → 600 → 800 → 1,000 → 1,200 epoch 걷기 −18.2 → −19.8 → −22.7 → −19.6 → −19.9 %, 달리기 −23.6 → −23.8 → −27.1 → −27.3 → −27.4 %. 교차(다른 동작): 걷기 전용 1,200 → 달리기 −13.3 %, 달리기 전용 1,200 → 걷기 −17.6 %(추가학습 전 −18.0 / −21.8 %).
- 교차 적용(평가 동작별 best checkpoint끼리, 2026-10-30): 달리기 전용 모델을 걷기에 쓰면 가장 좋아야 200 epoch −19.2 %(추가학습 전 −18.0 %)이고 오래 학습할수록 나빠진다(800 epoch −15.1 %); 걷기 70–90 % 방해(−6.0 / −8.2 N·m)가 그대로다. 걷기 전용 모델을 달리기에 쓰면 가장 좋아야 200 epoch −22.6 %(추가학습 전 −21.8 %)이고 1,200 epoch은 −13.3 %다; 600·1,000 epoch은 달리기 80–90 % 방해를 −9.7 / −6.8 N·m로 줄이지만 0–10 %·90–100 %에 새 방해(−1.2 ~ −5.6 N·m)가 생겨 이득이 없다. 두 동작 모두 추가학습 전보다 좋아지는 것은 L16뿐이다(걷기 800 epoch −22.7 %, 달리기 1,200 epoch −27.4 %). 보고서의 교차 적용 절에 epoch별 곡선(가로축은 총 epoch)과 best checkpoint 구간별 그래프(막대와 꺾은선).
- 영상(2026-10-30): 걷기 = 걷기 전용 1,200 epoch, 달리기 = L16 1,200 epoch을 평가와 같은 조건으로 정면·오른쪽 옆·오른쪽 앞 대각선에서 녹화해 `docs/ai_sbc/261029_L16_joint_finetune/videos/`(보고서 영상 절)에 두었다. 파일 이름이 `Walk_걷기전용모델_1200epoch_side.mp4`처럼 동작·모델·epoch·시점이고 영상 안에는 제목·범례가 없다(사용자가 발표 자료에 직접 적음). 뼈대 메시 = 보조 착용자, 하늘색 선 = 무보조 착용자(motion tracker W1)이고(화살표 없음) 오른쪽 위 반투명 상자에 좌·우 hip 보조 토크 막대(가운데 0, 오른쪽 = 굴곡/앞, 왼쪽 = 신전/뒤; 같은 순간 무보조 착용자의 토크와 같은 방향이면 파랑, 반대면 빨강; 보조 착용자 자신의 토크는 보조가 하중을 대신할 때 0 근처에서 부호가 흔들려 기준으로 쓰지 않음)·수치와 착용자 고관절 토크 1 s RMS를 보인다. 만드는 법은 `results/llp_L16_study_20261029/video/`(record.sh·record_more.sh → make_videos.sh; `run_system.py --record --no-torque-arrows --camera-azimuth/--camera-distance/--camera-height/--camera-look-height --window-pos 1760,731`; 뷰어 창이 세로 모니터에 뜨면 1440 폭으로 잘리므로 창 위치를 가로 모니터로 옮긴다). 2026-10-31 추가(대각선만): 추가학습 전 L15 걷기·달리기(`Walk_진자학습모델_diag.mp4`, `Run_진자학습모델_diag.mp4`), L16 걷기(`Walk_동시학습모델_800epoch_diag.mp4` = 걷기 평가 best checkpoint, `Walk_동시학습모델_1200epoch_diag.mp4` = 달리기 영상과 같은 checkpoint), 달리기 추가학습 모델 달리기(`Run_달리기전용모델_400epoch_diag.mp4` = 달리기 평가 best checkpoint).
- 발표 자료용 그림(2026-10-31): 보고서의 학습 보상 곡선, 동작별 τ² 변화 epoch 곡선, 모델 비교(평가 동작별 best checkpoint 네 모델) 구간별 감소량 막대·꺾은선, 진자모델만 학습 단독의 구간별 감소량 막대(걷기·달리기, `single_model_bins.py`), 보조 ON/OFF 고관절 토크 1 s RMS 시간 그래프(시뮬레이션 2~17 s, 리셋 과도·warmup 제외, 걷기·달리기 모두 걷기+달리기 추가학습 800 epoch, `rms_timeline.py`, 녹화 기록의 무보조 A = OFF·보조 B = ON)를 큰 검은 글자(축 이름 20 px 굵게 > 숫자 15 px, 범례는 그래프 안)로 `docs/ai_sbc/261029_L16_joint_finetune/figures/*.svg|png`(PNG 3배 해상도)에 저장한다. `report.py`의 `Figure.save`(SVG의 class를 속성으로 바꾸고 Chrome headless로 PNG) + `Svg.present = True`(넓은 여백); `build_report.sh`(train_curve.py·cross_section.py·`report.py finetune --figures`)가 보고서와 함께 다시 만든다. 그래프 규칙(2026-10-31 사용자 지시): 축 이름은 '이름 [단위]'만(세로축은 회전해 눈금 옆), '양수 = 부담 감소' 같은 설명은 HTML 보고서의 그래프 caption에만; 범례는 작은 상자로 그래프 안 왼쪽 아래(그 자리에 데이터가 있으면 왼쪽 위 → 오른쪽 아래 → 오른쪽 위 순으로 빈 모서리), 그래프 범위·크기는 그대로; 범례 요소는 항상 네 개(진자모델만 학습(파랑) → 걷기+달리기 추가학습(초록) → 다른 동작 학습 적용(주황, 교차) → 그 동작 추가학습(노랑)), epoch 꼬리표 없음. 본문 걷기·달리기 절의 그래프는 `report.py finetune --chart-walk/--chart-run "run 이름=범례 이름;…" --chart-colors`로 평가 동작별 best checkpoint만 올린다.
- 학습 보상 곡선(25 epoch 평균): 걷기 전용은 400 → 1,200 epoch에 62 → 89로 완만히 오르지만 평가는 600 epoch부터 평평하다. 달리기 전용은 400 epoch 181에서 800 epoch 90까지 내려갔다가 1,200 epoch 145로 돌아왔고, 학습 중 일찍 끝나는 에피소드가 있는 epoch 비율이 41 → 61 %로 늘었다. L16은 800 epoch 104 → 1,000 epoch 88 → 1,200 epoch 102.
- 보행 진행률 구간별 착용자 토크 실효값: 걷기 70–80 / 80–90 %의 방해는 추가학습 전 −7.1 / −7.9 N·m, 걷기 전용 400 epoch −1.5 / −1.9, 1,200 epoch +0.1 / −1.1(없어짐), L16 800 / 1,200 epoch −0.5 / −8.2, −2.1 / −8.5(80–90 %는 그대로). 달리기 80–90 % 방해는 추가학습 전 −16.4, 달리기 전용 400 / 1,200 epoch −13.6 / −12.5, L16 800 / 1,000 / 1,200 epoch −10.6 / −9.3 / −6.7(계속 줄어듦).
- 최대출력 95% 이상 사용 시간비율(|보조 토크| ≥ 19 N·m인 시간의 비율): 걷기 전용 1,200 epoch 걷기 68 %, 달리기 전용 1,200 epoch 달리기 97 %, L16 1,200 epoch 92 / 88 %. 낙상 0, 몸체 오차 변화 −2.8 ~ +1.2 %.
- 원본·코드: `results/llp_L16_study_20261029/`(평가 72건, `summary.json`, `train_curve.json`, 학습·평가 스크립트), 남긴 checkpoint `checkpoints/L15_ft_mlp_ppo_walk/epoch_1200.ckpt`, `checkpoints/L15_ft_mlp_ppo_run/epoch_400.ckpt`, `checkpoints/L16_mlp_ppo_ft_walkrun/epoch_{800,1200}.ckpt`(나머지 epoch은 2026-11-04에 지움). 새 코드: `llp_walk.py --motion-weights`, motion id별 영점(`skeleton_assist_control.py`, `angle_estimator.py`, `paired_skeleton_assist_env.py`). 평가 3건이 python 세그폴트로 기록 없이 끝나 다시 실행했다.

다음 작업은 위 '진자 사전학습의 효과' 절의 목록을 따른다(이 절의 seed 반복·배포 조합 결정은 거기에 합쳤다; L16 걷기 80–90 % 방해는 처음부터 학습 모델에도 남아 그쪽에서 본다).

## 20 N·m 진자 학습 L15와 예측기 토큰 자리 맞바꿈 (2026-10-29)

- L15 = 진자 학습(`mlp.py --recipe L15` = 토크 경로 수정 + 보조 한계 20 N·m, 4,096 환경·6,000 epoch·seed 0) → `checkpoints/L15_mlp_ppo_pendulum/epoch_6000.ckpt`. 최대출력 95% 이상 사용 시간비율(|보조 토크| ≥ 19 N·m인 시간의 비율) 76–84 %로 출력이 거의 항상 최대다. 새 H1에서 W1 걷기 −18.0 % · 달리기 −21.8 %. 이전 진자 모델들은 23–40 N·m로 학습했으므로 20 N·m로 학습한 현재 계보는 L15부터다.
- 추가학습: `llp_walk.py --initial-actor-checkpoint <진자 ckpt> --exclude-motion-ids 1|0 --finetune-assist-limit 20` (정책의 관측 구성과 20·tanh 출력은 그대로 두고 actor 가중치만 이어받으며 critic·optimizer는 새로 시작한다. 보상은 같은 난수로 나란히 걷는 무보조 착용자 대비 고관절 토크 감소. 기록의 앞 20 s 안에서 8 s씩 잘라 학습, 256 환경·seed 13, 착용자 영점은 걷기·달리기 각각 측정값).
- 예측기: DiT-X의 관측 토큰(4ch × 10)을 self-attention 열로, 노이즈 섞인 15프레임 action 토큰열을 cross-attention으로 보내고 출력을 위치 전용 15 query 토큰에서 읽는 구조(ManiFlow `obs_routing=swap`)를 옛 H1과 같은 데이터·12 epoch·seed로 학습해 ETRI 14동작 100 Hz에서 비교했다. denoise 1에서 평균 MAE 0.51° vs 0.56°, 지연 2.45 vs 2.20 ms. 정확도는 우열이 없지만 원래 구상한 구조이므로 **사용자 결정으로 이 모델을 H1로 채택했다.** 옛 H1(관측 cross-attention)의 가중치는 2026-11-04에 지웠다. 이 구조는 ManiFlow 저장소 커밋 `d9e536e`(`obs_routing`, `sample_as_context`)에 있다. 원본 `results/hlp_routing_swap_20261029/`, 채택 평가 `results/h1_swap_adoption_20261029/`.
- 코드: `llp_walk.py` 추가학습 옵션, `report.py finetune`. [보고서](261029_lim20_finetune/report.html).

## 목표와 현재 위치

**걷는 동작을 방해하지 않으면서 착용자의 고관절 부담을 줄이는 보조 제어기를 만든다.**
장기 목표는 다양한 보행에 대응하는 것이다. 하드웨어 확인은 사용자 담당이며, 현재 작업 범위는 시뮬레이션이다.

- 각도 예측기(HLP, H1): 최근 움직임으로 미래 고관절 각도를 예측한다.
- 보조 제어기(LLP, L 계열, 기본 L18): 예측과 관절 상태를 받아 좌우 고관절 보조 토크를 낸다.
- 가상 착용자(W1): 시뮬레이션에서 사람 대신 전신 동작을 따라가는 모델이다.

**기존 걷기·달리기 기록에서는 전신 시뮬레이션에서 처음부터 학습한 LLP(L17·L18)가 가장 좋다. 다양한 실제 보행에 쓸 수 있다는 검증은 아직 없다.**
2026-10 중순에 정한 우선순위는 진자만으로 학습하는 정책의 성능을 높이는 것이었다([개선안](archive/20261104/pendulum_pretrain_discussion.md)). 그 뒤 처음부터 학습한 전신 정책이 진자 사전학습 계보보다 좋았고, 기본 LLP를 L18로 정했다. 진자 학습과 전신 학습 중 무엇을 이어갈지는 위 절들의 남은 검증 결과로 정한다.
새 실제 보행은 W1이 무보조로도 따라가지 못해(16조건 중 15낙상) 보조 평가에 쓰지 못한다.
토크 제곱 평균은 시뮬레이션에서 계산한 부담 지표다. 대사량이나 실제 근활성을 측정한 값이 아니다.
실험별 조건·한계·근거는 [실험 결과 요약](experiment_summary.html)에 모았다. 상세 자료는 [보고서 목록](reports.html)에서 찾는다.

## 계속 작업할 때 필요한 파일

| 용도 | 위치 |
|---|---|
| 모델 이름·보존 경로 | [AI_SBC_VERSIONS.md](AI_SBC_VERSIONS.md), 각 `checkpoints/*/INFO.md` |
| 진자 학습에 적용한 FPO++ | [결과](261103_fpo_pendulum/report.html) · 원본 `results/llp_fpo_pendulum_20261103/` |
| L18(Transformer+FPO++) 대 MLP+PPO | [결과](261102_fpo_transformer/report.html) · 원본 `results/llp_fpo_study_20261102/` |
| L15_ft·L16·L17(처음부터 학습) | [결과](261029_L16_joint_finetune/report.html) · 원본 `results/llp_L16_study_20261029/` |
| L15·H1 토큰 자리 맞바꿈 | [결과](261029_lim20_finetune/report.html) · 원본 `results/llp_lim20_finetune_20261029/`, `results/hlp_routing_swap_20261029/` |
| L15 이전 계보 (가중치 삭제) | [보관 보고서](reports.html#past) · 정리 전 [HANDOFF](archive/20261104/HANDOFF.md)·[버전표](archive/20261104/AI_SBC_VERSIONS.md)·[요약](archive/20261104/experiment_summary.html) |
| 코드 구조·관측 계약·운영 주의 | [AI_SBC_FRAMEWORK.md](AI_SBC_FRAMEWORK.md) |
| 2026-10-26 코드 통합 이전 상태 | 스냅샷 커밋 `8058f3c`(`git worktree add ../AI-SBC-PROTOMOTIONS-8058f3c 8058f3c`). 닫힌 연구의 당시 스크립트는 `results/<study>/scripts/`(각 `SNAPSHOT.md`) |
| 2026-11-04 옛 계보 코드 정리 이전 상태 | 스냅샷 커밋 `e74e83e`. 뺀 파일의 원본은 `results/<연구>/scripts/retired_20261104/` |
| 새 보행 변환 / 무보조 평가 코드 | `data/scripts/retarget_soma23_to_suit.py` / `examples/experiments/assist_skeleton/evaluate.py native-wearer` |
| 실제 보행 변환본 | `results/real_walk_retarget_20261012_224957/` |
| 무보조 평가·참조 교체 실험 | `results/native_walk_validation_20261013/validation_summary.json` |

진자는 `examples/experiments/assist_pendulum/{mlp,runtime,evaluate}.py`(`mlp.py --recipe L15`),
스켈레톤은 `examples/experiments/assist_skeleton/{run_system,analysis,report,evaluate,plots,viz,llp_walk,tracker_human}.py`다.
연구 외란 제어는 `DisturbedAssistTargetControl` 하나다. LLP 추론은 `LLPPolicy`가 맡고, actor 하나가 양다리 관측을 함께 받아 토크 2개를 낸다.

실행 Python은 `/home/bak/miniconda3/envs/sbc/bin/python`이다. Full access에서 GPU 계산과 Newton 평가를 완료했다.
예전 접근 오류로 현재 실행을 막지 말고 실제 명령 결과를 확인한다. 남긴 모델·원본 결과는 보존한다.
