# L18 — Transformer+FPO++ 보조 제어기, 걷기·달리기 처음부터 학습

현재 기본 LLP다 (`run_system.py` 기본값, 2026-11-04 지정).
시뮬레이션 연구용이며 실기기 성능을 보증하지 않는다. 모델 목록은 [버전표](../../docs/ai_sbc/AI_SBC_VERSIONS.md)에 있다.

| 항목 | 내용 |
|---|---|
| 가중치 | `epoch_1200.ckpt`. actor는 EMA 가중치이고, 학습 중 원 가중치는 `fpo_actor_raw`에 있다 |
| 원본 실행 | `results/llp_fpo_tfm_20261102` (2026-11-02, 학습 1시간 31분) |
| 정책 | 관측 이력·이전 출력·HLP 예측을 시간별 토큰으로 읽는 Temporal Transformer(2블록) + flow 정책. 평가는 초기 잡음 0에서 Euler 16 step으로 적분한 결정적 출력 |
| 학습 | 가상 착용자 W1과 H1을 연결한 전신 시뮬레이션에서 FPO++로 학습. 걷기·달리기 반반, 진자 가중치 없이 무작위 초기화, 256 환경 × 32 step × 1,200 epoch, seed 13, 보상은 같은 난수로 나란히 걷는 무보조 착용자 대비 고관절 토크 감소 |
| 출력 | 20·tanh, 한계 ±20 N·m, 출력 배율 1, 10 ms 주기 |
| W1 + H1 평가 (40 s, 평가 seed 42·7 평균) | 착용자 고관절 τ² 걷기 −30.1 % / 달리기 −29.3 %. 보조 토크 실효값 10.1 / 11.6 N·m, 최대출력 95% 이상 사용 시간비율 0 %, 낙상 0, 몸체 오차 변화 −1.8 / −1.6 % |
| 추론 지연 | CPU batch 1 단일 스레드 0.98 ms |
| 로드 | `protomotions.assist.llp_policy.LLPPolicy(checkpoint)`. 같은 폴더의 `resolved_configs_inference.pt`가 필요하다 |
| 보고서 | [Transformer+FPO++ 대 MLP+PPO](../../docs/ai_sbc/261102_fpo_transformer/report.html) |

2026-11-04 기본값 확인: `run_system.py`를 인자 없이 실행한 걷기와 `--motion-id 1` 달리기(seed 42)에서
보조 토크 실효값 10.14 / 11.62 N·m로 학습 연구의 기록(10.13 / 11.62)과 같았다.

다시 학습: `results/llp_fpo_study_20261102/common.sh`의 `train tfm_fpo …`.
그때 `--initial-actor-checkpoint`는 관측 구성·출력 변환·PPO 설정의 틀로만 쓰며(가중치는 쓰지 않음),
지금 위치는 `checkpoints/L15_mlp_ppo_pendulum/epoch_6000.ckpt`다.

한계: 학습 seed 하나, W1 하나, 걷기·달리기 기록 하나씩. 1,200 epoch에서 아직 좋아지는 중이었다(수렴 전).
FPO++는 rollout당 gradient step 32회로 PPO 기준선(1회)과 갱신 횟수가 다르다.
