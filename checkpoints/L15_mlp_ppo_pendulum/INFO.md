# L15 — 진자에서 학습한 보조 제어기 (보조 한계 20 N·m)

진자 환경만으로 학습한 LLP다. L15_ft·L16의 출발 모델이며, 진자 학습 비교의 기준이다.
시뮬레이션 연구용이며 실기기 성능을 보증하지 않는다. 모델 목록은 [버전표](../../docs/ai_sbc/AI_SBC_VERSIONS.md)에 있다.

| 항목 | 내용 |
|---|---|
| 가중치 | `epoch_6000.ckpt` |
| 원본 실행 | `results/llp_R0lim20_s0_20261029` (2026-10-29) |
| 정책 | MLP(512·256·128) + PPO, 좌우 고관절을 한 actor가 출력 |
| 학습 | `examples/experiments/assist_pendulum/mlp.py --recipe L15` (랜덤화 진자 + 착용자 PD+FF 한 번만 제한하는 토크 경로 + 보조 한계 20 N·m), 4,096 환경, 6,000 epoch, seed 0. 레시피가 이 설정을 그대로 만드는지는 `protomotions/tests/test_pendulum_recipes.py`가 확인한다 |
| 출력 | 20·tanh, 한계 ±20 N·m, 출력 배율 1, 10 ms 주기 |
| W1 + H1 평가 (40 s, 평가 seed 42·7 평균, 2026-11-03) | 착용자 고관절 τ² 걷기 −18.0 % / 달리기 −21.8 %. 보조 토크 실효값 18.9 / 19.2 N·m, 최대출력 95% 이상 사용 시간비율 76.7 / 84.3 %, 낙상 0 |
| 로드 | `protomotions.assist.llp_policy.LLPPolicy(checkpoint)`. 같은 폴더의 `resolved_configs_inference.pt`가 필요하다 |
| 보고서 | [20 N·m 학습과 추가학습](../../docs/ai_sbc/261029_lim20_finetune/report.html), [진자 학습 정책 비교](../../docs/ai_sbc/261103_fpo_pendulum/report.html) |

다른 쓰임:
- 전신 학습(`llp_walk.py`)의 `--initial-actor-checkpoint`: 이어서 학습할 때는 출발 가중치, `--actor-init random`이면 관측 구성·출력 변환·PPO 설정의 틀로만 쓴다(L17·L18).
- 제어기 일치 검증 `python -m protomotions.assist.verify_controller`의 기본 모델.

같은 run을 1,200 epoch 이하로 다시 평가하면 1,000 epoch이 걷기 −24.1 % / 달리기 −22.1 %로 더 좋았지만,
그 checkpoint는 2026-11-04 정리에서 지웠다. 학습 seed 하나.
