# L16 — L15의 걷기·달리기 동시 추가학습 (800 · 1,200 epoch)

진자에서 학습한 L15를 전신 시뮬레이션에서 걷기·달리기 반반으로 이어서 학습한 모델이다.
시뮬레이션 연구용이며 실기기 성능을 보증하지 않는다. 모델 목록은 [버전표](../../docs/ai_sbc/AI_SBC_VERSIONS.md)에 있다.

| 항목 | 내용 |
|---|---|
| 가중치 | `epoch_800.ckpt` (걷기 best, 두 동작 균형), `epoch_1200.ckpt` (달리기 best) |
| 원본 실행 | `results/llp_L16_20261029` (2026-10-29 ~ 30). 800 epoch에서 같은 run을 이어 1,200 epoch까지 학습 |
| 정책 | MLP(512·256·128) + PPO. L15 actor 가중치를 이어받고 critic·optimizer는 새로 시작 |
| 학습 | 가상 착용자 W1과 H1을 연결한 전신 시뮬레이션, 걷기·달리기 반반, 동작별 착용자 영점, actor 학습률 3e-5, 256 환경 × 32 step, seed 13 |
| 출력 | 20·tanh, 한계 ±20 N·m, 출력 배율 1, 10 ms 주기 |
| W1 + H1 평가 (40 s, 평가 seed 42·7 평균) | 800 epoch: 착용자 고관절 τ² 걷기 −22.7 % / 달리기 −27.1 %, 최대출력 95% 이상 사용 시간비율 91 / 91 %. 1,200 epoch: −19.9 / −27.4 %, 92 / 88 %. 낙상 0 |
| 로드 | `protomotions.assist.llp_policy.LLPPolicy(checkpoint)`. 같은 폴더의 `resolved_configs_inference.pt`가 필요하다 |
| 보고서 | [L16 대 동작별 추가학습](../../docs/ai_sbc/261029_L16_joint_finetune/report.html) |

걷기 80–90 % 구간의 방해(−8.2 ~ −8.5 N·m)는 epoch을 늘려도 남았다.
같은 조건에서 진자 가중치 없이 처음부터 학습한 L17·L18이 두 동작 모두 더 좋다.
다시 학습: `results/llp_L16_study_20261029/`의 학습 스크립트(출발 모델은 지금 `checkpoints/L15_mlp_ppo_pendulum/epoch_6000.ckpt`).
한계: 학습 seed 하나, W1 하나, 걷기·달리기 기록 하나씩.
