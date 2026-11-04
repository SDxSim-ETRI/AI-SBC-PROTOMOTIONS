# L15_ft 달리기 — L15의 달리기 전용 추가학습 (400 epoch)

진자에서 학습한 L15를 전신 시뮬레이션에서 달리기만으로 이어서 학습한 모델이다. 달리기 전용 run의 best checkpoint다.
시뮬레이션 연구용이며 실기기 성능을 보증하지 않는다. 모델 목록은 [버전표](../../docs/ai_sbc/AI_SBC_VERSIONS.md)에 있다.

| 항목 | 내용 |
|---|---|
| 가중치 | `epoch_400.ckpt` |
| 원본 실행 | `results/llp_L15_ft_run_20261029` (2026-10-29 ~ 30, 1,200 epoch까지 학습한 run의 400 epoch) |
| 정책 | MLP(512·256·128) + PPO. L15 actor 가중치를 이어받고 critic·optimizer는 새로 시작 |
| 학습 | 가상 착용자 W1과 H1을 연결한 전신 시뮬레이션, 달리기(motion 1)만, actor 학습률 3e-5, 256 환경 × 32 step, seed 13 |
| 출력 | 20·tanh, 한계 ±20 N·m, 출력 배율 1, 10 ms 주기 |
| W1 + H1 평가 (40 s, 평가 seed 42·7 평균) | 착용자 고관절 τ² 달리기 −25.5 %. 낙상 0 |
| 로드 | `protomotions.assist.llp_policy.LLPPolicy(checkpoint)`. 같은 폴더의 `resolved_configs_inference.pt`가 필요하다 |
| 보고서 | [L16 대 동작별 추가학습](../../docs/ai_sbc/261029_L16_joint_finetune/report.html) |

400 epoch 뒤로는 물러났다가 1,200 epoch에 −24.9 %로 돌아왔다. 달리기 80–90 % 방해는 −13.6 N·m 남는다.
한계: 학습 seed 하나, W1 하나, 달리기 기록 하나.
