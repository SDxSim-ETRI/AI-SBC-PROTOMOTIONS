# L17 — MLP+PPO 보조 제어기, 걷기·달리기 처음부터 학습 (200 epoch)

진자 사전학습 없이 MLP+PPO로 걷기·달리기를 반반 섞어 처음부터 학습한 모델의 200 epoch이다.
시뮬레이션 연구용이며 실기기 성능을 보증하지 않는다. 모델 목록은 [버전표](../../docs/ai_sbc/AI_SBC_VERSIONS.md)에 있다.

| 항목 | 내용 |
|---|---|
| 가중치 | `epoch_200.ckpt` |
| 원본 실행 | `results/llp_L16scratch_20261102` (2026-11-02, 1,200 epoch 학습 40분 중 200 epoch) |
| 정책 | MLP(512·256·128) + PPO |
| 학습 | 가상 착용자 W1과 H1을 연결한 전신 시뮬레이션, 걷기·달리기 반반, actor 무작위 초기화, actor 학습률 1e-4, 256 환경 × 32 step, seed 13. L16과 다른 점은 actor 초기값과 학습률뿐이다 |
| 출력 | 20·tanh, 한계 ±20 N·m, 출력 배율 1, 10 ms 주기 |
| W1 + H1 평가 (40 s, 평가 seed 42·7 평균) | 착용자 고관절 τ² 걷기 −24.0 % / 달리기 −35.4 %. 보조 토크 실효값 16.4 / 17.2 N·m, 최대출력 95% 이상 사용 시간비율 16.1 / 39.3 %, 낙상 0 |
| 추론 지연 | CPU batch 1 단일 스레드 0.06 ms |
| 로드 | `protomotions.assist.llp_policy.LLPPolicy(checkpoint)`. 같은 폴더의 `resolved_configs_inference.pt`가 필요하다 |
| 보고서 | [L16 연구의 '진자 사전학습의 효과' 절](../../docs/ai_sbc/261029_L16_joint_finetune/report.html) |

200 epoch을 남긴 이유: 이 run에서 달리기가 가장 좋고(−35.4 %) 걷기도 1,200 epoch(−24.4 %)과 거의 같다.
걷기 80–90 % 구간의 방해(−8.9 N·m)는 이 run의 모든 checkpoint에 남아 있다.

다시 학습: `results/llp_L16_study_20261029/scratch_train_and_eval.sh`.
`--initial-actor-checkpoint`는 틀로만 쓰며 지금 위치는 `checkpoints/L15_mlp_ppo_pendulum/epoch_6000.ckpt`다.
한계: 학습 seed 하나, W1 하나, 걷기·달리기 기록 하나씩.
