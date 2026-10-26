# Checkpoint Info

| 항목 | 내용 |
|------|------|
| 소스 | `results/mimic_suit_passive_walk` |
| 보관일 | 2026-09-12 |
| Epoch | 4440 |
| 로봇 | `skeleton_torque_suit_passive_cable` (27 DOF, 슈트 slide 관절 4개 토크 0) |
| 시뮬레이터 | Newton, 평지, 물리 120 Hz / decimation 6 (env 20 Hz), 4096 env, batch 16384 |
| 모션 | walk 1클립(skeleton_torque_suit_walk.pt) |
| 메모 | 성공률 1.0(3400~4400 연속), 몸체 위치 오차 7.6 cm. 삭제된 v18_2 레시피(`examples/experiments/mimic/mlp.py`)를 처음부터 재학습(2026-09-12). 학습 프로세스가 외부 종료돼 epoch 4440에서 멈췄으나 목표 상태 도달 후였음. 주의: mimic 보상은 몸체 위치만 맞추므로 hip flexion의 mocap 추종은 10~14° 벗어남 (framework doc §3.5) |

## 학습 스크립트
`results/mimic_suit_passive_walk/scripts/train_tracker_walk.sh` / `results/mimic_suit_passive_motions15/scripts/train_tracker_motions15.sh` (2026-10-26 보관)

## 재사용 방법

```bash
# 착용자 대역으로 (HLP+LLP 통합 시스템)
python examples/experiments/assist_skeleton/run_system.py --human tracker \
    --tracker-checkpoint checkpoints/v30_newton_suit_passive_walk/last.ckpt --seconds 30
# 이어서 학습 (warm start)
python protomotions/train_agent.py ... --checkpoint checkpoints/v30_newton_suit_passive_walk/last.ckpt ...
```
