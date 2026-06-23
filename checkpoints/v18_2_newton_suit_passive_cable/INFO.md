# Checkpoint Info

| 항목 | 내용 |
|------|------|
| 소스 | `results/mimic_newton_suit_passive_cable_v18_2` |
| 보관일 | 2026-06-10 |
| Epoch | 5000 |
| 메모 | skeleton_torque_suit_passive_cable, 15모션(11+koo_4), Newton, epoch 5000, v18 warm_start, 케이블 토크=0 (passive), 성공률 100% |

## 학습 재사용 방법

```bash
python protomotions/train_agent.py \
    ... \
    --checkpoint checkpoints/v18_2_newton_suit_passive_cable/last.ckpt \
    ...
```
