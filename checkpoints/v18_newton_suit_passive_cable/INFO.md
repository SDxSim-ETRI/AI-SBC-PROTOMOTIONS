# Checkpoint Info

| 항목 | 내용 |
|------|------|
| 소스 | `results/mimic_newton_suit_passive_cable` |
| 보관일 | 2026-06-05 11:50 |
| Epoch | 1230 |
| 메모 | skeleton_torque_suit_passive_cable, 15모션(11+koo_4), Newton, epoch 1200, v17 warm_start, 케이블 토크=0 (passive) |

## 학습 재사용 방법

```bash
python protomotions/train_agent.py \
    ... \
    --checkpoint checkpoints/v18_newton_suit_passive_cable/last.ckpt \
    ...
```
