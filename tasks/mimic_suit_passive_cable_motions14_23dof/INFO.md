# mimic_suit_passive_cable_motions14_23dof

| 항목 | 내용 |
|------|------|
| robot | `skeleton_torque_suit_passive_cable` (27-DOF, passive cable — 케이블 토크=0) |
| 모션 | `skeleton_torque_suit_motions_11+koo_4.pt` (15클립: 11+koo_4) |

## Newton 결과

| 항목 | 내용 |
|------|------|
| 결과 | `output_newton/last.ckpt` — v18_2, epoch 5000, 100% |
| 최초 warm_start | `checkpoints/v18_newton_suit_passive_cable/last.ckpt` (미보관) |
| simulator | Newton, flat terrain |

## IsaacLab 결과 (sim-to-sim)

| 항목 | 내용 |
|------|------|
| 결과 | `output_isaaclab/last.ckpt` — v19, epoch 1600 |
| 최초 warm_start | `checkpoints/v18_2_newton_suit_passive_cable/score_based.ckpt` (Newton→IsaacLab sim-to-sim) |
| simulator | IsaacLab, flat terrain |

## 학습 계보

(v18 미보관) → **v18_2 Newton** (15모션, 100%)
→ **v19 IsaacLab** (sim-to-sim 전환)
→ v24 rough (walk 단일 모션 지형 강화)

## 스크립트

| 파일 | 설명 |
|------|------|
| `train_newton_initial.sh` | Newton 최초 학습 — v18 warm_start (v18 미보관 주의) |
| `train_newton_resume.sh` | Newton 재개 |
| `train_isaaclab_initial.sh` | IsaacLab 최초 학습 — v18_2 warm_start (sim-to-sim) |
| `train_isaaclab_resume.sh` | IsaacLab 재개 |
