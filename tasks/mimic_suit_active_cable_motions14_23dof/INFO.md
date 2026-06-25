# mimic_suit_active_cable_motions14_23dof

| 항목 | 내용 |
|------|------|
| robot | `skeleton_torque_suit_active_cable` (27-DOF, active cable — 케이블 토크 출력) |
| simulator | Newton, flat terrain |
| 모션 | `skeleton_torque_suit_motions14.pt` (14클립: constspeed 제외) |
| 결과 | `output_newton/last.ckpt` — v21_3, epoch 1100, 100% |
| 최초 warm_start | `checkpoints/v18_2_newton_suit_passive_cable/score_based.ckpt` (passive cable → active cable 전환) |

## 학습 계보

v18_2 (passive cable 15모션) → **v21_3 active cable** (active cable로 전환, 15모션)
→ v21_4 (walk 단일 모션 특화, `mimic_suit_active_cable_walk_23dof`)

## 스크립트

| 파일 | 설명 |
|------|------|
| `train_newton_initial.sh` | 최초 학습 — v18_2 passive cable warm_start (motions_11+koo_4, 15클립) |
| `train_newton_resume.sh` | 재개 — `output_newton/last.ckpt` (motions14, 14클립) |
| `play_newton.sh` | Newton mesh 인터랙티브 재생 |
| `record_newton.sh` | Newton mesh 자동 녹화 (14모션 × 20초) |
| `live_viz_newton.sh` | Newton 라이브 시각화 — 케이블 패널 포함 (`--active-cable`) |
| `collect_newton.sh` | Newton 데이터 수집 → `data/collected/*.zarr` (케이블 필드 포함) |
| `train_isaaclab_initial.sh` | IsaacLab 최초 학습 — Newton ckpt warm_start (motions_11+koo_4, 15클립) |
| `train_isaaclab_resume.sh` | IsaacLab 재개 — `output_isaaclab/last.ckpt` (motions14, 14클립) |
| `play_isaaclab.sh` | IsaacLab USD mesh 인터랙티브 재생 |
| `record_isaaclab.sh` | IsaacLab USD mesh 자동 녹화 (14모션 × 20초) |
