# Checkpoint Info — LLP (hip assist torque policy) on the walking skeleton wearer

| 항목 | 내용 |
|------|------|
| 소스 | `results/llp_skel_walk_ens_v5` |
| 보관일 | 2026-09-13 |
| Epoch | 1780 (평가 파워 평탄 후 중단) |
| 학습 env | `SkeletonAssistEnv` (env 10 ms, 물리 400 Hz/4), 착용자 = 동결 mimic tracker(5 step마다), HLP = 실 ManiFlow(배치, 20 ms마다), 4096 env |
| 설정 | walk 클립, 착용자 앙상블 {v30, v31} env별 무작위 배정, 40 N·m |
| 보상 | 몸체 오차 hinge 벌점(margin 0.09 m, w 10) + human_power −0.003 + assist_effort −5e-5 + action_rate −0.005 |
| 결과 | v30 walk **−31.7 %**, v31 walk **−25.6 %** (다른 구간·시드 −30.9/−26.0 %), 낙상 0 — 한 정책이 두 걸음 스타일 모두 보조 |
| 실기기 규약 | actor 입력 = hip flexion r/l 각·각속도(+히스토리 10), HLP 미래 창 4×50 ms, 이전 action 5; 출력 tanh×한계 [N·m]; `flexion_sign=+1`(관절각 = 데이터 flexion) |

## 사용
```bash
# 통합 시스템 평가 (착용자 tracker 위)
python examples/experiments/assist_skeleton/run_system.py --human tracker \
    --tracker-checkpoint checkpoints/v30_newton_suit_passive_walk/last.ckpt \
    --llp-checkpoint checkpoints/llp_v5_skel_walk_ensemble_40Nm/last.ckpt --seconds 30
# 모션별 A/B
python examples/experiments/assist_skeleton/eval_motions.py --llp-checkpoint checkpoints/llp_v5_skel_walk_ensemble_40Nm/last.ckpt \
    --tracker-checkpoint checkpoints/v31_newton_suit_passive_motions15/last.ckpt \
    --motion-file "data/motion_for_trackers/skeleton_torque_suit_motions_11+koo_4.pt" --calibration motion_all --out results/assist_skeleton_motions/<tag>
# 독립 제어기로 싣기: protomotions.assist.AssistController(AssistControllerConfig(llp_checkpoint="checkpoints/llp_v5_skel_walk_ensemble_40Nm/last.ckpt", joint_sign=+1))
```
