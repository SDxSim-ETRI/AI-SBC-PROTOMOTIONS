# Checkpoint Info — LLP (hip assist torque policy) on the walking skeleton wearer

| 항목 | 내용 |
|------|------|
| 소스 | `results/llp_skel_motions15_v4` |
| 보관일 | 2026-09-13 |
| Epoch | 1890 (평가 파워 평탄 후 중단) |
| 학습 env | `SkeletonAssistEnv` (env 10 ms, 물리 400 Hz/4), 착용자 = 동결 mimic tracker(5 step마다), HLP = 실 ManiFlow(배치, 20 ms마다), 4096 env |
| 설정 | 15모션(11+koo_4), 착용자 v31 tracker, 40 N·m |
| 보상 | 몸체 오차 hinge 벌점(margin 0.09 m, w 10) + human_power −0.003 + assist_effort −5e-5 + action_rate −0.005 |
| 결과 | 14모션 시간 가중 **−21.8 %**(84.3→66.0 W), 낙상 0; 미학습 착용자 v30 walk −11.3 % |
| 실기기 규약 | actor 입력 = hip flexion r/l 각·각속도(+히스토리 10), HLP 미래 창 4×50 ms, 이전 action 5; 출력 tanh×한계 [N·m]; `flexion_sign=+1`(관절각 = 데이터 flexion) |

## 사용
```bash
# 통합 시스템 평가 (착용자 tracker 위)
python examples/experiments/assist_skeleton/run_system.py --human tracker \
    --tracker-checkpoint checkpoints/v30_newton_suit_passive_walk/last.ckpt \
    --llp-checkpoint checkpoints/llp_v4_skel_motions15_40Nm/last.ckpt --seconds 30
# 모션별 A/B
python examples/experiments/assist_skeleton/evaluate.py motions --llp-checkpoint checkpoints/llp_v4_skel_motions15_40Nm/last.ckpt \
    --tracker-checkpoint checkpoints/v31_newton_suit_passive_motions15/last.ckpt \
    --motion-file "data/motion_for_trackers/skeleton_torque_suit_motions_11+koo_4.pt" --calibration motion_all --out results/assist_skeleton_motions/<tag>
# 독립 제어기로 싣기: protomotions.assist.AssistController(AssistControllerConfig(llp_checkpoint="checkpoints/llp_v4_skel_motions15_40Nm/last.ckpt", joint_sign=+1))
```
