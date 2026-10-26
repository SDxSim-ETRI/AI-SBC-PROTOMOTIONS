# Checkpoint Info — LLP (hip assist torque policy) on the walking skeleton wearer

| 항목 | 내용 |
|------|------|
| 소스 | `results/llp_skel_walk_v3` |
| 보관일 | 2026-09-13 |
| Epoch | 2500 (평가 파워 평탄 후 중단) |
| 학습 env | `SkeletonAssistEnv` (env 10 ms, 물리 400 Hz/4), 착용자 = 동결 mimic tracker(5 step마다), HLP = 실 ManiFlow(배치, 20 ms마다), 4096 env |
| 설정 | walk 클립, 착용자 v30 tracker, assist 한계 40 N·m (`llp_walk.py --reward-variant hinge --assist-limit 40`) |
| 보상 | 몸체 오차 hinge 벌점(margin 0.09 m, w 10) + human_power −0.003 + assist_effort −5e-5 + action_rate −0.005 |
| 결과 | 통합 평가(walk 30~60 s) 착용자 hip 파워 **−31~−32 %**, 낙상 0; 다른 착용자(v31)에서는 −2.4 % (착용자 특이성) |
| 실기기 규약 | actor 입력 = hip flexion r/l 각·각속도(+히스토리 10), HLP 미래 창 4×50 ms, 이전 action 5; 출력 tanh×한계 [N·m]; `flexion_sign=+1`(관절각 = 데이터 flexion) |

## 사용
```bash
# 통합 시스템 평가 (착용자 tracker 위)
python examples/experiments/assist_skeleton/run_system.py --human tracker \
    --tracker-checkpoint checkpoints/v30_newton_suit_passive_walk/last.ckpt \
    --llp-checkpoint checkpoints/llp_v3_skel_walk_40Nm/last.ckpt --seconds 30
# 모션별 A/B
python examples/experiments/assist_skeleton/evaluate.py motions --llp-checkpoint checkpoints/llp_v3_skel_walk_40Nm/last.ckpt \
    --tracker-checkpoint checkpoints/v31_newton_suit_passive_motions15/last.ckpt \
    --motion-file "data/motion_for_trackers/skeleton_torque_suit_motions_11+koo_4.pt" --calibration motion_all --out results/assist_skeleton_motions/<tag>
# 독립 제어기로 싣기: protomotions.assist.AssistController(AssistControllerConfig(llp_checkpoint="checkpoints/llp_v3_skel_walk_40Nm/last.ckpt", joint_sign=+1))
```
