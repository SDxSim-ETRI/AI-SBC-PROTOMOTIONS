# Checkpoint Info — LLP (hip assist torque policy), 보행 계열 전용

| 항목 | 내용 |
|------|------|
| 소스 | `results/llp_gait_v6` |
| 보관일 | 2026-09-14 |
| Epoch | 1901 (평가 파워 1400 epoch부터 평탄: 165.1 → 162.4 W) |
| 학습 env | `SkeletonAssistEnv` (env 10 ms, 물리 400 Hz/4), 착용자 = 동결 mimic tracker v31(5 step마다), HLP = 실 ManiFlow(배치, 20 ms마다), 4096 env |
| 학습 모션 | **보행 계열만** — walk 0.75 + run 0.25 (`make_gait_motions.py`; squat·제자리 걸음·한 걸음·런지 제외) |
| 설정 | assist 한계 40 N·m, `--action-rate-weight -0.005`(기본), 구동기 필터 없음 |
| 보상 | 몸체 오차 hinge 벌점(margin 0.09 m, w 10) + human_power −0.003 + assist_effort −5e-5 + action_rate −0.005 |
| 결과 (40 s, v31 착용자) | walk **174 → 127 W (−27.0 %)**, run **227 → 155 W (−31.7 %)**, 낙상 0 |
| 참고 | 같은 착용자에서 v5(walk 학습·앙상블)는 walk −25.6 %. 보행 계열로 좁히자 walk·run 둘 다 올라갔다 |
| 토크 성격 | 변화율 RMS 660 N·m/s, 한계 도달 14~23 %, 부호 전환 ≈5회/s — 빠르게 변하는 토크. 구동기 대역이 좁으면 `--torque-lpf-hz 12`(절감 2 포인트 손해) 또는 학습 때부터 필터를 넣은 v8 계열을 쓸 것 |
| 실기기 규약 | actor 입력 = hip flexion r/l 각·각속도(+히스토리 10), HLP 미래 창 4×50 ms, 이전 action 5; 출력 tanh×한계 [N·m]; `flexion_sign=+1`(관절각 = 데이터 flexion) |

## 사용
```bash
# 통합 시스템 A/B 평가 (walk = motion-id 0, run = 1)
python examples/experiments/assist_skeleton/run_system.py --human tracker \
    --tracker-checkpoint checkpoints/v31_newton_suit_passive_motions15/last.ckpt \
    --llp-checkpoint checkpoints/llp_v6_gait_40Nm/last.ckpt \
    --motion-file data/motion_for_trackers/skeleton_torque_suit_gait.pt --motion-id 0 --seconds 40
# 뷰어/녹화 (뼈 스켈레톤)
DISPLAY=:1 python examples/experiments/assist_skeleton/run_system.py ... --record
# 독립 제어기로 싣기
#   AssistController(AssistControllerConfig(llp_checkpoint="checkpoints/llp_v6_gait_40Nm/last.ckpt", joint_sign=+1))
```

## 학습 재현
```bash
python examples/experiments/assist_skeleton/make_gait_motions.py
python protomotions/train_agent.py --robot-name skeleton_torque_suit_passive_cable --simulator newton \
    --experiment-path examples/experiments/assist_skeleton/llp_walk.py --experiment-name llp_gait_v6 \
    --motion-file data/motion_for_trackers/skeleton_torque_suit_gait.pt --num-envs 4096 --batch-size 16384 \
    --tracker-checkpoint checkpoints/v31_newton_suit_passive_motions15/last.ckpt --assist-limit 40
```
