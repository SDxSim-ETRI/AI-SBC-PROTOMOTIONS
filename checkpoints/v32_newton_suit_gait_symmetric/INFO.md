# Checkpoint Info — 착용자(mimic tracker), 관절각 추종 + 보행 계열

| 항목 | 내용 |
|------|------|
| 소스 | `results/mimic_suit_gait_sym` |
| 보관일 | 2026-09-14 |
| Epoch | 8816 (v31 epoch 6440에서 warm start, 약 2400 epoch 추가) |
| 로봇 | `skeleton_torque_suit_passive_cable` (27 DOF) |
| 시뮬레이터 | Newton, 평지, 물리 120 Hz / decimation 6 (env 20 Hz), 4096 env, batch 16384 |
| 모션 | 보행 계열 `skeleton_torque_suit_gait.pt` (walk 0.75 + run 0.25) |
| 레시피 | `results/mimic_suit_gait_sym/scripts/tracker_gait_sym.py`(2026-10-26 보관) — mimic 보상 + **`dof_pos_rew`(관절각 추종, weight 0.5, coefficient −15)** |
| 결과 | 성공률 1.0, 몸체 오차 **6.8 cm**(v31 8.4), 관절 오차 **8.6°**(v31 9.8) |
| **걸음 대칭** | 고관절 가동범위 좌우 차이 walk **3.1°** (v31 21.9°), run 12.7° (v31 14.7°) — mocap 자체 차이는 4.7° |
| hip mocap 추종 | walk **6.3°** (v31 9.8), run **6.0°** (v31 12.2) |
| 무보조 hip 파워 | walk 179 W, run 360 W (v31: 174 / 227 — 새 착용자가 mocap에 가깝게 크게 걷는다) |

## 왜 다시 학습했나
기존 mimic 보상(gt/gr/gv/gav/rh)은 **몸체가 어디 있는지**만 본다. 같은 자세를 다른 관절 조합으로
만들어도 성공이라 v31은 좌우 고관절 가동범위가 22° 벌어진 비대칭 보행을 했다(보조를 꺼도 그대로).
`dof_pos_rew`(관절각 exp MSE)를 같은 급 weight로 넣자 대칭이 mocap 수준으로 돌아왔고 관절·몸체
오차도 함께 좋아졌다. 계수는 시작 오차(MSE 0.063 rad²)에서 보상이 0.39가 되도록 −15로 잡았다
(−50은 지수 꼬리에 눌려 신호가 없었다).

## 사용
```bash
# 통합 시스템 A/B (walk = motion-id 0, run = 1)
python examples/experiments/assist_skeleton/run_system.py --human tracker \
    --tracker-checkpoint checkpoints/v32_newton_suit_gait_symmetric/last.ckpt \
    --llp-checkpoint <llp>/last.ckpt \
    --motion-file data/motion_for_trackers/skeleton_torque_suit_gait.pt --motion-id 0 --seconds 40
# 재학습
bash results/mimic_suit_gait_sym/scripts/train_tracker_gait_sym.sh
```

## 주의
v6 LLP는 **v31 착용자**로 학습됐다 — 이 착용자에서는 walk −16.7 %, run −19.3 %로 떨어진다
(v31에서 −27.0 / −31.7 %). 이 착용자를 쓰려면 LLP를 다시 학습해야 한다.
