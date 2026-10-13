# Checkpoint Info — 가상 착용자 **W1** (옛 이름 v33; mimic tracker, 사람 범위 근력)

버전 체계(2026-09-29): H = HLP, L = LLP, W = 가상 착용자, 현재 계보는 1부터. `docs/AI_SBC_VERSIONS.md` §0.

| 항목 | 내용 |
|------|------|
| 소스 | `results/mimic_suit_human` (snap_12218 = last.ckpt @ epoch 12,218), unit `aisbc-tracker-v33` |
| 보관일 | 2026-09-28 |
| Epoch | 12,218 (v32 epoch 8,810에서 warm start, 3,400 epoch 추가; 11,200 이후 평탄) |
| 로봇 | **`skeleton_torque_suit_human`** (27 DOF) — passive_cable과 같되 **hip 3관절 effort_limit 600 → 200 N·m** |
| 시뮬레이터 | Newton, 평지, 물리 120 Hz / decimation 6 (env 20 Hz), 4096 env, batch 16384 |
| 모션 | 보행 계열 `skeleton_torque_suit_gait.pt` (walk 0.75 + run 0.25) |
| 레시피 | `examples/experiments/assist_skeleton/tracker_human.py` — v32 레시피 + `dof_pos_rew` 0.5 → **1.0**, `pow_rew` −1e-5 → **−1e-4**, `torque_sq_rew` **−1e-6** 추가 |
| 학습 지표 | 성공률 1.0, 몸체 오차 0.060 m (v32 0.068), 관절각 추종 보상 0.565 → 0.666, 전신 파워 1,324 → 658 W |

## 왜 다시 학습했나
v32(hip 한계 600)는 PD 목표각을 토크 손잡이로 써서 착지마다 −257~−332 N·m(2.8~3.7 N·m/kg)를 냈다 —
문헌 보행 hip 모멘트 피크는 약 1 N·m/kg. 또 무보조 hip 스윙이 54.5°로 mocap(46°)보다 8° 커서 보조를 켜면
정책과 무관하게 |ω|가 12 % 줄었다(착용자 산물). LLP의 τ² 절감률이 "사람보다 센 착용자" 기준으로 재어지고
진자 학습 분포(하중 ≤120 N·m)와도 어긋났다. v33은 시험대를 사람 쪽으로 옮긴 것이다.

## 착용자 잣대 (`eval_wearer.py`, 무보조 env A, 40 s)

| | hip 스윙 A / mocap | hip 파워 | 토크 RMS | 스트라이드 신전 피크 | 주기 평균 신전 / 굴곡 | 피크/RMS | 몸체 오차 |
|---|---|---|---|---|---|---|---|
| v32 걷기 | 54.5° / 46.0° | 179 W | 75 N·m | −257 N·m (2.8 N·m/kg) | −152 / +65 | 3.4 | 0.069 m |
| **v33 걷기** | **48.1° / 46.0°** | **77 W** | **39 N·m** | **−150 N·m (1.65)** | **−86 / +34** | 3.9 | **0.060 m** |
| v32 달리기 | 65.3° / 58.9° | 360 W | 83 | −180 (2.0) | −126 / +86 | 2.2 | 0.064 |
| **v33 달리기** | **57.2° / 58.9°** | **170 W** | **56** | **−127 (1.4)** | **−112 / +76** | 2.3 | **0.061** |

주기 평균 신전 피크 0.95 N·m/kg = 문헌 값. 스윙은 mocap ±2° 안. 피크/RMS 3.9는 200 N·m 안에서 여전히
슬램이 있다는 뜻(토크² 벌점 −1e-6은 이를 못 눌렀다) — 남은 과제.

## 이 착용자에서 LLP는 어떻게 되나 (같은 실행의 env B)

| v33 위에 | 걷기 τ² | 달리기 τ² | 몸체 오차 | 보조 RMS |
|---|---|---|---|---|
| L1 (옛 llp_v13) 그대로 (40 N·m) | −0.1 % | **4.4 s 낙상** | — | 33 / 35 N·m |
| **L1 × 0.5** (`--assist-scale 0.5`) | **−19.3 %** | **−23.9 %** | −1.6 / −0.3 % | 17 / 19 |
| β 오라클 0.5 (특권 상한) | −35.1 % | −40.8 % | −1.9 / −0.1 % | 14 / 20 |

**40 N·m 그대로는 쓰지 말 것.** 사람 범위 착용자에겐 ~20 N·m가 맞고, 정책은 착용자 근력을 관측하지 못해 크기를
스스로 줄이지 못한다(`docs/NEXT_SESSION.md` §2-H). β와의 차이는 유각 말기(70~90 %) 브레이크 오방향.

## 사용
```bash
# 통합 시스템 A/B (walk = motion-id 0, run = 1). 영점은 착용자 스트림에서 잰다(--calibration wearer).
python examples/experiments/assist_skeleton/run_system.py --human tracker \
    --tracker-checkpoint checkpoints/W1_wearer_human/last.ckpt \
    --llp-checkpoint checkpoints/L1_llp_pendulum/last.ckpt --llp-dt 0.01 --assist-scale 0.5 \
    --motion-file data/motion_for_trackers/skeleton_torque_suit_gait.pt --motion-id 0 --seconds 40 \
    --calibration wearer
python examples/experiments/assist_skeleton/eval_wearer.py results/oracle/<run>   # 착용자 잣대
# 재학습
bash examples/experiments/assist_skeleton/train_tracker_human.sh
```
`run_system.py`·`llp_walk.py`는 로봇 config를 이 체크포인트의 `resolved_configs_inference.pt`에서 읽으므로
`--robot-name`을 따로 줄 필요가 없다. 뷰어/영상은 `skeleton_torque_suit_human`이 메시 표에 등록돼 있다.
