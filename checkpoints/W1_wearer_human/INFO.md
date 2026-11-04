# W1 — 시뮬레이션 가상 착용자

W1은 전신 동작을 따라가는 동결 mimic tracker다. 실제 장치에서 사용하는 HLP·LLP와 별도다.
옛 이름은 `tracker v33`이며, 걷기·달리기 하나의 모델로 LLP 평가와 전신 학습의 착용자 역할을 한다.
모델 목록은 [버전표](../../docs/ai_sbc/AI_SBC_VERSIONS.md), 다음 작업은 [HANDOFF](../../docs/ai_sbc/HANDOFF.md)를 참조한다.

## 보관 정보

| 항목 | 내용 |
|---|---|
| 가중치 | `last.ckpt`, 학습 12,218회 시점 |
| 원본 실행 | `results/mimic_suit_human`, `snap_12218` |
| 보관일 | 2026-09-28 |
| 시작 모델 | 이전 v32의 학습 8,810회 체크포인트 |
| 로봇 | `skeleton_torque_suit_human`, 27 DOF |
| 고관절 한계 | 좌우 고관절 각 축의 effort limit 200 N·m |
| 학습 시간 | Newton 물리 120 Hz, decimation 6, 착용자 20 Hz |
| 통합 평가 시간 | Newton 물리 400 Hz, 착용자 20 Hz, 보조 100 Hz |
| 학습 모션 | `skeleton_torque_suit_gait.pt`, 걷기 0.75 / 달리기 0.25 |
| 학습 구성 | `experiment_config.py`, `resolved_configs*.pt/yaml`, `config.yaml` |
| 원래 실행 스크립트 | `examples/experiments/assist_skeleton/train_tracker_human.sh` |

## 사용 범위

기존 걷기·달리기 기록(`skeleton_torque_suit_gait.pt` motion 0·1)에서 LLP를 평가하고 전신 학습할 때의 가상 착용자다.
기본 조합은 W1 + H1 + L18이다(`run_system.py` 기본값). 2026-10-01의 첫 기준 보고서는
[보관 자료](../../docs/ai_sbc/archive/reports/l1_w1_report.html)로 옮겼다.
고관절 힘 한계를 줄인 시뮬레이션 모델이라는 뜻이며, 실제 사람 전반을 검증한 모델은 아니다.

2026-10-13의 새 실제 보통·느린 보행 16조건에서는 15낙상,
1조건은 골반이 내려가는 중 클립 끝에 도달했다.
따라서 새 보행에서 보조 모델을 판정하는 시험대로는 아직 쓸 수 없다.
[착용자 추종 보고서](../../docs/ai_sbc/walking_retarget_report.html)에서 데이터 표현과 추종 한계를 확인한다.

## 실행

저장소 루트의 `sbc` 환경에서 실행한다.

```bash
# W1 + H1 + L18의 기존 걷기 비교 (달리기는 --motion-id 1)
python examples/experiments/assist_skeleton/run_system.py --seconds 40

# 새 실제 보행의 무보조 추종만 확인. 출력은 새 폴더를 사용한다.
python examples/experiments/assist_skeleton/evaluate.py native-wearer \
    --tracker-checkpoint checkpoints/W1_wearer_human/last.ckpt \
    --motion-file results/real_walk_retarget_20261012_224957/native_pairs_v1/suit_native_walk.pt \
    --starts 0 0.5 --max-seconds 8 --out results/native_walk_check
```

로봇 설정은 체크포인트의 `resolved_configs_inference.pt`에서 읽는다.
기존 실험 수치·학습 동기·초기 비교표는 [보관 기록](../../docs/ai_sbc/archive/20261013/W1_INFO.md)에 보존했다.
새 착용자 학습은 현재 데이터 표현과 발 접촉을 확인한 뒤 별도 판단한다.
