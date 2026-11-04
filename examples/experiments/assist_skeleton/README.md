# AI-SBC 전신 보조 평가와 실행

현재 작업은 [HANDOFF](../../../docs/ai_sbc/HANDOFF.md), 결과는 [보고서 목록](../../../docs/ai_sbc/reports.html),
모델 경로는 [버전표](../../../docs/ai_sbc/AI_SBC_VERSIONS.md)를 기준으로 한다.
이 문서는 실행법과 파일 역할만 다룬다. 초기 실험 일지는 [보관본](../../../docs/ai_sbc/archive/20261013/assist_skeleton_README.md)에 있다.

## 기본 비교 기준

`run_system.py`는 같은 초기 상태의 무보조 A와 보조 B를 Newton에서 비교한다.
기본값은 아래와 같다. 모든 명령은 저장소 루트의 `sbc` 환경에서 실행한다.

| 항목 | 기본값 |
|---|---|
| 가상 착용자 | W1, `checkpoints/W1_wearer_human/last.ckpt` |
| 각도 예측기 | H1, `checkpoints/H1_maniflow_swap/epoch_0008.ckpt` (flexion100-h15-swap의 EMA 가중치 사본) |
| 보조 제어기 | L18, `checkpoints/L18_tf_fpo_scratch_walkrun/epoch_1200.ckpt` (Transformer+FPO++), 출력 배율 1 |
| 동작 | `data/motion_for_trackers/skeleton_torque_suit_gait.pt`, 걷기 0번 (달리기 1번) |
| 시간 | 물리 400 Hz, 착용자 20 Hz, 보조 100 Hz, 평가 40초 |
| 예측 | 입력 100 Hz, 현재부터 50 ms까지 6점, LLP가 쓰는 미래 창은 0/10/20 ms |
| 영점 | 첫 1초의 무보조 착용자 관측 평균 (`--calibration wearer`) |

보관한 LLP(L15–L18)는 모두 20·tanh 출력·한계 ±20 N·m로 학습했으므로 출력 배율 1(`--assist-scale` 기본값)로 쓴다.
`--torque-limit`은 배율 적용 전 제어기 한계다. 두 값을 혼동하지 않는다.
다른 LLP는 `--llp-checkpoint checkpoints/<버전>/epoch_N.ckpt`로 고른다(같은 폴더에 `resolved_configs_inference.pt` 필요).

## 실행

```bash
# 기본 W1 + H1 + L18, 기존 걷기 40초. 기본은 headless.
python examples/experiments/assist_skeleton/run_system.py

# 같은 기준의 달리기
python examples/experiments/assist_skeleton/run_system.py --motion-id 1

# 화면 / 영상 (그래픽 디스플레이 필요)
python examples/experiments/assist_skeleton/run_system.py --viewer --record

# 문서용 영상: 착용자를 따라가는 궤도 카메라 (방위각 0 = 정면, 270 = 오른쪽 옆, 315 = 오른쪽 앞 대각선), 토크 화살표 없이
python examples/experiments/assist_skeleton/run_system.py --viewer --record --no-torque-arrows \
    --camera-azimuth 315 --camera-distance 3.4 --camera-height 0.5 --camera-look-height 0

# 새 보행의 착용자 추종만 검사: 짧은 클립을 이어 붙이지 않는 전용 평가기
python examples/experiments/assist_skeleton/evaluate.py native-wearer \
    --motion-file results/real_walk_retarget_20261012_224957/native_pairs_v1/suit_native_walk.pt \
    --starts 0 0.5 --max-seconds 8 --out results/native_walk_check
```

전용 평가기의 출력 폴더는 실행마다 새 경로를 사용한다. 기존 결과를 덮어쓰지 않는다.
새 실제 보행은 W1의 무보조 추종부터 확인한다. 이 단계를 통과하기 전에는 보조 성능을 판정하지 않는다.

다른 보관 모델이나 연구 후보를 비교할 때는 체크포인트·동작·seed를 명시한다. 예:

```bash
python examples/experiments/assist_skeleton/run_system.py \
    --llp-checkpoint checkpoints/L17_mlp_ppo_scratch_walkrun/epoch_200.ckpt \
    --motion-id 1 --seconds 40 --seed 7
```

이 명령은 기존 달리기 기록의 평가다. 범용 보행이나 실기기 성능을 입증하지 않는다.
골반 고정 진단은 `--human mocap_pd`로 선택한다. 지면을 딛고 걷는 W1 평가와 별도 조건이다.
과거 결과를 재현하려면 당시 `meta.json`의 모델·모션·영점·배율·시간 옵션을 모두 명시한다.
이전 기본값(v3, 초기 착용자, 단일 walk 파일, motion 영점, 30초)에 의존한 명령은 그대로 재현되지 않는다.

## 출력과 해석

`results/assist_skeleton/<시각>_<착용자>[_태그]/`에 `meta.json`, `traces.npz`,
`metrics.json`, `report.md`, `plots.png`를 저장한다. `--record`는 영상을 추가한다.
`meta.json`에 실제 사용한 모델·시간·배율·토크 한계·실행 명령이 남는다.

토크 제곱 평균, 몸체 추종 오차, 전진 속도, 낙상, 다른 관절의 부담을 함께 본다.
토크 지표는 시뮬레이션 부담이며 실제 근활성·대사량이 아니다.
HLP 계산 시간은 별도 기록한다. `--hlp-latency-ms 0`은 계산 지연을 물리 시간에 추가하지 않는 설정이다.
`--oracle`과 `--mask-window`는 참조·착용자 토크를 사용하는 원인 진단용이다.

`--swing-gate 0.20`은 관절각만 보는 유각 브레이크 게이트다 (2026-10-14, `protomotions/assist/controller.py`).
각 hip 관절각이 자기 중간값(지수이동평균)을 굴곡 방향으로 지나는 순간부터 같은 hip의 직전 교차 간격 × 0.20
동안 신전 보조를 0으로 둔다. 걷기에서 유각 중·후반(참조 위상 68–92 %)의 다리 감속은 무릎 결합·중력이 내고
착용자 hip 토크는 ≈0인데, 진자에서 배운 정책이 감속을 보고 브레이크를 걸어 착용자가 같은 크기로 맞서던 구간을
끊는 진단이다(2026-10-14, 지금은 지운 진자 모델로 측정: [보관 보고서](../../../docs/ai_sbc/archive/reports/walking_swing_gate.html)).
손으로 만든 위상 규칙이라 **진단 기준값**이지 배포 후보가 아니다.

```bash
python examples/experiments/assist_skeleton/run_system.py --swing-gate 0.20                 # 걷기
python examples/experiments/assist_skeleton/run_system.py --swing-gate 0.20 --motion-id 1   # 달리기
```

## 파일 역할

| 파일 | 역할 |
|---|---|
| `run_system.py` | 통합 비교 실행. `SystemRunner`(조립→제어기→훅→루프→기록), `SystemHook`(10 ms 서브스텝 훅), `MocapGaitPhase`(오라클용 특권 위상표), `FullBodyDiagnostic`(`--fullbody-diagnostic`) |
| `analysis.py` | 기록의 지표 계산 (numpy만). 실행 직후 지표·그림·report.md와 `RunRecord`·`ObjectiveComparison`·`TorqueAlignment`·`PhaseScreen`·`DecelerationMasks`·`WalkingPhaseDigest`·`SwingGateBins` 등 |
| `evaluate.py` | 시뮬레이션을 돌리는 평가: `motions`(모션별 A/B), `native-wearer`(새 보행의 무보조 착용자), `wearer`(착용자 사람다움 표) |
| `report.py` | HTML 보고서: `comparison`, `checkpoint`, `torque`, `finetune`(추가학습 전후·모델 비교: 보행 진행률 구간별 감소량 막대·꺾은선, `--figures`로 발표용 .svg/.png), `swing-gate`. 생성 뒤 `python docs/ai_sbc/organize_reports.py` |
| `plots.py` | 문서용 PNG: `gait-symmetry`, `timing`, `torque-phase`, `compare`(터미널 표 + 오라클 그림) |
| `viz.py` | Newton 뷰어 오버레이 (A 고스트 라인, 보조 토크 화살표) |
| `tracker_human.py`, `train_tracker_human.sh` | W1 착용자 학습 구성·실행 |
| `llp_walk.py` | W1 + H1 전신 시뮬레이션에서 LLP 학습: 진자 정책 추가학습(`--initial-actor-checkpoint`, `--finetune-assist-limit 20`, `--motion-weights 0.5,0.5`, motion별 `--hlp-calibration`), 같은 조건으로 처음부터 학습(`--actor-init random`), 정책 선택 `--policy mlp_ppo|mlp_fpo|tfm_fpo`(FPO++·Temporal Transformer, `protomotions/agents/fpo/`). 옛 L13/L14 변형은 `results/llp_L13_fullbody_walk_20261011/scripts/`에 보관 |

보행 클립 부분집합은 `data/scripts/make_gait_motions.py`로 만든다. 공통 제어기는 `protomotions/assist/`,
ManiFlow 연결은 `protomotions/maniflow/`에 있다. 진자 쪽 학습·평가는 `../assist_pendulum/`
(`mlp.py` 프리셋, `runtime.py`, `evaluate.py`, `study.py`, `report.py`).
구체적인 코드 흐름과 주의점은 [프레임워크 문서](../../../docs/ai_sbc/AI_SBC_FRAMEWORK.md)를 참조한다.
닫힌 연구의 실행기는 `results/<연구>/scripts/`에 있고(각 `SNAPSHOT.md`), 그대로 다시 시작하지 않는다.
