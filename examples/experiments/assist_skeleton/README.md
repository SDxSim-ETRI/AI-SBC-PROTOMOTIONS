# HLP + LLP 통합 시스템 — 스켈레톤 착용자 (Newton)

ManiFlow flexion 예측기(HLP)와 assist torque RL 정책(LLP)을 **하나의 제어기**로 묶어
인체 스켈레톤(`skeleton_torque_suit`, 슈트 관절 4개는 토크 0)의 고관절에 보조
토크를 전달하는 폐루프 시스템. 2026-09-12 구축.

```
      스켈레톤 상태 (Newton, 물리 400 Hz)
          │ 10 ms마다 (서브스텝 훅)
          ▼
   hip_flexion_r/l 각·각속도 + trunk pitch/roll  ──▶  AssistController  ──▶  τ_assist (2ch)
                                                      ├ HLP: 링버퍼→25 ms 재표집→ManiFlow→chunk(7 knot, 0~150 ms)
                                                      └ LLP: 히스토리·미래 창·이전 action → actor → tanh×25 N·m
          ▲                                                                        │
          └──────────── qfrc_applied 주입 (JointTorqueOverride, hip flexion) ◀──────┘
   착용자 대역: mocap PD(골반 고정) 또는 mimic tracker(보행), env 20 Hz
```

## 구성 파일

| 파일 | 역할 |
|------|------|
| `protomotions/assist/controller.py` | **`AssistController`** — env 비의존 제어기. 입력 관절각·각속도·몸통자세 스트림(10 ms), 출력 assist 토크. 실기기/시뮬 공용 |
| `protomotions/assist/llp_policy.py` | `LLPPolicy` — `assist_pendulum` 체크포인트에서 actor + 관측 레이아웃만 복원 (env/Fabric 없이) |
| `protomotions/assist/skeleton_features.py` | 스켈레톤 상태 → HLP 관측 4ch (학습 데이터 추출기와 동일 규약) + 영점 보정 |
| `protomotions/assist/verify_controller.py` | 진자 학습 env와의 step 단위 등가성 검증 (`python -m protomotions.assist.verify_controller`) |
| `protomotions/simulator/newton/substep_hook.py` | `SubstepHookMixin` — env step(50 ms) 안의 서브스텝 루프에서 10 ms마다 콜백 (LLP 실행 + qfrc 갱신 + 창 단위 PD 토크 평균) |
| `run_system.py` | 폐루프 러너 (A: 보조 없음 / B: 보조, paired) → traces·metrics·report·plots·mp4 |
| `analysis.py` | 지표·그림·리포트 |
| `viz.py` | 뷰어 오버레이 (A 고스트 라인, B 메시 + assist 화살표) |
| `train_tracker_walk.sh`, `train_tracker_motions15.sh` | 착용자 대역(mimic tracker) 재학습 |

## 실행 (sbc env)

```bash
# 1) 독립 제어기 ↔ 진자 env 등가성 (한 번 확인) — PASS 기대
python -m protomotions.assist.verify_controller --checkpoint results/assist_pendulum_v3/last.ckpt

# 2) 골반 고정 대역 (tracker 불필요). 기본: 착용자 PD를 진자 학습 플랜트와 같은 동역학 영역으로 (×12.4/×6.2)
python examples/experiments/assist_skeleton/run_system.py --human mocap_pd --seconds 30
#    강체 다리(무릎·발목 잠금) 진단 / assist 배율 스윕
python examples/experiments/assist_skeleton/run_system.py --human mocap_pd --lock-knee-ankle --assist-scale 3

# 3) 보행 착용자 (tracker 체크포인트 필요; train_tracker_walk.sh)
python examples/experiments/assist_skeleton/run_system.py --human tracker \
    --tracker-checkpoint results/mimic_suit_passive_walk/last.ckpt --seconds 30

# 4) 뷰어 / 녹화 (DISPLAY 필요; A=고스트 라인, B=메시, 초록 화살표=assist)
DISPLAY=:1 python examples/experiments/assist_skeleton/run_system.py --human mocap_pd --record
```

출력: `results/assist_skeleton/<timestamp>_<human>[_tag]/{report.md, metrics.json, plots.png, traces.npz, meta.json, sim.mp4}`.

## 설계 결정

- **시간 구조**: tracker는 env dt 50 ms(120 Hz/6)로 학습됐고 관측(미래 레퍼런스 프레임)이
  dt에 묶여 있어 env 주기를 못 바꾼다. LLP는 10 ms. 그래서 물리 400 Hz / decimation 20 /
  hook_every 4 로 두고 LLP를 서브스텝 루프 안에서 돌린다. 진자 학습도 물리 400 Hz였다.
- **부호 규약**: 데이터/HLP는 +θ = 굴곡, LLP(진자)는 +θ = 신전. `AssistController(joint_sign)`이
  양방향 변환 (skeleton +1, hip_pendulum −1). 등가성 검증에서 진자 env와 bit-exact 일치 확인.
- **HLP 영점 보정**: 재생 클립의 4ch 평균을 SEED 학습 데이터 평균(`SEED_CHANNEL_MEAN_DEG`)에
  맞추는 오프셋을 관측에 가산 (eval_abc `--align-ref-zarr`와 같은 방식). walk 클립 기준
  flexion +12.3°/+10.1°.
- **골반 고정 대역의 게인**: 로봇 config 게인(hip Kp 200/Kd 20)은 다리 관성(hip 기준
  3.53 kg·m²)에 비해 물러 ζ≈0.38로 12° 오버슈트 → HLP 분포 밖. 기본값은 진자 플랜트
  (I=0.43, Kp 300/Kd 15 → ω_n 26.5 rad/s, ζ 0.66)와 같은 영역이 되도록 Kp ×12.4, Kd ×6.2
  (추종 오차 0.78°).
- **CUDA graph 끔**: 콜백이 서브스텝 사이에 끼므로. 소수 env 평가용.

## 결과 (2026-09-12, walk 클립 30 s, LLP v3 그대로)

착용자 hip flexion 파워 |τ·ω| 합 (A 대비 B), `assist_vs_agentA_corr` = assist 토크와 착용자
단독 토크의 상관 (방향·타이밍 일치도).

| 착용자 대역 | assist 배율 | 착용자 파워 A→B [W] | 절감 | corr |
|---|---|---|---|---|
| 골반 고정, 관절 다리 (무릎·발목 mocap 추종) | 1 | 117.6 → 112.3 | **−4.5 %** | 0.19 |
| 〃 | 3 | 116.6 → 111.9 | −4.0 % (토크 RMS +3 %) | 0.19 |
| 골반 고정, **강체 다리** (무릎·발목 잠금) | 1 | 169.3 → 157.6 | −7.0 % | **0.80** |
| 〃 | 2 | 169.3 → 146.5 | −13.5 % | 0.79 |
| 〃 | 3 | 169.3 → 136.5 | −19.4 % | 0.79 |
| 〃 | 4 | 169.3 → 127.4 | −24.8 % | 0.78 |
| 보행 tracker (학습 중, epoch ≈2600, 30 s) | 1 | 226.5 → 236.0 | +4.2 % | −0.11 |

해석:

1. **시스템은 닫힌다** — HLP 예측(150 ms MAE 6.3°, suit14 오프라인 5.7°와 같은 수준)이 chunk로
   들어가고 LLP가 10 ms마다 토크를 내며, 추론 시간은 HLP 7.9 ms + LLP 0.4 ms (batch 2).
2. **진자 LLP는 "강체 진자" 플랜트에는 모양이 맞는다** (corr 0.80, 배율에 따라 −7 → −25 %
   단조 절감). 크기만 부족하다: 스켈레톤 다리는 진자보다 관성이 8.3배라 같은 궤적에 8배
   토크가 필요한데 LLP 출력 한계는 25 N·m.
3. **무릎이 움직이면 맞지 않는다** (corr 0.19). 스윙 중 무릎 굴곡(≤65°)이 다리 관성을 바꾸고
   무릎 PD 반력이 hip 토크에 ±100 N·m 스파이크를 만든다 — 진자에는 없던 성분. 보행
   tracker에서는 지면 반력까지 더해져 corr ≈ 0.
   → LLP를 실제 플랜트(관절 다리 + 지면 접촉, 실제 토크 스케일)에서 재학습해야 한다는 뜻.
   관측 계약(hip-local)은 그대로 두고 플랜트·사람 모델만 바꾸면 된다 — 이 러너가 그 학습
   환경의 평가 하네스로 그대로 쓰인다.

tracker 항목은 학습 중간 체크포인트 기준이며 400 Hz 물리(학습 120 Hz)에서 30 s 낙상 없이
걸었다. epoch 1300→2600에서 tracker 자체 지표(자체 평가 성공률 1.0, 몸체 위치 오차 8.6→7.5 cm)는
좋아졌지만 **hip flexion의 mocap 추종은 10.8°→14.2°로 나빠졌고** HLP 150 ms MAE도 9.5°→11.0°
— mimic 보상은 몸체 위치를 맞추지 관절각을 맞추지 않으므로, hip 궤적이 사람 같지 않아도
성공으로 친다. 착용자 대역으로 쓰려면 관절각(특히 hip) 추종 항을 강화한 tracker가 필요하다.
(epoch 1300 결과: 167.1→173.3 W, +3.7 %, corr −0.07.) 학습이 끝나면 같은 명령으로 갱신할 것.

## LLP 재학습 — 보행 착용자 위에서 (2026-09-12 시작)

위 결과의 결론대로 LLP를 실제 플랜트에서 다시 학습한다. "따로 보행 비슷하게 만든 플랜트"
대신 **통합 시스템 그대로**를 학습 env로 쓴다: 착용자 = 동결 mimic tracker(v30), HLP = 실
ManiFlow(배치 예측), LLP만 학습. 관측 계약(hip-local)은 진자와 동일하므로 학습된 정책은
`AssistController`가 그대로 싣는다 (`flexion_sign`·보조 관절 목록은 체크포인트 config에서 읽음).

| 파일 | 역할 |
|------|------|
| `protomotions/envs/base_env/skeleton_assist_env.py` | `SkeletonAssistEnv` — env 10 ms, policy action 2ch(hip), 착용자 PD 목표는 tracker가 5 step마다 갱신, τ_agent = hip substep 평균 PD 토크 |
| `protomotions/envs/control/skeleton_assist_control.py` | `TrackerPolicy`(동결 tracker actor + PD 목표 변환), `TrackerMimicControl`(미래 레퍼런스 +50 ms), `SkeletonAssistControl`(실 HLP chunk, θ_g = mocap hip, hip 부분집합 컨텍스트) |
| `llp_walk.py` | 실험 파일. tracker config에서 관측(`max_coords_obs`, `mimic_target_poses`)·종료(`tracking_error`)·motion manager·보상(`gt_rew`, `rh_rew`) 객체를 그대로 가져옴 |
| `check_train_env.py` | 학습 env 점검: 내장 tracker가 걷는지(몸체 오차), 신호 스케일, HLP 오차 |

```bash
python protomotions/train_agent.py --robot-name skeleton_torque_suit_passive_cable --simulator newton \
    --experiment-path examples/experiments/assist_skeleton/llp_walk.py --experiment-name llp_skel_walk_v1 \
    --motion-file data/motion_for_trackers/skeleton_torque_suit_walk.pt --num-envs 4096 --batch-size 16384 \
    --tracker-checkpoint checkpoints/v30_newton_suit_passive_walk/last.ckpt   # [--assist-limit 25 --human-power-weight -0.0015]
python examples/experiments/assist_skeleton/check_train_env.py --run results/llp_skel_walk_v1 --num-envs 16 --steps 800
# 평가 (통합 시스템, 새 LLP)
python examples/experiments/assist_skeleton/run_system.py --human tracker --llp-checkpoint results/llp_skel_walk_v1/last.ckpt --seconds 30
```

설계 결정:

- **보상**: 진자의 "hip 목표 추종" 항을 **착용자 몸체 추종**(tracker 자신의 `gt_rew` 0.5·`rh_rew` 0.2)으로
  바꿨다. 보행 착용자의 의도를 hip 각도 하나로 정의할 수 없어서다 — tracker의 PD 목표는 실제 각도와
  15~20° 차이 나는 토크 명령이고, mocap hip 각도는 tracker 자체가 16° 벗어나 따라간다. "보조가
  보행 과제를 깨뜨리지 않는다"를 직접 재는 편이 맞다. 파워 항은 −0.0015 (보행 hip 파워 240 W →
  −0.36/step, 몸체 추종 보상 ≈ +0.4/step와 균형). θ_g = mocap hip flexion은 평가·critic 관측에만 쓴다.
- **시간 구조**: 학습 env 주기 10 ms(LLP), tracker는 5 step마다. tracker 관측은 tracker config의
  MdpComponent 객체를 그대로 env에 넣어 매 step 계산하고(정책은 5 step마다 소비), 미래 레퍼런스만
  `TrackerMimicControl`이 +50 ms로 만든다. 이전 action은 tracker 자체 버퍼(두 결정 전 action —
  state history 규약과 동일).
- **점검 결과 (assist 0, 16 env × 8 s)**: 종료 0, 몸체 오차 7.9 cm(단독 tracker 7.6 cm) → 내장 tracker가
  400 Hz 물리에서 정상 보행. τ_agent RMS 79 N·m, hip 파워 240 W, HLP 150 ms MAE 10.6°.
- **속도**: 4096 env에서 ≈3 s/epoch (실 HLP 배치 4096 예측 포함).
- **v1 → v2 (22:57)**: v1(tracker의 gt/rh 추종 보상, 파워 −0.0015)은 850 epoch에서 assist를 한계(RMS 23/25 N·m)까지
  써서 착용자 몸체 추종을 0.86→0.94로 "개선"하는 해에 머물렀고 파워는 그대로였다(통합 평가 −0.8 %, corr −0.04).
  추종을 좋게 만드는 데 보상을 주면 최대 토크로 착용자를 mocap 쪽으로 끌어당기는 것이 이득이기 때문.
  v2는 몸체 오차가 무보조 수준(0.09 m)을 **넘는 만큼만 벌점**(hinge, weight 10) + 파워 −0.003 + effort −5e-5.
  v1은 1270 epoch에서 중단(평탄), v2는 계속.
- **오라클 상한 (23:45, `check_train_env.py --oracle-beta`)**: τ_assist = β·τ_agent(직전 step 계측, ±25 N·m clamp)를
  넣으면 착용자 hip 파워 240 W → β 0.3: 208.5 W (−13 %), β 0.6: 197.8 W (−18 %), 몸체 오차 7.9→8.2 cm.
  즉 **25 N·m 한계에서는 완벽한 위상의 보조라도 −15~−20 %가 상한**이다 (β 실험의 −31 %는 한계 없이).
  40 N·m 한계에서는 β 0.3: −14 %, 0.6: −22 %, 1.0: −26 % (178 W) — 한계를 올려도 tracker가 적응해 상한은 완만히 오른다.
  → v3 (23:55 시작): hinge 목적함수 + `--assist-limit 40`. v2(25 N·m)와 비교.

### 중간 결과 (2026-09-13 01:30, 학습 계속 중 — 스냅샷 평가, walk 30 s, 통합 시스템 `run_system.py --human tracker`)

| LLP | 한계 | epoch | 착용자 파워 A→B [W] | 절감 | corr | assist RMS | hip mocap 오차 A→B |
|---|---|---|---|---|---|---|---|
| 진자 v3 (참고) | 25 | — | 225.5 → 236.0 | +4.2 % | −0.11 | 12.7 | 14.2 → 15.0° |
| v1 (추종 보상) | 25 | 840 | 225.6 → 223.7 | −0.8 % | −0.04 | 23.1 | 17.0 → 15.8° |
| **v2 (hinge)** | 25 | 1270 | 225.5 → 180.5 | **−20.0 %** | 0.54 | 19.9 | 17.0 → 18.3° |
| **v3 (hinge)** | 40 | 900 | 225.7 → 164.7 | −27.0 % | 0.53 | 27.4 | 17.0 → 18.4° |
| **v3 (hinge)** | 40 | 1890 | 225.7 → 156.1 | **−30.9 %** | 0.52 | 30.4 | 17.0 → 17.7° |
| 오라클 β0.6 (참고) | 25 / 40 | — | 240 → 198 / 188 | −18 % / −22 % | | | |

- 통합 env에서 학습한 LLP는 **오라클 상한 수준 이상**의 절감을 낙상 없이 낸다 (v2 −20 % vs 25 N·m 오라클
  −18 %; v3 −31 % vs 40 N·m 오라클 β1.0 −26 % — 오라클은 직전 step 토크의 배수라 위상이 한 step 늦고
  tracker 적응을 이용하지 못한다). 진자 LLP(+4 %)와 대비된다. v3는 다른 클립 구간·시드(시작 30 s,
  seed 7, epoch 900)에서도 −27.8 %로 일관. v2는 평가 파워 ≈196 W에서 평탄해 2295 epoch에서 중단.
- **v4 (03:30 시작, `results/llp_skel_motions15_v4/`)**: 15모션 + v31 tracker, hinge, 40 N·m — 일반화.
- 토크 패턴은 보행 위상에 맞춘 거의 bang-bang(±한계)이다 — 파워 목적함수만으로는 부드러움을 요구하지
  않으므로. 실기기용이라면 effort/action-rate 벌점을 키워 재학습하거나 출력에 저역통과를 두어야 한다.
- 착용자의 hip mocap 추종은 1.3° 나빠졌고 몸체 추종은 hinge 범위(≈8 cm) 안에 머물렀다(학습 중 벌점 ≈0).
- **train_agent.py 함정**: 스크립트 디렉터리 `protomotions/`가 sys.path[0]이라 `import maniflow`가
  `protomotions/maniflow`(통합 레이어)로 잡혀 ManiFlow 로딩이 깨졌다 — `loader.py`가 진짜 namespace
  패키지를 `sys.modules`에 명시 등록하도록 고쳤다.
