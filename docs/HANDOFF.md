# AI-SBC 인수인계 — 처음 오는 에이전트가 먼저 읽을 문서 (2026-10-01 기준)

이 레포를 처음 보는 사람·에이전트가 **지금 어디까지 왔고, 무엇이 확정됐고, 무엇이 열려 있는지**를
한 번에 파악하도록 쓴 요약이다. 세부 수치와 판단 근거는 아래 문서로 내려간다.
기존 문서·커밋·run 이름에 남아 있는 옛 이름은 **v13 → L1, v33 → W1, v32 → W0**으로 읽는다.

읽는 순서:

| 순서 | 문서 | 내용 |
|---|---|---|
| 1 | 이 문서 | 전체 상황 요약 |
| 2 | [`AI_SBC_VERSIONS.md`](AI_SBC_VERSIONS.md) (브라우저용 [`versions.html`](versions.html)) | 버전 체계·용어·지금 쓸 것 |
| 3 | [`NEXT_SESSION.md`](NEXT_SESSION.md) | 실험 일지(시간순). 각 실험의 수치·판단·중단 이유 |
| 4 | [`AI_SBC_FRAMEWORK.md`](AI_SBC_FRAMEWORK.md) | 설계 원리, 파일 지도(§4), 함정 모음(§8), 코드 읽기 체크리스트(§9) |
| 5 | [`../examples/experiments/assist_skeleton/README.md`](../examples/experiments/assist_skeleton/README.md) | 통합 시스템(스켈레톤 A/B) 사용법 |
| 6 | [`../CLAUDE.md`](../CLAUDE.md) | 명령·코드 규약·gotcha |
| 7 | `../checkpoints/*/INFO.md` | 보관본별 학습 설정·결과·사용 명령 |

---

## 1. 무엇을 만드는가

고관절 보조 장치(AI-SBC)의 제어기. **실기기에는 두 모델만 올라간다.**

| 구성 | 하는 일 | 학습 | 위치 |
|---|---|---|---|
| **HLP** (High-Level Predictor) | 지난 100 ms의 고관절 각도·몸통 자세로 **앞으로 0~50 ms의 고관절 굴곡각**(10 ms 격자 6점)을 예측 | 사람 모션 데이터, 지도학습(ManiFlow) | sibling 레포 `~/Projects/ManiFlow_Policy`, 이 레포에서는 `protomotions/maniflow/`로만 접근 |
| **LLP** (Low-Level Policy) | HLP 예측 + hip 각도·각속도 히스토리로 **매 10 ms 좌우 hip 굴곡 보조 토크**를 냄 | 시뮬레이션 PPO | `checkpoints/L*`, 학습 코드 `examples/experiments/assist_pendulum/` |
| **가상 착용자** (W, mimic tracker) | 시뮬 안에서 사람 역할. mocap을 따라 걷고 뛴다. 실기기에서는 진짜 사람 | 시뮬레이션 PPO | `checkpoints/W*`, 학습 코드 `examples/experiments/assist_skeleton/tracker_*.py` |

사용자가 못 박은 목표(원문): **"assist torque를 출력하는 RL모델이 어떤 input이 들어오던간에 올바른 torque assist를
해주는게 목표"**. 특정 착용자·특정 동작에 맞춘 정책이 아니라, 입력이 무엇이든 올바른 보조를 내는 정책이다.
그래서 LLP는 hip-local 최소 관측(각도·각속도·예측)만 쓰고, 착용자나 보행 데이터를 학습에서 보지 않는 쪽으로 왔다.

고정된 범위: **보행 계열(걷기·달리기)만**. 인체 모델은 해부학적 뼈 스켈레톤(`skeleton_torque_suit.xml`).
착용자 특이성(여러 사람 대응)은 뒤로 미룸.

---

## 2. 지금 결론 (2026-10-01)

**배포·평가 후보 = H1 + L1 × 0.5 @ W1.** 걷기 τ² −19.3 %, 달리기 −23.9 %, 낙상 0, 보조 RMS 17 / 19 N·m.
(τ² = 착용자 hip 토크 제곱의 합, 근활성 프록시. 무보조 A 대비 보조 B의 변화. §5 참고.)

| W1 위에서 (40 s A/B) | 걷기 τ² | 달리기 τ² | 비고 |
|---|---|---|---|
| **H1 + L1 × 0.5** | **−19.3 %** | **−23.9 %** | 현재 후보. `--assist-scale 0.5` ≈ 한계 20 N·m |
| H1 + L1 그대로 (40 N·m) | −0.1 % | 4.4 s 낙상 | 사람 범위 착용자에겐 과보조. **쓰지 말 것** |
| L2 (한계 U[10,50] 랜덤화 + 질량∝관성) | −16 % | −24 % | 포화 90~97 %(bang-bang), 구동기 불가 |
| L3 (+ 하중 절반, 반전 점성 브레이크) | −12.5 % | −23.7 % | |
| L4 (L3 + chunk를 실제 각도에 정박) | +0.9 % | −7.8 % | 학습할수록 악화(2k → 4k) |
| L5 (환경이 반전 관성을 대신 냄) | 무효 | 무효 | 정책 출력 ≈ 0 (RMS 7, 상관 0) |
| β 오라클 0.5 (특권 상한) | −35.1 % | −40.8 % | 착용자 토크의 절반을 그대로 내는 규칙. 정책이 아님 |
| 마스크 진단: L1×0.5 + 위상 70~100 % 보조 0 | −15.6 % | −30.3 % | 보행 위상을 아는 특권 진단. 정책이 아님 |

- 옛 개인화 정책(스켈레톤 위에서 추가 학습한 llp_v3~v12)은 τ²로 재면 **달리기를 한 번도 못 도왔다**(v10 +21, v11 +230, v12 +9 %).
- W0(옛 v32, 사람보다 1.7~3.7배 센 착용자) 위의 L1 −30.5 / −25.5 %는 **잣대에서 뺐다**. 사람 범위 착용자 W1이 잣대.
- 2026-09-29 밤 진자 변형 L2~L5 네 개를 돌렸고 모두 L1×0.5 아래 → 전부 중단. 지금 **GPU 비어 있음, systemd 유닛 없음**.
- **사용자 결정 대기**: (가) L1×0.5를 통합 결과로 확정·정리 vs (나) 달리기 유각 말기 시점만 겨냥한 진자 실험 — §7.
- 2026-09-30에 사용자가 "현재로선 L1 × 0.5가 가장 좋다"는 판단을 확인했다. 단 배율 0.5 자체는 최적값이 아니라 처음 시도한 값이다
  (이전 W1 스냅샷 걷기에서 0.3 < 0.5 한 번 확인, 0.4~0.7 미탐색).

---

## 3. 시스템 구조 (한 화면)

```
 스켈레톤 착용자 W1 (Newton, 물리 400 Hz)         ← 사람 역할. mocap 추종 PPO. hip 한계 200 N·m
      │ 10 ms마다 (env step 50 ms 안의 서브스텝 훅, protomotions/simulator/newton/substep_hook.py)
      ▼
 hip_flexion_r/l 각·각속도 + trunk pitch/roll ──▶ AssistController (protomotions/assist/controller.py)
                                                  ├ HLP H1: 과거 100 ms → 미래 0~50 ms, 10 ms 격자 6점 (3.2~4.9 ms)
                                                  └ LLP L1: 히스토리 250 ms + 미래 창 {0,10,20 ms} + 이전 action 100 ms → tanh × 40 N·m × 배율 (0.36 ms)
      ▲                                                                              │
      └────────── qfrc_applied 주입 (hip flexion r/l) ◀──────────────────────────────┘
 A env = 보조 0, B env = 보조 켬. 같은 mocap·같은 seed로 짝지어 40 s 돌리고 착용자 토크를 비교한다 (= 통합 평가).
```

| 주기 | HLP 10 ms | LLP 10 ms | 물리 2.5 ms (400 Hz, decimation 4) | tracker env 50 ms (관측이 묶여 있어 못 바꿈) |
|---|---|---|---|---|

**LLP가 보는 것**: hip 2ch 각도·각속도 + 히스토리 25 step + HLP 예측 창 {0, 10, 20 ms} + 이전 action 10 step. **못 보는 것**:
착용자 근력·관성·하중·보행 위상·모션 종류. 그래서 같은 mocap을 따르는 두 착용자(W0, W1)에게 같은 토크를 낸다.

**용어 (2026-09-29 사용자 지시)**
- **통합** = HLP·LLP를 각각 따로 학습한 뒤 그대로 붙여 평가·배포. `run_system.py`의 스켈레톤 A/B가 곧 통합 평가. 기본 목표.
- **개인화** = 통합 뒤 특정 착용자 위에서 LLP를 추가 학습(`SkeletonAssistEnv`, `llp_walk.py`; 옛 v3~v12). 지금은 안 씀.

**버전 체계 (2026-09-29)**: 구성 요소 접두어 + 번호, 현재 계보는 1부터. 조합은 "H1+L1 @ W1"처럼 쓴다.

| | 현재 | 옛 이름 | 위치 | 내용 |
|---|---|---|---|---|
| H | **H1** | flexion100-h15 | ManiFlow 레포 run, `run_system.py`가 자동 탐색 | 100 fps 예측기, denoise 1. H0 = flexion40(50 ms 격자, 폐기) |
| L | **L1** | llp_v13 = 진자 v4 @ epoch 6,000 | `checkpoints/L1_llp_pendulum/` | 랜덤화 진자 학습, 40 N·m, 10 ms 제어. L2~L5는 실패(results/에만) |
| W | **W1** | tracker v33 | `checkpoints/W1_wearer_human/` | 사람 범위 근력(hip 한계 200, 주기 피크 0.95 N·m/kg). W0 = v32(기준선용) |

**학습 플랜트 = 랜덤화 진자, 스켈레톤 = 시험장.** L1은 스켈레톤·mocap·걷기·달리기를 한 번도 보지 않았다. 진자(고정 골반 hip 1축)에
관성 ×1~25, 착용자 PD ω_n 12~24, 입각 사건(외부 하중 ≤120 N·m + 굳힘), 작은 섭동, 다양한 의도 궤적(sine + 고조파 + hold)을
섞고, HLP는 배포 형식(10 ms × 6점)으로 에뮬레이션했다. 비용 = 추종 + |τ·ω| + τ² + 보조 토크 절약 + action 변화.
세부: `checkpoints/L1_llp_pendulum/INFO.md`, `examples/experiments/assist_pendulum/mlp_v4.py` docstring.

---

## 4. 어떻게 여기까지 왔나 (타임라인)

| 날짜 | 일 | 결과 |
|---|---|---|
| ~09-08 | 진자 LLP v3 + 실 HLP 결합(Phase C) | 진자 안에서는 동작. 스켈레톤에 옮기면 방향이 안 맞음(상관 0.19) |
| 09-12 | 통합 시스템 구축: 독립 제어기 `protomotions/assist/`, 서브스텝 훅, 스켈레톤 A/B 러너 | 폐루프 동작. 착용자 tracker 재학습(v30 walk, v31 15모션) |
| 09-13~14 | **개인화** 시도: 스켈레톤 위에서 LLP 추가 학습 llp_v3~v6, 두 속도 구조(HLP 10 ms, LLP 5 ms) v10, 대칭 보행 착용자 W0(v32) | \|τ·ω\| 기준 −22~−32 %로 보였음 |
| 09-15 | **오라클 진단** | \|τ·ω\|는 각속도만 줄어도 내려감. τ²로 재면 v10은 +7 %. **목적함수가 틀렸었다.** β 오라클 상한 τ² −32 % |
| 09-16 | v11(τ² 항 + 한계 해제), v12(80 N·m) | 걷기는 되지만 **달리기는 아무도 못 도움**(v11 2 s 낙상, v12 +9 %) |
| 09-17~22 | 진자 단계 재점검, 착용자 토크 스파이크 분석 | 스파이크는 사람이 아니라 tracker + 강성 PD의 산물. 진자가 "밀기만 하는 그네"를 가르쳤음 |
| 09-22 | **진자 v4 = 랜덤화 플랜트** 설계·학습 시작 | 하중·관성·의도 다양화, 배포 HLP 형식 일치 |
| 09-23 | **전이 확인** | 스켈레톤을 본 적 없는 v4 epoch 6,000이 W0에서 걷기 τ² −30.5 / 달리기 −25.5 %. 달리기를 도운 첫 정책. 10k·14k는 물러남 |
| 09-28 | v4 epoch 6,000 보관(L1). v5(토크 절약 항 축소) | v5도 후기 하락 → 절약 항이 원인 아님. 중단 |
| 09-28 | **사람 범위 착용자 W1** 학습·보관(hip 한계 200) | W1에서 L1 그대로는 걷기 무효·달리기 낙상, **×0.5면 −19.3 / −23.9 %**. β −35 / −41 |
| 09-29 | 버전 체계 H/L/W, 용어 통합·개인화 정리 | |
| 09-29 밤 | 진자 변형 **L2~L5** 동시 학습·W1 평가, 유각 말기 마스크 진단 | 전부 L1×0.5 아래. 교훈 §5. 전부 중단 |
| 09-30 | (가)/(나) 선택지 설명, "현재 최선 = L1×0.5" 확인. HLP 축소 실험은 불필요로 취소 | 결정 대기 |

---

## 5. 확정된 사실·교훈 (다시 배우지 말 것)

1. **잣대는 착용자 hip 토크의 τ²**(근활성 프록시). \|τ·ω\|(파워)는 착용자 다리가 덜 흔들리기만 해도 내려가 속는다.
   항상 τ², 전진 거리, 몸체 추종 오차, 낙상을 **한 표에서** 본다(`compare_runs.py`). 재실행 산포는 1~2 %.
2. **비교는 A(무보조) vs B(보조)만.** mocap과 비교하지 않는다. 변화는 "두 값 그대로"나 절대 단위로 쓰고 "pt"는 쓰지 않는다(사용자 지시).
3. **정책은 착용자 근력을 못 본다.** 각도만 보므로 보조 크기는 착용자별 배율로 정한다. W1에서 0.5. 사람 범위에 40 N·m는 과보조.
4. **진자 정책의 쓸모는 "하중에 맞서 의도의 토크를 착용자 대신 내는 것"에서 온다.** 하중을 줄이거나(L3), 환경이 대신 내거나(L5),
   chunk에서 하중 신호를 감추면(L4) 학습 신호가 사라진다. **진자를 "쉽게" 만드는 방향은 다시 시도하지 말 것.**
5. **bang-bang(포화)은 필요량 > 한계의 결과**다. 한계를 랜덤화해도(L2) 안 바뀐다.
6. **유각 말기 브레이크**: L1은 의도 반전 전에 관성을 되돌리는 토크를 미리 낸다. 걷기에선 착지 직전(90~100 %) 도움, 달리기에선
   70~100 % 전체가 손해(막으면 −24 → −30). β와의 나머지 차이는 착지 구간 비례성(β 이득 −486 vs L1 −211, RMS는 비슷).
7. **β 오라클·마스크 진단은 특권 정보**(착용자 토크, 보행 위상)를 쓴다. 상한과 인과를 보는 도구지 배포 후보가 아니다.
8. **HLP 형식은 학습·배포가 같아야 한다.** 옛 형식(25 ms × 7점)으로 학습한 v4는 전이가 안 됐고, 배포 형식(10 ms × 6점)으로 맞춘 뒤 됐다.
9. **평가는 학습 제어 주기와 `--llp-dt`를 맞춘다**(L1 = 0.01). 영점은 `--calibration wearer`(착용자 스트림에서 측정).
10. **40 N·m는 스펙이 아니라 우리가 고른 값**(25 → 40). 실제 구동기의 피크·연속 토크·대역폭을 받으면 한계와 `--torque-lpf-hz`를 그 값으로.
    측정: 12 Hz 저역통과는 거의 무손실, 6 Hz −4~5, 3 Hz 붕괴.
11. **W0 수치는 비교 불가.** 사람보다 센 착용자가 과보조·오방향을 받아 준 결과. 모든 LLP 평가는 W1에서.
12. **Newton은 실행 간 bit-reproducible이 아니다.** 회귀 판정은 재실행 산포로.
13. 진자 v4 전용 세그폴트(매 step 게인 통지가 MuJoCo 상수 재계산)는 408f0b6으로 고쳤다. 스켈레톤 run 세그폴트는 원인 미확정(CUDA graph 1순위).
14. HLP H1(6블록 × 256, 10.5 M)은 이미 작다. 크기 축소 실험은 불필요로 취소(2026-09-30). ARDY는 HLP 후보에서 제외(2026-09-08).

---

## 6. 평가·학습 방법

모두 conda env `sbc` (`~/miniconda3/envs/sbc/bin/python`). ManiFlow **학습**만 env `maniflow`.
추론·평가에서는 `--robot-name`을 주지 않는다(로봇 config는 체크포인트의 `resolved_configs_inference.pt`에서 읽음).
음수 CLI 값은 `--flag=-5e-6` 형태로.

```bash
# 통합 평가 (= 스켈레톤 A/B). motion-id 0 = 걷기, 1 = 달리기. 결과는 results/oracle/<날짜>_tracker_<태그>_m<모션>/
python examples/experiments/assist_skeleton/run_system.py --human tracker \
    --tracker-checkpoint checkpoints/W1_wearer_human/last.ckpt \
    --llp-checkpoint checkpoints/L1_llp_pendulum/last.ckpt --llp-dt 0.01 --assist-scale 0.5 \
    --motion-file data/motion_for_trackers/skeleton_torque_suit_gait.pt --motion-id 0 --seconds 40 \
    --calibration wearer --out results/oracle --tag L1s05        # --record = mp4, --torque-lpf-hz 12 = 구동기 대역
# 여러 실행을 한 표에 (τ², |τ·ω|, 속도, 몸체 오차, 보조 RMS, 상관)
python examples/experiments/assist_skeleton/compare_runs.py results/oracle/*_tracker_v33e12218_*_m0
# 착용자 자체가 얼마나 사람 같은가 (스윙, 토크 피크 N·m/kg, 피크/RMS)
python examples/experiments/assist_skeleton/eval_wearer.py results/oracle/<run>
# 보고서: 여러 실행 비교 HTML / 체크포인트 하나 깊게 (위상표·3보 시계열·학습 곡선)
python examples/experiments/assist_skeleton/report_html.py --out docs/transfer_report.html --walk <runs...> --run <runs...>
python examples/experiments/assist_skeleton/report_checkpoint_html.py --out docs/llp_v13_report.html --walk <run> --run <run>
# 특권 탐침: β 오라클(상한), 위상 창 토크 강제, 정책 토크 위상 마스크
python examples/experiments/assist_skeleton/run_system.py ... --oracle beta --oracle-beta 0.5
python examples/experiments/assist_skeleton/run_system.py ... --mask-window 70,100
# 학습 중 체크포인트가 생기면 W1에서 자동 평가 + 비교표 (LIMITS, TRACKER, CALIB 환경변수)
bash examples/experiments/assist_skeleton/eval_when_ready.sh results/llp_L6_pendulum/epoch_2000.ckpt L6e2k aisbc-llp-L6
```

```bash
# LLP 학습 (랜덤화 진자). 장시간 학습은 반드시 systemd 사용자 서비스로 (세그폴트 자동 재시작)
systemd-run --user --unit=aisbc-llp-L6 --working-directory="$PWD" \
  --property=Restart=on-failure --property=RestartSec=15 --property=StartLimitIntervalSec=0 \
  --property=StandardOutput=append:$PWD/results/_pids/llp_L6.log \
  --property=StandardError=append:$PWD/results/_pids/llp_L6.log --setenv=PYTHONFAULTHANDLER=1 \
  ~/miniconda3/envs/sbc/bin/python protomotions/train_agent.py --robot-name hip_pendulum --simulator newton \
    --experiment-path examples/experiments/assist_pendulum/mlp_v4.py --experiment-name llp_L6_pendulum \
    --motion-file none --num-envs 4096 --batch-size 16384      # + mlp_v4.py의 CLI 옵션 (기본값 = L1)
systemctl --user status aisbc-llp-L6 ; systemctl --user stop aisbc-llp-L6
# 가상 착용자 학습 (W1 레시피, W0에서 warm start)
bash examples/experiments/assist_skeleton/train_tracker_human.sh
# 학습 전 진자 env 점검 그림
python examples/experiments/assist_pendulum/check_env_v4.py
```

L1 학습 속도: 4096 env에서 epoch당 약 6 s, 6,000 epoch ≈ 10 h. 평가는 한 실행 40 s ≈ 3분. 전이 성능은 epoch 6,000 근방이
정점이고 그 뒤 물러나므로 **2k / 4k / 6k에서 W1 평가**를 보고 판단한다.

---

## 7. 열린 과제와 다음 선택지

**결정 대기 (사용자)**

- **(가) H1 + L1 × 0.5 @ W1을 통합 결과로 확정.** 학습 없이 평가·문서만: 배율 0.3~0.7 탐색(실행당 3분), `--torque-lpf-hz 12`에서도
  유지되는지, W1 기준 보고서(지금 보고서는 W0 기준 + W1 부록 한 줄), 버전 문서에 "배포 후보 = H1+L1×배율, W1에서 0.5" 명시.
  남는 한계: β 상한의 절반 남짓, 배율은 사람마다 손으로 맞춰야 함(목표의 "크기" 부분 양보), 달리기 유각 말기 오방향은 그대로.
- **(나) 달리기 유각 말기 시점만 겨냥한 진자 변형(L6).** 가설: "반전이 느리면(걷기) 브레이크, 빠르면(달리기) 브레이크 없음"을
  정책이 chunk 모양에서 구분하게 한다. L5의 "환경이 반전 관성을 대신 냄"을 **의도 반전 가속도가 문턱 이상일 때만** 켠다. 문턱은
  W1 걷기·달리기 기록에서 chunk 반전 가속도를 재어 두 분포 사이에 둔다. **학습 전에** 같은 규칙을 L1 위 추론 시 규칙으로 얹어
  (정책 관측만 사용) W1에서 확인하고, 달리기가 −30 쪽으로 가고 걷기가 유지될 때만 학습을 건다. 상한은 마스크 진단의 달리기 −30 %,
  걷기는 현상 유지. 위험: L5처럼 학습 신호 소실, 걷기·달리기 반전 속도 분포 겹침.
- 순서 제안: (가)의 배율 탐색·대역 확인은 학습이 없으니 어느 쪽이든 먼저. 그 뒤 (나)의 학습 전 시험.

**그 외 (우선순위 낮음)**
- `protomotions/assist/verify_controller.py`: 배포 HLP(10 ms × 6점)와 v3 제어기 형식 불일치로 실패 중. L1 기준으로 다시 PASS시키고
  스켈레톤 env ↔ `AssistController` 등가성 검사도 만들 것.
- 구동기 스펙(피크·연속 토크·대역폭) 반영. 달리기는 40 N·m에서 한계 도달 64 %라 한계가 병목.
- HLP 학습셋의 계단 부족(74,488클립 중 98개) → 계단·전환에서 오차 2.3~2.6배.
- W1의 피크/RMS 3.9: 200 N·m 안에서도 착지 슬램이 남아 있음(τ² 벌점 −1e-6으로는 못 눌렀다).
- 스켈레톤 run 세그폴트 원인.

---

## 8. 파일 지도 (요약 — 전체는 FRAMEWORK §4)

| 경로 | 내용 |
|---|---|
| `checkpoints/L1_llp_pendulum/`, `W1_wearer_human/` | 현재 보관본 + INFO.md. 나머지 `llp_v3~v10`, `v18~v32`는 legacy |
| `results/` (git 미추적) | 학습 run과 평가 기록. `results/oracle/` = 통합 평가 104회(`<날짜>_tracker_<착용자>_<태그>_m<모션>/` 안에 traces·metrics·report). `results/_pids/*.log` = systemd 로그. `llp_L2~L5_pendulum/`, `assist_pendulum_v4/v5`, `mimic_suit_human/`(W1 run) |
| `examples/experiments/assist_pendulum/` | 진자 LLP 학습: `mlp_v4.py`(L1 + L2~L5 옵션, docstring에 run 기록), `check_env_v4.py`, `eval_abc.py` |
| `examples/experiments/assist_skeleton/` | 통합 평가·착용자: `run_system.py`(A/B 러너, 오라클·마스크·배율·LPF 옵션), `compare_runs.py`, `report_html.py`, `report_checkpoint_html.py`, `eval_wearer.py`, `eval_when_ready.sh`, `tracker_human.py` + `train_tracker_human.sh`(W1), `llp_walk.py`(개인화, 미사용), `make_gait_motions.py`, `README.md` |
| `protomotions/assist/` | 독립 제어기(`controller.py`), LLP actor 로더, 스켈레톤 특징량·영점, 등가성 검증 |
| `protomotions/envs/base_env/assist_env.py`, `protomotions/envs/control/assist_*_control.py` | 진자 env(관성·질량·한계 랜덤화, 보조 주입), 외란(입각 하중·굳힘·반전 브레이크), 의도·chunk 생성 |
| `protomotions/envs/base_env/skeleton_assist_env.py` | 개인화 env(미사용) |
| `protomotions/simulator/newton/substep_hook.py` | env step 안 10 ms 콜백(LLP) |
| `protomotions/robot_configs/skeleton_torque_suit_human.py`, `hip_pendulum*.py` | W1 로봇 config(hip 한계 200), 진자 |
| `protomotions/data/assets/mjcf/skeleton_torque_suit.xml` | 인체 모델(뼈 메시 그리기 전용 + 캡슐 충돌; headless는 메시 안 읽음) |
| `data/motion_for_trackers/skeleton_torque_suit_gait.pt` | 보행 세트(walk 0.75 / run 0.25). motion 0 = 걷기, 1 = 달리기 |
| `protomotions/maniflow/` | HLP 접근 계층(README 참고). `import maniflow` 직접 금지 |
| `docs/llp_v13_report.html`, `docs/transfer_report.html` | L1 깊은 보고서(W0 기준 + W1 부록), 여러 체크포인트 비교. 같은 내용의 claude.ai artifact: `A1enWCcYbiRaEE6rFNn6qB`, `LhmuUcPkK1iZxuKRmCvoNp` |

Claude Code 에이전트 메모리(이 레포 전용): `~/.claude/projects/-home-bak-Projects-AI-SBC-PROTOMOTIONS/memory/` — 교훈·사용자 지시가
파일 하나당 하나씩. 다른 에이전트도 읽어도 된다. 핵심은 이 문서 §5·§9에 옮겨 두었다.

---

## 9. 운영 규칙과 사용자 방침

- **장시간 학습은 systemd 사용자 서비스**(§6 명령)로. `nohup`/`setsid`는 세그폴트에 죽은 채 방치된다. `pkill -f <이름>`은 셸까지 죽인다.
- 학습 중 산출물은 `results/<exp>/`, 보관은 `checkpoints/<버전>/ + INFO.md`(가중치는 LFS). 보관하지 않은 run은 지워도 된다.
- 커밋: `git commit -s`, gitmoji + 영어 제목, 본문 끝 `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. 1인 레포라 main에 직접.
- `pre-commit run --files <파일>`만. `--all-files` 금지. Apache 헤더는 기존 파일에서 그대로 복사.
- **사용자와는 한국어.** 설명은 은어·단계 라벨 없이 내용으로("스모크", "Phase D" 같은 말 금지), 기술 용어는 통용 영어 그대로.
  코드 설명은 계층 분류 대신 실행 순서의 file:line 트레이스로.
- 사용자 성향: 하드웨어 한계를 고정하기보다 학습에서 **랜덤화**하는 쪽을 선호. 필터는 자발적으로 걸지 않는다(구동기 스펙이 오면 그 값으로).
  먼저 제안하지 말 것: ARDY, HLP 크기 축소, 진자를 쉽게 만드는 변형.
- 사용자는 원격 tmux로도 작업한다. 남의 세션·유닛을 끌 때는 먼저 확인.

---

## 10. 보고서·기록 위치

| 무엇 | 어디 |
|---|---|
| 실험 일지(가장 상세) | `docs/NEXT_SESSION.md` §2~§2-L (오라클 → v11/v12 → 진자 v4 → v5 → W1 → L2~L5), §3 한계 출처, §4 후보, §5 운영, §6 방침 |
| 버전·용어 | `docs/AI_SBC_VERSIONS.md` §0~§5, `docs/versions.html` |
| 설계·함정·코드 읽기 | `docs/AI_SBC_FRAMEWORK.md` |
| 보관본 설정·결과 | `checkpoints/L1_llp_pendulum/INFO.md`, `checkpoints/W1_wearer_human/INFO.md` |
| 커밋 메시지 | `git log --date=short` — 2026-09-13 이후 각 실험의 결론이 제목에 있다 |
| 그림 | `docs/*.png` (hlp_llp_timing, torque_phase, pendulum_vs_walking_torque, oracle, landing_torque_zoom) |
