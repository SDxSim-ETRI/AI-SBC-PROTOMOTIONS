# 다음 세션 인계 메모 (2026-09-15 저녁 기준)

새 세션에서 이 파일 + `AI_SBC_VERSIONS.md`만 읽으면 이어서 할 수 있게 정리한 것.
설계 배경은 `AI_SBC_FRAMEWORK.md`, 통합 시스템 사용법은
`../examples/experiments/assist_skeleton/README.md`.

## 1. 지금 시스템 (전부 보관 완료, 작업 트리 깨끗)

```
착용자(시뮬 속 사람)  checkpoints/v32_newton_suit_gait_symmetric   ← 대칭 보행
예측기 HLP            ManiFlow flexion100-h15 (자동 탐색, 지정 불필요)
보조 정책 LLP         checkpoints/llp_v10_gait_5ms_40Nm
인체 모델             mjcf/skeleton_torque_suit.xml (뼈 메시 + 캡슐 충돌)
```

| | 주기 | 연산 시간 | 비고 |
|---|---|---|---|
| HLP | 10 ms | 4.9 ms (denoise 1) | 과거 100 ms → 미래 0~50 ms, 10 ms 격자 6점 |
| LLP | **5 ms** | 0.25 ms | 미래 창은 {0, 10, 20 ms}만 읽음 |
| 물리 | 2.5 ms | — | Newton 400 Hz, decimation 2 |

성능(40 s A/B, 착용자 v32, 낙상 0): **걷기 181 → 135 W (−25.4 %)**, 달리기 362 → 309 W (−14.6 %).

### 재현 명령 (sbc env)

```bash
# 평가 — 학습과 같은 5 ms 제어로 돌려야 한다 (--llp-dt 0.005 빠뜨리면 다른 시스템이 된다)
python examples/experiments/assist_skeleton/run_system.py --human tracker \
    --tracker-checkpoint checkpoints/v32_newton_suit_gait_symmetric/last.ckpt \
    --llp-checkpoint checkpoints/llp_v10_gait_5ms_40Nm/last.ckpt --llp-dt 0.005 \
    --motion-file data/motion_for_trackers/skeleton_torque_suit_gait.pt --motion-id 0 --seconds 40 \
    --calibration "5.70,2.63,-14.43,0.84"        # motion-id 1 = 달리기, --record = 영상
# 주기 평균 토크·파워 그림
python examples/experiments/assist_skeleton/plot_torque_phase.py <run_dir>
```

## 2. 지금 가장 중요한 미해결 문제

**보조가 파워 최대 구간을 놀고 있다.** 보행 한 주기를 평균해 보면:

| 구간 | 착용자 토크 | 착용자 파워 | 보조 토크 |
|---|---|---|---|
| 0~10 % (뒤꿈치 착지·하중 수용) | −166 N·m (피크 −200) | **229 W ← 최대** | +22 N·m (반대 방향) |
| 10~65 % (입각 후반~유각) | +11 N·m | 96 W | +34 N·m (거의 포화) |

보조 토크 방향 통계: **굴곡(+) 71 % / 신전(−) 28 %**, +40 포화 39 % vs −40 포화 3 %
→ 정책이 사실상 **한 방향(굴곡)으로만 쓰는 해**에 수렴했다. 그림: `docs/torque_phase.png`.

가설 두 가지, 아직 미확정:
1. **보상**: 착지 때 신전 토크를 밀면 몸체 추종이 흔들려 hinge 벌점에 걸린다.
2. **한계**: 40 N·m는 그 순간 피크 200 N·m의 20 %라, 같은 토크를 유각에 쓰는 편이 이득이었다.

### 가르는 실험 — 오라클 진단

"오라클"은 학습된 정책 대신 **정해진 규칙의 토크를 강제로 주입**해서 상한과 인과를 보는 진단이다
(정책이 아니라 탐침). 이미 있는 것: `check_train_env.py --oracle-beta β`
= τ_assist를 β×(착용자가 직전 step에 실제로 낸 토크)로 넣는다. 이전 측정: 25 N·m 한계에서 −18 %,
40 N·m에서 −26 %가 상한이었다.

다음에 할 것: **착지 구간(위상 0~10 %)에만 −40 N·m를 주입**하고 A/B로
(a) 착용자 파워가 실제로 줄어드는지 (b) 몸체 추종 오차가 얼마나 나빠지는지 잰다.
- 파워 ↓ & 오차 유지 → 보상/학습이 기회를 놓친 것 → 보상 재설계(위상별 가중, 한계 상향 등)
- 오차 ↑ → 1번 가설이 맞고, 착지 구간은 원래 건드리면 안 되는 구간

구현은 `check_train_env.py`에 위상 기반 오라클을 추가하거나(발 접촉/최대 굴곡 기준),
`run_system.py`에 `--oracle-phase-window` 같은 인자를 새로 다는 쪽이 간단하다.

## 3. 토크 한계 40 N·m의 출처 — 스펙이 아니라 우리가 고른 값

- 진자 단계에서 25 N·m로 시작(내가 "착용 장치 현실 범위"로 가정한 값, 근거 문서 없음).
- 25 N·m에서 오라클 상한이 −18~20 %로 막혀서 40 N·m로 올렸고 −32 %가 나왔다.
- 즉 **실제 구동기 스펙에서 온 숫자가 아니다.** 걷기 중 사람 고관절 피크 토크가 200 N·m
  수준이므로 40 N·m는 그 20 %다.
- 한계를 올리면 절감이 거의 단조 증가한다(측정: 25 → −20 %, 40 → −32 %). 올릴수록 구동기가
  무겁고 커지고, 보행을 흔들 위험도 커진다.
- **필요한 것**: 실제 장치의 (1) 피크 토크 (2) 연속 토크 (3) 토크 대역폭[Hz].
  대역폭이 정해지면 `--torque-lpf-hz`에 넣고 그 조건에서 성능을 다시 읽어야 한다
  (측정: 12 Hz 거의 무손실, 6 Hz −4~5 포인트, 3 Hz 붕괴).

## 4. 다음 단계 후보 (우선순위 순)

1. **착지 구간 오라클 진단** (위 2절) — 지금 가장 정보량이 큰 실험.
2. **달리기 하락 보완** — v10은 걷기 −25.4 %인데 달리기 −14.6 %(v6은 −19.3 %였다).
   학습 비중이 walk 0.75 / run 0.25다. 0.5/0.5로 다시 학습해 비교.
3. **구동기 스펙 반영** — 피크·연속 토크·대역폭 받으면 한계와 필터를 실제 값으로.
4. **계단 데이터 보완** — HLP 학습셋 74,488클립 중 계단이 98개뿐이라 계단·전환에서 오차 2.3~2.6배.
5. **세그폴트 원인 규명** — 다음 학습부터 `PYTHONFAULTHANDLER=1`을 켜고, CUDA graph를 끈 판과 비교.

## 5. 반드시 지킬 운영 규칙

- **장시간 학습은 systemd 사용자 서비스로**, 자동 재시작을 걸 것. `nohup`/`setsid`로는 세그폴트에
  죽은 채 방치된다(실제로 하룻밤 날렸다).
  ```bash
  systemd-run --user --unit=aisbc-<이름> --working-directory="$PWD" \
    --property=Restart=on-failure --property=RestartSec=15 --property=StartLimitIntervalSec=0 \
    --property=StandardOutput=append:$PWD/results/_pids/<이름>.log \
    --property=StandardError=append:$PWD/results/_pids/<이름>.log \
    ~/miniconda3/envs/sbc/bin/python protomotions/train_agent.py ...
  systemctl --user status aisbc-<이름>     # 상태·NRestarts
  systemctl --user stop   aisbc-<이름>
  ```
- 파이썬 세그폴트가 3일간 8회 났다(6회 python3.11 내부, 2회 libc10_cuda.so). 주소가 매번 달라
  특정 버그라기보다 네이티브 확장 메모리 문제로 보이고, CUDA graph가 1순위 용의자다.
- `pkill -f <실험이름>`은 **명령을 친 셸까지 죽인다**. PID 파일이나 systemd로 끌 것.
- 평가할 때 LLP 학습 주기와 `--llp-dt`를 반드시 맞출 것(v10은 0.005).
- 학습 중 산출물은 `results/<exp>/`(git 미추적), 보관본은 `checkpoints/<버전>/` + `INFO.md`(LFS).

## 6. 사용자 지시로 고정된 방침

- 목표는 **보행 계열(걷기·달리기)만**. squat·제자리걸음·런지는 대상 아님.
- 인체 모델은 **해부학적 뼈 스켈레톤**으로 보여야 한다(캡슐 렌더 금지).
- 착용자 특이성(여러 사람 대응)은 **뒤로 미룸**.
- 필터는 자발적으로 걸지 않는다 — 구동기 스펙이 나오면 그 값으로.
