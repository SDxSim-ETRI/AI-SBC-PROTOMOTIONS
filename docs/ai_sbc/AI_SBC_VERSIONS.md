# AI-SBC 모델·데이터 이름과 경로

2026-10-20 기준. 현재 작업은 [HANDOFF.md](HANDOFF.md), 실험 결론은 [실험 결과 요약](experiment_summary.html).
이 문서의 모델은 시뮬레이션 연구용이다. 일반 보행이나 실제 장치의 성능을 보증하지 않는다.

## 이름 읽는 법

| 표기 | 뜻 |
|---|---|
| H1 | 미래 고관절 각도 예측기. HLP(High-Level Predictor), ManiFlow 모델 |
| L1, L9, L13R, L14, L15, L16 등 | 보조 토크 제어기의 실험 버전. LLP(Low-Level Policy). 큰 번호가 항상 더 좋은 모델은 아님 |
| W1 | 시뮬레이션에서 사람 역할을 하는 가상 착용자. 보조 제어기와 별도 모델 |
| A443 같은 Axxx | BONES-SEED 원본 메타데이터의 `actor_uid`, 촬영한 사람을 구분하는 코드. 실험·모델 번호가 아님 |
| 무보조 A / 보조 B | 같은 조건에서 보조 유무를 비교하는 두 시뮬레이션. 위의 동작 수행자 코드와 무관 |
| s0 / s1 | 학습 난수 초기값 0의 첫 학습 / 초기값 1의 독립 재학습. 각자 처음부터 학습한 모델이며 좌우 구분이 아님. G1·G4만 두 실행 수행 |
| epoch_400 | 해당 학습 실행에서 400회 갱신한 저장 모델. 실행마다 수집 데이터 양이 달라 번호만으로 학습량을 비교하지 않음 |

과거 이름 대응: `llp_v13`→L1, `tracker v33`→W1, `tracker v32`→W0.
`L13`은 전신 추가학습 실험이며 과거 `llp_v13`과 다른 모델이다.

## 지금 보존하고 비교할 모델

| 설명 이름 | 원래 이름 | 경로 | 역할과 한계 |
|---|---|---|---|
| 각도 예측기 | H1 | 아래 H1 경로 | 최근 실험에서 동결. 100 Hz 입력, 현재부터 50 ms 예측 |
| 가상 착용자 | W1 | `checkpoints/W1_wearer_human/last.ckpt` | 기존 걷기·달리기 평가 기준. 새 실제 보행은 추종 실패 |
| 진자 학습 기준 모델 | L1, 출력 0.5배 | `checkpoints/L1_llp_pendulum/last.ckpt` | 기존 걷기·달리기 비교 기준. 적용 한계 약 ±20 N·m. 과거 토크 경로로 학습 |
| 진자 학습 후보 기준 모델 (2026-10-20) | R0, 출력 0.5배 | `results/llp_relative_R0_s0_20261020/epoch_6000.ckpt` | L1 레시피 + 올바른 토크 경로만. 0.5배 ±20 N·m에서 이득 262.03 (N·m)²·저부담 3.27·최악 7.57 N·m(보조 RMS 8.29); 같은 보조 크기의 L1 ±10보다 이득 +15 %, 배율 1 클립에서는 L1 ±20과 같은 이득·손해. 2026-10-21: 같은 걷기 400–440 s 구간에서도 L1 곡선 대비 +35 %로 유지, **달리기에서는 −34 %(±20 이득 436 vs L1 756)로 유지 안 됨**. 학습 seed 하나. 걷기 후보 기준, 달리기 기준은 L1 유지 |
| 기준 모델 + 유각 게이트 | L1, 출력 0.5배, `--swing-gate 0.20` | 같은 체크포인트 + `AssistController(swing_gate_fraction=0.20)` | 2026-10-14 진단·옵션(기본 off, 배포 후보 아님). 관절각만으로 유각 중·후반 신전 보조를 끊으면 걷기 −26.5 / 달리기 −31.0 % — 그 구간의 손해 크기를 보여 주는 기준값. 현재는 진자 학습 개선이 우선 |
| 전신 추가학습의 시작 모델 | L9, 2,000회 | `results/llp_L9_opposition/epoch_2000.ckpt` | 비교·학습 계보용. 구간별 방해가 남음 |
| 부담 비율 반영 모델 | L13R, 400회 | `results/llp_L13R_walk_continue/epoch_400.ckpt` | 기본 걷기 연구 후보. 느린 재생 조건 실패 |
| 세 재생 속도 혼합 모델 | L14, 200회 | `results/llp_L14_walk_tempo/epoch_200.ckpt` | 기본 걷기에서 개선. 느린 재생 조건과 허리·무릎 부담 미해결 |

L9·L13R·L14 평가는 배율 1, 최종 토크 한계 ±100 N·m를 사용했다.
L1과는 출력 한계와 학습 조건도 다르므로 성능 차이를 보상식 하나의 효과로 해석하지 않는다.

H1의 검증된 저장 경로 (2026-10-29부터, flexion100-h15-swap: 관측 토큰 self-attention · 노이즈 action 열 cross-attention):
`/home/bak/Projects/ManiFlow_Policy/ManiFlow/data/outputs/hip_flexion_seed-maniflow_lowdim_policy_walking-flexion100-h15-swap-run01_seed42/checkpoints/epoch=0008-val_loss=0.000834.ckpt`

옛 H1(관측 cross-attention)은 2026-10-29에 폐기했다. 그 전의 모든 결과는 이 파일로 측정했으며 재현용으로만 남긴다:
`/home/bak/Projects/ManiFlow_Policy/ManiFlow/data/outputs/hip_flexion_seed-maniflow_lowdim_policy_walking-flexion100-h15-run01_seed42/checkpoints/epoch=0006-val_loss=0.000762.ckpt`

## 진자 비교 결과와 보존 경로

**2026-10-22: 20 N·m·실제 동작·단일 관절 공유로 다시 학습한 네 모델(A·Δ 2×2)은 W1에서 L1 ±20보다 덜 줄였다(학습 때 예측 신호와 실제 H1의 차이). L1 ×0.5가 비교 기준으로 남는다.** **2026-10-20 R0/R1: 토크 경로만 바로잡은 R0는 같은 보조 크기에서 L1보다 이득이 15 % 크고 출력이 필요 토크에 비례하지만, 저부담 상태의 방향 문제는 그대로다(학습 seed 하나). R1·F1/F2·G0–G4·PA 계열은 L1 대체 기준에 미달했다.** L1은 기존 ×0.5(±20 N·m)를 비교 기준으로 유지하고 R0 ×0.5를 후보 기준으로 병기한다.

| 실험 | 경로 | 상태 |
|---|---|---|
| L15 · 진자 학습, 보조 한계 20 N·m (2026-10-29) | `results/llp_R0lim20_s0_20261029/epoch_6000.ckpt` | seed 0·6,000 epoch. 배율 1 ±20: W1 걷기 τ² −17.8 %·달리기 −22.4 %, 최대출력 95% 이상 사용 시간비율 76 / 84 %. 20 N·m로 학습한 현재 계보의 출발 모델 |
| L15 걷기 전용 / 달리기 전용 추가학습 (2026-10-29, 옛 H1) | `results/llp_lim20ft_{walk,run}_20261029/epoch_200.ckpt` | 가상 착용자 W1과 예측기를 연결한 전신 시뮬레이션에서 200 epoch 이어서 학습. 걷기 −22.3 %, 달리기 −25.0 %; 최대출력 95% 이상 사용 시간비율 78–84 % 유지 |
| R0 ×0.5 걷기/달리기 전신 추가학습 (기준 계보, 2026-10-29) | `results/llp_R0ft_{walk,run}_20261029/epoch_200.ckpt` | 출력 20·tanh·한계 20으로 R0 actor 이어받음. 걷기 −26.5 %, 달리기 −25.6 %, 걷기 모델→달리기 −30.2 %. 걷기 후보 |
| 위 연구 원본 (2026-10-29) | `results/llp_lim20_finetune_20261029/` · `results/llp_R0_finetune_20261029/` | 평가 기록·학습 로그·보고서 스크립트. [결과](261029_lim20_finetune/report.html) |
| 예측기 토큰 자리 맞바꿈 (2026-10-29) | `/home/bak/Projects/ManiFlow_Policy/ManiFlow/data/outputs/hip_flexion_seed-maniflow_lowdim_policy_walking-flexion100-h15-swap-run01_seed42/checkpoints/epoch=0008-val_loss=0.000834.ckpt` · `results/hlp_routing_swap_20261029/` | 옛 H1과 같은 조건 학습. ETRI 14동작 MAE 0.51° (옛 H1 0.56°, denoise 1), 지연 2.45 vs 2.20 ms. 정확도는 같은 수준이며 원래 구상한 구조이므로 사용자 결정으로 2026-10-29 H1로 채택, 옛 H1 폐기 |
| 새 H1 기준값 (2026-10-29) | `results/h1_swap_adoption_20261029/` (`summary.json`, `eval_all.sh`) | W1 + 새 H1, 40 s, seed 42·7, denoise 1: L1 ×0.5 걷기 −19.0 / 달리기 −23.3 %, R0 ×0.5 걷기 추가학습 모델 −26.7 / −29.7 % (옛 H1: −18.8 / −24.1, −26.5 / −30.2). 낙상 0 |
| L15 걷기 전용 / 달리기 전용 추가학습 400 epoch (2026-10-29, 새 H1) | `results/llp_L15ft_{walk,run}_20261029/epoch_400.ckpt` | W1 걷기 −24.9 % / 달리기 −25.5 %(각자 동작). 다른 동작에 쓰면 −18.9 / −18.6 % |
| L16 · L15 걷기·달리기 동시 추가학습 800 epoch (2026-10-29) | `results/llp_L16_20261029/epoch_800.ckpt` | 두 동작 반반, motion별 영점. W1 걷기 −22.7 % / 달리기 −27.1 %, 최대출력 95% 이상 사용 시간비율 91 / 91 %. 한 모델로 두 동작을 맡길 때 가장 균형 잡힌 checkpoint |
| L15 걷기 전용 / 달리기 전용 추가학습 1,200 epoch (2026-10-30, 400에서 이어서) | `results/llp_L15ft_{walk,run}_20261029/epoch_1200.ckpt` | 같은 run을 400 epoch에서 이어받음(원본 설정 `resolved_configs.pt.ep400`). 걷기 전용: 걷기 −26.6 %(600 epoch과 같음), 최대출력 95% 이상 사용 시간비율 68 %, 걷기 70–90 % 방해 없음 → L15 계보의 걷기 후보. 달리기 전용: 달리기 −24.9 %(400 epoch −25.5 %보다 못함) |
| L16 1,200 epoch (2026-10-30, 800에서 이어서) | `results/llp_L16_20261029/epoch_1200.ckpt` | 같은 run을 800 epoch에서 이어받음(원본 설정 `resolved_configs.pt.ep800`). W1 걷기 −19.9 % / 달리기 −27.4 %, 최대출력 95% 이상 사용 시간비율 92 / 88 %. 달리기는 수렴(800 → 1,200 epoch −27.1 → −27.4 %) → L15 계보의 달리기 후보; 걷기는 800 epoch보다 못함 |
| 위 연구 원본 (2026-10-29 ~ 30) | `results/llp_L16_study_20261029/` | 평가 72건·`summary.json`·`train_curve.json`(학습 보상 곡선)·`cross_summary.json`(교차 적용)·학습/평가/이어서 학습 스크립트. [결과](261029_L16_joint_finetune/report.html) |
| 발표 편집본 3장·고부담 출력 진단 (2026-10-23) | `results/presentation_three_slides_20261023/index.html` · 보관 스크립트 `results/presentation_three_slides_20261023/scripts/` | 기본형=L1, 효율형=R0. 보조 최대 ±20 조건만. 기존 영상·수치 재구성; 새 학습/추론 없음. 원본·진단 `source_manifest.json`, `diagnostics/` |
| 발표용 ON → OFF → ON 회복 시연 (2026-10-23) | `results/presentation_toggle_20261023/` · `results/presentation_toggle_20261023/scripts/presentation_toggle_capture.py`(보관) | 기존 가중치·W1 힘 한계·피드백 유지. 영상 5초 OFF/10초 ON. 전환 2건+저장 대조군 2건 모두 낙상 0. 영상·계측·원본 해시 `deliverables/` |
| 발표용 고관절 최대 힘 제한 ON/OFF (2026-10-23) | `results/presentation_cap_20261023/` · `results/presentation_cap_20261023/scripts/presentation_cap_capture.py`(보관) | 기존 L1/R0 ×0.5·H1·W1 동결. 착용자 hip_flexion 두 축만 기존 ON 절대 최대값으로 제한, 피드백 유지. 4건 완주·낙상 0. 조건·실행 `protocol.json`, `runs.json`; 영상·검증 `deliverables/` |
| R0 올바른 토크 경로만 (대조) | `results/llp_relative_R0_s0_20261020/epoch_6000.ckpt` | seed 0·6,000 epoch 완료. L1 공동 actor·±40 N·m에 사람 PD+FF 단일 제한·독립 주입만 적용. W1 0.5배 ±20: 이득 262.03 / 저부담 3.27 / 최악 7.57 / 보조 RMS 8.29 / 포화 0.5 %; 같은 보조 크기의 L1 ±10 대비 이득 +15 %, 손해 +1.0 N·m; 배율 1 클립 277.05 / 8.41 / 14.69. 가장 유망한 후보 기준이나 저부담 방향 문제는 미해결. 진자 실제 H1은 L1보다 조금 나쁨 |
| R1 무보조 대비 토크 증가 비용 | `results/llp_relative_R1_s0_20261020/epoch_6000.ckpt` | seed 0·6,000 epoch 완료(epoch 4,850에서 SIGSEGV 뒤 재개). R0 + 무보조 대비 토크 증가 비용: 학습 중 같은 목표 궤적·관성·외란의 무보조 진자를 함께 계산하고, 착용자 토크²가 무보조보다 커진 만큼을 (무보조 토크²의 최근 50 ms 평균 + 10²)로 나눠 가중치 −0.5로 벌점. W1 ±20 이득 135.59·저부담 3.79, 부호 일치 하락. 채택 안 함 |
| LLP 20 N·m 단일 관절·공유·실제 동작 (2026-10-22) | `results/llp20_motion_{A0D0,A1D1,A1D0,A0D1}_right_s0_20261021/epoch_6000.ckpt` | 학습 seed 0·6,000 epoch 4개 완료. 레시피 `mlp.py --recipe motion_leg`(R0 + 20·tanh + SEED 동작 의도·가벼운 다리 집단 + 오른쪽 관절 단일 학습). 이름의 A·D는 보조 토크² 벌점(−2e-5)·명령 변화 벌점(−0.005)의 켬(1)/끔(0). 배포·평가는 같은 가중치를 양다리에 적용(`LLPPolicy(share_legs=True)`, `run_system.py --llp-share-legs`). W1 ±20 기존 걷기 τ² 감소 149 / −123 / −132 / 81, 달리기 637 / 413 / 229 / 155 (L1 278·756). L1 대체 실패 — 학습 때 예측 신호와 실제 H1의 차이. `results/llp20_motion_A0D0_left_s1_20261021/`은 사용자 결정으로 중지한 미완료 학습(평가 안 함). [결과](261022_llp20_motion_shared/report.html) |
| 위 연구 원본 (2026-10-21/22) | `results/llp20_motion_20261021/` | `manifest.json`(학습 계획·소스 해시), `decision_weight_sharing.json`, `execution_right.json`, `eval_manifest.json`, `eval_runs.json`(74건), `pendulum/`, `walking/`, `diagnostics/`(예측 신호 진단 12건), `training_distribution_audit.json`, `eval_data/`(ETRI 13동작 평가용 사본: SEED 영점·±86° 클램프), `verification.json`(통과), 실패·재실행 기록. 코드(2026-10-26 통합) `examples/experiments/assist_pendulum/{mlp,study,evaluate,report}.py`; 당시 실행기 원본은 `results/llp20_motion_20261021/scripts/` |
| R0 일반화 평가 원본 (2026-10-21) | `results/llp_r0_generalization_20261021/` | 새 학습 없음. 선별 4건(달리기·400–440 s 걷기 통과, 제자리 걷기·무릎 높이 들기는 무보조 W1 낙상) + L1 ±10/15/20·R0 ±15/20 × 평가 난수 42/7 × 2조건 = 신규 22건. `manifest.json`, `analysis_plan.json`(결과 전 고정 판정 기준), `screening.json`, `runs.json`, `comparison.json`, `runtime_failures.json`(SIGILL 1회 재실행), 소스 사본. 코드 `results/llp_r0_generalization_20261021/scripts/{evaluate_generalization,report_generalization}.py`(보관). [결과](261021_r0_generalization/report.html) |
| R0/R1 계획·평가 원본 | `results/llp_relative_20261020/` | 2026-10-20 완료. 신규 W1 8건·진자 6조건, 기존 기준선 재평가 없음. `manifest.json`, `execution.json`, `evidence.json`, `config_audit.json`, `l1_need_bins.json`, `verification.json`, 소스 사본·로그. 추가 평가(배율 1·±20 클립)는 `walking_native/`. [결과](261020_relative/report.html) |
| F1 관성 제한 C2 전환 | `results/llp_feasible_F1_s0_20261020/epoch_6000.ckpt` | seed 0·6,000 epoch 완료. L1 공동 actor·±40 N·m, 올바른 독립 토크 경로 공통 적용. W1 저부담 방해와 평균 이득이 함께 감소, L1 대체 실패 |
| F2 흡수 파워 비용 추가 | `results/llp_feasible_F2_s0_20261020/epoch_6000.ckpt` | seed 0·6,000 epoch 완료. F1 + 흡수 파워 비용 −0.0032만 변경. W1 ±20에서 작은 보조 RMS로 비슷한 고관절 이득, 저부담 손해·진자 일반화 악화. 후속 연구 후보이며 채택하지 않음. seed 1 미실행 |
| F1/F2 계획·평가 원본 | `results/llp_feasible_20261020/` | 2026-10-20 완료. 신규 W1 8건·진자 6조건, 기존 기준선 재평가 없음. `manifest.json`, `execution.json`, `evidence.json`, `verification.json`, 소스 사본·로그·복구 기록 보존. [결과](261020_feasible/report.html) |
| 공통 출력 크기·저부담 감속 진단 | `results/llp_output_scale_20261016/` | 24건 완료·정상 저장 실행 모두 40초 완주. L1·G0 s0·G4 s0/s1 × 10/15/20 N·m × 평가 seed 42/7. 새 학습 없음. SIGSEGV 1회 재실행 기록·계획·모델/소스 해시·원본·검증 보존. [결과](261016_output_scale/report.html) |
| G0–G4 seed 0 | `results/llp_general23_G{0,1,2,3,4}_s0_20261015_all/epoch_3000.ckpt` | 5개 모두 완료. 선정 G4이나 L1 대체 실패 |
| G1·G4 seed 1 | `results/llp_general23_G{1,4}_s1_20261015_all/epoch_3000.ckpt` | 2개 새 초기화 학습 완료. 각 3,000 epoch. G0/G2/G3 seed 1은 실행하지 않음 |
| 일반화 실험 원본 | `results/pendulum_general23_20261015_all/` | 진자 40평가·W1 걷기 8평가 완료. `manifest.json`, `comparison.json`, `confirmation.json`, `walking/runs.json`, 소스·JSON/NPZ 보존. 새 출력 ±23, 비교 L1 ±20 |
| C0/S1/A1/T1 | `results/llp_pendulum20_{C0,S1,A1,T1}_20261014_nativeff/` | 각 2,000 epoch·총 24평가 완료. 새 모델은 L1 기준 미달. C0/S1/T1 보조 실패 0회, A1 14–24/256회. 보조 ±20, 전체 사람 ±150 N·m |
| C0 연장 | `results/llp_pendulum20_C0_20261015_e6000/` | 총 6,000 epoch 완료. 실제 H1 최선은 @5000 τ² 2003.4, L1 1878.7에 미달 |
| AB1 | `results/llp_pendulum20_AB1_20261015/` | A1 + raw action bounds loss 0.001. 2,000 epoch 완료. 포화·실패 남아 제외 |
| 첫 후속 원본 | `results/pendulum_research_followup_20261015/` | 10월 15일 10:42, 29평가 완료. 통과 후보 없음. 조건부 추가 seed/W1 전이 미실행 |
| L1 진자 추가학습 | `results/llp_pendulum20_{PA_E,PA_H1}_20261015/` | L1 actor에서 합성 예측/실제 동결 H1 비교. 512환경·2,000 epoch·±20 N·m. critic/optimizer 새 초기화 |
| 이전 추가학습 원본 | `results/pendulum_adaptation_20261015/` | 최신 지시로 후속 실행 중단. 예비검사·명령·로그·평가 보존. 현재 후보에서 제외 |
| 학습 한계 대조 실험 | `results/llp_limit_{historical,corrected}_{20,40}_s{0,1}_20261015/` | 8개 설정 검사 완료. 23 N·m 실험 우선으로 중단. 기존 경로 20·seed 0은 epoch 1640 저장; 나머지는 학습 미완료 |
| 학습 한계 대조 원본 | `results/pendulum_limit_control_20261015/` | `config_audit.json`, 고정 계획·소스 사본·실행 상태. 기존 L1/C0 비교와 구별하며 성능 결과는 아직 없음 |
| 실행·평가 원본 | `results/pendulum_research_nativeff_20261014/` | `sequence_status.json`, 소스 사본, 학습 로그, 평가 NPZ/JSON |
| 초기 토크 경로 진단 | `results/llp_pendulum20_{C0,A1,C0c}_20261014/` | 중단·보존. 합산 제한과 사람 PD/FF 별도 포화의 영향을 확인한 진단 실행; 최종 후보에서 제외 |
| 환경 검증 원본 | `results/pendulum_research_20261014/` | `human_limit_audit.json`, `eval_L1_separate/`, `eval_L1_combined_human/` 등 |

## 현재 데이터

| 자료 | 위치 | 용도 |
|---|---|---|
| 기존 suit 걷기·달리기 | `data/motion_for_trackers/skeleton_torque_suit_gait.pt` | 기존 결과 재현. motion 0=걷기, 1=달리기 |
| 기존 걷기 재생 속도 변형 | `results/llp_walk_curriculum_20261011/walk_tempo_variants.pt` | 같은 동작을 0.85–1.15배로 재생. 실제 속도별 촬영 자료가 아님 |
| 실제 보통·느린 보행 변환 후보 | `results/real_walk_retarget_20261012_224957/native_pairs_v1/suit_native_walk.pt` | 4명 표기의 각 보통/느림, 총 8클립. 무보조 추종 미확보 |
| 빠른 보행 변환 후보 | `results/real_walk_retarget_20261012_224957/fast_windows_v1/suit_native_walk.pt` | 다른 3명 표기의 4초 구간. 걷기/달리기 분류 미확정 |

새 후보의 원본은 H1 원본 자료 목록과 겹친다. 시스템 전체가 보지 않은 평가 자료라고 부르지 않는다.
동작 수행자가 달라도 시뮬레이션에서는 같은 suit 체형과 같은 W1을 사용했다. 여러 체형의 착용자를 검증한 것이 아니다.

## 과거 버전

L2–L11의 중간 결과와 더 오래된 버전은 [요약의 실험 이력](experiment_summary.html#history)과
[보관한 버전표](archive/20261013/AI_SBC_VERSIONS.md)에 있다. L12는 코드 준비 후 학습하지 않았다.
종료된 실험을 자동 재개하지 않는다. 가중치·원본 수치·실패 로그는 기존 위치에 보존한다.
