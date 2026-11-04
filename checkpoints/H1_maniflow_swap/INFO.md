# H1 — 미래 고관절 각도 예측기 (ManiFlow flexion100-h15-swap)

최근 100 ms의 고관절·몸통 각도로 앞으로 50 ms의 좌우 고관절 굴곡각을 예측하는 HLP다.
2026-10-29부터 쓰는 H1이며, 시뮬레이션 연구용으로 검증했다. 실기기 성능을 보증하지 않는다.
모델 목록은 [버전표](../../docs/ai_sbc/AI_SBC_VERSIONS.md)에 있다.

## 보관 정보

| 항목 | 내용 |
|---|---|
| 가중치 | `epoch_0008.ckpt` (42.5 MB). 원본 체크포인트에서 EMA 가중치·학습 설정·epoch 정보만 남긴 사본 |
| 원본과의 관계 | 2026-11-04 확인: 가중치 164개가 원본과 같고, 같은 입력·같은 난수에서 예측 출력도 같다 |
| SHA-256 | `24355171cf3b7dd87fc0a32cebacfba27773145acd313930328f247af8d90223` |
| 원본 | ManiFlow_Policy `ManiFlow/data/outputs/hip_flexion_seed-maniflow_lowdim_policy_walking-flexion100-h15-swap-run01_seed42/checkpoints/epoch=0008-val_loss=0.000834.ckpt` (169 MB, 학습용 가중치·optimizer 포함, 로컬에만 있음) |
| 모델 코드 | ManiFlow_Policy 저장소 커밋 `d9e536e` 이후 (`obs_routing: swap`). 체크포인트의 설정으로 모델을 만들므로 ManiFlow 코드가 필요하다. 원본 run 폴더와 학습 자료는 ManiFlow_Policy에 남아 있다 |
| 학습 자료 | `hip-flexion-seed-locomotion-100fps.zarr` (SEED 보행 계열, 100 fps), 12 epoch, seed 42, val_loss 0.000834 |

## 모델

| 항목 | 내용 |
|---|---|
| 구조 | DiT-X 6블록 × 256, 파라미터 10.6 M. flow matching(+consistency) 정책 |
| 토큰 배치 | 관측 토큰 10개(step당 1개)는 self-attention 앞자리, 노이즈 섞인 미래 action 토큰 15개는 cross-attention 컨텍스트, 출력은 위치 정보만 가진 query 토큰 15개에서 읽는다 |
| 입력 | 100 Hz, 4채널 × 10 step(과거 100 ms): 좌우 고관절 굴곡각, 몸통 pitch·roll |
| 프레임 간격 | 10 ms. `ManiFlowAngleEstimator`가 학습 zarr의 `frame_dt`를 읽고, 학습 자료가 없는 환경(실기기 등)에서는 기본값 `FLEXION_FRAME_DT` = 10 ms를 쓴다(2026-11-05 전에는 지운 40 fps 판 기준 25 ms였다) |
| 출력 | 좌우 고관절 굴곡각 15 step 중 현재~50 ms의 6 step을 쓴다 |
| 영점 | 착용자의 warmup 구간 평균을 `skeleton_features.SEED_CHANNEL_MEAN_DEG` = (10.41, 10.42, 7.39, −0.58)°에 맞추는 오프셋 (`run_system.py --calibration wearer`). 이 값은 H1 학습 자료의 실제 평균 (15.66, 15.67, 13.26, −1.44)°와 약 5° 다르다(2026-11-05 float64로 확인, 출처 기록 없음). L15–L18 평가는 모두 앞의 값을 썼고, 학습 자료 평균으로 바꿔도 W1 + L18의 착용자 τ² 변화는 걷기 −29.8 → −30.1 %, 달리기 −29.0 → −29.4 %로 재실행 산포 안이다(`run_system.py --calibration-target`, `results/calibration_target_20261105/`) |
| 정확도 | ETRI suit14 100 Hz 교차평가, denoise 1: 10 / 20 / 50 ms 앞 MAE 0.23 / 0.34 / 1.03°, 지연 2.45 ms |

## 사용

`protomotions.maniflow.loader.DEFAULT_H1_CHECKPOINT`가 이 파일을 가리킨다.
`run_system.py`, `AssistController`, 진자 env의 실제 HLP 연결은 경로를 주지 않으면 이 파일을 쓴다.

```python
from protomotions.maniflow.loader import DEFAULT_H1_CHECKPOINT, load_maniflow_policy
policy, cfg, info = load_maniflow_policy(DEFAULT_H1_CHECKPOINT, device="cuda:0")  # EMA 가중치
```

ManiFlow 저장소는 `~/Projects/ManiFlow_Policy/ManiFlow`에서 찾는다.
다른 위치면 `maniflow_root` 인자나 `protomotions/maniflow/loader.py`의 탐색 규칙을 따른다.
필요한 패키지는 `torch, einops, timm, termcolor, zarr, dill, hydra-core, omegaconf`다.
