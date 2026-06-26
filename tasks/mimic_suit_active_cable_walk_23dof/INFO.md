# mimic_suit_active_cable_walk_23dof

| 항목 | 내용 |
|------|------|
| robot | `skeleton_torque_suit_active_cable` (27-DOF, active cable) |
| simulator | Newton (평지) / IsaacLab (평지 + 비평지) |
| 모션 | `skeleton_torque_suit_walk.pt` (walk 1클립) |
| 최초 warm_start | `tasks/mimic_suit_active_cable_motions14_23dof/output_newton/score_based.ckpt` |

---

## 학습 계보

```
mimic_suit_active_cable_motions14_23dof (14클립)
  └─ output_newton ──warm_start──▶ output_newton_flat   (Newton, 평지 walk 특화)
                   ──warm_start──▶ output_isaaclab_flat  (IsaacLab, 평지 walk 특화)
                                          │
                                   ┌──────┴───────┐
                                   ▼              ▼
                             discrete05       (추후 확장)
                                   │
                                   ▼
                             discrete15
                             ┌─────┴───────────────┐
                             ▼                     ▼
                          sand05          smooth_slope13
                             │                     │
                             └──────────┬──────────┘
                                        ▼
                             mixed_slope13_discrete15
```

---

## flat 재학습 경로

`output_isaaclab_flat/`에서 추가 학습 시 사용하는 경로:

| 항목 | 경로 |
|------|------|
| experiment-path | `tasks/mimic_suit_active_cable_walk_23dof/mlp/mlp_isaaclab_suit_active_cable_flat.py` |
| save-dir | `tasks/mimic_suit_active_cable_walk_23dof/output_isaaclab_flat` |
| checkpoint (resume) | `tasks/mimic_suit_active_cable_walk_23dof/output_isaaclab_flat/last.ckpt` |
| checkpoint (warm_start) | `tasks/mimic_suit_active_cable_motions14_23dof/output_newton/score_based.ckpt` |
| motion-file | `data/motion_for_trackers/skeleton_torque_suit_walk.pt` |

> `last.ckpt`가 존재하면 자동 resume, 없으면 `--checkpoint`의 warm_start가 적용됨.

`output_newton_flat/` 재학습:

| 항목 | 경로 |
|------|------|
| experiment-path | `tasks/mimic_suit_active_cable_walk_23dof/mlp/mlp_newton_suit_active_cable_flat.py` |
| save-dir | `tasks/mimic_suit_active_cable_walk_23dof/output_newton_flat` |

---

## 폴더 구조

```
tasks/mimic_suit_active_cable_walk_23dof/
├── mlp/                                   ← 실험 설정 파이썬 파일 (이 task 전용)
│   ├── mlp_isaaclab_suit_active_cable_flat.py
│   ├── mlp_newton_suit_active_cable_flat.py
│   ├── mlp_isaaclab_suit_active_cable_discrete05.py
│   ├── mlp_isaaclab_suit_active_cable_discrete15.py
│   ├── mlp_isaaclab_suit_active_cable_sand05.py
│   ├── mlp_isaaclab_suit_active_cable_smooth13.py
│   └── mlp_isaaclab_suit_active_cable_mixed_slope13_discrete15.py
│
├── output_newton_flat/                    ← Newton 평지 학습 결과
├── output_isaaclab_flat/                  ← IsaacLab 평지 학습 결과 (train_isaaclab_initial.sh)
│
├── output_isaaclab_discrete/              ← IsaacLab 불규칙 장애물 계열 (train_isaaclab_discrete.sh)
│   ├── discrete05/                        ←   discrete 5cm  │ warm_start: output_isaaclab_flat
│   └── discrete15/                        ←   discrete 15cm │ warm_start: discrete05
│
├── output_isaaclab_rough_slope/           ← IsaacLab 거친 경사 계열 (train_isaaclab_rough_slope.sh)
│   └── sand05/                            ←   rough_slope 5.7° ±5cm │ warm_start: discrete15
│
├── output_isaaclab_smooth_slope/          ← IsaacLab 완만한 경사 계열 (train_isaaclab_smooth_slope.sh)
│   └── smooth_slope13/                    ←   smooth_slope 13° │ warm_start: discrete15
│
├── output_isaaclab_mixed/                 ← IsaacLab 혼합 지형 계열 (train_isaaclab_mixed.sh)
│   └── mixed_slope13_discrete15/          ←   smooth_slope 40% + discrete 40% + flat 20%
│
└── zarr_data/                             ← zarr 에피소드 수집 결과
    └── {terrain_tag}/
        └── {terrain_tag}-{timestamp}.zarr
```

각 `output_*/하위폴더/`에는 `config.yaml`이 있어 학습 파라미터를 기록.  
`-in <하위폴더>` 옵션으로 해당 config.yaml을 읽어 학습 실행.

---

## terrain 이름 생성 규칙

이름은 **비율(proportion)이 아닌 물리적 특성값**을 사용한다.

| terrain 종류 | 특성값 기준 | 예시 |
|-------------|------------|------|
| discrete obstacles | 최대 장애물 높이 (cm) | `discrete05` (5cm), `discrete15` (15cm) |
| rough_slope (모래) | amplitude (cm) | `sand05` (±5cm) |
| smooth_slope | 최대 경사각 (°) | `smooth_slope13` (13°) |
| 혼합 지형 | 구성 지형 특성값 조합 | `mixed_slope13_discrete15` |

규칙 요약:
- 단일 지형: `{종류}{특성값}` — `discrete15`, `smooth_slope13`
- 혼합 지형: `mixed_{지형A특성값}_{지형B특성값}` — 주요 지형 순서로 나열
- 비율(30%, 80% 등)은 이름에 쓰지 않음 → 비율은 mlp 파일 내 `TerrainConfig`에서 확인

---

## -in 패턴으로 학습 실행

```bash
cd /home/user/ProtoMotions

# discrete 계열
bash tasks/mimic_suit_active_cable_walk_23dof/train_isaaclab_discrete.sh -in discrete05
bash tasks/mimic_suit_active_cable_walk_23dof/train_isaaclab_discrete.sh -in discrete15

# rough_slope 계열
bash tasks/mimic_suit_active_cable_walk_23dof/train_isaaclab_rough_slope.sh -in sand05

# smooth_slope 계열
bash tasks/mimic_suit_active_cable_walk_23dof/train_isaaclab_smooth_slope.sh -in smooth_slope13

# mixed 계열
bash tasks/mimic_suit_active_cable_walk_23dof/train_isaaclab_mixed.sh -in mixed_slope13_discrete15
```

스크립트가 읽는 config.yaml 위치: `output_isaaclab_{type}/{dirname}/config.yaml`

---

## zarr 수집

```bash
cd /home/user/ProtoMotions

# 기본 (10 envs)
bash tasks/mimic_suit_active_cable_walk_23dof/collect_walk_zarr.sh <checkpoint> <terrain_tag>

# 1024 envs (대량 수집)
bash tasks/mimic_suit_active_cable_walk_23dof/collect_walk_zarr.sh \
    tasks/mimic_suit_active_cable_walk_23dof/output_isaaclab_flat/score_based.ckpt flat 1024

bash tasks/mimic_suit_active_cable_walk_23dof/collect_walk_zarr.sh \
    tasks/mimic_suit_active_cable_walk_23dof/output_isaaclab_discrete/discrete05/score_based.ckpt discrete05 1024
```

출력: `zarr_data/{terrain_tag}/{terrain_tag}-{timestamp}.zarr`

---

## 참고: TerrainConfig 비율 배열 인덱스 순서

```python
terrain_proportions = [smooth_slope, rough_slope, stairs_up, stairs_down, discrete, stepping, poles, flat]
#                       0             1            2          3             4         5         6      7
```

Newton은 비평지 지형 미지원 → IsaacLab 전용 학습.
