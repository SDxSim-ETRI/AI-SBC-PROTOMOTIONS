# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ProtoMotions3: GPU-accelerated simulation + RL framework for physically simulated humanoids and humanoid robots. Simulators: IsaacGym, IsaacLab, Newton, Genesis, MuJoCo. Algorithms: PPO, AMP, ASE, MaskedMimic. Python 3.8+, Apache-2.0.

This fork hosts the **AI-SBC hip-assist research** (HLP = ManiFlow flexion predictor, LLP = assist-torque RL). Single reference point for structure, terminology, file map, experiment history, roadmap and the code-study checklist: **`docs/AI_SBC_FRAMEWORK.md`**. 어떤 버전이 무엇이고 지금 무엇을 쓰는지: **`docs/AI_SBC_VERSIONS.md`** (브라우저용 `docs/versions.html`). ManiFlow integration layer: `protomotions/maniflow/` (see its README) — simulation code must go through this package, never `import maniflow` directly.

## Commands

### Setup
```bash
pip install -e .
pip install -r requirements_<sim>.txt   # isaacgym | isaaclab | newton | genesis | mujoco
```
- Newton: tested against 1.0.0 (`pip install "newton[examples]"`). MuJoCo backend: CPU-only, `num_envs=1`, for quick checks.
- Conda envs: `sbc` = ProtoMotions/Newton + ManiFlow online inference; `maniflow` = ManiFlow training (framework doc §4.3).

### Training / inference (generic)
```bash
python protomotions/train_agent.py --robot-name g1 --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp.py --experiment-name my_exp \
    --motion-file data/motion_for_trackers/g1_bones_seed_mini.pt --num-envs 4096 --batch-size 16384 \
    --overrides agent.config.learning_rate=0.0001   # overrides are baked into resolved_configs.pt

python protomotions/inference_agent.py --checkpoint <run>/last.ckpt \
    --motion-file <motions>.pt --simulator newton --num-envs 16   # simulator may differ from training (sim2sim)
```
Never pass `--robot-name` at inference — the robot config is loaded from the checkpoint's `resolved_configs_inference.pt`.

### ETRI skeleton suit (Newton) — 착용자 tracker (2026-09-12 재학습 완료)

suit mimic tracker 가중치는 2026-09-03에 전부 삭제됐었다(`checkpoints/<ver>/INFO.md`,
`resolved_configs*.yaml`, `tasks/*/` 스크립트만 남음). 2026-09-11에 **새 로봇이 넘어오기
전까지 이전 인체모델(`skeleton_torque_suit`, 슈트 관절 4개 토크 0 = `_passive_cable`)로
통합 시스템을 완성**하기로 재조정 → 같은 레시피로 다시 학습해 `checkpoints/v30`(walk),
`v31`(15모션)로 보관했다. 재현 스크립트:

```bash
bash examples/experiments/assist_skeleton/train_tracker_walk.sh       # walk 단일 → results/mimic_suit_passive_walk/
bash examples/experiments/assist_skeleton/train_tracker_motions15.sh  # 15모션, walk에서 warm start
```
- 에셋: `protomotions/data/assets/mjcf/skeleton_torque_suit.xml` — **해부학적 뼈 메시
  (그리기 전용, contype=0) + 캡슐/박스 충돌**이 한 파일에 들어 있다(2026-09-13 통합).
  충돌·질량·관성은 그 전과 같아 이전 체크포인트가 모두 유효하다. Newton은 headless일 때
  시각 geom을 아예 읽지 않는다(`parse_visuals=not headless`) — 안 그러면 env당 90개씩
  늘어 4096 env 학습이 3.5 → 14 s/epoch로 느려진다. `_mesh.xml`은 옛 시각화 전용 쌍둥이.
  suit 모션(27 DOF) `data/motion_for_trackers/skeleton_torque_suit_*.pt`.
  plain skeleton XML은 레포에 없음. robot config의 `passive/active_cable`은 슈트
  slide 관절 4개의 토크 0/구동 차이일 뿐 — 보조는 항상 hip qfrc 주입.
- 체크포인트 보관 규약: 학습 중 `results/<exp>/`(로컬, gitignore) → 보관본은
  `checkpoints/<version>/` + `INFO.md`(git/LFS).

### AI-SBC LLP (hip pendulum, Newton, 물리 400 Hz / policy 100 Hz)
```bash
python protomotions/train_agent.py --robot-name hip_pendulum --simulator newton \
    --experiment-path examples/experiments/assist_pendulum/mlp.py \
    --experiment-name assist_pendulum --motion-file none --num-envs 4096 --batch-size 16384
python protomotions/inference_agent.py --checkpoint results/assist_pendulum_v2/last.ckpt \
    --simulator newton --num-envs 16 --headless --full-eval   # baseline: --overrides "env.assist_torque_limit=0"
```
LLP v3(실 HLP 150 ms chunk 형식) = `results/assist_pendulum_v3/`, 실험 파일 `mlp_v3.py`,
A/B/C 평가 `eval_abc.py` (framework doc §6 Phase C).

### AI-SBC 통합 시스템 — HLP + LLP → 스켈레톤 (framework doc §3.5, 버전 표 `docs/AI_SBC_VERSIONS.md`)
독립 제어기 `protomotions/assist/AssistController`(env 비의존, 실기기 후보) + 서브스텝 훅
러너. 상세는 `examples/experiments/assist_skeleton/README.md`.
```bash
python -m protomotions.assist.verify_controller --checkpoint results/assist_pendulum_v3/last.ckpt  # 진자 env와 등가성 (PASS)
python examples/experiments/assist_skeleton/run_system.py --human mocap_pd --seconds 30            # 골반 고정 대역
python examples/experiments/assist_skeleton/run_system.py --human tracker \
    --tracker-checkpoint results/mimic_suit_passive_walk/last.ckpt --seconds 30                    # 보행 착용자
DISPLAY=:1 python examples/experiments/assist_skeleton/run_system.py --human mocap_pd --record     # 뷰어/mp4
python examples/experiments/assist_skeleton/run_system.py ... --torque-lpf-hz 12                    # 출력 저역통과(구동기 대역)
# 보행 계열 모션만 모은 학습·평가 세트 (walk 0.75 / run 0.25)
python examples/experiments/assist_skeleton/make_gait_motions.py
# LLP 학습 (착용자 = 동결 tracker, 실 HLP; env 10 ms, 4096 env ≈3.5 s/epoch)
python protomotions/train_agent.py --robot-name skeleton_torque_suit_passive_cable --simulator newton \
    --experiment-path examples/experiments/assist_skeleton/llp_walk.py --experiment-name llp_gait_v6 \
    --motion-file data/motion_for_trackers/skeleton_torque_suit_gait.pt --num-envs 4096 --batch-size 16384 \
    --tracker-checkpoint checkpoints/v31_newton_suit_passive_motions15/last.ckpt --assist-limit 40
```
결과 요지: 진자 LLP는 강체 다리에서만 방향이 맞고(corr 0.80) 관절 다리·보행에서는 맞지
않는다(0.19/≈0) → 통합 env(`SkeletonAssistEnv`)에서 재학습해 walk −32 %(v3), 15모션
−22 %(v4), 착용자 앙상블 −32/−26 %(v5).

평가는 **여러 비용 지표를 한 표에** 올려 읽는다 (아래 gotcha 참고):
```bash
python examples/experiments/assist_skeleton/compare_runs.py --plot docs/oracle.png results/oracle/*
# 오라클 탐침 (학습된 정책 대신 규칙 토크를 강제 주입 — 상한과 인과를 본다)
python examples/experiments/assist_skeleton/run_system.py ... --oracle beta --oracle-beta 0.5
python examples/experiments/assist_skeleton/run_system.py ... --oracle phase --oracle-window 0,10 --oracle-torque -40
``` 보관: `checkpoints/llp_v{3,4,5}_*`, tracker
`checkpoints/v30_newton_suit_passive_walk`·`v31_newton_suit_passive_motions15`.

### Tests / lint
```bash
pytest protomotions/tests/                  # or one file
pre-commit run --files <file1> <file2>      # NEVER `pre-commit run --all-files` (100+ unrelated diffs)
```
If pre-commit/ruff are absent from the active env, fall back to `python -m py_compile` + running the code.
ONNX export of BeyondMimic trackers: `deployment/export_bm_tracker_onnx.py` (adapt obs keys for other configs).

## Architecture — where to look

These files carry the design docs; read the relevant one before changing that area.
- `protomotions/train_agent.py` — config pipeline: `robot_factory()`/`simulator_factory()` → experiment file (`configure_robot_and_simulator`, `env_config`, `agent_config`; slot signatures in `examples/experiments/format.py` — but its `apply_inference_overrides` signature is stale, so copy a working experiment such as `assist_pendulum/mlp.py` instead) → CLI `--overrides` → pickled `resolved_configs.pt`. Resume loads the pickle and does **not** re-run the experiment file. `apply_inference_overrides()` runs at **training** time (`train_agent.py:853`) and is baked into `resolved_configs_inference.pt`; inference just loads that pickle → `apply_backward_compatibility_fixes` → CLI overrides (never re-runs the experiment file).
- `protomotions/envs/base_env/env.py` — `step()`: action processing → `simulator.step()` (decimated substeps) → state/context update → control components → obs → rewards → terminations → reset.
- `protomotions/envs/mdp_component.py`, `context_views.py`, `component_factories.py` — obs/reward/termination = `MdpComponent(compute_func, dynamic_vars={...: EnvContext.current.dof_pos}, static_params={...})`, managed by `ComponentManager`; `FieldPath` yields a path string at class level and the tensor at instance level.
- `protomotions/simulator/base_simulator/simulator.py`, `simulator_state.py` — abstract `Simulator` + `RobotState`/`StateConversion` (COMMON vs SIMULATOR body/DOF ordering). Common quaternion = xyzw (IsaacGym/IsaacLab convert from wxyz). Control modes `BUILT_IN_PD` / `PROPORTIONAL` / `TORQUE`. Friction combine: PhysX AVERAGE vs Newton MAX (`convert_friction_for_simulator`).
- `protomotions/agents/` — `BaseAgent.fit()` (rollout → advantages → minibatch optimize → periodic eval); PPO ← AMP ← ASE, Mimic/ADD ← AMP; MaskedMimic = distillation. Models are `TensorDictModuleBase` using `nn.LazyLinear` (shapes inferred on first forward).
- `protomotions/robot_configs/` — per-robot assets/control/body mapping; `KinematicInfo` extracted from MJCF in `__post_init__`.
- `protomotions/components/` — `motion_lib.py`, `pose_lib.py`, `scene_lib.py`, `terrains/`.
- `protomotions/utils/simulator_imports.py` — IsaacGym/IsaacLab must be imported before torch.
- `protomotions/assist/` — HLP+LLP 독립 제어기(`controller.py`), actor 로더(`llp_policy.py`), 스켈레톤 특징량(`skeleton_features.py`), 등가성 검증. `protomotions/simulator/newton/substep_hook.py` — env step 안에서 10 ms 콜백(LLP)을 돌리는 Newton 확장(CUDA graph 끔).

## Gotchas

- `resolved_configs*.pt` are pickles → `torch.load(..., weights_only=False)`. Resume ignores CLI `--overrides`.
- `common_naming_to_robot_body_names` values must be **lists**, not strings.
- `MdpComponent` `static_params` named `threshold`/`weight`/`min_value` are metadata and never reach the compute fn.
- Newton: `TORQUE`/`PROPORTIONAL` wiring is broken (`_update_torques` never called) → use `BUILT_IN_PD` + qfrc injection; marker updates are no-op (draw with viewer `log_lines/log_arrows`).
- tracker env dt(50 ms)는 관측이 묶여 있어 못 바꾼다 — 다른 주기의 제어기는 `SubstepHookMixin`으로 서브스텝 안에서 돌릴 것. 훅 시뮬레이터는 CUDA graph를 시도하지 않는다(캡처 중 간헐 크래시).
- Pre-existing F822 errors in `component_factories.py` `__all__` — leave them.
- suit MJCF의 뼈 메시는 **그리기 전용**(contype=0)이고 headless에서는 로드되지 않는다.
  대규모 학습이 갑자기 4배 느려지면 `parse_visuals` 경로부터 볼 것.
- **`|τ·ω|` 절감률만 보면 속는다.** 이 값은 착용자 다리가 덜 흔들리기만 해도 내려간다 —
  v10의 "−25.4 %"는 대부분 각속도 감소였고, 근활성 프록시 τ²로 재면 +7 %로 나빠진다
  (2026-09-15 오라클 진단, `docs/NEXT_SESSION.md` §2). 평가할 때 τ²·전진 속도·몸체
  추종 오차를 **항상 같이** 볼 것 — `compare_runs.py`가 한 표로 찍어 준다.
  비교 수치는 백분율점(pt) 대신 두 값을 그대로 적거나 절대 단위로 쓴다 (사용자 지시).
- Research-side pitfalls (RNG pairing, fps, zarr versions): framework doc §8.

## Code Standards

- Pre-commit: Ruff lint/format, `typos`, Apache-2.0 header on every `.py` except setup.py — copy the header **verbatim** from an existing file (e.g. `protomotions/maniflow/angle_estimator.py`); no abbreviated form.
- Commits: `git commit -s` (DCO); gitmoji prefix + English subject.
- OOP (`nn.Module` subclasses) for model architectures; functional style for data pipelines.
