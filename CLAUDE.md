# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ProtoMotions3: GPU-accelerated simulation + RL framework for physically simulated humanoids and humanoid robots. Simulators: IsaacGym, IsaacLab, Newton, Genesis, MuJoCo. Algorithms: PPO, AMP, ASE, MaskedMimic. Python 3.8+, Apache-2.0.

This fork hosts the **AI-SBC hip-assist research** (HLP = ManiFlow flexion predictor, LLP = assist-torque RL). Single reference point for structure, terminology, file map, experiment history, roadmap and the code-study checklist: **`docs/AI_SBC_FRAMEWORK.md`**. ManiFlow integration layer: `protomotions/maniflow/` (see its README) — simulation code must go through this package, never `import maniflow` directly.

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

### ETRI skeleton suit (Newton) — mimic tracker = 시뮬 속 "착용자" 대역
```bash
# 재생 (mesh 에셋으로 시각화, 20초 자동 전환)
python protomotions/inference_agent.py \
    --checkpoint checkpoints/v18_2_newton_suit_passive_cable/last.ckpt \
    --motion-file data/motion_for_trackers/skeleton_torque_suit_motions_11+koo_4.pt \
    --simulator newton --num-envs 1 --cycle-seconds 20 \
    --overrides "robot.asset.asset_file_name=mjcf/skeleton_torque_suit_mesh.xml"

# v18_2에서 warm start 재학습
python protomotions/train_agent.py --robot-name skeleton_torque_suit_passive_cable --simulator newton \
    --experiment-path examples/experiments/mimic/mlp.py --experiment-name <new_exp> \
    --checkpoint checkpoints/v18_2_newton_suit_passive_cable/last.ckpt \
    --motion-file data/motion_for_trackers/skeleton_torque_suit_motions_11+koo_4.pt \
    --num-envs 2048 --batch-size 8192
```
- suit 체크포인트에는 suit 모션 파일(27 DOF)만. 에셋(`protomotions/data/assets/mjcf/`):
  `skeleton_torque_suit.xml` = 학습용(hip ring capsule, OOM 방지) / `skeleton_torque_suit_mesh.xml` = 시각화용(cylinder); `31dof/`도 같은 쌍.
- 비suit(plain) skeleton의 XML·체크포인트·모션 파일은 **레포에 없음** (`skeleton.py`/`skeleton_torque.py`가 참조만 함) — skeleton 작업은 전부 suit 로봇 기준.
- 체크포인트: 학습 중 `results/<exp>/`(로컬, gitignore) → 보관본은 `checkpoints/<version>/` + `INFO.md`(git/LFS). 현존: `v18_newton_suit_passive_cable`, `v18_2_newton_suit_passive_cable`.

### AI-SBC LLP (hip pendulum, Newton, 물리 400 Hz / policy 100 Hz)
```bash
python protomotions/train_agent.py --robot-name hip_pendulum --simulator newton \
    --experiment-path examples/experiments/assist_pendulum/mlp.py \
    --experiment-name assist_pendulum --motion-file none --num-envs 4096 --batch-size 16384
python protomotions/inference_agent.py --checkpoint results/assist_pendulum_v2/last.ckpt \
    --simulator newton --num-envs 16 --headless --full-eval   # baseline: --overrides "env.assist_torque_limit=0"
```

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

## Gotchas

- `resolved_configs*.pt` are pickles → `torch.load(..., weights_only=False)`. Resume ignores CLI `--overrides`.
- `common_naming_to_robot_body_names` values must be **lists**, not strings.
- `MdpComponent` `static_params` named `threshold`/`weight`/`min_value` are metadata and never reach the compute fn.
- Newton: `TORQUE`/`PROPORTIONAL` wiring is broken (`_update_torques` never called) → use `BUILT_IN_PD` + qfrc injection; marker updates are no-op (draw with viewer `log_lines/log_arrows`).
- Pre-existing F822 errors in `component_factories.py` `__all__` — leave them.
- Research-side pitfalls (RNG pairing, fps, zarr versions): framework doc §8.

## Code Standards

- Pre-commit: Ruff lint/format, `typos`, Apache-2.0 header on every `.py` except setup.py — copy the header **verbatim** from an existing file (e.g. `protomotions/maniflow/angle_estimator.py`); no abbreviated form.
- Commits: `git commit -s` (DCO); gitmoji prefix + English subject.
- OOP (`nn.Module` subclasses) for model architectures; functional style for data pipelines.
