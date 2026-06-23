# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Test trained agents and visualize their behavior.

This script loads trained checkpoints and runs agents in the simulation environment
for inference, visualization, and analysis. It supports interactive controls,
video recording, and motion playback.

Motion Playback
---------------

For kinematic motion playback (no physics simulation)::

    PYTHON_PATH protomotions/inference_agent.py \\
        --config-name play_motion \\
        +robot=smpl \\
        +simulator=isaacgym \\
        +motion_file=data/motions/walk.motion

Inference Config System
------------------------

Inference loads frozen configs from resolved_configs_inference.pt and applies inference-specific overrides.

Override Priority:

1. CLI overrides (--overrides) - Highest (runtime control)
2. Experiment inference overrides (apply_inference_overrides) - High (experiment-specific inference settings)
3. Frozen configs from resolved_configs.pt - Lowest (exact training configs)

Note: configure_robot_and_simulator() is NOT called during inference (already baked into frozen configs).

Keyboard Controls
-----------------

During inference, these controls are available:

- **J**: Apply random forces to test robustness
- **R**: Reset all environments
- **O**: Toggle camera view
- **[** / **]**: Orbit camera left/right (15° per press)
- **B**: Rear view (camera behind robot, 180°)
- **N**: Front view (camera in front of robot, 0°)
- **L**: Start/stop video recording
- **Q**: Quit

Example
-------
>>> # Test with custom settings
>>> # PYTHON_PATH protomotions/inference_agent.py \\
>>> #     +robot=smpl \\
>>> #     +simulator=isaacgym \\
>>> #     +checkpoint=results/tracker/last.ckpt \\
>>> #     motion_file=data/motions/test.pt \\
>>> #     num_envs=16
"""


def create_parser():
    """Create and configure the argument parser for inference."""
    parser = argparse.ArgumentParser(
        description="Test trained reinforcement learning agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint file to test"
    )
    # Optional arguments
    parser.add_argument(
        "--full-eval",
        action="store_true",
        default=False,
        help="Run full evaluation instead of simple inference",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run simulation in headless mode",
    )
    parser.add_argument(
        "--simulator",
        type=str,
        required=True,
        help="Simulator to use (e.g., 'isaacgym', 'isaaclab', 'newton', 'genesis')",
    )
    parser.add_argument(
        "--num-envs", type=int, default=1, help="Number of parallel environments to run"
    )
    parser.add_argument(
        "--motion-file",
        type=str,
        required=False,
        default=None,
        help="Path to motion file for inference. If not provided, will use the motion file from the checkpoint.",
    )
    parser.add_argument(
        "--scenes-file", type=str, default=None, help="Path to scenes file (optional)"
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Config overrides in format key=value (e.g., env.max_episode_length=5000 simulator.headless=True)",
    )
    parser.add_argument(
        "--auto-record",
        action="store_true",
        default=False,
        help="Automatically start recording video on launch and exit after --record-steps steps",
    )
    parser.add_argument(
        "--record-steps",
        type=int,
        default=300,
        help="Number of steps to record when --auto-record is set (default: 300 ≈ 10s at 30fps)",
    )
    parser.add_argument(
        "--recording-path",
        type=str,
        default=None,
        help="Override recording output directory (default: output/renderings/<experiment_name>-<datetime>)",
    )
    parser.add_argument(
        "--cycle-seconds",
        type=float,
        default=0.0,
        help="Auto-advance to next motion every N seconds (0 = manual R key only)",
    )
    parser.add_argument(
        "--use-skin",
        action="store_true",
        default=False,
        help="Newton only: open a separate MuJoCo window showing bone mesh skin alongside the Newton simulation",
    )
    parser.add_argument(
        "--use-skin-cable",
        action="store_true",
        default=False,
        help="Newton only: bone mesh skin + cable tendon lines (implies --use-skin)",
    )

    return parser


# Parse arguments first (argparse is safe, doesn't import torch)
import argparse  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402

parser = create_parser()
args, unknown_args = parser.parse_known_args()

# Import simulator before torch - isaacgym/isaaclab must be imported before torch
# This also returns AppLauncher if using isaaclab, None otherwise
from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch(args.simulator)

# Now safe to import everything else including torch
import logging  # noqa: E402
from pathlib import Path  # noqa: E402
import torch  # noqa: E402
from protomotions.utils.hydra_replacement import get_class  # noqa: E402
from protomotions.utils.fabric_config import FabricConfig  # noqa: E402
from lightning.fabric import Fabric  # noqa: E402
from dataclasses import asdict  # noqa: E402
from protomotions.utils.config_utils import clean_dict_for_storage  # noqa: E402

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

log = logging.getLogger(__name__)


# def tmp_enable_domain_randomization(robot_cfg, simulator_cfg, env_cfg):
#     """Temporary function to enable domain randomization for testing.

#     TODO: find a better way for sophisticated tmp inference overrides beyond CLI.
#     """
#     from protomotions.simulator.base_simulator.config import (
#         # FrictionDomainRandomizationConfig,
#         CenterOfMassDomainRandomizationConfig,
#         DomainRandomizationConfig,
#     )

#     # env_cfg.terrain.sim_config.static_friction = 0.01
#     # env_cfg.terrain.sim_config.dynamic_friction = 0.01

#     simulator_cfg.domain_randomization = DomainRandomizationConfig(
#         # Uncomment to enable action noise and friction randomization:
#         # action_noise=ActionNoiseDomainRandomizationConfig(
#         #     action_noise_range=(-0.01, 0.01),
#         #     dof_names=[".*"],
#         #     dof_indices=None
#         # ),
#         # friction=FrictionDomainRandomizationConfig(
#         #     num_buckets=64,
#         #     static_friction_range=(0.0, 1.0),
#         #     dynamic_friction_range=(0.0, 1.0),
#         #     restitution_range=(0.0, 0.0),
#         #     body_names=[".*"],
#         #     body_indices=None
#         # ),
#     )
#     log.info("Enabled domain randomization for testing")


def main():
    # Re-use the parser and args from module level
    global parser, args
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)

    # Load frozen configs from resolved_configs.pt (exact reproducibility)
    resolved_configs_path = checkpoint.parent / "resolved_configs_inference.pt"
    assert (
        resolved_configs_path.exists()
    ), f"Could not find resolved configs at {resolved_configs_path}"

    log.info(f"Loading resolved configs from {resolved_configs_path}")
    resolved_configs = torch.load(
        resolved_configs_path, map_location="cpu", weights_only=False
    )

    robot_config = resolved_configs["robot"]
    simulator_config = resolved_configs["simulator"]
    terrain_config = resolved_configs.get("terrain")
    scene_lib_config = resolved_configs["scene_lib"]
    motion_lib_config = resolved_configs["motion_lib"]
    env_config = resolved_configs["env"]
    agent_config = resolved_configs["agent"]

    # Check if we need to switch simulators
    # Extract simulator name from current config's _target_
    current_simulator = simulator_config._target_.split(
        "."
    )[
        -3
    ]  # e.g., "isaacgym" from "protomotions.simulator.isaacgym.simulator.IsaacGymSimulator"

    if args.simulator != current_simulator:
        log.info(
            f"Switching simulator from '{current_simulator}' (training) to '{args.simulator}' (inference)"
        )
        from protomotions.simulator.factory import update_simulator_config_for_test

        simulator_config = update_simulator_config_for_test(
            current_simulator_config=simulator_config,
            new_simulator=args.simulator,
            robot_config=robot_config,
        )
    # Apply backward compatibility fixes for old checkpoints
    from protomotions.utils.inference_utils import apply_backward_compatibility_fixes

    apply_backward_compatibility_fixes(robot_config, simulator_config, env_config)

    # # Temporary: Enable domain randomization for testing (uncomment to use)
    # tmp_enable_domain_randomization(robot_config, simulator_config, env_config)

    # from protomotions.robot_configs.base import ControlType
    # robot_config.control.control_type = ControlType.PROPORTIONAL

    # Apply CLI runtime overrides
    if args.num_envs is not None:
        log.info(f"CLI override: num_envs = {args.num_envs}")
        simulator_config.num_envs = args.num_envs

    if args.motion_file is not None:
        log.info(f"CLI override: motion_file = {args.motion_file}")
        motion_lib_config.motion_file = args.motion_file  # Always present

    if args.scenes_file is not None:
        log.info(f"CLI override: scenes_file = {args.scenes_file}")
        scene_lib_config.scene_file = args.scenes_file  # Always present

    if args.headless is not None:
        log.info(f"CLI override: headless = {args.headless}")
        simulator_config.headless = args.headless

    # Parse and apply general CLI overrides
    from protomotions.utils.config_utils import (
        parse_cli_overrides,
        apply_config_overrides,
    )

    cli_overrides = parse_cli_overrides(args.overrides) if args.overrides else None

    if cli_overrides:
        apply_config_overrides(
            cli_overrides,
            env_config,
            simulator_config,
            robot_config,
            agent_config,
            terrain_config,
            motion_lib_config,
            scene_lib_config,
        )

    # Create fabric config for inference (simplified)
    # MuJoCo is CPU-only, so force CPU accelerator
    accelerator = "cpu" if args.simulator == "mujoco" else "gpu"
    fabric_config = FabricConfig(
        accelerator=accelerator,
        devices=1,
        num_nodes=1,
        loggers=[],  # No loggers needed for inference
        callbacks=[],  # No callbacks needed for inference
    )
    fabric: Fabric = Fabric(**asdict(fabric_config))
    fabric.launch()

    # Setup IsaacLab simulation_app if using IsaacLab simulator
    simulator_extra_params = {}
    if args.simulator == "isaaclab":
        app_launcher_flags = {"headless": args.headless, "device": str(fabric.device)}
        app_launcher = AppLauncher(app_launcher_flags)
        simulator_extra_params["simulation_app"] = app_launcher.app
        # Suppress the "Simulation Settings" overlay that pops up by default
        import carb
        carb.settings.get_settings().set(
            "/persistent/physics/visualizationSimulationOutput", False
        )

    # Convert friction for simulator compatibility
    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator

    terrain_config, simulator_config = convert_friction_for_simulator(
        terrain_config, simulator_config
    )

    # Create components
    from protomotions.utils.component_builder import build_all_components

    save_dir_for_weights = (
        getattr(env_config, "save_dir", None)
        if hasattr(env_config, "save_dir")
        else None
    )
    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=fabric.device,
        save_dir=save_dir_for_weights,
        **simulator_extra_params,  # simulation_app for IsaacLab
    )

    terrain = components["terrain"]
    scene_lib = components["scene_lib"]
    motion_lib = components["motion_lib"]
    simulator = components["simulator"]

    # Create env (auto-initializes simulator)
    from protomotions.envs.base_env.env import BaseEnv

    EnvClass = get_class(env_config._target_)
    env: BaseEnv = EnvClass(
        config=env_config,
        robot_config=robot_config,
        device=fabric.device,
        terrain=terrain,
        scene_lib=scene_lib,
        motion_lib=motion_lib,
        simulator=simulator,
    )

    # Determine root_dir for agent based on checkpoint path
    agent_kwargs = {}
    checkpoint_path = Path(args.checkpoint)
    agent_kwargs["root_dir"] = checkpoint_path.parent

    # Create agent
    from protomotions.agents.base_agent.agent import BaseAgent

    # agent_config.evaluator.eval_metric_keys = [
    #     "gt_err",
    #     "gr_err_degrees",
    #     "pow_rew",
    #     "gt_left_foot_contact",
    #     "gt_right_foot_contact",
    #     "pred_left_foot_contact",
    #     "pred_right_foot_contact"
    # ]
    AgentClass = get_class(agent_config._target_)
    agent: BaseAgent = AgentClass(
        config=agent_config, env=env, fabric=fabric, **agent_kwargs
    )

    agent.setup()
    agent.load(args.checkpoint, load_env=False)

    try:
        if args.full_eval:
            agent.evaluator.eval_count = 0
            evaluation_log, evaluated_score = agent.evaluator.evaluate()

            # Print evaluation metrics
            print("\n" + "=" * 60)
            print("EVALUATION RESULTS")
            print("=" * 60)
            for key, value in sorted(evaluation_log.items()):
                print(f"  {key}: {value:.6f}")
            print("=" * 60)
            if evaluated_score is not None:
                print(f"  Overall Score: {evaluated_score:.6f}")
            print("=" * 60 + "\n")
        elif args.auto_record:
            _cycle = args.cycle_seconds if args.cycle_seconds > 0 else 20.0
            if getattr(args, "recording_path", None):
                _rp = os.path.abspath(args.recording_path)
                os.makedirs(_rp, exist_ok=True)
                env.simulator._user_recording_video_path = os.path.join(_rp, "%s")
            _run_auto_record(agent, env, args.record_steps, cycle_seconds=_cycle)
        else:
            # --use-skin / --use-skin-cable: non-Newton 시뮬레이터에서만 별도 MuJoCo 창 오픈.
            # Newton은 --overrides "robot.asset.asset_file_name=mjcf/..._mesh.xml" 로 자체 처리.
            skin = None
            show_cables = False
            _cls = type(robot_config).__name__
            _stripped = re.sub(r"RobotConfig$", "", _cls)
            _rname = re.sub(r"(?<!^)(?=[A-Z])", "_", _stripped).lower()
            use_skin = getattr(args, "use_skin", False) or getattr(args, "use_skin_cable", False)
            show_cables = getattr(args, "use_skin_cable", False)
            if use_skin and args.simulator != "newton" and _rname in _SKIN_ASSET:
                _asset_root = robot_config.asset.asset_root
                skin = _make_skin_viewer(_rname, _asset_root)
            elif use_skin and args.simulator == "newton":
                log.info(
                    "[use-skin] Newton: --overrides 'robot.asset.asset_file_name=mjcf/..._mesh.xml' 으로 mesh 시각화하세요."
                )
            _run_interactive(agent, env, cycle_seconds=args.cycle_seconds,
                             skin=skin, robot_name=_rname, show_cables=show_cables)
    finally:
        # Ensure simulator viewer is properly closed (prevents hangs)
        if hasattr(env.simulator, "shutdown"):
            env.simulator.shutdown()


def _get_motion_name(env, env_id: int = 0) -> str:
    """Return the current motion name for the given environment."""
    import os

    if env.motion_manager is None:
        return "N/A"
    motion_id = env.motion_manager.motion_ids[env_id].item()
    motion_file = env.motion_lib.motion_files[motion_id]
    return os.path.splitext(os.path.basename(motion_file))[0]


def _print_motion_status(env, step: int, record_steps: int = 0) -> None:
    """Print current motion name, time, and progress to stdout."""
    import os

    if env.motion_manager is None:
        return
    motion_id = env.motion_manager.motion_ids[0].item()
    motion_time = env.motion_manager.motion_times[0].item()
    motion_file = env.motion_lib.motion_files[motion_id]
    motion_name = os.path.splitext(os.path.basename(motion_file))[0]
    motion_length = env.motion_lib.motion_lengths[motion_id].item()
    progress = motion_time / motion_length * 100
    step_info = f"{step+1:4d}/{record_steps}" if record_steps else f"step {step+1:5d}"
    print(
        f"\r[{step_info}] motion: {motion_name:<30s} "
        f"{motion_time:5.2f}s / {motion_length:5.2f}s ({progress:5.1f}%)",
        end="",
        flush=True,
    )


_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
# ImGui menu font ≈ 16px → 2× = 32px subtitle
_SUBTITLE_FONT_SIZE = 32


def _overlay_subtitles_on_frames(frame_dir: str, labels: list) -> None:
    """Burn motion-name subtitles into each PNG frame (top center, 2× menu font)."""
    import glob
    import os

    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype(_FONT_PATH, _SUBTITLE_FONT_SIZE)
    except OSError:
        font = ImageFont.load_default()

    frames = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    print(f"\nOverlaying subtitles on {len(frames)} frames...")

    # determine canonical size from first valid frame
    ref_size = None
    for fpath in frames:
        try:
            ref_size = Image.open(fpath).size
            break
        except Exception:
            continue

    for i, fpath in enumerate(frames):
        label = labels[i] if i < len(labels) else labels[-1]
        try:
            img = Image.open(fpath).convert("RGB")
        except Exception:
            continue

        # ensure all frames match the reference size
        if ref_size and img.size != ref_size:
            img = img.resize(ref_size, Image.LANCZOS)

        draw = ImageDraw.Draw(img)
        w, h = img.size

        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (w - text_w) // 2
        y = 14  # top margin

        # semi-transparent dark background for legibility
        pad = 8
        bg = Image.new("RGBA", img.size, (0, 0, 0, 0))
        bg_draw = ImageDraw.Draw(bg)
        bg_draw.rounded_rectangle(
            [x - pad, y - pad, x + text_w + pad, y + text_h + pad],
            radius=6,
            fill=(0, 0, 0, 160),
        )
        img = img.convert("RGBA")
        img = Image.alpha_composite(img, bg).convert("RGB")
        draw = ImageDraw.Draw(img)

        # white text
        draw.text((x, y), label, font=font, fill=(255, 255, 255))

        img.save(fpath)
    print("Subtitle overlay complete.")


def _build_title(state: dict, env, num_motions: int, cycle_seconds: float) -> str:
    """Build unified title string for all simulators.

    Format: [N/total] MotionName    step: NNN/TTT
    """
    if env.motion_manager is None:
        return ""
    mid = env.motion_manager.motion_ids[0].item()
    mname = os.path.splitext(os.path.basename(env.motion_lib.motion_files[mid]))[0]
    idx_str = f"[{state['cursor']+1}/{num_motions}]"
    total = state.get("total_steps", 0)
    return f"{idx_str} {mname}    step:{state['clip_step']:4d}/{total}"


def _register_motion_title_ui(simulator, state: dict, env, num_motions: int, cycle_seconds: float) -> None:
    """Register Newton ImGui overlay that reads the unified title from state."""

    def _motion_title_ui(imgui):
        title = state.get("current_title", "")
        if not title:
            return
        io = imgui.get_io()
        dw = io.display_size[0]
        imgui.set_next_window_pos(
            imgui.ImVec2(dw / 2, 10),
            pivot=imgui.ImVec2(0.5, 0.0),
        )
        imgui.set_next_window_bg_alpha(0.6)
        flags = (
            imgui.WindowFlags_.no_decoration.value
            | imgui.WindowFlags_.always_auto_resize.value
            | imgui.WindowFlags_.no_saved_settings.value
            | imgui.WindowFlags_.no_focus_on_appearing.value
            | imgui.WindowFlags_.no_nav.value
            | imgui.WindowFlags_.no_move.value
        )
        if imgui.begin("##motion_title", flags=flags):
            imgui.push_font(None, 13.0)
            imgui.text(title)
            imgui.pop_font()
        imgui.end()

    if hasattr(simulator, "viewer") and simulator.viewer is not None and hasattr(simulator.viewer, "register_ui_callback"):
        simulator.viewer.register_ui_callback(_motion_title_ui, position="free")


def _run_auto_record(agent, env, record_steps: int, cycle_seconds: float = 20.0) -> None:
    """Run inference for record_steps steps with automatic video recording."""
    import os
    import sys

    simulator = env.simulator
    agent.eval()

    num_motions_ar = len(env.motion_lib.motion_files)

    def _get_motion_name(motion_id: int) -> str:
        return os.path.splitext(os.path.basename(env.motion_lib.motion_files[motion_id]))[0]

    # step-based cycle counter — use actual policy fps (1/dt) instead of hardcoded 30
    _policy_fps = round(1.0 / simulator.dt) if hasattr(simulator, "dt") and simulator.dt > 0 else 30
    cycle_steps = int(cycle_seconds * _policy_fps) if cycle_seconds > 0 and num_motions_ar > 1 else 0
    cursor = [0]
    steps_in_cycle = [0]  # counts steps since last cycle reset

    # patch sample_motions to always use current cursor motion
    original_sample = env.motion_manager.sample_motions
    def _patched_sample(env_ids, new_motion_ids=None):
        if new_motion_ids is None:
            new_motion_ids = torch.full(
                (len(env_ids),), cursor[0], device=env.device, dtype=torch.long
            )
        original_sample(env_ids, new_motion_ids)
    env.motion_manager.sample_motions = _patched_sample

    _auto_record_state = {"cursor": 0, "clip_step": 0, "cycle_start_time": time.perf_counter(), "current_title": ""}
    _register_motion_title_ui(simulator, _auto_record_state, env, num_motions_ar, 0.0)

    print(f"Auto-record: will capture {record_steps} steps then save video.")
    if cycle_steps > 0:
        print(f"  Motion cycle: every {cycle_seconds:.0f}s (~{cycle_steps} steps), {num_motions_ar} motions")
        for i in range(num_motions_ar):
            print(f"    [{i+1}/{num_motions_ar}] {_get_motion_name(i)}")
    simulator._toggle_video_record()  # start recording

    def _update_title():
        name = _get_motion_name(cursor[0])
        title = f"[{cursor[0]+1}/{num_motions_ar}] {name}  step:{steps_in_cycle[0]:4d}/{cycle_steps if cycle_steps else record_steps}"
        _auto_record_state["current_title"] = title
        try:
            simulator.set_window_title(title)
        except Exception:
            pass

    motion_labels: list = []  # subtitle text per step
    done_indices = None
    for step in range(record_steps):
        if (
            hasattr(simulator, "viewer")
            and simulator.viewer is not None
            and hasattr(simulator.viewer, "is_running")
            and not simulator.viewer.is_running()
        ) or (
            hasattr(simulator, "is_app_running")
            and not simulator.is_app_running()
        ):
            print("\nViewer window closed — stopping recording.")
            break

        # cycle motion when cycle_steps reached
        if cycle_steps > 0 and steps_in_cycle[0] >= cycle_steps:
            cursor[0] = (cursor[0] + 1) % num_motions_ar
            steps_in_cycle[0] = 0
            print(f"\n→ [{cursor[0]+1}/{num_motions_ar}] {_get_motion_name(cursor[0])}")
            done_indices = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

        obs, _ = env.reset(done_indices)
        obs = agent.add_agent_info_to_obs(obs)
        obs_td = agent.obs_dict_to_tensordict(obs)
        model_outs = agent.model(obs_td)
        action = model_outs.get("mean_action", model_outs["action"])
        obs, _rewards, dones, _terminated, _extras = env.step(action)
        done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
        steps_in_cycle[0] += 1
        _print_motion_status(env, step, record_steps)
        _update_title()

        # collect subtitle label for this frame (full title = same as window title)
        motion_labels.append(_auto_record_state.get("current_title", _get_motion_name(cursor[0])))

    print()  # newline after progress line

    # burn subtitles into PNG frames before video compilation
    if motion_labels and hasattr(simulator, "_curr_user_recording_name"):
        frames_dir = os.path.join(simulator._curr_user_recording_name, "_frames")
        _overlay_subtitles_on_frames(frames_dir, motion_labels)

    simulator._toggle_video_record()  # stop recording + trigger mp4 save
    # one extra render call so the mixin flushes the video to disk
    env.simulator.render()


# ---------------------------------------------------------------------------
# Skin-viewer helpers (Newton + MuJoCo bone-mesh window)
# ---------------------------------------------------------------------------

# Maps robot_name → (skin_mjcf_relative_path, needs_separate_viewer)
# needs_separate_viewer=True  → mesh MJCF has no collision geoms; Newton keeps
#   plain MJCF for physics and opens a synchronized passive MuJoCo window.
_SKIN_ASSET = {
    "skeleton_torque":                    ("mjcf/skeleton_torque_mesh.xml",             False),
    "skeleton_torque_suit":               ("mjcf/skeleton_torque_suit_mesh.xml",        False),
    "skeleton_torque_suit_passive_cable": ("mjcf/skeleton_torque_suit_mesh.xml",        True),
    "skeleton_torque_suit_active_cable":  ("mjcf/skeleton_torque_suit_muscle_mesh.xml", True),
    "skeleton_torque_suit_muscle":        ("mjcf/skeleton_torque_suit_muscle_mesh.xml", True),
}

# DOF remapping: active_cable / passive_cable (27 DOF) → muscle_mesh (31 DOF)
# muscle_mesh inserts subtalar_r(idx 5), mtp_r(idx 6), subtalar_l(idx 12), mtp_l(idx 13)
# These joints are equality-constrained to 0 in the mesh XML.
_ACTIVE_TO_MUSCLE_DOF_MAP = [0, 1, 2, 3, 4,   # hip_r ×5
                              None, None,        # subtalar_r, mtp_r (→ 0)
                              5, 6, 7, 8, 9,    # hip_l ×5
                              None, None,        # subtalar_l, mtp_l (→ 0)
                              10, 11, 12,        # lumbar ×3
                              13, 14, 15, 16, 17,# arm_r ×5
                              18, 19, 20, 21, 22,# arm_l ×5
                              23, 24, 25, 26]    # slide ×4  → total 31


def _make_skin_viewer(robot_name: str, asset_root: str):
    """Return (mj_model, mj_data, mj_viewer) for the bone-mesh skin window, or None."""
    import mujoco
    import mujoco.viewer as mj_viewer

    entry = _SKIN_ASSET.get(robot_name)
    if entry is None:
        print(f"[use-skin] No skin asset mapping for robot '{robot_name}', skipping.")
        return None
    rel_path, _ = entry
    mjcf_path = str(asset_root) + "/" + rel_path
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    viewer = mj_viewer.launch_passive(model, data)
    print(f"[use-skin] MuJoCo skin viewer opened: {mjcf_path}")
    return model, data, viewer


def _sync_skin_viewer(skin, dof_pos, root_pos, root_rot_xyzw, robot_name: str,
                      show_cables: bool = False, env=None):
    """Sync Newton pose → MuJoCo skin viewer.

    Handles DOF remapping for robots whose skin XML has more DOFs than
    the policy (e.g. active_cable 27 → muscle_mesh 31).
    """
    import mujoco
    import numpy as np

    model, data, viewer = skin

    # Root pose
    data.qpos[0:3] = root_pos.cpu().numpy()
    r = root_rot_xyzw.cpu().numpy()          # xyzw → MuJoCo wxyz
    data.qpos[3] = r[3]
    data.qpos[4] = r[0]
    data.qpos[5] = r[1]
    data.qpos[6] = r[2]

    dp = dof_pos.cpu().numpy()
    skin_ndof = model.nq - 7               # free-joint takes qpos[0:7]

    if robot_name in ("skeleton_torque_suit_active_cable",
                      "skeleton_torque_suit_passive_cable") and skin_ndof == 31:
        # Remap 27 → 31 DOFs
        qpos_dof = np.zeros(31, dtype=np.float64)
        for skin_idx, src_idx in enumerate(_ACTIVE_TO_MUSCLE_DOF_MAP):
            if src_idx is not None:
                qpos_dof[skin_idx] = dp[src_idx]
        data.qpos[7:38] = qpos_dof
    else:
        n = min(len(dp), skin_ndof)
        data.qpos[7: 7 + n] = dp[:n]

    mujoco.mj_forward(model, data)
    viewer.sync()

    # Cable lines: draw tendon site positions as line segments
    if show_cables and env is not None:
        _draw_cable_lines(skin, env)


def _draw_cable_lines(skin, env):
    """Draw active-cable tendon paths as colored lines in the MuJoCo skin viewer.

    Uses the Newton simulator's rigid_body_pos to compute approximate
    cable attachment points and draws them via MuJoCo's geom overlay.
    (Stub — extend with actual site positions if needed.)
    """
    pass  # Cable lines visible natively via Newton tendon rendering


# ---------------------------------------------------------------------------


def _run_interactive(agent, env, cycle_seconds: float = 0.0,
                     skin=None, robot_name: str = "", show_cables: bool = False) -> None:
    """Interactive inference loop.

    R key cycles through motions in reverse-alphabetical order.
    If cycle_seconds > 0, motions advance automatically every N wall-clock seconds.
    Motion name is shown as a large ImGui overlay at the top-center of the viewer.
    skin: optional (mj_model, mj_data, mj_viewer) tuple for bone-mesh overlay.
    """
    import os
    import sys
    import time
    import torch

    simulator = env.simulator
    agent.eval()

    # ── build reverse-alphabetical motion order ───────────────────────────────
    motion_names = [
        os.path.splitext(os.path.basename(f))[0]
        for f in env.motion_lib.motion_files
    ]
    # sort by name descending; each entry is (lib_index, name)
    ordered = sorted(enumerate(motion_names), key=lambda x: x[1], reverse=True)
    ordered_ids = [lib_idx for lib_idx, _ in ordered]   # lib indices in Z→A order
    ordered_names = [name for _, name in ordered]
    num_motions = len(ordered_ids)

    # cycle pointer + shared reset flag + wall-clock cycle timer + clip step counter
    state = {"cursor": 0, "force_reset": False, "cycle_start_time": time.perf_counter(), "clip_step": 0, "total_steps": 0}

    # ── patch sample_motions to always use the ordered list ───────────────────
    original_sample = env.motion_manager.sample_motions

    def _patched_sample_motions(env_ids, new_motion_ids=None):
        if new_motion_ids is None:
            lib_idx = ordered_ids[state["cursor"]]
            new_motion_ids = torch.full(
                (len(env_ids),), lib_idx, device=env.device, dtype=torch.long
            )
        original_sample(env_ids, new_motion_ids)

    env.motion_manager.sample_motions = _patched_sample_motions

    def _on_r_key():
        state["cursor"] = (state["cursor"] + 1) % num_motions
        name = ordered_names[state["cursor"]]
        lib_idx = ordered_ids[state["cursor"]]
        print(f"\n→ [{state['cursor']+1:2d}/{num_motions}] {name}  (lib idx {lib_idx})")
        # Signal the inference loop to reset all envs on the next iteration.
        # (simulator.user_requested_reset is cleared at the start of simulator.step()
        # so it cannot be used here; we use a shared flag instead.)
        state["force_reset"] = True
        # Reset wall-clock cycle timer and clip step counter on every motion change
        state["cycle_start_time"] = time.perf_counter()
        state["clip_step"] = 0
        # Update total_steps for the new motion (_sim_dt captured by closure)
        if cycle_seconds > 0:
            state["total_steps"] = max(1, int(cycle_seconds / _sim_dt))
        elif env.motion_lib is not None:
            state["total_steps"] = max(1, int(env.motion_lib.motion_lengths[lib_idx].item() / _sim_dt))

    if hasattr(simulator, "_custom_key_handlers"):
        simulator._custom_key_handlers["r"] = _on_r_key

    # ── ImGui title overlay (Newton) ──────────────────────────────────────────
    _register_motion_title_ui(simulator, state, env, num_motions, cycle_seconds)

    # ── print startup info ────────────────────────────────────────────────────
    print("Evaluating policy... (Ctrl+C to stop)")
    print(f"  R → cycle motions (reverse-alphabetical, {num_motions} total)")
    print("  Order:")
    for i, (lib_idx, name) in enumerate(zip(ordered_ids, ordered_names)):
        print(f"    {i+1:2d}. {name}  (lib idx {lib_idx})")

    # ── inference loop ────────────────────────────────────────────────────────
    # Control period: decimation / fps (e.g., 6/120 = 50ms at 20Hz for Newton)
    _sim_dt = getattr(env.simulator, "dt", 1.0 / 20.0)

    # Initialize total_steps now that _sim_dt is known
    if cycle_seconds > 0:
        state["total_steps"] = max(1, int(cycle_seconds / _sim_dt))
    elif env.motion_manager is not None and env.motion_lib is not None:
        mid = env.motion_manager.motion_ids[0].item()
        state["total_steps"] = max(1, int(env.motion_lib.motion_lengths[mid].item() / _sim_dt))

    if cycle_seconds > 0:
        print(f"  Auto-cycle: {cycle_seconds:.0f}s per motion (wall-clock)")

    done_indices = None
    step = 0

    def _status_line():
        """Print status using ordered list name so it always matches the arrow line."""
        if env.motion_manager is None:
            return
        name = ordered_names[state["cursor"]]
        total = state.get("total_steps", 0)
        timing = f"step:{state['clip_step']:4d}/{total}"
        print(
            f"\r[{state['cursor']+1:2d}/{num_motions}] {name:<25s} {timing}",
            end="",
            flush=True,
        )

    try:
        while True:
            _t0 = time.perf_counter()

            # Exit cleanly when the viewer window is closed with the X button
            if (
                hasattr(simulator, "viewer")
                and simulator.viewer is not None
                and hasattr(simulator.viewer, "is_running")
                and not simulator.viewer.is_running()
            ) or (
                hasattr(simulator, "is_app_running")
                and not simulator.is_app_running()
            ):
                print("\nViewer window closed.")
                sys.exit(0)

            # MuJoCo R key: viewer calls _requested_reset() → user_requested_reset=True.
            # env.step() clears it, so check it HERE before stepping.
            if getattr(simulator, "user_requested_reset", False):
                simulator.user_requested_reset = False
                _on_r_key()

            obs, _ = env.reset(done_indices)
            obs = agent.add_agent_info_to_obs(obs)
            obs_td = agent.obs_dict_to_tensordict(obs)
            model_outs = agent.model(obs_td)
            action = model_outs.get("mean_action", model_outs["action"])
            obs, _rewards, dones, _terminated, _extras = env.step(action)
            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
            step += 1
            state["clip_step"] += 1

            # Sync bone-mesh skin viewer (Newton + MuJoCo overlay)
            if skin is not None:
                try:
                    _rs = env.simulator.get_robot_state()
                    _sync_skin_viewer(
                        skin,
                        _rs.dof_pos[0],
                        _rs.rigid_body_pos[0, 0],
                        _rs.rigid_body_rot[0, 0],
                        robot_name,
                        show_cables=show_cables,
                        env=env,
                    )
                except Exception as _skin_err:
                    log.warning(f"[skin sync] {_skin_err}")
                    skin = None  # disable after first error to avoid log spam

            # real-time sync: sleep to match control period
            _elapsed = time.perf_counter() - _t0
            _remaining = _sim_dt - _elapsed
            if _remaining > 0.001:
                time.sleep(_remaining)

            # auto-advance when wall-clock elapsed time exceeds cycle_seconds
            if cycle_seconds > 0:
                if time.perf_counter() - state["cycle_start_time"] >= cycle_seconds:
                    _on_r_key()  # resets cycle_start_time and sets force_reset

            # Apply pending reset (from auto-cycle or manual R key press)
            # Note: simulator.user_requested_reset is cleared at the START of
            # simulator.step(), so we force reset by overriding done_indices.
            if state["force_reset"]:
                state["force_reset"] = False
                done_indices = torch.arange(
                    env.num_envs, device=env.device, dtype=torch.long
                )

            _status_line()

            # Update title for all simulators (Newton: OS title, MuJoCo: overlay, IsaacLab: OS title)
            title = _build_title(state, env, num_motions, cycle_seconds)
            state["current_title"] = title
            if hasattr(simulator, "set_window_title"):
                try:
                    simulator.set_window_title(title)
                except Exception:
                    pass  # viewer window closed or X11 handle released
    except KeyboardInterrupt:
        print(f"\nStopped after {step} steps.")


if __name__ == "__main__":
    main()
