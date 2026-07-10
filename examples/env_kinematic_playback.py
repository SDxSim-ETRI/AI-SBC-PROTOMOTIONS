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
"""
Environment Kinematic Playback Script

This script allows you to visualize reference motions in kinematic playback mode without training.
It uses the KinematicReplayControl component to directly set robot state to reference motion poses,
bypassing physics simulation entirely.

Usage:
    python examples/env_kinematic_playback.py \
        --experiment-path=examples/experiments/mimic/mlp.py \
        --motion-file=xxx.pt \
        --robot-name=g1 \
        --simulator=isaacgym \
        --num-envs=14 \
        --scenes-file=xxx.pt
"""


def create_parser():
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Visualize environment in kinematic playback mode",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--robot-name",
        type=str,
        required=True,
        help="Name of the robot (e.g., 'h1', 'g1', 'smpl')",
    )
    parser.add_argument(
        "--simulator",
        type=str,
        required=True,
        help="Simulator to use (e.g., 'isaacgym', 'isaaclab', 'newton', 'genesis')",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        required=True,
        help="Number of parallel environments to run",
    )
    parser.add_argument(
        "--motion-file",
        type=str,
        required=True,
        help="Path to motion file for playback",
    )
    parser.add_argument(
        "--experiment-path",
        type=str,
        required=True,
        help="File path to experiment configuration (e.g., 'examples/experiments/mimic/mlp.py')",
    )

    # Optional arguments
    parser.add_argument(
        "--scenes-file", type=str, default=None, help="Path to scenes file (optional)"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="kinematic_playback",
        help="Name of the experiment for logging",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run simulation in headless mode",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        default=False,
        help="Use CPU only for simulation (experimental, GPU is default)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Config overrides in key=value format (e.g. robot.asset.asset_file_name=mjcf/...)",
    )
    parser.add_argument(
        "--auto-record",
        action="store_true",
        default=False,
        help="Automatically start recording and exit after --record-steps steps",
    )
    parser.add_argument(
        "--record-steps",
        type=int,
        default=5600,
        help="Number of steps to record when --auto-record is set",
    )
    parser.add_argument(
        "--cycle-seconds",
        type=float,
        default=20.0,
        help="Seconds per motion clip when cycling through multiple motions",
    )

    return parser


# Parse arguments first (argparse is safe, doesn't import torch)
import argparse  # noqa: E402

parser = create_parser()
args, unknown_args = parser.parse_known_args()

# Import simulator before torch - isaacgym/isaaclab must be imported before torch
# This also returns AppLauncher if using isaaclab, None otherwise
from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch(args.simulator)

# Now safe to import everything else including torch
from pathlib import Path  # noqa: E402
import logging  # noqa: E402
import importlib.util  # noqa: E402
import torch  # noqa: E402

log = logging.getLogger(__name__)


def main():
    # Re-use the parser and args from module level
    global parser, args

    device = torch.device("cuda:0") if not args.cpu_only else torch.device("cpu")

    # Dynamically import the module from file path
    experiment_path = Path(args.experiment_path)
    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment file not found: {experiment_path}")

    spec = importlib.util.spec_from_file_location("experiment_module", experiment_path)
    experiment_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment_module)

    args, _ = parser.parse_known_args()

    # Apply overrides to configs after building
    _overrides = {kv.split("=")[0]: kv.split("=", 1)[1] for kv in (args.overrides or []) if "=" in kv}

    print("\n=== Environment Kinematic Playback Configuration ===")
    print(f"Experiment path: {args.experiment_path}")
    print(f"Robot: {args.robot_name}")
    print(f"Simulator: {args.simulator}")
    print(f"Number of environments: {args.num_envs}")
    print(f"Motion file: {args.motion_file}")
    print(f"Scenes file: {args.scenes_file}")
    print(f"Device: {device}")
    print(f"Headless: {args.headless}")

    # Extra simulator parameters
    extra_simulator_params = {}
    if args.simulator == "isaaclab":
        app_launcher_flags = {"headless": args.headless, "device": str(device)}
        app_launcher = AppLauncher(app_launcher_flags)
        simulation_app = app_launcher.app
        extra_simulator_params["simulation_app"] = simulation_app

    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

    # Get config functions from experiment module
    from protomotions.utils.config_builder import build_standard_configs
    from protomotions.simulator.base_simulator.config import SimulatorConfig
    from protomotions.envs.base_env.config import EnvConfig
    from protomotions.robot_configs.base import RobotConfig

    # Build configs from experiment (without agent for kinematic playback)
    print("\n=== Building Configuration from Experiment ===")

    # Get required config functions
    terrain_config_fn = getattr(experiment_module, "terrain_config")
    scene_lib_config_fn = getattr(experiment_module, "scene_lib_config")
    motion_lib_config_fn = getattr(experiment_module, "motion_lib_config")
    env_config_fn = getattr(experiment_module, "env_config")

    # Get optional config functions
    configure_robot_and_simulator_fn = getattr(
        experiment_module, "configure_robot_and_simulator", None
    )

    configs = build_standard_configs(
        args=args,
        terrain_config_fn=terrain_config_fn,
        scene_lib_config_fn=scene_lib_config_fn,
        motion_lib_config_fn=motion_lib_config_fn,
        env_config_fn=env_config_fn,
        configure_robot_and_simulator_fn=configure_robot_and_simulator_fn,
        agent_config_fn=None,  # No agent needed for kinematic playback
    )

    robot_config: RobotConfig = configs["robot"]
    simulator_config: SimulatorConfig = configs["simulator"]

    # Apply CLI overrides (e.g. robot.asset.asset_file_name=mjcf/31dof/skeleton_torque_mesh.xml)
    if _overrides:
        from protomotions.utils.config_utils import apply_config_overrides
        apply_config_overrides(
            _overrides,
            env_config=configs.get("env"),
            simulator_config=simulator_config,
            robot_config=robot_config,
        )
        print(f"Applied overrides: {_overrides}")
    terrain_config = configs["terrain"]
    scene_lib_config = configs["scene_lib"]
    motion_lib_config = configs["motion_lib"]
    env_config: EnvConfig = configs["env"]

    print(f"Robot config class: {type(robot_config).__name__}")
    print(f"Simulator config class: {type(simulator_config).__name__}")
    print(f"Environment config class: {type(env_config).__name__}")

    if args.motion_file is not None:
        print(f"Motion library configured from: {args.motion_file}")

    if args.scenes_file is not None:
        print(f"Scene library configured from: {args.scenes_file}")

    # Enable kinematic playback mode using KinematicReplayControl
    from protomotions.envs.control.kinematic_replay_control import (
        KinematicReplayControlConfig,
    )
    
    print("Enabling kinematic playback via KinematicReplayControl component")
    env_config.show_terrain_markers = False
    
    # Add kinematic replay control component (replaces any existing control components)
    env_config.control_components = {
        "kinematic_replay": KinematicReplayControlConfig(),
    }
    
    # Disable terminations - kinematic replay should run indefinitely
    env_config.termination_components = {}
    
    # Disable observations - not needed for kinematic playback
    env_config.observation_components = {}
    
    # Disable rewards - not needed for kinematic playback
    env_config.reward_components = {}

    print("\n=== Creating Environment ===")

    # Convert friction settings for simulator compatibility
    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator

    terrain_config, simulator_config = convert_friction_for_simulator(
        terrain_config, simulator_config
    )

    # Create components using configs from build_standard_configs
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
        device=device,
        save_dir=save_dir_for_weights,
        **extra_simulator_params,
    )

    terrain = components["terrain"]
    scene_lib = components["scene_lib"]
    motion_lib = components["motion_lib"]
    simulator = components["simulator"]

    # Create environment - use BaseEnv directly for kinematic playback
    from protomotions.envs.base_env.env import BaseEnv

    env: BaseEnv = BaseEnv(
        config=env_config,
        robot_config=robot_config,
        device=device,
        terrain=terrain,
        scene_lib=scene_lib,
        motion_lib=motion_lib,
        simulator=simulator,
    )

    print("Environment created successfully")
    print(f"Environment class: {type(env).__name__}")
    print(f"Motion library loaded: {env.motion_lib is not None}")
    print(f"  - Number of motions: {env.motion_lib.num_motions()}")
    print(f"  - Motion file: {env.motion_lib.motion_file}")
    print(f"Scene library loaded: {env.scene_lib is not None}")
    print(f"  - Number of scenes: {env.scene_lib.num_scenes()}")
    if hasattr(env.scene_lib, "scenes_file"):
        print(f"  - Scenes file: {env.scene_lib.scenes_file}")
    print(f"Motion manager created: {env.motion_manager is not None}")
    if env.motion_manager is not None:
        print(f"  - Motion manager type: {type(env.motion_manager).__name__}")

    # Reset the environment
    print("\n=== Resetting Environment ===")
    env.reset()
    print("Environment reset complete")

    if env.motion_manager is not None:
        print(f"Motion IDs assigned: {env.motion_manager.motion_ids}")
        print(f"Motion times initialized: {env.motion_manager.motion_times}")

    # Run simulation loop
    print("\n=== Starting Kinematic Playback ===")
    print("This will play back the reference motion kinematically")
    print("The humanoid will follow the motion capture data exactly")
    print("\nCamera controls:")
    print("  L - start/stop recording")
    print("  ; - cancel recording")
    print("  O - toggle camera target")
    print("  Q - close simulator")

    actions = torch.zeros(env.num_envs, robot_config.number_of_actions, device=device)

    if args.auto_record:
        import os, time as _time
        simulator = env.simulator
        num_motions = env.motion_lib.num_motions()
        policy_fps = round(1.0 / simulator.dt) if hasattr(simulator, "dt") and simulator.dt > 0 else 20
        cycle_steps = int(args.cycle_seconds * policy_fps) if args.cycle_seconds > 0 and num_motions > 1 else 0
        cursor = [0]
        steps_in_cycle = [0]

        def _get_motion_name(mid):
            base = os.path.splitext(os.path.basename(env.motion_lib.motion_files[mid]))[0]
            names = {"02-constspeed_reduced_humanoid": "constspeed", "walk": "walk (ETRI)", "walk_koo": "walk_koo (김범호)"}
            return names.get(base, base)

        def _reset_to_motion(mid):
            env.motion_manager.motion_ids[:] = mid
            env.motion_manager.motion_times[:] = 0.0
            env_ids = torch.arange(env.num_envs, dtype=torch.long, device=device)
            env.reset(env_ids)

        print(f"\nAuto-record: {args.record_steps} steps, {num_motions} motions, cycle={args.cycle_seconds:.0f}s ({cycle_steps} steps)")
        for i in range(num_motions):
            print(f"  [{i+1}/{num_motions}] {_get_motion_name(i)}")

        # inference_agent.py와 동일한 패턴: ImGui 라이브 오버레이 + 녹화 후 PIL 자막 처리
        _FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        _SUBTITLE_FONT_SIZE = 32

        def _overlay_subtitles_on_frames(frame_dir, labels):
            import glob
            from PIL import Image, ImageDraw, ImageFont
            try:
                font = ImageFont.truetype(_FONT_PATH, _SUBTITLE_FONT_SIZE)
            except OSError:
                font = ImageFont.load_default()
            frames = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
            print(f"\nOverlaying subtitles on {len(frames)} frames...")
            for i, fpath in enumerate(frames):
                label = labels[i] if i < len(labels) else labels[-1]
                try:
                    img = Image.open(fpath).convert("RGB")
                except Exception:
                    continue
                draw = ImageDraw.Draw(img)
                w, h = img.size
                bbox = draw.textbbox((0, 0), label, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                x, y, pad = (w - tw) // 2, 14, 8
                bg = Image.new("RGBA", img.size, (0, 0, 0, 0))
                bg_draw = ImageDraw.Draw(bg)
                bg_draw.rounded_rectangle([x-pad, y-pad, x+tw+pad, y+th+pad], radius=6, fill=(0, 0, 0, 160))
                img = Image.alpha_composite(img.convert("RGBA"), bg).convert("RGB")
                draw = ImageDraw.Draw(img)
                draw.text((x, y), label, font=font, fill=(255, 255, 255))
                img.save(fpath)
            print("Subtitle overlay complete.")

        def _register_motion_title_ui(sim, state):
            def _motion_title_ui(imgui):
                title = state.get("current_title", "")
                if not title:
                    return
                io = imgui.get_io()
                dw = io.display_size[0]
                imgui.set_next_window_pos(imgui.ImVec2(dw / 2, 10), pivot=imgui.ImVec2(0.5, 0.0))
                imgui.set_next_window_bg_alpha(0.6)
                flags = (imgui.WindowFlags_.no_decoration.value | imgui.WindowFlags_.always_auto_resize.value |
                         imgui.WindowFlags_.no_saved_settings.value | imgui.WindowFlags_.no_focus_on_appearing.value |
                         imgui.WindowFlags_.no_nav.value | imgui.WindowFlags_.no_move.value)
                if imgui.begin("##motion_title", flags=flags):
                    imgui.push_font(None, 26.0)
                    imgui.text(title)
                    imgui.pop_font()
                imgui.end()
            if hasattr(sim, "viewer") and sim.viewer is not None and hasattr(sim.viewer, "register_ui_callback"):
                sim.viewer.register_ui_callback(_motion_title_ui, position="free")

        _auto_record_state = {"current_title": ""}
        _register_motion_title_ui(simulator, _auto_record_state)

        _reset_to_motion(0)
        simulator._toggle_video_record()
        print("Recording started.")

        motion_labels: list = []

        try:
            for step in range(args.record_steps):
                if cycle_steps > 0 and steps_in_cycle[0] >= cycle_steps:
                    cursor[0] = (cursor[0] + 1) % num_motions
                    steps_in_cycle[0] = 0
                    _reset_to_motion(cursor[0])
                    print(f"\n→ [{cursor[0]+1}/{num_motions}] {_get_motion_name(cursor[0])}")

                title = f"[{cursor[0]+1}/{num_motions}] {_get_motion_name(cursor[0])}"
                _auto_record_state["current_title"] = title
                try:
                    simulator.set_window_title(title)
                except Exception:
                    pass

                obs, rewards, dones, terminated, infos = env.step(actions)
                steps_in_cycle[0] += 1
                motion_labels.append(title)

                name = _get_motion_name(cursor[0])
                dur = env.motion_lib.get_motion_length(cursor[0]).item() if hasattr(env.motion_lib, "get_motion_length") else args.cycle_seconds
                t = steps_in_cycle[0] / policy_fps
                print(f"\r[{step+1:5d}/{args.record_steps}] motion: {name:<30s} {t:5.2f}s / {dur:.2f}s ({t/dur*100:5.1f}%)", end="", flush=True)
        except KeyboardInterrupt:
            print("\nRecording interrupted.")
        finally:
            # PIL 자막 오버레이 → Newton이 MP4 컴파일하기 전에 PNG에 굽기
            if motion_labels and hasattr(simulator, "_curr_user_recording_name"):
                frames_dir = os.path.join(simulator._curr_user_recording_name, "_frames")
                _overlay_subtitles_on_frames(frames_dir, motion_labels)
            simulator._toggle_video_record()
            print("\nRecording saved.")
            env.simulator.render()
            env.close()
    else:
        import os as _os

        def _motion_name_from_idx(idx):
            files = env.motion_lib.motion_files if hasattr(env.motion_lib, "motion_files") else []
            if idx < len(files):
                return _os.path.splitext(_os.path.basename(files[idx]))[0]
            return f"motion_{idx}"

        _ui_state = {"title": ""}

        def _motion_title_ui(imgui):
            title = _ui_state.get("title", "")
            if not title:
                return
            io = imgui.get_io()
            dw = io.display_size[0]
            imgui.set_next_window_pos(imgui.ImVec2(dw / 2, 10), pivot=imgui.ImVec2(0.5, 0.0))
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
                imgui.push_font(None, 26.0)
                imgui.text(title)
                imgui.pop_font()
            imgui.end()

        sim = env.simulator
        if hasattr(sim, "viewer") and sim.viewer is not None and hasattr(sim.viewer, "register_ui_callback"):
            sim.viewer.register_ui_callback(_motion_title_ui, position="free")

        # Sequential cycling: cycle through motions in order when --cycle-seconds is set
        num_motions = env.motion_lib.num_motions()
        policy_fps = round(1.0 / sim.dt) if hasattr(sim, "dt") and sim.dt > 0 else 20
        cycle_steps = int(args.cycle_seconds * policy_fps) if args.cycle_seconds > 0 else 0
        seq_cursor = 0
        steps_in_cycle = 0

        def _reset_to_motion(mid):
            env.motion_manager.motion_ids[:] = mid
            env.motion_manager.motion_times[:] = 0.0
            env_ids = torch.arange(env.num_envs, dtype=torch.long, device=device)
            env.reset(env_ids)

        if cycle_steps > 0:
            # Patch sample_motions so internal resets don't pick a random motion
            _original_sample = env.motion_manager.sample_motions

            def _sequential_sample(env_ids):
                env.motion_manager.motion_ids[env_ids] = seq_cursor

            env.motion_manager.sample_motions = _sequential_sample
            _reset_to_motion(0)
            print(f"Sequential mode: {num_motions} motions, {cycle_steps} steps each ({args.cycle_seconds:.0f}s)")

        try:
            step_count = 0
            while env.is_simulation_running():
                # Sequential cycling: advance motion when cycle_steps reached
                if cycle_steps > 0 and steps_in_cycle >= cycle_steps:
                    seq_cursor = (seq_cursor + 1) % num_motions
                    steps_in_cycle = 0
                    _reset_to_motion(seq_cursor)
                    print(f"\n→ [{seq_cursor+1}/{num_motions}] {_motion_name_from_idx(seq_cursor)}")

                obs, rewards, dones, terminated, infos = env.step(actions)
                step_count += 1
                steps_in_cycle += 1

                if env.motion_manager is not None:
                    mid = seq_cursor if cycle_steps > 0 else int(env.motion_manager.motion_ids[0].item())
                    num = num_motions
                    name = _motion_name_from_idx(mid)
                    title = f"[{mid+1}/{num}] {name}"
                    _ui_state["title"] = title
                    try:
                        sim.set_window_title(title)
                    except Exception:
                        pass
        except KeyboardInterrupt:
            print("\n\nSimulation stopped by user")
        finally:
            env.close()

    print("\n=== Playback Complete ===")
    print(f"Total steps: {step_count if not args.auto_record else args.record_steps}")
    print("Environment closed successfully")


if __name__ == "__main__":
    main()
