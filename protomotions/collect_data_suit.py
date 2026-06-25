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
"""Collect simulation data from a suit robot policy, saved to zarr.

Extension of collect_data.py for skeleton_torque_suit models.
When --active-cable is set, adds cable-specific fields using DOFC A버전 formula.

Output zarr layout
------------------
::

    dataset.zarr/
    ├── data/
    │   ├── dof_pos          [N, n_dof]       joint positions (rad)
    │   ├── dof_vel          [N, n_dof]       joint velocities (rad/s)
    │   ├── dof_forces       [N, n_dof]       joint torques (N·m)
    │   ├── body_pos         [N, n_bodies, 3] link positions (m)
    │   ├── body_rot         [N, n_bodies, 4] link rotations, xyzw quaternion
    │   ├── body_vel         [N, n_bodies, 3] link linear velocities (m/s)
    │   ├── body_ang_vel     [N, n_bodies, 3] link angular velocities (rad/s)
    │   ├── contact_forces   [N, n_bodies, 3] contact forces per link (N)
    │   ├── contacts         [N, n_bodies]    binary contact flags
    │   ├── actions          [N, n_actions]   raw PPO policy actions
    │   ├── cable_pos        [N, 4]           slide1-4 positions (m)       [--active-cable]
    │   ├── cable_forces     [N, 4]           slide1-4 torques (N·m)       [--active-cable]
    │   ├── hip_angles       [N, 2]           [hip_r, hip_l] (rad)         [--active-cable]
    │   ├── dofc_targets     [N, 2]           [slide2_tgt, slide4_tgt] (m) [--active-cable]
    │   └── dofc_balance     [N, 1]           y = sin(-hip_r)-sin(-hip_l)  [--active-cable]
    └── meta/
        ├── episode_ends     [n_episodes]     exclusive end index in flat N
        └── episode_env_ids  [n_episodes]     source env index

    root attrs: fps, dt, body_names, dof_names, num_envs, total_timesteps,
                num_episodes, checkpoint, active_cable
                [+ cable_dof_names, dofc_params when --active-cable]

Example
-------
::

    # Active cable data collection
    python protomotions/collect_data_suit.py \\
        --checkpoint tasks/mimic_suit_active_cable_motions14_23dof/output_newton/score_based.ckpt \\
        --motion-file data/motion_for_trackers/skeleton_torque_suit_motions14.pt \\
        --simulator newton --num-envs 16 --num-steps 5000 --active-cable \\
        --output data/collected/active_cable_newton.zarr

    # Passive cable (no cable fields)
    python protomotions/collect_data_suit.py \\
        --checkpoint tasks/mimic_suit_passive_cable_motions14_23dof/output_newton/score_based.ckpt \\
        --motion-file data/motion_for_trackers/skeleton_torque_suit_motions_11+koo_4.pt \\
        --simulator newton --num-envs 16 --num-steps 5000 \\
        --output data/collected/passive_cable_newton.zarr
"""

# Ensure this project's protomotions package takes precedence over any installed version.
# torch.load(weights_only=False) unpickles class references that need the local robot configs.
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


def create_parser():
    import argparse

    p = argparse.ArgumentParser(
        description="Collect suit robot simulation data to zarr",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--motion-file", type=str, default=None)
    p.add_argument(
        "--simulator",
        type=str,
        default="newton",
        choices=["isaacgym", "isaaclab", "newton", "genesis", "mujoco"],
    )
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--num-steps", type=int, default=5000,
                   help="Number of control steps to collect (all envs run this many steps)")
    p.add_argument("--output", type=str, default="dataset_suit.zarr")
    p.add_argument("--headless", action="store_true", default=False)
    p.add_argument(
        "--active-cable",
        action="store_true",
        default=False,
        help="Add cable-specific fields: cable_pos, cable_forces, hip_angles, "
             "dofc_targets (DOFC A버전), dofc_balance",
    )
    p.add_argument("--overrides", nargs="*", default=None)
    return p


# ── module-level setup (simulator must be imported before torch) ──────────────
import argparse as _ap

_parser = create_parser()
_args, _unknown = _parser.parse_known_args()

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

_AppLauncher = import_simulator_before_torch(_args.simulator)

import logging  # noqa: E402
from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402
from lightning.fabric import Fabric  # noqa: E402
from tqdm import tqdm  # noqa: E402

from protomotions.utils.fabric_config import FabricConfig  # noqa: E402
from protomotions.utils.hydra_replacement import get_class  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
log = logging.getLogger(__name__)

# Cable DOF range: slide1(23), slide2(24), slide3(25), slide4(26)
_CABLE_SLICE = slice(23, 27)
_HIP_R_IDX = 0
_HIP_L_IDX = 5


# ── helpers ───────────────────────────────────────────────────────────────────


def _maybe_numpy(tensor):
    return tensor.cpu().numpy() if tensor is not None else None


def _append(buf: dict, key: str, arr):
    if arr is None:
        return
    if key not in buf:
        buf[key] = []
    buf[key].append(arr)


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    args = _args
    checkpoint_path = Path(args.checkpoint)

    for cfg_name in ("resolved_configs_inference.pt", "resolved_configs.pt"):
        cfg_path = checkpoint_path.parent / cfg_name
        if cfg_path.exists():
            break
    resolved_configs = torch.load(str(cfg_path), weights_only=False)

    robot_config = resolved_configs["robot"]
    simulator_config = resolved_configs["simulator"]
    terrain_config = resolved_configs.get("terrain")
    scene_lib_config = resolved_configs["scene_lib"]
    motion_lib_config = resolved_configs["motion_lib"]
    env_config = resolved_configs["env"]
    agent_config = resolved_configs["agent"]

    current_simulator = simulator_config._target_.split(".")[-3]
    if args.simulator != current_simulator:
        log.info(f"Switching simulator '{current_simulator}' → '{args.simulator}'")
        from protomotions.simulator.factory import update_simulator_config_for_test

        simulator_config = update_simulator_config_for_test(
            current_simulator_config=simulator_config,
            new_simulator=args.simulator,
            robot_config=robot_config,
        )

    from protomotions.utils.inference_utils import apply_backward_compatibility_fixes

    apply_backward_compatibility_fixes(robot_config, simulator_config, env_config)

    simulator_config.num_envs = args.num_envs
    simulator_config.headless = args.headless
    if args.motion_file is not None:
        motion_lib_config.motion_file = args.motion_file

    if args.overrides:
        from protomotions.utils.config_utils import (
            apply_config_overrides,
            parse_cli_overrides,
        )

        apply_config_overrides(
            parse_cli_overrides(args.overrides),
            env_config,
            simulator_config,
            robot_config,
            agent_config,
            terrain_config,
            motion_lib_config,
            scene_lib_config,
        )

    accelerator = "cpu" if args.simulator == "mujoco" else "gpu"
    fabric = Fabric(**asdict(FabricConfig(
        accelerator=accelerator, devices=1, num_nodes=1, loggers=[], callbacks=[]
    )))
    fabric.launch()

    simulator_extra_params = {}
    if args.simulator == "isaaclab":
        app_launcher = _AppLauncher({"headless": True, "device": str(fabric.device)})
        simulator_extra_params["simulation_app"] = app_launcher.app

    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator
    from protomotions.utils.component_builder import build_all_components

    terrain_config, simulator_config = convert_friction_for_simulator(
        terrain_config, simulator_config
    )
    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=fabric.device,
        save_dir=None,
        **simulator_extra_params,
    )

    from protomotions.envs.base_env.env import BaseEnv

    env: BaseEnv = get_class(env_config._target_)(
        config=env_config,
        robot_config=robot_config,
        device=fabric.device,
        terrain=components["terrain"],
        scene_lib=components["scene_lib"],
        motion_lib=components["motion_lib"],
        simulator=components["simulator"],
    )

    from protomotions.agents.base_agent.agent import BaseAgent

    agent: BaseAgent = get_class(agent_config._target_)(
        config=agent_config, env=env, fabric=fabric, root_dir=checkpoint_path.parent,
    )
    agent.setup()
    agent.load(args.checkpoint, load_env=False)
    agent.eval()

    sim = env.simulator

    dofc_fn = None
    dofc_meta = {}
    if args.active_cable:
        from protomotions.envs.base_env.active_cable_env import (
            _KAPPA,
            _EXT_GAIN,
            _PULLEY_RADIUS,
            _STIFFNESS,
            _dofc_a_target_pos,
        )

        dofc_fn = _dofc_a_target_pos
        dofc_meta = {
            "kappa": _KAPPA,
            "ext_gain": _EXT_GAIN,
            "pulley_radius": _PULLEY_RADIUS,
            "stiffness": _STIFFNESS,
        }
        log.info(
            f"Active cable mode ON — DOFC A버전 params: {dofc_meta}"
        )

    # ── collection loop ───────────────────────────────────────────────────────
    step_bufs: dict = {}
    done_indices = None
    log.info(f"Collecting {args.num_steps} steps × {args.num_envs} envs …")

    with torch.no_grad():
        for _ in tqdm(range(args.num_steps)):
            obs, _ = env.reset(done_indices)
            obs = agent.add_agent_info_to_obs(obs)
            obs_td = agent.obs_dict_to_tensordict(obs)

            model_outs = agent.model(obs_td)
            actions = model_outs.get("mean_action", model_outs["action"])

            obs, _rewards, dones, _terminated, _extras = env.step(actions)

            robot_state = sim.get_robot_state()
            try:
                contact_buf = sim.get_bodies_contact_buf()
                contact_forces = contact_buf.rigid_body_contact_forces
            except Exception:
                contact_forces = None

            _append(step_bufs, "dof_pos",       _maybe_numpy(robot_state.dof_pos))
            _append(step_bufs, "dof_vel",       _maybe_numpy(robot_state.dof_vel))
            _append(step_bufs, "dof_forces",    _maybe_numpy(robot_state.dof_forces))
            _append(step_bufs, "body_pos",      _maybe_numpy(robot_state.rigid_body_pos))
            _append(step_bufs, "body_rot",      _maybe_numpy(robot_state.rigid_body_rot))
            _append(step_bufs, "body_vel",      _maybe_numpy(robot_state.rigid_body_vel))
            _append(step_bufs, "body_ang_vel",  _maybe_numpy(robot_state.rigid_body_ang_vel))
            _append(step_bufs, "contacts",      _maybe_numpy(robot_state.rigid_body_contacts))
            _append(step_bufs, "contact_forces", _maybe_numpy(contact_forces))
            _append(step_bufs, "actions",       _maybe_numpy(actions))
            _append(step_bufs, "_dones",        _maybe_numpy(dones.float()))

            if dofc_fn is not None:
                dof_pos = robot_state.dof_pos
                s2, s4 = dofc_fn(dof_pos)
                s2 = s2.clamp(min=0.0, max=0.51)
                s4 = s4.clamp(min=0.0, max=0.51)
                hip_r = dof_pos[:, _HIP_R_IDX]
                hip_l = dof_pos[:, _HIP_L_IDX]
                balance_y = torch.sin(-hip_r) - torch.sin(-hip_l)

                _append(step_bufs, "cable_pos",
                        _maybe_numpy(dof_pos[:, _CABLE_SLICE]))
                _append(step_bufs, "cable_forces",
                        _maybe_numpy(
                            robot_state.dof_forces[:, _CABLE_SLICE]
                            if robot_state.dof_forces is not None else None
                        ))
                _append(step_bufs, "hip_angles",
                        _maybe_numpy(torch.stack([hip_r, hip_l], dim=-1)))
                _append(step_bufs, "dofc_targets",
                        _maybe_numpy(torch.stack([s2, s4], dim=-1)))
                _append(step_bufs, "dofc_balance",
                        _maybe_numpy(balance_y.unsqueeze(-1)))

            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)

    # ── build zarr ────────────────────────────────────────────────────────────
    all_data = {k: np.stack(v) for k, v in step_bufs.items()}
    T, n_env = all_data["dof_pos"].shape[:2]
    all_dones = all_data.pop("_dones").astype(bool)

    log.info(f"Splitting {T} steps × {n_env} envs into episodes …")

    flat: dict = {k: [] for k in all_data}
    episode_ends: list = []
    episode_env_ids: list = []
    flat_offset = 0

    for env_idx in range(n_env):
        env_dones = all_dones[:, env_idx]
        done_steps = np.where(env_dones)[0]
        start = 0
        for done_step in done_steps:
            end = int(done_step) + 1
            if end > start:
                for k in flat:
                    flat[k].append(all_data[k][start:end, env_idx])
                flat_offset += end - start
                episode_ends.append(flat_offset)
                episode_env_ids.append(env_idx)
            start = end
        if start < T:
            for k in flat:
                flat[k].append(all_data[k][start:, env_idx])
            flat_offset += T - start
            episode_ends.append(flat_offset)
            episode_env_ids.append(env_idx)

    log.info(
        f"Writing {flat_offset} timesteps across {len(episode_ends)} episodes → {args.output}"
    )

    store = zarr.open(args.output, mode="w")
    data_grp = store.require_group("data")
    meta_grp = store.require_group("meta")
    compressor = zarr.Blosc(cname="lz4", clevel=4, shuffle=zarr.Blosc.BITSHUFFLE)

    for key, chunks in flat.items():
        arr = np.concatenate(chunks, axis=0)
        chunk_t = min(10_000, arr.shape[0])
        data_grp.create_dataset(
            key,
            data=arr,
            chunks=(chunk_t,) + arr.shape[1:],
            compressor=compressor,
            dtype=np.float32,
        )
        log.info(f"  data/{key}: {arr.shape}")

    meta_grp.create_dataset("episode_ends",    data=np.array(episode_ends, dtype=np.int64))
    meta_grp.create_dataset("episode_env_ids", data=np.array(episode_env_ids, dtype=np.int64))

    fps = 1.0 / sim.dt
    attrs = {
        "fps": fps,
        "dt": sim.dt,
        "body_names": list(sim._body_names),
        "dof_names": list(sim._dof_names),
        "num_envs": n_env,
        "num_steps_collected": T,
        "num_episodes": len(episode_ends),
        "total_timesteps": flat_offset,
        "checkpoint": str(args.checkpoint),
        "active_cable": args.active_cable,
    }
    if dofc_fn is not None:
        attrs["cable_dof_names"] = [sim._dof_names[i] for i in range(23, 27)]
        attrs["dofc_params"] = dofc_meta
    store.attrs.update(attrs)
    log.info(f"Done. fps={fps:.1f}")

    if hasattr(sim, "shutdown"):
        sim.shutdown()


if __name__ == "__main__":
    main()
