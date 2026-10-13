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
"""Evaluate a frozen W1 wearer on short native clips, with zero assistance.

No clips are looped. Evaluation stops 100 ms before the clip end so both the
current reference and the 50 ms future observation remain inside the source.
Two simultaneous environments are repeatability checks, both unassisted.
"""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from run_system import build_configs, build_env_agent, configure_for_system, pin_spawn_location
import numpy as np
import torch
from lightning.fabric import Fabric


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motion-file", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tracker-checkpoint", default="checkpoints/W1_wearer_human/last.ckpt")
    p.add_argument("--motion-ids", nargs="+", type=int)
    p.add_argument("--starts", nargs="+", type=float, default=[0.0])
    p.add_argument("--max-seconds", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "summary.json").exists():
        raise FileExistsError(args.out / "summary.json")
    run = SimpleNamespace(human="tracker", tracker_checkpoint=args.tracker_checkpoint,
                          motion_file=args.motion_file, physics_fps=400, env_dt=0.05,
                          llp_dt=0.01, viewer=False, fixed_base_z=0.0)
    cfgs = build_configs(run)
    configure_for_system(cfgs, run)
    torch.manual_seed(args.seed)
    fabric = Fabric(accelerator="gpu", devices=1, num_nodes=1, loggers=[], callbacks=[])
    fabric.launch()
    env, agent = build_env_agent(cfgs, run, fabric)
    pin_spawn_location(env)
    sim = env.simulator
    ids = args.motion_ids if args.motion_ids is not None else range(len(env.motion_lib.motion_files))
    results = {"motion_file": args.motion_file, "tracker_checkpoint": args.tracker_checkpoint,
               "seed": args.seed, "physics_hz": 400, "policy_hz": 20, "assist_torque_Nm": 0,
               "body_error_definition": "mean world position error across all 28 bodies, after first 0.5 seconds",
               "fall_definition": "pelvis z <0.5m for 0.5 seconds; stop on either copy",
               "short_clip_boundary_margin_s": 0.1,
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "trials": []}
    for mid in ids:
        for start in args.starts:
            duration = min(args.max_seconds, float(env.motion_lib.motion_lengths[mid])-start-0.1)
            nsteps = int(np.floor((duration+1e-6)/0.05))
            if nsteps < 20:
                raise ValueError("Less than one second of valid source remaining")
            torch.manual_seed(args.seed)
            env.motion_manager.motion_ids[:] = mid
            env.motion_manager.motion_times[:] = start
            obs, _ = env.reset(disable_motion_resample=True)
            initial = sim.get_robot_state()
            initial_arrays = {
                f"initial_{key}": getattr(initial, key).detach().cpu().numpy().copy()
                for key in ("rigid_body_pos", "rigid_body_rot", "rigid_body_vel",
                            "rigid_body_ang_vel", "dof_pos", "dof_vel")
            }
            log = {k: [] for k in ("time", "motion_time", "body_error", "root_pos", "ref_root_pos", "dof_pos", "ref_dof_pos", "dof_vel")}
            fallen = torch.zeros(2, dtype=torch.long, device=env.device)
            stopped = "clip_window_complete"
            with torch.no_grad():
                for step in range(nsteps):
                    td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))
                    output = agent.model(td)
                    obs, _, done, _, _ = env.step(output["mean_action"])
                    ctx = env._current_context
                    state, ref = ctx.current, ctx.mimic.ref_state
                    root = state.rigid_body_pos[:, 0]
                    t = (step+1)*0.05
                    values = {"time": np.array(t), "motion_time": env.motion_manager.motion_times,
                              "body_error": (state.rigid_body_pos-ref.rigid_body_pos).norm(dim=-1).mean(-1),
                              "root_pos": root, "ref_root_pos": ref.rigid_body_pos[:, 0],
                              "dof_pos": state.dof_pos, "ref_dof_pos": ref.dof_pos, "dof_vel": state.dof_vel}
                    for key, value in values.items():
                        log[key].append(value.detach().cpu().numpy().copy() if torch.is_tensor(value) else value)
                    if bool(done.any()) or not torch.allclose(env.motion_manager.motion_times, torch.full_like(env.motion_manager.motion_times,start+t), atol=2e-4):
                        raise RuntimeError("Unexpected reset/end or reference time mismatch")
                    if not torch.isfinite(root).all():
                        stopped = "nonfinite_state"
                        break
                    fallen = torch.where(root[:, 2] < 0.5, fallen+1, 0)
                    if int(fallen.max()) >= 10:
                        stopped = "fall"
                        break
            if stopped == "clip_window_complete" and int(fallen.max()) > 0:
                stopped = "clip_end_low_pelvis"
            arrays = {k: np.asarray(v) for k,v in log.items()}
            name = f"motion_{mid:02d}_start_{start:g}"
            np.savez_compressed(args.out / f"{name}.npz", **arrays, **initial_arrays)
            valid = arrays["time"] >= 0.5
            hip_error = arrays["dof_pos"]-arrays["ref_dof_pos"]
            trial = {"motion_id": int(mid), "source": env.motion_lib.motion_files[mid], "start_s": start,
                     "planned_s": nsteps*0.05, "elapsed_s": float(arrays["time"][-1]), "status": stopped,
                     "body_error_mean_m": arrays["body_error"][valid].mean(0).tolist(),
                     "body_error_p95_m": np.percentile(arrays["body_error"][valid],95,axis=0).tolist(),
                     "hip_rmse_deg": np.rad2deg(np.sqrt((hip_error[valid][:,:,[0,5]]**2).mean(axis=(0,2)))).tolist(),
                     "root_min_z_m": arrays["root_pos"][:,:,2].min(0).tolist(),
                     "root_end_z_m": arrays["root_pos"][-1,:,2].tolist(),
                     "trailing_low_pelvis_s": (fallen * 0.05).cpu().tolist(),
                     "root_xy_displacement_m": np.linalg.norm(arrays["root_pos"][-1,:,:2]-arrays["root_pos"][0,:,:2],axis=-1).tolist(),
                     "ref_root_xy_displacement_m": np.linalg.norm(arrays["ref_root_pos"][-1,:,:2]-arrays["ref_root_pos"][0,:,:2],axis=-1).tolist(),
                     "trace": str((args.out / f"{name}.npz").resolve())}
            results["trials"].append(trial)
            (args.out / "summary.json").write_text(json.dumps(results,indent=2))
            print(json.dumps(trial), flush=True)
    print("Unassisted native-clip evaluation complete", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
