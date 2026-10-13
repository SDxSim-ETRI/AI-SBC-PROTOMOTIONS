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
"""Match the relative opposition weight to L8 on synthetic pendulum samples."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from protomotions.utils.simulator_imports import import_simulator_before_torch

import_simulator_before_torch("newton")
import numpy as np
import torch
from runtime_utils import build_env_agent
from protomotions.envs.rewards.assist import (
    compute_assist_opposition_cost,
    compute_relative_assist_opposition_cost,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="results/llp_L8_opposition/last.ckpt")
    parser.add_argument("--out", default="results/llp_L9_opposition_sequence/calibration.json")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--torque-floor", type=float, default=10.0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_cfg = torch.load(Path(args.checkpoint).parent / "resolved_configs.pt",
                           map_location="cpu", weights_only=False)
    env, agent = build_env_agent(args.checkpoint, args.num_envs, overrides={
        "env.max_episode_length": train_cfg["env"].max_episode_length})
    env._lazy_init_assist()
    obs, _ = env.reset()
    actor = agent.model._actor
    wearer, assist, valid = [], [], []
    for _ in range(args.steps):
        td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))
        with torch.no_grad():
            action = actor.mu(td)[actor.config.mu_key]
        obs, _, done, _, _ = env.step(action)
        ctx = env._current_context
        wearer.append(ctx.assist.tau_agent.detach().cpu().clone())
        assist.append(ctx.assist.tau_assist.detach().cpu().clone())
        valid.append(((env.assist_component._episode_time() >= 1.0) & ~done.bool()).cpu().clone())
        if done.any():
            env.reset(done.nonzero(as_tuple=False).flatten())
            obs = env.get_obs()
    h, u, keep = torch.stack(wearer), torch.stack(assist), torch.stack(valid)
    raw = compute_assist_opposition_cost(h[keep], u[keep])
    relative = compute_relative_assist_opposition_cost(h[keep], u[keep], args.torque_floor)
    if not keep.any() or not torch.isfinite(raw).all() or not torch.isfinite(relative).all() or relative.mean() <= 0:
        raise RuntimeError("Invalid calibration samples")
    raw_mean, relative_mean = float(raw.mean()), float(relative.mean())
    weight = -1e-4 * raw_mean / relative_mean
    result = {"checkpoint": str(Path(args.checkpoint).resolve()),
              "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
              "seed": args.seed, "num_envs": args.num_envs, "steps": args.steps,
              "valid_env_steps": int(keep.sum()), "torque_floor_Nm": args.torque_floor,
              "source": "synthetic randomized training pendulum; deterministic L8 final actor; no W1 data",
              "raw_cost_mean_Nm2": raw_mean, "relative_cost_mean": relative_mean,
              "old_weight": -1e-4, "matched_relative_weight": weight,
              "opposition_form": "log1p(relu(-human*assist)/torque_floor^2), per joint then sum"}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.with_suffix(".npz"), tau_agent=h.numpy(), tau_assist=u.numpy(), valid=keep.numpy())
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
