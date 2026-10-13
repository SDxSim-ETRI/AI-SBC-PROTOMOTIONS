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
"""Paired synthetic probes of wearer velocity feedforward, without episode resets.

Run each condition in its own process with the same seed. Compare only the common
prefix before any condition terminates. This probes a training-domain mismatch;
it does not measure full-body W1 performance or justify a causal W1 claim.
"""

import argparse
from pathlib import Path

from protomotions.utils.simulator_imports import import_simulator_before_torch

import_simulator_before_torch("newton")
import numpy as np
import torch

from runtime_utils import build_env_agent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="results/llp_L9_opposition/epoch_2000.ckpt")
    p.add_argument("--feedforward", type=int, choices=(0, 1), required=True)
    p.add_argument("--assist", type=int, choices=(0, 1), required=True)
    p.add_argument("--disable-prob", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env, agent = build_env_agent(args.checkpoint, args.num_envs, overrides={
        "env.human_velocity_feedforward": bool(args.feedforward),
        "env.human_feedforward_disable_prob": args.disable_prob,
        "env.max_episode_length": 1000})
    torch.manual_seed(args.seed)
    env._lazy_init_assist()
    obs, _ = env.reset()
    enabled = env._human_ff_enabled
    if enabled is None:
        enabled = torch.full((args.num_envs, 1), bool(args.feedforward), device=env.device)
    enabled = enabled.clone()
    actor = agent.model._actor
    alive = torch.ones(args.num_envs, dtype=torch.bool, device=env.device)
    logs = {k: [] for k in ("q", "velocity", "target", "wearer", "assist", "ff", "valid")}
    for _ in range(args.steps):
        td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))
        with torch.no_grad():
            action = actor.mu(td)[actor.config.mu_key]
        if not args.assist:
            action = torch.zeros_like(action)
        obs, _, done, _, _ = env.step(action)
        alive &= ~done.bool()
        c = env._current_context
        wearer = c.assist.tau_agent
        pd = env.simulator.get_substep_mean_dof_forces()
        values = dict(q=c.current.dof_pos, velocity=c.current.dof_vel,
                      target=c.assist.theta_g, wearer=wearer,
                      assist=c.assist.tau_assist, ff=wearer-pd,
                      valid=alive & (env.assist_component._episode_time() >= 1.0))
        for key, value in values.items():
            logs[key].append(value.detach().cpu().clone())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: torch.stack(v).numpy() for k, v in logs.items()}
    disabled = ~enabled[:, 0].cpu().numpy().astype(bool)
    if disabled.any() and np.abs(arrays["ff"][:, disabled]).max() > 1e-5:
        raise RuntimeError("Disabled wearers received velocity feedforward")
    if env._human_ff_enabled is not None:
        env.reset(torch.arange(0, args.num_envs, 2, device=env.device))
        if not torch.equal(env._human_ff_enabled, enabled):
            raise RuntimeError("Wearer type changed on partial reset")
    np.savez_compressed(out, **arrays,
                        kp=env._ke_base.cpu().numpy(), kd=env._kd_base.cpu().numpy(),
                        inertia=env.hip_inertia_eff().cpu().numpy(),
                        ff_enabled=enabled.cpu().numpy(),
                        **{f"arg_{k}": v for k, v in vars(args).items()})
    print(f"saved {out}; {int(alive.sum())}/{args.num_envs} envs without termination", flush=True)


if __name__ == "__main__":
    main()
