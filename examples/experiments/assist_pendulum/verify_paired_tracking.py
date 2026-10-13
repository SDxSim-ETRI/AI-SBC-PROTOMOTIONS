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
"""GPU integration checks for L12 before any training service is started."""
import argparse
import json
import time
from pathlib import Path

from protomotions.utils.simulator_imports import import_simulator_before_torch

import_simulator_before_torch("newton")
import numpy as np
import torch

from runtime_utils import build_env_agent
from protomotions.envs.context_views import EnvContext
from protomotions.envs.rewards.assist import compute_paired_tracking_rew


def check_exogenous(env):
    ref = env.reference_env
    for name in ("_inertia_scale", "_mass_scale", "_ke_base", "_kd_base", "_human_ff_enabled"):
        a, b = getattr(ref, name, None), getattr(env, name, None)
        assert (a is None) == (b is None), name
        if a is not None:
            assert torch.equal(a, b), name
    ca, cb = ref.assist_component, env.assist_component
    for name in ca._params:
        assert torch.equal(ca._params[name], cb._params[name]), name
    for name in ("_seg_start", "_blend_dur", "_ev_start", "_ev_end", "_ev_load", "_ev_next",
                 "_pt_start", "_pt_end", "_pt_amp", "_pt_next", "_tau_ext", "_kp_scale", "_kd_scale"):
        assert torch.equal(getattr(ca, name), getattr(cb, name)), name
    assert torch.equal(ca.get_pd_targets(), cb.get_pd_targets()), "PD targets"
    assert torch.equal(ref.context.assist.theta_d_future, env.context.assist.theta_d_future), "HLP emulation"
    assert torch.count_nonzero(ca._tau_assist).item() == 0, "reference assistance"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="results/llp_L11_opposition/epoch_2000.ckpt")
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--steps", type=int, default=350)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    torch.manual_seed(17)
    np.random.seed(17)
    env, agent = build_env_agent(args.checkpoint, args.num_envs, overrides={
        "env._target_": "protomotions.envs.base_env.paired_assist_env.PairedAssistEnv"})
    tracking = env.config.reward_components["tracking"]
    tracking.compute_func = compute_paired_tracking_rew
    tracking.dynamic_vars["reference_dof_pos"] = EnvContext.assist.reference_dof_pos
    assert all("reference_dof_pos" not in str(c.dynamic_vars)
               for c in env.config.observation_components.values())
    report = {"pass": False, "num_envs": args.num_envs, "steps_per_condition": args.steps}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        env.max_episode_length = 79  # Test PPO-side curriculum propagation.
        obs, _ = env.reset()
        ff = env._human_ff_enabled.clone()
        check_exogenous(env)
        report["conditions"] = {}
        for mode in ("zero", "different_action"):
            env.reset()
            worst = {"q_rad": 0., "wearer_Nm": 0., "tracking_reward": 0.}
            reset_count = 0
            start = time.perf_counter()
            for step in range(args.steps):
                action = torch.zeros(args.num_envs, 2, device=env.device)
                if mode == "different_action":
                    # Applied 8 Nm, enough to separate B while preserving exogenous inputs.
                    action[:, 0] = .2 * np.sin(step * .13)
                    action[:, 1] = -.2 * np.cos(step * .11)
                _, reward, done, term, extras = env.step(action)
                check_exogenous(env)
                assert env.reference_env.max_episode_length == env.max_episode_length
                assert torch.isfinite(reward).all()
                if mode == "zero":
                    a, b = env.reference_env.context, env.context
                    worst["q_rad"] = max(worst["q_rad"], float((a.current.dof_pos-b.current.dof_pos).abs().max()))
                    worst["wearer_Nm"] = max(worst["wearer_Nm"], float((a.assist.tau_agent-b.assist.tau_agent).abs().max()))
                    r = compute_paired_tracking_rew(b.current.dof_pos, a.current.dof_pos, b.assist.theta_g)
                    worst["tracking_reward"] = max(worst["tracking_reward"], float((r-1).abs().max()))
                ids = done.nonzero().flatten()
                if step % 43 == 0:
                    ids = torch.unique(torch.cat((ids, torch.arange(0, args.num_envs, 2, device=env.device))))
                if ids.numel():
                    reset_count += ids.numel()
                    env.reset(ids)
                    check_exogenous(env)
                    assert torch.equal(env._human_ff_enabled, ff)
            torch.cuda.synchronize()
            report["conditions"][mode] = dict(max_difference=worst, reset_count=reset_count,
                                              steps_per_s=args.steps/(time.perf_counter()-start))
            if mode == "zero":
                # Same-run pair should match far more closely than cross-run W1 noise.
                assert worst["q_rad"] < 1e-4, worst
                assert worst["wearer_Nm"] < .1, worst
                assert worst["tracking_reward"] < 1e-3, worst

        # Force A-only and B-only failure through the actual termination hook.
        for side in ("reference", "assisted"):
            env.reset()
            target = env.reference_env if side == "reference" else env
            original = target.check_resets_and_terminations
            def forced(context):
                done, term = original(context)
                done, term = done.clone(), term.clone()
                done[0] = True
                term[0] = True
                return done, term
            target.check_resets_and_terminations = forced
            try:
                _, _, done, term, _ = env.step(torch.zeros(args.num_envs, 2, device=env.device))
                assert done[0] and bool(term[0]) == (side == "assisted")
                env.reset(torch.tensor([0], device=env.device))
                assert env.progress_buf[0] == env.reference_env.progress_buf[0] == 0
            finally:
                target.check_resets_and_terminations = original
        report["peak_torch_memory_MiB"] = torch.cuda.max_memory_allocated() / 2**20
        report["pass"] = True
    finally:
        out.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
