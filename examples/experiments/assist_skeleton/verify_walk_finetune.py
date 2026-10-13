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
"""Preflight L13 on real physics; no training before this succeeds."""

import argparse
import json
import time
from pathlib import Path

from protomotions.utils.simulator_imports import import_simulator_before_torch
import_simulator_before_torch("newton")

import torch
from tensordict import TensorDict

from check_train_env import build_env
from protomotions.assist.llp_policy import LLPPolicy
from protomotions.utils.hydra_replacement import get_class


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="results/llp_L13_fullbody_walk")
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--steps", type=int, default=350)
    p.add_argument("--expected-motion-ids", default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    torch.manual_seed(13)
    out = Path(args.out)
    report = {"pass": False, "num_envs": args.num_envs, "steps": args.steps}
    run = Path(args.run)
    try:
        env = build_env(run, args.num_envs)
        cfg = torch.load(run / "resolved_configs.pt", map_location="cpu", weights_only=False)["agent"]
        from lightning.fabric import Fabric
        fabric = Fabric(accelerator="gpu", devices=1, loggers=[], callbacks=[])
        fabric.launch()
        agent = get_class(cfg._target_)(config=cfg, env=env, fabric=fabric, root_dir=run)
        agent.setup()
        agent.model.eval()
        source = LLPPolicy(cfg.initial_actor_checkpoint, env.device)
        maxima = {"actor_raw": 0., "zero_q": 0., "zero_tau": 0.}
        zero_q, zero_tau, zero_reward = [], [], []
        obs, _ = env.reset()
        assert sum(obs[k].shape[-1] for k in source.layout.in_keys) == 137
        assert env.dt == .01
        assert env.config.action_config["torque_scale"] == 40
        assert env.config.action_config["action_transform"] is None
        report["conditions"] = {}
        seen_motion_ids = set()
        for mode in ("zero", "L9"):
            obs, _ = env.reset()
            torch.testing.assert_close(env.context.current.dof_pos,
                                       env.reference_env.context.current.dof_pos, rtol=0, atol=1e-6)
            start = time.perf_counter()
            failures = resets = 0
            for step in range(args.steps):
                with torch.no_grad():
                    td = TensorDict({k: obs[k] for k in source.layout.in_keys}, batch_size=env.num_envs)
                    raw = agent.model._actor.mu(td)[cfg.model.actor.mu_key]
                    mirrored = {k: -obs[k] for k in source.layout.in_keys}
                    mirrored["assist_target"][:, -1] = obs["assist_target"][:, -1]
                    expected = -source.act(mirrored) if source.layout.flexion_sign == -1 else source.act(obs)
                    maxima["actor_raw"] = max(maxima["actor_raw"], float((raw - expected).abs().max()))
                action = torch.zeros_like(raw) if mode == "zero" else raw
                obs, rew, done, term, extras = env.step(action)
                env.check_pairing()
                seen_motion_ids.update(env.motion_manager.motion_ids.cpu().tolist())
                if args.expected_motion_ids:
                    expected_ids = {int(v) for v in args.expected_motion_ids.split(",")}
                    assert seen_motion_ids <= expected_ids, "held-out motion used by training"
                assert torch.isfinite(rew).all()
                if mode == "zero":
                    a, b = env.reference_env.context, env.context
                    maxima["zero_q"] = max(maxima["zero_q"], float((a.current.dof_pos - b.current.dof_pos).abs().max()))
                    maxima["zero_tau"] = max(maxima["zero_tau"], float((a.assist.tau_agent - b.assist.tau_agent).abs().max()))
                    zero_q.append((a.assist.dof_pos - b.assist.dof_pos).cpu().clone())
                    zero_tau.append((a.assist.tau_agent - b.assist.tau_agent).cpu().clone())
                    zero_reward.append(rew.cpu().clone())
                failures += int(term.sum())
                ids = done.nonzero().flatten()
                # Exercise partial resets, independently of any natural endings.
                if step == 177:
                    ids = torch.unique(torch.cat((ids, torch.arange(0, env.num_envs, 2, device=env.device))))
                if ids.numel():
                    resets += ids.numel()
                    obs, _ = env.reset(ids)
            report["conditions"][mode] = {"failures": failures, "resets": resets,
                "steps_per_second": args.steps / (time.perf_counter() - start)}
        assert maxima["actor_raw"] < 1e-5, maxima
        report["observed_motion_ids"] = sorted(seen_motion_ids)
        if args.expected_motion_ids:
            assert seen_motion_ids == expected_ids, "preflight did not cover all training motions"
        # Compare RMS with the measured 8s repeatability in the prior diagnostic
        # (hip q 0.00368 rad, wearer 2.73 Nm), rather than rejecting single peaks.
        q_rms = float(torch.stack(zero_q).square().mean().sqrt())
        tau_rms = float(torch.stack(zero_tau).square().mean().sqrt())
        report["zero_assist_repeatability"] = dict(hip_q_RMS_rad=q_rms, wearer_RMS_Nm=tau_rms,
                                                   mean_reward=float(torch.stack(zero_reward).mean()))
        report["max_difference"] = maxima
        torch.save({"q_difference": torch.stack(zero_q), "tau_difference": torch.stack(zero_tau),
                    "reward": torch.stack(zero_reward)}, out.with_suffix(".pt"))
        assert q_rms < .005 and tau_rms < 3., report
        # Test end semantics through actual termination hooks.
        for side in ("reference", "assisted"):
            env.reset()
            target = env.reference_env if side == "reference" else env
            original = target.check_resets_and_terminations
            def forced(context):
                done, term = original(context)
                done, term = done.clone(), term.clone()
                done[0] = term[0] = True
                return done, term
            target.check_resets_and_terminations = forced
            try:
                _, _, done, term, _ = env.step(torch.zeros(env.num_envs, 2, device=env.device))
                assert done[0] and bool(term[0]) == (side == "assisted")
            finally:
                target.check_resets_and_terminations = original
        report["max_difference"] = maxima
        report["peak_torch_MiB"] = torch.cuda.max_memory_allocated() / 2**20
        # Export a reviewable initial policy without changing any source checkpoint.
        # Standalone Fabric has no distributed process group. This artifact is
        # actor initialization only; do not save transient forced-failure env state.
        torch.save(agent.get_state_dict({}), run / "initial.ckpt")
        report["pass"] = True
    finally:
        out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
