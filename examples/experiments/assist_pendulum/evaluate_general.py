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
"""Paired generalization evaluation, phase diagnostics and actual reward magnitudes.

The original research evaluator stays unchanged for already-running studies.
"""
import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
from protomotions.utils.simulator_imports import import_simulator_before_torch
import_simulator_before_torch("newton")
import numpy as np  # noqa: E402
import torch  # noqa: E402
from runtime_utils import build_env_agent  # noqa: E402
from evaluate_research import IsolatedH1Forecast, seed_all, summarize  # noqa: E402
from protomotions.assist.llp_policy import LLPPolicy  # noqa: E402
from protomotions.envs.control.pendulum_general_control import PendulumGeneralControlConfig  # noqa: E402
from protomotions.envs.obs.assist_obs import compute_assist_target_obs  # noqa: E402


def reward_audit(arrays, reward_configs, warmup):
    mask = arrays["valid"] & (arrays["time_s"] >= warmup)
    result = {}
    for name, component in reward_configs.items():
        raw = arrays["reward_raw_" + name][mask]
        params = component.static_params
        weighted = raw * params.get("weight", 0.)
        effective = np.clip(weighted, params.get("min_value", -np.inf), params.get("max_value", np.inf))
        result[name] = dict(weight=params.get("weight", 0.), raw_mean=float(raw.mean()),
            weighted_mean=float(weighted.mean()), effective_mean=float(effective.mean()),
            effective_p01_p50_p99=np.quantile(effective, [.01, .5, .99]).tolist(),
            clipped_fraction=float(np.mean(weighted != effective)))
    return result


def general_metrics(arrays, dt, warmup):
    result = summarize(arrays, dt, warmup)
    # Intent-driven event times are shared; use the unassisted membership only.
    for label in ("swing", "shock"):
        for side in (None, 0, 1):
            subset = dict(arrays)
            event = arrays["reference_" + label] > 0
            if side is not None:
                event = event.copy()
                event[..., 1-side] = False
            subset["reference_damping"] = event.astype(float)
            group = summarize(subset, dt, warmup).get("damping")
            if group:
                result[label + ("" if side is None else ("_right" if side == 0 else "_left"))] = group
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--checkpoint", default=None, help="Omit for a paired zero-action identity check")
    p.add_argument("--output-scale", type=float, default=1.)
    p.add_argument("--evaluation-limit", type=float, choices=[20., 23.], default=23.,
                   help="New models use 23; historical L1 x0.5 retains 20 Nm")
    p.add_argument("--condition", choices=["legacy", "structured", "phase", "hold50"], default="legacy")
    p.add_argument("--independent-limits", action="store_true")
    p.add_argument("--combined-human-limit", action="store_true",
                   help="Limit total human PD+FF inside the built-in actuator")
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--cohorts", type=int, default=2)
    p.add_argument("--plant-seed", type=int, default=34011)
    p.add_argument("--trajectory-seed", type=int, default=41011)
    p.add_argument("--warmup", type=float, default=.5)
    p.add_argument("--hlp-checkpoint", default=None,
                   help="Frozen real H1; otherwise use the synthetic intention forecast")
    p.add_argument("--hlp-seed", type=int, default=777)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    resolved = torch.load(Path(args.base_checkpoint).parent / "resolved_configs_inference.pt",
                          map_location="cpu", weights_only=False)
    old = resolved["env"].control_components["assist"]
    cfg = PendulumGeneralControlConfig(**{
        f.name: getattr(old, f.name) for f in fields(old) if f.name != "_target_"})
    cfg.structured_probability = .5 if args.condition in ("structured", "phase") else 0.
    cfg.phase_probability = .5 if args.condition == "phase" else 0.
    cfg.human_target_hold_steps = 5 if args.condition == "hold50" else 1
    seed_all(args.plant_seed)
    def configure_evaluation(resolved):
        resolved["env"].control_components["assist"] = cfg

    env, _ = build_env_agent(args.base_checkpoint, args.num_envs, overrides={
        "env._target_": "protomotions.envs.base_env.paired_assist_env.PairedAssistEnv",
        "env.assist_torque_limit": args.evaluation_limit,
        "env.action_config.torque_scale": args.evaluation_limit,
        "env.clamp_combined_injection": not args.independent_limits,
        "env.human_feedforward_inside_pd": args.combined_human_limit,
    }, config_transform=configure_evaluation)
    env._lazy_init_assist()
    ref = env.reference_env
    env.max_episode_length = args.steps + 2
    policy = LLPPolicy(args.checkpoint, env.device) if args.checkpoint else None
    hlp = IsolatedH1Forecast(args.hlp_checkpoint, env) if args.hlp_checkpoint else None
    if policy:
        assert policy.layout.action_transform == "tanh"
        assert abs(policy.layout.torque_scale*args.output_scale-args.evaluation_limit) < 1e-6, "Inconsistent effective torque mapping"
        assert policy.layout.torque_limit >= policy.layout.torque_scale
        assert env.config.action_config["torque_scale"] == args.evaluation_limit
    reward_configs = env.config.reward_components
    if args.checkpoint:
        trained = torch.load(Path(args.checkpoint).parent / "resolved_configs.pt", map_location="cpu", weights_only=False)
        reward_configs = trained["env"].reward_components
    all_logs = []
    failures = {"reference": [], "assisted": []}
    clip_max, clip_sum, clip_steps = 0., 0., 0
    reference_digest = hashlib.sha256()
    with torch.no_grad():
        for cohort in range(args.cohorts):
            seed_all(args.trajectory_seed + cohort)
            obs, _ = env.reset()
            if hlp:
                hlp.reset(env.context.current.dof_pos, args.hlp_seed + cohort)
            first = {name: torch.full((args.num_envs,), -1, device=env.device, dtype=torch.long)
                     for name in failures}
            logs = {}
            for step in range(args.steps):
                valid = (first["reference"] < 0) & (first["assisted"] < 0)
                load = ref.assist_component.in_stance_event().clone()
                damping = ref.assist_component._tau_damping.abs().clone()
                swing = ref.assist_component._tau_swing.abs().clone()
                shock = ref.assist_component._tau_shock.abs().clone()
                a_vel = ref.context.current.dof_vel.clone()
                b_vel = env.context.current.dof_vel.clone()
                a_ext = ref.assist_component.external_torque().clone()
                b_ext = env.assist_component.external_torque().clone()
                a_goal, a_goal_vel = ref.assist_component.get_pd_targets_and_vel()
                b_goal, b_goal_vel = env.assist_component.get_pd_targets_and_vel()
                a_goal, b_goal = a_goal.clone(), b_goal.clone()
                a_goal_vel, b_goal_vel = a_goal_vel.clone(), b_goal_vel.clone()
                predicted = env.context.assist.theta_d_future.clone()
                if hlp:
                    predicted = hlp.predict(env.context.current.dof_pos,
                                            step * env.dt, observe=step > 0)
                    obs["assist_target"] = compute_assist_target_obs(
                        predicted, env.context.current.dof_pos,
                        torch.zeros(args.num_envs, 1, device=env.device),
                    )
                action = policy.act(obs) if policy else torch.zeros(args.num_envs, 2, device=env.device)
                obs, _, _, terminal, extras = env.step(action)
                for name, terminated in (("reference", ref.terminate_buf), ("assisted", terminal)):
                    newly = terminated & (first[name] < 0)
                    first[name][newly] = step
                ca, cb = ref.context, env.context
                values = dict(valid=valid, time_s=torch.full_like(valid, (step+1)*env.dt, dtype=torch.float32),
                    reference_q=ca.current.dof_pos, assisted_q=cb.current.dof_pos,
                    intent=ca.assist.theta_g, reference_tau=ca.assist.tau_agent,
                    assisted_tau=cb.assist.tau_agent, assist_tau=cb.assist.tau_assist,
                    reference_velocity=(a_vel+ca.current.dof_vel)*.5,
                    assisted_velocity=(b_vel+cb.current.dof_vel)*.5,
                    reference_load_event=load, reference_damping=damping,
                    reference_external_tau=a_ext, assisted_external_tau=b_ext,
                    reference_pd_target=a_goal, assisted_pd_target=b_goal,
                    predicted_angles=predicted, reference_swing=swing, reference_shock=shock)
                assert torch.equal(ca.assist.theta_g, cb.assist.theta_g), "Paired intention drift"
                for event in ("_swing_c", "_shock_c", "_swing_start", "_shock_start"):
                    assert torch.equal(getattr(ref.assist_component, event), getattr(env.assist_component, event)), "Paired event drift"
                for name in env.config.reward_components:
                    values["reward_raw_" + name] = extras["raw_r/" + name]
                    values["reward_effective_" + name] = extras["scaled_r/" + name]
                for name, plant, velocity in (("reference", ref, a_goal_vel),
                                               ("assisted", env, b_goal_vel)):
                    ff = (plant._human_kd * velocity).clamp(-plant._effort_limit, plant._effort_limit)
                    if (not plant.config.human_velocity_feedforward
                            or plant.config.human_feedforward_inside_pd):
                        ff = torch.zeros_like(ff)
                    if plant._human_ff_enabled is not None:
                        ff = ff * plant._human_ff_enabled
                    values[name + "_ff_tau"] = ff
                    values[name + "_pd_tau"] = values[name + "_tau"] - ff
                for key, value in values.items():
                    logs.setdefault(key, []).append(value.detach().cpu().numpy().copy())
                for ex in (extras, ref.extras):
                    clip_max = max(clip_max, float(ex["assist/injection_error_max_Nm"]))
                    clip_sum += float(ex["assist/injection_clipped_fraction"])
                    clip_steps += 1
            block = {k: np.stack(v) for k, v in logs.items()}
            assert all(np.isfinite(v).all() for v in block.values()), "Nonfinite rollout, including failed plants"
            reference_digest.update(block["reference_q"].tobytes())
            reference_digest.update(block["intent"].tobytes())
            all_logs.append(block)
            for name in failures:
                failures[name].extend(first[name].cpu().tolist())
            print(f"cohort {cohort+1}/{args.cohorts} complete", flush=True)
    arrays = {k: np.concatenate([b[k] for b in all_logs], axis=1) for k in all_logs[0]}
    np.savez_compressed(out / "rollout.npz", **arrays,
        reference_first_failure=np.asarray(failures["reference"]),
        assisted_first_failure=np.asarray(failures["assisted"]),
        hip_inertia=env.hip_inertia_eff().cpu().numpy(),
        human_kp=env.assist_component._kp_base.cpu().numpy())
    report = dict(protocol=vars(args), policy_epoch=policy.epoch if policy else None,
        reference_sha256=reference_digest.hexdigest(),
        failures={name: sum(x >= 0 for x in times) for name, times in failures.items()},
        total_episodes=args.num_envs*args.cohorts,
        common_survival_fraction=float(arrays["valid"].mean()),
        injection_error_max_Nm=clip_max, injection_clipped_fraction=clip_sum/clip_steps,
        metrics=general_metrics(arrays, env.dt, args.warmup),
        reward_audit=reward_audit(arrays, reward_configs, args.warmup),
        assist_max_abs_Nm=float(np.abs(arrays["assist_tau"]).max()),
        assist_above_95pct_limit_fraction=float(np.mean(
            np.abs(arrays["assist_tau"])[arrays["valid"] & (arrays["time_s"] >= args.warmup)] > .95 * args.evaluation_limit)),
        phase_model="Intent-linked dissipative swing/reversal torque; fixed inertia, no foot contact",
        caveat="Paired burden metrics are conditional on common survival; failures are reported separately. Work uses endpoint-average angular velocity.")
    if policy is None:
        report["zero_action_max_q_difference_rad"] = float(np.max(np.abs(arrays["reference_q"]-arrays["assisted_q"])))
        assert report["zero_action_max_q_difference_rad"] < 1e-4
    (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
