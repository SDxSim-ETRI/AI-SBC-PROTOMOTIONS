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
"""Fixed-cohort paired pendulum evaluation, without failure-triggered resampling.

Build all evaluation plants from one baseline config, independently of the actor.
Both plants run the full scheduled horizon. Metrics use common survival up to
first failure, INCLUDING that failure's transition; first failures and censored
exposure are reported separately. This conditional metric is not an unconditional
performance guarantee. No failure can silently start a different random episode.
"""

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import random

from protomotions.utils.simulator_imports import import_simulator_before_torch

import_simulator_before_torch("newton")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from runtime_utils import build_env_agent  # noqa: E402

from protomotions.assist.llp_policy import LLPPolicy  # noqa: E402
from protomotions.envs.control.pendulum_research_control import (  # noqa: E402
    PendulumResearchControlConfig,
)
from protomotions.envs.obs.assist_obs import compute_assist_target_obs  # noqa: E402


class IsolatedH1Forecast:
    """Frozen H1 observations with a private RNG stream, separate from the plant.

    The emulated control component still runs in both paired plants. Only the
    assisted actor's target observation is replaced, so H1 cannot change the
    synthetic intention or the external disturbance random sequence.
    """

    def __init__(self, checkpoint, env):
        from protomotions.maniflow.angle_estimator import ManiFlowAngleEstimator
        from protomotions.maniflow.chunk_source import ManiFlowChunkSource

        self.device = torch.device(env.device)
        estimator = ManiFlowAngleEstimator.from_checkpoint(
            checkpoint, num_envs=env.num_envs, device=self.device, denoise_steps=1,
        )
        self.source = ManiFlowChunkSource(
            estimator, env.num_envs, env.dt, self.device, flexion_sign=-1.,
        )
        assert abs(self.source.knot_dt - env.dt) < 1e-6
        assert self.source.num_knots >= 3
        self.env_ids = torch.arange(env.num_envs, device=self.device)
        cfg = env.assist_component.config
        assert cfg.obs_future_samples == 3 and abs(cfg.obs_future_dt - env.dt) < 1e-6
        assert cfg.chunk_refresh_steps == 1
        kin = env.robot_config.kinematic_info
        self.low = kin.dof_limits_lower.to(self.device) * cfg.dof_limit_margin
        self.high = kin.dof_limits_upper.to(self.device) * cfg.dof_limit_margin

    def reset(self, q, seed):
        self.source.reset(self.env_ids, q, trunk=None)
        self.cpu_rng = torch.Generator().manual_seed(seed).get_state()
        self.cuda_rng = torch.Generator(device=self.device).manual_seed(seed).get_state()

    def predict(self, q, time_s, observe=True):
        if observe:
            self.source.observe(q, trunk=None)
        with torch.random.fork_rng(devices=[self.device]):
            torch.set_rng_state(self.cpu_rng)
            torch.cuda.set_rng_state(self.cuda_rng, self.device)
            knots, _ = self.source.predict_knots(
                self.env_ids, torch.full((len(q),), time_s, device=self.device),
            )
            self.cpu_rng = torch.get_rng_state()
            self.cuda_rng = torch.cuda.get_rng_state(self.device)
        return knots[:, :3].clamp(self.low, self.high)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def summarize(arrays, dt, warmup):
    valid = arrays["valid"] & (arrays["time_s"] >= warmup)
    # Masks refer to the unassisted member only, to avoid action-dependent bins.
    moving = np.abs(arrays["reference_velocity"]) >= .3
    groups = {"all": np.ones_like(moving, dtype=bool),
              "moving": moving, "near_stationary": ~moving,
              "heavy_load": arrays["reference_load_event"],
              "outside_load": ~arrays["reference_load_event"],
              "damping": arrays["reference_damping"] > 0}
    result = {}
    for name, mask in groups.items():
        mask = mask & valid[..., None]
        count = int(mask.sum())
        if not count:
            continue
        a, b = arrays["reference_tau"][mask], arrays["assisted_tau"][mask]
        ea = (arrays["reference_q"] - arrays["intent"])[mask]
        eb = (arrays["assisted_q"] - arrays["intent"])[mask]
        assist = arrays["assist_tau"][mask]
        am, bm = float(np.mean(a*a)), float(np.mean(b*b))
        result[name] = dict(joint_samples=count, exposure_joint_s=count*dt,
            reference_tau_sq_Nm2=am, assisted_tau_sq_Nm2=bm,
            torque_sq_change_percent=100*(bm/am-1),
            reference_abs_work_W=float(np.mean(np.abs(a*arrays["reference_velocity"][mask]))),
            assisted_abs_work_W=float(np.mean(np.abs(b*arrays["assisted_velocity"][mask]))),
            reference_tracking_rmse_deg=float(np.sqrt(np.mean(ea*ea))*180/np.pi),
            assisted_tracking_rmse_deg=float(np.sqrt(np.mean(eb*eb))*180/np.pi),
            assist_rms_Nm=float(np.sqrt(np.mean(assist*assist))),
            assist_above_19Nm_fraction=float(np.mean(np.abs(assist)>19)))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--checkpoint", default=None, help="Omit for a paired zero-action identity check")
    p.add_argument("--output-scale", type=float, default=1.)
    p.add_argument("--condition", choices=["legacy", "structured", "hold50"], default="legacy")
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
    cfg = PendulumResearchControlConfig(**{
        f.name: getattr(old, f.name) for f in fields(old) if f.name != "_target_"})
    cfg.structured_probability = .5 if args.condition == "structured" else 0.
    cfg.human_target_hold_steps = 5 if args.condition == "hold50" else 1
    seed_all(args.plant_seed)
    def configure_evaluation(resolved):
        resolved["env"].control_components["assist"] = cfg

    env, _ = build_env_agent(args.base_checkpoint, args.num_envs, overrides={
        "env._target_": "protomotions.envs.base_env.paired_assist_env.PairedAssistEnv",
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
        assert abs(policy.layout.torque_scale*args.output_scale-20) < 1e-6, "Require effective 20*tanh mapping"
        assert policy.layout.torque_limit >= policy.layout.torque_scale
        assert env.config.action_config["torque_scale"] == 20
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
                    predicted_angles=predicted)
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
        metrics=summarize(arrays, env.dt, args.warmup),
        caveat="Paired burden metrics are conditional on common survival; failures are reported separately. Work uses endpoint-average angular velocity.")
    if policy is None:
        report["zero_action_max_q_difference_rad"] = float(np.max(np.abs(arrays["reference_q"]-arrays["assisted_q"])))
        assert report["zero_action_max_q_difference_rad"] < 1e-4
    (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
