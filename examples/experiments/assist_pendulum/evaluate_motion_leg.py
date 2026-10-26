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
"""진자 짝 평가 (2026-10-21): 무보조 진자와 보조 진자를 같은 의도·플랜트·외란으로 나란히 돌린다.

조건
* legacy / phase / hold50: 2026-10-20 R0 진자 평가와 같은 합성 궤적 프로토콜·seed
  (`evaluate_feasible.py`). L1·R0 결과는 저장본을 쓰고 새 모델만 돌린다.
* seed_val: SEED 고관절 동작 중 H1·새 LLP 모두 학습하지 않은 클립 (H1 validation 분할).
  학습과 같은 두 플랜트 집단(가벼운 다리 50 %: 달리기·조깅·점프 / 일반 50 %: 걷기·조깅·점프·
  앉기류·기타)과 같은 실현 가능성 규칙.
* etri: ETRI 슈트 13동작 평가용 사본 (`prepare_motion_eval_data.py`), 같은 두 플랜트 집단에
  동작별 같은 시간.
모든 조건에서 보조 진자의 미래 목표 관측은 동결 H1의 실제 예측이다 (동작 조건은 데이터의 몸통
기울기도 H1에 준다). 단일 관절 모델은 `--share-legs`로 양쪽 다리에 같은 가중치로 적용한다.
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
from evaluate_research import IsolatedH1Forecast, seed_all, summarize  # noqa: E402
from runtime_utils import build_env_agent  # noqa: E402

from protomotions.assist.llp_pair import LLPLegPair, LLPSharedLeg  # noqa: E402
from protomotions.assist.llp_policy import LLPPolicy  # noqa: E402
from protomotions.envs.control.assist_disturbance_control import (  # noqa: E402
    DisturbedAssistTargetControlConfig,
)
from protomotions.envs.control.pendulum_general_control import PendulumGeneralControlConfig  # noqa: E402
from protomotions.envs.obs.assist_obs import compute_assist_target_obs  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
SYNTHETIC = ("legacy", "phase", "hold50")
SEED_ZARR = "/home/bak/Projects/ManiFlow_Policy/ManiFlow/data/hip-flexion-seed-locomotion-100fps.zarr"
SEED_METADATA = str(ROOT / "data/seed/metadata/seed_metadata_v004.parquet")
ETRI_DIR = ROOT / "results/llp20_motion_20261021/eval_data"
SEED_GROUPS = (
    ("running", r"^running"),
    ("jogging", r"^jogging"),
    ("deep", r"sitting|kneel|transition|crouch"),
    ("jumping", r"^jumping"),
    ("walking", r"^walking(, turning)?$"),
    ("other", r".*"),
)
ETRI_GROUPS = (
    ("run", r"^run$"), ("walk", r"^walk$"), ("squat", r"^squat$"), ("lunge", r"^lunge"),
    ("highknee", r"^highknee$"), ("stepinplace", r"^stepinplace"), ("onestep", r"^onestep"),
    ("hop", r"^onehopforward$"),
)
LIGHT = (1.0, 3.9)  # hip 축 0.42–0.80 kg·m² (학습의 가벼운 다리 집단)
GENERAL = (1.0, 25.0)
NEED_EDGES = (0, 5, 10, 20, 40, 80, np.inf)


def motion_setup(condition, trajectory_seed):
    if condition == "seed_val":
        pools = (("light", .5, "motion", *LIGHT, (("running", .5), ("jogging", .3), ("jumping", .2))),
                 ("general", .5, "motion", *GENERAL,
                  (("walking", .3), ("jogging", .2), ("jumping", .15), ("deep", .25), ("other", .1))))
        return dict(motion_zarr_path=SEED_ZARR, motion_split="val", motion_metadata_parquet=SEED_METADATA,
                    motion_groups=SEED_GROUPS, intent_pools=pools)
    equal = tuple((name, 1.0 / len(ETRI_GROUPS)) for name, _ in ETRI_GROUPS)
    pools = (("light", .5, "motion", *LIGHT, equal), ("general", .5, "motion", *GENERAL, equal))
    return dict(motion_zarr_path=str(ETRI_DIR / "etri_suit13_seedzero_clamp86_100fps.zarr"), motion_split="all",
                motion_metadata_parquet=str(ETRI_DIR / "etri_suit13_metadata.parquet"),
                motion_groups=ETRI_GROUPS, intent_pools=pools)


class TrunkH1Forecast(IsolatedH1Forecast):
    """`IsolatedH1Forecast` + 데이터의 몸통 pitch/roll 입력 (동작 조건)."""

    def reset(self, q, seed, trunk=None):
        self.source.reset(self.env_ids, q, trunk=trunk)
        self.cpu_rng = torch.Generator().manual_seed(seed).get_state()
        self.cuda_rng = torch.Generator(device=self.device).manual_seed(seed).get_state()

    def predict(self, q, time_s, observe=True, trunk=None):
        if observe:
            self.source.observe(q, trunk=trunk)
        with torch.random.fork_rng(devices=[self.device]):
            torch.set_rng_state(self.cpu_rng)
            torch.cuda.set_rng_state(self.cuda_rng, self.device)
            knots, _ = self.source.predict_knots(
                self.env_ids, torch.full((len(q),), time_s, device=self.device))
            self.cpu_rng = torch.get_rng_state()
            self.cuda_rng = torch.cuda.get_rng_state(self.device)
        return knots[:, :3].clamp(self.low, self.high)


def _group_stats(ref_tau, asst_tau, assist, ref_q, asst_q, intent, ref_vel):
    if ref_tau.size == 0:
        return None
    a2, b2 = float(np.mean(ref_tau ** 2)), float(np.mean(asst_tau ** 2))
    ea, eb = ref_q - intent, asst_q - intent
    active = np.abs(assist) > 0.5
    agree = (np.sign(assist) == np.sign(ref_tau)) & active
    return dict(
        joint_samples=int(ref_tau.size), reference_tau_sq_Nm2=a2, assisted_tau_sq_Nm2=b2,
        benefit_Nm2=a2 - b2, torque_sq_change_percent=100 * (b2 / a2 - 1) if a2 > 0 else None,
        reference_tracking_rmse_deg=float(np.degrees(np.sqrt(np.mean(ea ** 2)))),
        assisted_tracking_rmse_deg=float(np.degrees(np.sqrt(np.mean(eb ** 2)))),
        assist_rms_Nm=float(np.sqrt(np.mean(assist ** 2))),
        saturation_fraction=float(np.mean(np.abs(assist) > 19.0)),
        sign_agreement=float(agree.sum() / max(active.sum(), 1)),
        reference_speed_mean=float(np.mean(np.abs(ref_vel))))


def _edge(value):
    """JSON용 구간 끝 (무한대는 None)."""
    return None if np.isinf(value) else float(value)


def extended_metrics(arrays, dt, warmup, group_names=None, pool_names=None):
    """need bin·각속도·동작 그룹·집단별 이득, 방향 일치, 명령 변화, 고주파 성분."""
    valid = arrays["valid"] & (arrays["time_s"] >= warmup)
    mask = np.repeat(valid[..., None], 2, axis=2)
    pick = {k: arrays[k][mask] for k in ("reference_tau", "assisted_tau", "assist_tau", "reference_q",
                                         "assisted_q", "intent", "reference_velocity")}

    def stats(sel):
        return _group_stats(pick["reference_tau"][sel], pick["assisted_tau"][sel], pick["assist_tau"][sel],
                            pick["reference_q"][sel], pick["assisted_q"][sel], pick["intent"][sel],
                            pick["reference_velocity"][sel])

    need = np.abs(pick["reference_tau"])
    speed = np.abs(pick["reference_velocity"])
    result = dict(all=stats(np.ones_like(need, dtype=bool)), fast_over_4rad_s=stats(speed > 4.0),
                  slow_under_4rad_s=stats(speed <= 4.0), fast_time_fraction=float(np.mean(speed > 4.0)))
    result["need_bins"] = [dict(lo=float(lo), hi=_edge(hi), **(stats((need >= lo) & (need < hi)) or {}))
                           for lo, hi in zip(NEED_EDGES[:-1], NEED_EDGES[1:])]
    result["speed_bins"] = [dict(lo=lo, hi=_edge(hi), **(stats((speed >= lo) & (speed < hi)) or {}))
                            for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 6), (6, 8), (8, np.inf))]
    if group_names is not None:
        groups = np.repeat(arrays["clip_group"][..., None], 2, axis=2)[mask]
        result["groups"] = {name: stats(groups == j) for j, name in enumerate(group_names)}
    if pool_names is not None:
        pools = np.repeat(np.broadcast_to(arrays["env_pool"][None, :], valid.shape)[..., None], 2, axis=2)[mask]
        result["pools"] = {name: stats(pools == j) for j, name in enumerate(pool_names)}
    # 명령 변화: 연속 두 step이 모두 유효한 표본의 |Δτ_assist| (10 ms)
    tau = arrays["assist_tau"]
    pair = valid[1:] & valid[:-1]
    change = np.abs(np.diff(tau, axis=0))[np.repeat(pair[..., None], 2, axis=2)]
    need_pair = np.abs(arrays["reference_tau"][1:])[np.repeat(pair[..., None], 2, axis=2)]
    result["command_change"] = dict(
        mean_Nm_per_step=float(change.mean()), p99_Nm_per_step=float(np.percentile(change, 99)),
        over_5Nm_fraction=float(np.mean(change > 5.0)),
        by_need=[dict(lo=float(lo), hi=_edge(hi),
                      mean_Nm_per_step=float(change[(need_pair >= lo) & (need_pair < hi)].mean())
                      if ((need_pair >= lo) & (need_pair < hi)).any() else None)
                 for lo, hi in zip(NEED_EDGES[:-1], NEED_EDGES[1:])])
    # 고주파 성분: 전 구간 유효한 env의 보조 토크 스펙트럼에서 10 Hz 초과 비율 (평균 제거)
    start = int(round(warmup / dt))
    whole = arrays["valid"][start:].all(axis=0)
    if whole.any():
        x = tau[start:, whole, :]
        x = x - x.mean(axis=0, keepdims=True)
        power = np.abs(np.fft.rfft(x, axis=0)) ** 2
        freq = np.fft.rfftfreq(x.shape[0], d=dt)
        result["assist_power_above_10Hz_fraction"] = float(power[freq > 10.0].sum() / max(power[1:].sum(), 1e-12))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--checkpoint", default=None, help="생략하면 무보조 짝 동일성 점검")
    p.add_argument("--left-checkpoint", default=None, help="다리별 모델: 왼쪽 (--checkpoint = 오른쪽)")
    p.add_argument("--share-legs", action="store_true", help="단일 관절 모델 하나를 양쪽 다리에 같은 가중치로 적용")
    p.add_argument("--output-scale", type=float, default=1.)
    p.add_argument("--evaluation-limit", type=float, default=20.)
    p.add_argument("--condition", choices=[*SYNTHETIC, "seed_val", "etri"], required=True)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--cohorts", type=int, default=2)
    p.add_argument("--plant-seed", type=int, default=63031)
    p.add_argument("--trajectory-seed", type=int, default=52021)
    p.add_argument("--warmup", type=float, default=.5)
    p.add_argument("--hlp-checkpoint", required=True)
    p.add_argument("--hlp-seed", type=int, default=991)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    motion = args.condition not in SYNTHETIC
    resolved = torch.load(Path(args.base_checkpoint).parent / "resolved_configs_inference.pt",
                          map_location="cpu", weights_only=False)
    old = resolved["env"].control_components["assist"]
    if motion:
        names = {f.name for f in fields(DisturbedAssistTargetControlConfig)}
        cfg = DisturbedAssistTargetControlConfig(**{
            f.name: getattr(old, f.name) for f in fields(old) if f.name != "_target_" and f.name in names})
        setup = motion_setup(args.condition, args.trajectory_seed)
        for key, value in setup.items():
            setattr(cfg, key, value)
        cfg.intent_source = "motion_data"
        cfg.motion_accel_frac = 0.8
        cfg.motion_accel_check_dofs = ()
        cfg.motion_seed = args.trajectory_seed
        cfg.intent_pool_seed = args.plant_seed
    else:
        cfg = PendulumGeneralControlConfig(**{f.name: getattr(old, f.name) for f in fields(old) if f.name != "_target_"})
        cfg.structured_probability = .5 if args.condition == "phase" else 0.
        cfg.phase_probability = .5 if args.condition == "phase" else 0.
        cfg.human_target_hold_steps = 5 if args.condition == "hold50" else 1
    seed_all(args.plant_seed)

    def configure_evaluation(resolved):
        resolved["env"].control_components["assist"] = cfg

    env, _ = build_env_agent(args.base_checkpoint, args.num_envs, overrides={
        "env._target_": "protomotions.envs.base_env.paired_assist_env.PairedAssistEnv",
        "env.assist_torque_limit": args.evaluation_limit,
        "env.action_config.torque_scale": args.evaluation_limit,
        "env.clamp_combined_injection": False,
        "env.human_feedforward_inside_pd": True,
    }, config_transform=configure_evaluation)
    env._lazy_init_assist()
    ref = env.reference_env
    env.max_episode_length = args.steps + 2
    if args.share_legs:
        assert args.checkpoint and not args.left_checkpoint
        policy = LLPSharedLeg(args.checkpoint, env.device)
    elif args.left_checkpoint:
        policy = LLPLegPair(args.checkpoint, args.left_checkpoint, env.device)
    else:
        policy = LLPPolicy(args.checkpoint, env.device) if args.checkpoint else None
    if policy:
        lay = policy.layout
        assert lay.action_transform == "tanh"
        assert abs(lay.torque_scale * args.output_scale - args.evaluation_limit) < 1e-6, "Inconsistent torque mapping"
        assert lay.torque_limit >= lay.torque_scale
        assert lay.history_steps == env.config.num_state_history_steps
    hlp = TrunkH1Forecast(args.hlp_checkpoint, env)
    comp_a, comp_b = ref.assist_component, env.assist_component
    group_names = [str(g[0]) for g in cfg.motion_groups] if motion else None
    pool_names = [s["name"] for s in comp_b.pool_summary] if motion else None
    if motion:
        from protomotions.envs.control.motion_intent import group_clip_probabilities

        table = comp_b._intent.table
        _, group_of_clip = group_clip_probabilities(table.clip_type, table.clip_len.cpu().numpy(),
                                                    cfg.motion_groups, [(group_names[0], 1.0)],
                                                    comp_b._intent.min_frames)
        group_of_clip = torch.as_tensor(group_of_clip, device=env.device)
    all_logs, failures = [], {"reference": [], "assisted": []}
    clip_max = 0.
    reference_digest = hashlib.sha256()
    zeros_age = torch.zeros(args.num_envs, 1, device=env.device)
    with torch.no_grad():
        for cohort in range(args.cohorts):
            seed_all(args.trajectory_seed + cohort)
            obs, _ = env.reset()
            trunk = comp_b._intent.trunk_at(comp_b._episode_time()) if motion else None
            hlp.reset(env.context.current.dof_pos, args.hlp_seed + cohort, trunk)
            first = {name: torch.full((args.num_envs,), -1, device=env.device, dtype=torch.long) for name in failures}
            logs = {}
            for step in range(args.steps):
                valid = (first["reference"] < 0) & (first["assisted"] < 0)
                load = comp_a.in_stance_event().clone()
                a_vel = ref.context.current.dof_vel.clone()
                b_vel = env.context.current.dof_vel.clone()
                if motion:
                    trunk = comp_b._intent.trunk_at(comp_b._episode_time())
                    clip = comp_a._intent._clip[0]
                    hold = comp_a._intent._hold[0]
                    group = torch.where(hold, torch.full_like(clip, -1), group_of_clip[clip])
                predicted = hlp.predict(env.context.current.dof_pos, step * env.dt, observe=step > 0, trunk=trunk)
                obs["assist_target"] = compute_assist_target_obs(predicted, env.context.current.dof_pos, zeros_age)
                action = policy.act(obs) if policy else torch.zeros(args.num_envs, 2, device=env.device)
                obs, _, _, terminal, extras = env.step(action)
                for name, terminated in (("reference", ref.terminate_buf), ("assisted", terminal)):
                    newly = terminated & (first[name] < 0)
                    first[name][newly] = step
                ca, cb = ref.context, env.context
                assert torch.equal(ca.assist.theta_g, cb.assist.theta_g), "Paired intention drift"
                values = dict(valid=valid, time_s=torch.full_like(valid, (step + 1) * env.dt, dtype=torch.float32),
                              reference_q=ca.current.dof_pos, assisted_q=cb.current.dof_pos,
                              intent=ca.assist.theta_g, reference_tau=ca.assist.tau_agent,
                              assisted_tau=cb.assist.tau_agent, assist_tau=cb.assist.tau_assist,
                              reference_velocity=(a_vel + ca.current.dof_vel) * .5,
                              assisted_velocity=(b_vel + cb.current.dof_vel) * .5,
                              reference_load_event=load, predicted_angles=predicted,
                              reference_damping=torch.zeros_like(load, dtype=torch.float32))
                if motion:
                    values["clip_group"] = group
                for key, value in values.items():
                    logs.setdefault(key, []).append(value.detach().cpu().numpy().copy())
                for ex in (extras, ref.extras):
                    clip_max = max(clip_max, float(ex["assist/injection_error_max_Nm"]))
            block = {k: np.stack(v) for k, v in logs.items()}
            assert all(np.isfinite(v).all() for v in block.values()), "Nonfinite rollout"
            reference_digest.update(block["reference_q"].tobytes())
            reference_digest.update(block["intent"].tobytes())
            all_logs.append(block)
            for name in failures:
                failures[name].extend(first[name].cpu().tolist())
            print(f"cohort {cohort + 1}/{args.cohorts} complete", flush=True)
    arrays = {k: np.concatenate([b[k] for b in all_logs], axis=1) for k in all_logs[0]}
    if motion:
        arrays["env_pool"] = np.tile(comp_b.env_pool.cpu().numpy(), args.cohorts)
    np.savez_compressed(out / "rollout.npz", **arrays,
                        reference_first_failure=np.asarray(failures["reference"]),
                        assisted_first_failure=np.asarray(failures["assisted"]),
                        hip_inertia=env.hip_inertia_eff().cpu().numpy())
    report = dict(
        protocol=vars(args), policy_epoch=None if policy is None else getattr(policy, "epochs", policy.epoch),
        reference_sha256=reference_digest.hexdigest(),
        failures={name: sum(x >= 0 for x in times) for name, times in failures.items()},
        total_episodes=args.num_envs * args.cohorts, common_survival_fraction=float(arrays["valid"].mean()),
        injection_error_max_Nm=clip_max, metrics=summarize(arrays, env.dt, args.warmup),
        extended=extended_metrics(arrays, env.dt, args.warmup, group_names, pool_names),
        assist_max_abs_Nm=float(np.abs(arrays["assist_tau"]).max()),
        pools=comp_b.pool_summary if motion else None,
        fallback_clips=int(comp_b._intent.fallback_count) if motion else None)
    if policy is None:
        # 의도는 매 step 같음을 확인했다. 남는 차이는 Newton의 비결정적 GPU 연산이 10 s 동안 쌓인 것
        # (한 env에서 1e-4 rad 남짓, 2026-10-21 실측)이라 1e-3 rad를 기준으로 하고 분포를 남긴다.
        diff = np.abs(arrays["reference_q"] - arrays["assisted_q"])
        report["zero_action_max_q_difference_rad"] = float(diff.max())
        report["zero_action_envs_over_1e-4_rad"] = int((diff.max(axis=(0, 2)) > 1e-4).sum())
        report["zero_action_max_tau_difference_Nm"] = float(np.abs(arrays["reference_tau"] - arrays["assisted_tau"]).max())
        assert report["zero_action_max_q_difference_rad"] < 1e-3
    (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in ("failures", "common_survival_fraction")}, indent=2))


if __name__ == "__main__":
    main()
