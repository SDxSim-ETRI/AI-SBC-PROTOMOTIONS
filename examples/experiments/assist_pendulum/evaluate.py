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
"""진자 짝 평가 — 무보조 진자와 보조 진자를 같은 의도·플랜트·외란으로 나란히 돌린다.

`PairedPendulumEvaluation` (`paired`)
  조건
  * legacy / phase / hold50: 합성 궤적 프로토콜·seed (2026-10-20 진자 평가와 같음). phase = 구조화 힘장·
    유각 감속 외란 50 %, hold50 = 착용자 명령 50 ms 유지 (`DisturbedAssistTargetControlConfig`의 스위치).
  * seed_val: SEED 고관절 동작 중 H1·LLP 모두 학습하지 않은 클립 (H1 validation 분할). 학습과 같은
    두 플랜트 집단(가벼운 다리 50 %: 달리기·조깅·점프 / 일반 50 %)과 같은 실현 가능성 규칙.
  * etri: ETRI 슈트 13동작 평가용 사본 (`prepare-etri`), 같은 두 집단에 동작별 같은 시간.
  모든 조건에서 보조 진자의 미래 목표 관측은 동결 H1의 실제 예측이다 (`H1Forecast`, 플랜트와 분리된
  난수 스트림; 동작 조건은 데이터의 몸통 기울기도 H1에 준다). `--checkpoint`를 생략하면 무보조 짝 동일성 점검이다.
`EtriEvalData` (`prepare-etri`): ETRI 슈트 14동작 → 평가용 사본 (SEED 영점 정렬, 고관절 ±86° 클램프,
  복제본 제거) + 동작 이름을 종류로 쓰는 metadata parquet.

    python examples/experiments/assist_pendulum/evaluate.py paired \\
        --base-checkpoint checkpoints/L15_mlp_ppo_pendulum/epoch_6000.ckpt --checkpoint <ckpt> \\
        --condition seed_val --hlp-checkpoint checkpoints/H1_maniflow_swap/epoch_0008.ckpt --out <dir>
    python examples/experiments/assist_pendulum/evaluate.py prepare-etri --out <dir>
"""

import argparse
from dataclasses import fields
import hashlib
import json
import random
import sys
from pathlib import Path

from protomotions.utils.simulator_imports import import_simulator_before_torch

import_simulator_before_torch("newton")
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlp import MOTION_GROUPS, SEED_METADATA, SEED_ZARR  # noqa: E402
from runtime import build_env_agent  # noqa: E402

from protomotions.assist.llp_policy import LLPPolicy  # noqa: E402
from protomotions.envs.control.assist_disturbance_control import (  # noqa: E402
    DisturbedAssistTargetControlConfig,
)
from protomotions.envs.obs.assist_obs import compute_assist_target_obs  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
SYNTHETIC = ("legacy", "phase", "hold50")
MOTION = ("seed_val", "etri")
ETRI_EVAL_DIR = ROOT / "results/llp20_motion_20261021/eval_data"  # prepare-etri 기본 출력 = paired 기본 입력
ETRI_GROUPS = (
    ("run", r"^run$"), ("walk", r"^walk$"), ("squat", r"^squat$"), ("lunge", r"^lunge"),
    ("highknee", r"^highknee$"), ("stepinplace", r"^stepinplace"), ("onestep", r"^onestep"),
    ("hop", r"^onehopforward$"),
)
LIGHT = (1.0, 3.9)  # hip 축 0.42–0.80 kg·m² (학습의 가벼운 다리 집단)
GENERAL = (1.0, 25.0)
NEED_EDGES = (0, 5, 10, 20, 40, 80, np.inf)
PAIRED_ENV = "protomotions.envs.base_env.paired_assist_env.PairedAssistEnv"


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class H1Forecast:
    """동결 H1 예측 — 플랜트와 분리된 전용 RNG 스트림.

    에뮬레이션 control 컴포넌트는 두 짝 플랜트에서 그대로 돈다. 보조 actor의 목표 관측만 바꾸므로
    H1이 합성 의도나 외란 난수열을 바꿀 수 없다. 동작 조건은 데이터의 몸통 pitch/roll도 입력한다.
    """

    def __init__(self, checkpoint, env):
        from protomotions.maniflow.angle_estimator import ManiFlowAngleEstimator
        from protomotions.maniflow.chunk_source import ManiFlowChunkSource

        self.device = torch.device(env.device)
        estimator = ManiFlowAngleEstimator.from_checkpoint(
            checkpoint, num_envs=env.num_envs, device=self.device, denoise_steps=1,
        )
        self.source = ManiFlowChunkSource(estimator, env.num_envs, env.dt, self.device, flexion_sign=-1.)
        assert abs(self.source.knot_dt - env.dt) < 1e-6
        assert self.source.num_knots >= 3
        self.env_ids = torch.arange(env.num_envs, device=self.device)
        cfg = env.assist_component.config
        assert cfg.obs_future_samples == 3 and abs(cfg.obs_future_dt - env.dt) < 1e-6
        assert cfg.chunk_refresh_steps == 1
        kin = env.robot_config.kinematic_info
        self.low = kin.dof_limits_lower.to(self.device) * cfg.dof_limit_margin
        self.high = kin.dof_limits_upper.to(self.device) * cfg.dof_limit_margin

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
            knots, _ = self.source.predict_knots(self.env_ids, torch.full((len(q),), time_s, device=self.device))
            self.cpu_rng = torch.get_rng_state()
            self.cuda_rng = torch.cuda.get_rng_state(self.device)
        return knots[:, :3].clamp(self.low, self.high)


# ---------------------------------------------------------------------------
# 요약 통계 (rollout.npz → summary.json)
# ---------------------------------------------------------------------------
def paired_summary(arrays, dt, warmup):
    valid = arrays["valid"] & (arrays["time_s"] >= warmup)
    # 마스크는 무보조 쪽만 참조한다 (action 의존 구간 분류를 피한다).
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
        am, bm = float(np.mean(a * a)), float(np.mean(b * b))
        result[name] = dict(
            joint_samples=count, exposure_joint_s=count * dt,
            reference_tau_sq_Nm2=am, assisted_tau_sq_Nm2=bm,
            torque_sq_change_percent=100 * (bm / am - 1),
            reference_abs_work_W=float(np.mean(np.abs(a * arrays["reference_velocity"][mask]))),
            assisted_abs_work_W=float(np.mean(np.abs(b * arrays["assisted_velocity"][mask]))),
            reference_tracking_rmse_deg=float(np.sqrt(np.mean(ea * ea)) * 180 / np.pi),
            assisted_tracking_rmse_deg=float(np.sqrt(np.mean(eb * eb)) * 180 / np.pi),
            assist_rms_Nm=float(np.sqrt(np.mean(assist * assist))),
            assist_above_19Nm_fraction=float(np.mean(np.abs(assist) > 19)))
    return result


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


# ---------------------------------------------------------------------------
# 짝 평가
# ---------------------------------------------------------------------------
class PairedPendulumEvaluation:
    """무보조 진자 A와 보조 진자 B를 같은 의도·플랜트·외란으로 나란히 돌리고 요약한다."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.motion = args.condition not in SYNTHETIC
        self.out = Path(args.out)
        self.etri_dir = Path(args.etri_dir)

    def motion_setup(self):
        """동작 조건의 데이터·집단 설정."""
        if self.args.condition == "seed_val":
            pools = (("light", .5, "motion", *LIGHT, (("running", .5), ("jogging", .3), ("jumping", .2))),
                     ("general", .5, "motion", *GENERAL,
                      (("walking", .3), ("jogging", .2), ("jumping", .15), ("deep", .25), ("other", .1))))
            return dict(motion_zarr_path=SEED_ZARR, motion_split="val", motion_metadata_parquet=SEED_METADATA,
                        motion_groups=MOTION_GROUPS, intent_pools=pools)
        equal = tuple((name, 1.0 / len(ETRI_GROUPS)) for name, _ in ETRI_GROUPS)
        pools = (("light", .5, "motion", *LIGHT, equal), ("general", .5, "motion", *GENERAL, equal))
        return dict(motion_zarr_path=str(self.etri_dir / "etri_suit13_seedzero_clamp86_100fps.zarr"), motion_split="all",
                    motion_metadata_parquet=str(self.etri_dir / "etri_suit13_metadata.parquet"),
                    motion_groups=ETRI_GROUPS, intent_pools=pools)

    def control_config(self, old) -> DisturbedAssistTargetControlConfig:
        """기준 체크포인트의 control config를 조건에 맞게 바꾼다 (plain 필드만 복사)."""
        args = self.args
        names = {f.name for f in fields(DisturbedAssistTargetControlConfig)}
        cfg = DisturbedAssistTargetControlConfig(**{
            f.name: getattr(old, f.name) for f in fields(old) if f.name != "_target_" and f.name in names})
        if self.motion:
            for key, value in self.motion_setup().items():
                setattr(cfg, key, value)
            cfg.intent_source = "motion_data"
            cfg.motion_accel_frac = 0.8
            cfg.motion_accel_check_dofs = ()
            cfg.motion_seed = args.trajectory_seed
            cfg.intent_pool_seed = args.plant_seed
        else:
            cfg.structured_probability = .5 if args.condition == "phase" else 0.
            cfg.phase_probability = .5 if args.condition == "phase" else 0.
            cfg.human_target_hold_steps = 5 if args.condition == "hold50" else 1
        return cfg

    def build_env(self):
        args = self.args
        resolved = torch.load(Path(args.base_checkpoint).parent / "resolved_configs_inference.pt",
                              map_location="cpu", weights_only=False)
        cfg = self.control_config(resolved["env"].control_components["assist"])
        seed_all(args.plant_seed)

        def configure_evaluation(resolved):
            resolved["env"].control_components["assist"] = cfg

        env, _ = build_env_agent(args.base_checkpoint, args.num_envs, overrides={
            "env._target_": PAIRED_ENV,
            "env.assist_torque_limit": args.evaluation_limit,
            "env.action_config.torque_scale": args.evaluation_limit,
            "env.clamp_combined_injection": False,
            "env.human_feedforward_inside_pd": True,
        }, config_transform=configure_evaluation)
        env._lazy_init_assist()
        env.max_episode_length = args.steps + 2
        self.cfg = cfg
        return env

    def policy(self, env):
        args = self.args
        policy = LLPPolicy(args.checkpoint, env.device) if args.checkpoint else None
        if policy:
            lay = policy.layout
            assert lay.action_transform == "tanh"
            assert abs(lay.torque_scale * args.output_scale - args.evaluation_limit) < 1e-6, "Inconsistent torque mapping"
            assert lay.torque_limit >= lay.torque_scale
            assert lay.history_steps == env.config.num_state_history_steps
        return policy

    def run(self):
        args = self.args
        self.out.mkdir(parents=True, exist_ok=True)
        env = self.build_env()
        ref = env.reference_env
        policy = self.policy(env)
        hlp = H1Forecast(args.hlp_checkpoint, env)
        comp_a, comp_b = ref.assist_component, env.assist_component
        motion, cfg = self.motion, self.cfg
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
        np.savez_compressed(self.out / "rollout.npz", **arrays,
                            reference_first_failure=np.asarray(failures["reference"]),
                            assisted_first_failure=np.asarray(failures["assisted"]),
                            hip_inertia=env.hip_inertia_eff().cpu().numpy())
        return self.report(env, policy, arrays, failures, clip_max, reference_digest, group_names, pool_names, comp_b)

    def report(self, env, policy, arrays, failures, clip_max, reference_digest, group_names, pool_names, comp_b):
        args, motion = self.args, self.motion
        report = dict(
            protocol=vars(args), policy_epoch=None if policy is None else policy.epoch,
            reference_sha256=reference_digest.hexdigest(),
            failures={name: sum(x >= 0 for x in times) for name, times in failures.items()},
            total_episodes=args.num_envs * args.cohorts, common_survival_fraction=float(arrays["valid"].mean()),
            injection_error_max_Nm=clip_max, metrics=paired_summary(arrays, env.dt, args.warmup),
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
        (self.out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False, ensure_ascii=False))
        print(json.dumps({k: report[k] for k in ("failures", "common_survival_fraction")}, indent=2))
        return report


class EtriEvalData:
    """ETRI 슈트 14동작 → 진자 평가용 사본 (학습에 쓰지 않는다; W1 평가 걷기·달리기와 같은 출처)."""

    DATA = Path("/home/bak/Projects/ManiFlow_Policy/ManiFlow/data")
    CLAMP_RAD = float(np.radians(86.0))
    DUPLICATES = {"walk_koo.npz"}  # 같은 기록의 복제본(walk_koo = walk)

    def __init__(self, etri, seed_ref, out):
        self.etri, self.seed_ref, self.out = etri, seed_ref, Path(out)

    def run(self):
        import pandas as pd
        import zarr

        from protomotions.envs.control.motion_intent import channel_mean_offset

        out = self.out
        out.mkdir(parents=True, exist_ok=True)
        # 4채널을 SEED 영점으로 맞추고 (HLP 입력 규약), 고관절 2채널은 진자 관절 범위 ±86° 안으로 자른다.
        # 한계를 넘는 클립(높이 들기·런지 등)을 통째로 빼지 않기 위해서이며 잘린 비율을 동작별로 기록한다.
        offset = np.asarray(channel_mean_offset(self.seed_ref, self.etri), dtype=np.float32)
        src = zarr.open(self.etri, mode="r")
        names = list(src.attrs["motion_names"])
        ends = np.asarray(src["meta/episode_ends"][:], dtype=np.int64)
        starts = np.concatenate([[0], ends[:-1]])
        state = np.asarray(src["data/state"][:], dtype=np.float32) + offset[None, :]
        clips, kept, info = [], [], {}
        for name, a, b in zip(names, starts, ends):
            if name in self.DUPLICATES:
                continue
            x = state[a:b].copy()
            clipped = np.abs(x[:, :2]) > self.CLAMP_RAD
            x[:, :2] = np.clip(x[:, :2], -self.CLAMP_RAD, self.CLAMP_RAD)
            clips.append(x)
            kept.append(name)
            info[name] = dict(seconds=float((b - a) / 100.0),
                              clamped_frame_fraction=float(clipped.any(axis=1).mean()),
                              peak_flexion_deg=float(np.degrees(np.abs(state[a:b, :2]).max())))
        zarr.config.set({"default_zarr_format": 2})
        path = out / "etri_suit13_seedzero_clamp86_100fps.zarr"
        root = zarr.open(str(path), mode="w")
        data = np.concatenate(clips).astype(np.float32)
        new_ends = np.cumsum([len(c) for c in clips]).astype(np.int64)
        root.create_group("data").create_array("state", shape=data.shape, dtype="float32")[:] = data
        root.create_group("meta").create_array("episode_ends", shape=new_ends.shape, dtype="int64")[:] = new_ends
        for key in ("features", "obs_spec", "action_dof_names", "convention", "embodiment"):
            if key in src.attrs:
                root.attrs[key] = src.attrs[key]
        root.attrs["fps"] = 100.0
        root.attrs["motion_names"] = kept
        root.attrs["note"] = "ETRI suit 13 motions, SEED neutral offset applied, hip channels clipped to ±86 deg; evaluation only"
        stems = [Path(n).stem for n in kept]
        pd.DataFrame({"filename": stems, "content_type_of_movement": stems}).to_parquet(out / "etri_suit13_metadata.parquet")
        report = dict(source=self.etri, offset_rad=offset.tolist(), clamp_deg=86.0, motions=info,
                      zarr=str(path.relative_to(ROOT)), metadata=str((out / "etri_suit13_metadata.parquet").relative_to(ROOT)))
        (out / "etri_eval_data.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("paired", help="무보조/보조 진자 짝 평가 (동결 H1 예측)")
    s.add_argument("--base-checkpoint", required=True, help="플랜트·관측 config의 출처 (예: L15 epoch_6000.ckpt)")
    s.add_argument("--checkpoint", default=None, help="생략하면 무보조 짝 동일성 점검")
    s.add_argument("--output-scale", type=float, default=1.)
    s.add_argument("--evaluation-limit", type=float, default=20.)
    s.add_argument("--condition", choices=[*SYNTHETIC, *MOTION], required=True)
    s.add_argument("--num-envs", type=int, default=128)
    s.add_argument("--steps", type=int, default=1000)
    s.add_argument("--cohorts", type=int, default=2)
    s.add_argument("--plant-seed", type=int, default=63031)
    s.add_argument("--trajectory-seed", type=int, default=52021)
    s.add_argument("--warmup", type=float, default=.5)
    s.add_argument("--hlp-checkpoint", required=True)
    s.add_argument("--hlp-seed", type=int, default=991)
    s.add_argument("--etri-dir", default=str(ETRI_EVAL_DIR), help="prepare-etri 출력 폴더 (etri 조건)")
    s.add_argument("--out", required=True)

    s = sub.add_parser("prepare-etri", help="ETRI 슈트 동작의 진자 평가용 사본 만들기")
    s.add_argument("--etri", default=str(EtriEvalData.DATA / "hip-flexion-etri-suit14-100fps.zarr"))
    s.add_argument("--seed-ref", default=str(EtriEvalData.DATA / "hip-flexion-seed-locomotion-100fps.zarr"))
    s.add_argument("--out", default=str(ETRI_EVAL_DIR))

    args = p.parse_args(argv)
    if args.command == "paired":
        PairedPendulumEvaluation(args).run()
    else:
        EtriEvalData(args.etri, args.seed_ref, args.out).run()


if __name__ == "__main__":
    main()
