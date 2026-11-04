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
"""`AssistController` ↔ 진자 학습 env 등가성 검증.

독립 제어기가 LLP 학습 env(`AssistEnv` + `MotionAssistTargetControl`,
chunk_source=maniflow)와 **같은 관측을 조립해 같은 action을 내는지** step 단위로
대조한다. 진자 env를 실 HLP + mocap 의도로 돌리면서 매 step의 관절 상태·trunk를
제어기에도 그대로 먹인다.

1부 — LLP 경로 (chunk 주입): env가 만든 chunk를 제어기에 복사해 넣고 action을
       비교한다. HLP의 확률성(denoise 노이즈)이 배제되므로 **부동소수점 수준**
       으로 일치해야 한다. 히스토리·previous action·미래 창 interpolation·
       정규화·tanh 스케일 전부가 검증 대상.
2부 — HLP 입력 경로: 제어기와 env의 HLP 링 버퍼(25 ms 재표집 전 원본)가 매
       step 동일한지 비교 → 같은 관측 창을 모델에 준다는 뜻.
3부 — 전체 경로 (RNG 페어링): env.step 직전과 제어기 observe 직전에 같은
       seed를 걸어 denoise 노이즈까지 맞춘 뒤 chunk knot·action을 비교한다.
       ManiFlow가 전역 RNG를 쓰면 정확히 일치하고, 아니면 통계적 차이만 남는다
       (결과에 어느 쪽인지 기록).

실행 (sbc env):
    python -m protomotions.assist.verify_controller \
        --checkpoint checkpoints/L15_mlp_ppo_pendulum/epoch_6000.ckpt \
        --zarr ~/Projects/ManiFlow_Policy/ManiFlow/data/hip-flexion-seed-locomotion-100fps.zarr
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

import_simulator_before_torch("newton")

import torch  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "examples" / "experiments" / "assist_pendulum"))
from runtime import build_env_agent  # noqa: E402

from protomotions.assist.controller import AssistController, AssistControllerConfig  # noqa: E402

DEFAULT_ZARR = os.path.expanduser(
    "~/Projects/ManiFlow_Policy/ManiFlow/data/hip-flexion-seed-locomotion-100fps.zarr"
)
# 1부/2부 허용오차: 같은 연산을 같은 순서로 하므로 float32 잡음 수준이어야 한다
TOL_ACTION = 1e-4
TOL_RING = 1e-6
TOL_TORQUE = 1e-4
TOL_AGE = 1e-6


def _parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="checkpoints/L15_mlp_ppo_pendulum/epoch_6000.ckpt")
    p.add_argument("--hlp-checkpoint", default=None)
    p.add_argument("--hlp-denoise-steps", type=int, default=1)
    p.add_argument("--zarr", default=DEFAULT_ZARR)
    p.add_argument("--split", default="val")
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--num-steps", type=int, default=600, help="1·2부 step 수")
    p.add_argument("--num-steps-paired", type=int, default=200, help="3부 step 수")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reset-every", type=int, default=137, help="1부에서 부분 reset 검증 주기 (0이면 자연 종료만)")
    p.add_argument("--out", default="results/assist_controller_verify")
    return p


def _overrides(args):
    a = "env.control_components.assist."
    overrides = {
        a + "intent_source": "motion_data",
        a + "motion_zarr_path": args.zarr,
        a + "motion_split": args.split,
        a + "motion_seed": 0,
        a + "chunk_source": "maniflow",
        a + "maniflow_ckpt": _hlp_checkpoint(args),
        a + "maniflow_denoise_steps": args.hlp_denoise_steps,
    }
    return overrides


def _env_state(env, comp):
    """env의 현재 (dof_pos, dof_vel, trunk) — MotionAssistTargetControl과 같은 소스."""
    dof_pos = env.state_history.dof_pos[:, 0]
    dof_vel = env.state_history.dof_vel[:, 0]
    trunk = comp._intent.trunk_at(comp._episode_time())
    return dof_pos, dof_vel, trunk


def _reset_state(env, comp, env_ids):
    dof_pos = env.state_history.dof_pos[env_ids, 0]
    dof_vel = env.state_history.dof_vel[env_ids, 0]
    t0 = torch.zeros(len(env_ids), device=env.device)
    trunk = comp._intent.trunk_at(t0, env_ids)
    return dof_pos, dof_vel, trunk


def _hlp_checkpoint(args) -> str:
    """env의 chunk 소스와 같은 HLP 체크포인트 (ManiFlowChunkSource.from_config 규칙)."""
    if args.hlp_checkpoint:
        return str(Path(args.hlp_checkpoint).expanduser().resolve())
    from protomotions.maniflow.loader import DEFAULT_H1_CHECKPOINT

    return str(DEFAULT_H1_CHECKPOINT)


def _make_controller(env, comp, args, external_chunk: bool) -> AssistController:
    ctrl = AssistController(
        AssistControllerConfig(
            llp_checkpoint=str(Path(args.checkpoint).resolve()),
            hlp_checkpoint=comp.config.maniflow_ckpt,
            hlp_denoise_steps=int(comp.config.maniflow_denoise_steps),
            joint_sign=-1.0,  # hip_pendulum: +θ = 신전
            dt=env.dt,
        ),
        num_envs=env.num_envs,
        device=env.device,
    )
    ctrl.external_chunk = external_chunk
    if ctrl.num_knots != comp.config.chunk_knots or not math.isclose(ctrl.hlp_dt, comp.config.hlp_dt):
        raise ValueError("Environment and controller HLP grids differ")
    # Capture the actual actor inputs, including history ordering, before normalization.
    original_act = ctrl.llp.act

    def record_obs(obs):
        ctrl.verification_obs = {key: value.clone() for key, value in obs.items()}
        return original_act(obs)

    ctrl.llp.act = record_obs
    return ctrl


def _mean_action(agent, obs):
    td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))
    with torch.no_grad():
        actor = agent.model._actor
        out = actor.mu(td)
    return out[actor.config.mu_key]


def _max_diff(left, right):
    diff = (left - right).abs()
    return float(diff.max()) if torch.isfinite(diff).all() else float("inf")


def _obs_diff(ctrl, obs):
    return max(_max_diff(value, obs[key]) for key, value in ctrl.verification_obs.items())


def run_injected(env, agent, comp, args):
    """1부 + 2부: chunk 주입 상태에서 action·링 버퍼 비교."""
    ctrl = _make_controller(env, comp, args, external_chunk=True)
    env._lazy_init_assist()
    torch.manual_seed(args.seed)
    obs, _ = env.reset()
    ids = torch.arange(env.num_envs, device=env.device)
    dof_pos, dof_vel, trunk = _reset_state(env, comp, ids)
    ctrl.reset(ids, dof_pos, dof_vel, trunk)
    ctrl.set_chunk(comp._chunk.clone(), comp._chunk_t0.clone())

    max_act, max_tau, max_ring, max_age = 0.0, 0.0, 0.0, 0.0
    a_env = _mean_action(agent, obs)
    tau_ctrl = ctrl.act()
    max_obs = _obs_diff(ctrl, obs)
    max_act = _max_diff(ctrl.last_raw_action, a_env)
    num_resets = 0
    forced_resets = 0
    for step in range(args.num_steps):
        obs, _rew, done, _term, _extras = env.step(a_env)
        # Compare with the torque actually applied by AssistEnv, including its
        # action transform and clamp, rather than reimplementing that transform.
        max_tau = max(max_tau, _max_diff(tau_ctrl, comp._tau_assist))
        dof_pos, dof_vel, trunk = _env_state(env, comp)
        ctrl.observe(dof_pos, dof_vel, trunk)
        # 2부: 링 버퍼 동일성 (재표집 전 원본)
        max_ring = max(max_ring, _max_diff(comp._chunk_src._ring, ctrl.hlp._ring))
        # 1부: env chunk 주입 후 action 비교
        ctrl.set_chunk(comp._chunk.clone(), comp._chunk_t0.clone())
        a_env = _mean_action(agent, obs)
        tau_ctrl = ctrl.act()
        max_obs = max(max_obs, _obs_diff(ctrl, obs))
        max_act = max(max_act, _max_diff(ctrl.last_raw_action, a_env))
        # 관측 보조값: chunk age
        age_env = comp._episode_time() - comp._chunk_t0
        max_age = max(max_age, _max_diff(age_env, ctrl.chunk_age))

        done_ids = done.nonzero(as_tuple=False).flatten()
        if args.reset_every > 0 and (step + 1) % args.reset_every == 0:
            forced_ids = torch.tensor([(step // args.reset_every) % env.num_envs], device=env.device)
            done_ids = torch.unique(torch.cat((done_ids, forced_ids)))
            forced_resets += len(forced_ids)
        if len(done_ids) > 0:
            num_resets += len(done_ids)
            env.reset(done_ids)
            obs = env.get_obs()
            dof_pos, dof_vel, trunk = _reset_state(env, comp, done_ids)
            ctrl.reset(done_ids, dof_pos, dof_vel, trunk)
            ctrl.set_chunk(comp._chunk[done_ids].clone(), comp._chunk_t0[done_ids].clone(), done_ids)
            a_env = _mean_action(agent, obs)
            tau_ctrl = ctrl.act()
            max_obs = max(max_obs, _obs_diff(ctrl, obs))
            max_act = max(max_act, _max_diff(ctrl.last_raw_action, a_env))
    return {
        "steps": args.num_steps,
        "resets": num_resets,
        "forced_resets": forced_resets,
        "max_abs_obs_diff": max_obs,
        "max_abs_raw_action_diff": max_act,
        "max_abs_torque_diff_Nm": max_tau,
        "max_abs_chunk_age_diff_s": max_age,
        "max_abs_ring_diff_rad": max_ring,
        "pass_action": max_act < TOL_ACTION,
        "pass_ring": max_ring < TOL_RING,
        "pass_obs": max_obs < TOL_RING,
        "pass_torque": max_tau < TOL_TORQUE,
        "pass_age": max_age < TOL_AGE,
    }


def run_paired(env, agent, comp, args):
    """3부: RNG 페어링으로 HLP 예측까지 포함한 전체 경로 비교."""
    ctrl = _make_controller(env, comp, args, external_chunk=False)
    env._lazy_init_assist()
    torch.manual_seed(args.seed)
    obs, _ = env.reset()
    ids = torch.arange(env.num_envs, device=env.device)
    dof_pos, dof_vel, trunk = _reset_state(env, comp, ids)
    # reset 시 env는 이미 예측을 1회 했다 — 같은 seed로 재현되지 않으므로 chunk만 맞춘다
    ctrl.external_chunk = True
    ctrl.reset(ids, dof_pos, dof_vel, trunk)
    ctrl.set_chunk(comp._chunk.clone(), comp._chunk_t0.clone())
    ctrl.external_chunk = False
    a_env = _mean_action(agent, obs)
    ctrl.act()

    knot_diffs, act_diffs = [], []
    n_refresh = 0
    for step in range(args.num_steps_paired):
        seed = args.seed * 1000 + step
        torch.manual_seed(seed)
        obs, _rew, done, _term, _extras = env.step(a_env)
        if done.any():
            break  # 리셋이 끼면 페어링이 깨진다 — 짧게 돌리므로 여기서 중단
        dof_pos, dof_vel, trunk = _env_state(env, comp)
        torch.manual_seed(seed)
        ctrl.observe(dof_pos, dof_vel, trunk)
        refreshed = (env.progress_buf % comp.config.chunk_refresh_steps == 0).any().item()
        if refreshed:
            n_refresh += 1
            knot_diffs.append(_max_diff(comp._chunk, ctrl._chunk))
        a_env = _mean_action(agent, obs)
        ctrl.act()
        act_diffs.append(_max_diff(ctrl.last_raw_action, a_env))
    knot_max = max(knot_diffs) if knot_diffs else float("nan")
    act_max = max(act_diffs) if act_diffs else float("nan")
    return {
        "steps": len(act_diffs),
        "chunk_refreshes": n_refresh,
        "max_abs_knot_diff_deg": math.degrees(knot_max) if knot_diffs else None,
        "max_abs_raw_action_diff": act_max,
        "hlp_rng_paired_exact": bool(knot_diffs) and knot_max < 1e-5,
        "pass_action": act_max < TOL_ACTION,
        "completed": len(act_diffs) == args.num_steps_paired,
    }


def main():
    args = _parser().parse_args()
    if min(args.num_envs, args.num_steps, args.num_steps_paired, args.hlp_denoise_steps) <= 0 or args.reset_every < 0:
        raise ValueError("Environment/step counts must be positive and reset interval nonnegative")

    env, agent = build_env_agent(args.checkpoint, args.num_envs, headless=True, overrides=_overrides(args))
    comp = env.control_manager.components["assist"]
    assert comp._chunk_src is not None, "env must run with chunk_source=maniflow"

    print(
        f"env: dt={env.dt} hlp_dt={comp.config.hlp_dt} knots={comp.config.chunk_knots} "
        f"refresh={comp.config.chunk_refresh_steps} limit={env.config.assist_torque_limit}"
    )
    injected = run_injected(env, agent, comp, args)
    print("[1·2부] chunk 주입 비교:", json.dumps(injected, indent=2))
    paired = run_paired(env, agent, comp, args)
    print("[3부] RNG 페어링 비교:", json.dumps(paired, indent=2))

    verdict = all(injected[key] for key in ("pass_action", "pass_ring", "pass_obs", "pass_torque", "pass_age"))
    verdict = verdict and paired["completed"] and paired["hlp_rng_paired_exact"] and paired["pass_action"]
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "hlp_checkpoint": comp.config.maniflow_ckpt,
        "zarr": args.zarr,
        "split": args.split,
        "num_envs": args.num_envs,
        "injected_chunk": injected,
        "rng_paired": paired,
        "PASS": verdict,
    }
    out_dir = Path(args.out) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\n{'PASS' if verdict else 'FAIL'} — saved {out_dir / 'report.json'}")
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
