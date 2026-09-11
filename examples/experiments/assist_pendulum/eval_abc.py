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
"""A/B/C paired 평가 — assist off / 에뮬 chunk / 실 ManiFlow chunk.

AI-SBC Phase C의 검증 논리: 에뮬레이션 chunk로 학습된 LLP가 **실 HLP chunk**
에서도 성능을 유지하면 노이즈·지연 에뮬레이션 설계가 옳았다는 뜻이다.

| 조건 | 의도 궤적 | chunk | assist |
|------|----------|-------|--------|
| A | mocap | 에뮬 | **차단** (베이스라인 = 사람 단독) |
| B | mocap | 에뮬 | 정책 |
| C | mocap | **실 ManiFlow** | 정책 |

세 조건은 같은 mocap 클립·같은 사람 게인을 받는다 (의도 소스가 전용 난수
generator를 쓰므로 ManiFlow denoise가 전역 RNG를 소비해도 어긋나지 않는다).
조건마다 **별도 프로세스**로 돌린다 — chunk 소스는 env 생성 시점에 정해지고
Newton/Fabric을 한 프로세스에서 재생성하지 않기 위함이다.

실행 (sbc env, PYTHONPATH=<worktree>):

    # SEED 미학습분 (HLP val split)
    python examples/experiments/assist_pendulum/eval_abc.py \
        --checkpoint results/assist_pendulum_v3/last.ckpt \
        --zarr ~/Projects/ManiFlow_Policy/ManiFlow/data/hip-flexion-seed-locomotion-40fps.zarr \
        --split val --num-envs 32 --num-steps 3000 --out results/eval_abc/seed_val

    # ETRI suit14 (SEED 규약으로 영점 정렬해서 사용)
    python examples/experiments/assist_pendulum/eval_abc.py \
        --checkpoint results/assist_pendulum_v3/last.ckpt \
        --zarr ~/Projects/ManiFlow_Policy/ManiFlow/data/hip-flexion-etri-suit14-40fps.zarr \
        --split all --align-ref-zarr <SEED zarr> --out results/eval_abc/suit14
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

import_simulator_before_torch("newton")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from runtime_utils import build_env_agent  # noqa: E402

CONDITIONS = {
    "A": "assist off (사람 단독)",
    "B": "에뮬 chunk",
    "C": "실 ManiFlow chunk",
}


# ---------------------------------------------------------------------------
# 단일 조건 실행
# ---------------------------------------------------------------------------
def condition_overrides(cond: str, args, channel_offset):
    """조건별 config override."""
    a = "env.control_components.assist."
    ov = {
        a + "intent_source": "motion_data",
        a + "motion_zarr_path": args.zarr,
        a + "motion_split": args.split,
        a + "motion_seed": args.motion_seed,
        a + "chunk_source": "maniflow" if cond == "C" else "emulated",
    }
    if args.metadata:
        ov[a + "motion_metadata_parquet"] = args.metadata
    if args.type_filter:
        ov[a + "motion_type_filter"] = args.type_filter
    if channel_offset is not None:
        ov[a + "motion_channel_offset"] = tuple(channel_offset)
    if args.refresh_steps:
        ov[a + "chunk_refresh_steps"] = args.refresh_steps
    if cond == "A":
        ov["env.assist_torque_limit"] = 0.0
    return ov


def run_condition(cond: str, args, channel_offset, out_dir: Path):
    """한 조건을 rollout하고 시계열을 npz로 저장."""
    env, agent = build_env_agent(
        args.checkpoint,
        args.num_envs,
        headless=True,
        overrides=condition_overrides(cond, args, channel_offset),
    )
    comp = env.control_manager.components["assist"]
    cfg = comp.config
    print(
        f"[{cond}] control={type(comp).__name__} intent={type(comp._intent).__name__} "
        f"chunk={type(comp._chunk_src).__name__ if comp._chunk_src else 'emulated'} "
        f"hlp_dt={cfg.hlp_dt} knots={cfg.chunk_knots} refresh={cfg.chunk_refresh_steps} "
        f"assist_limit={env.config.assist_torque_limit}"
    )

    # 게인 랜덤화(전역 RNG)를 시딩 전에 소진 — 조건 간 동일 게인 보장
    env._lazy_init_assist()
    torch.manual_seed(args.seed)
    obs, _ = env.reset()
    td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))

    keys = (
        "dof_pos",
        "dof_vel",
        "theta_g",
        "theta_d",
        "tau_agent",
        "tau_assist",
    )
    logs = {k: [] for k in keys}
    chunk_log, chunk_t0_log, chunk_gt_log, trans_log, time_log = [], [], [], [], []
    num_resets = 0

    for _ in range(args.num_steps):
        with torch.no_grad():
            outs = agent.model(td)
        actions = outs.get("mean_action", outs.get("action"))
        obs, _rew, done, _term, _extras = env.step(actions)

        ctx = env._current_context
        logs["dof_pos"].append(ctx.current.dof_pos.detach().cpu().clone())
        logs["dof_vel"].append(ctx.current.dof_vel.detach().cpu().clone())
        logs["theta_g"].append(ctx.assist.theta_g.detach().cpu().clone())
        logs["theta_d"].append(ctx.assist.theta_d.detach().cpu().clone())
        logs["tau_agent"].append(ctx.assist.tau_agent.detach().cpu().clone())
        logs["tau_assist"].append(ctx.assist.tau_assist.detach().cpu().clone())

        # chunk knot과 그 시각의 "실제" 의도 — 온라인 HLP 오차 계산용
        t_now = comp._episode_time()
        knot_t = comp._chunk_t0.unsqueeze(1) + torch.arange(
            cfg.chunk_knots, device=env.device, dtype=torch.float32
        ).unsqueeze(0)*cfg.hlp_dt
        chunk_log.append(comp._chunk.detach().cpu().clone())
        chunk_t0_log.append(comp._chunk_t0.detach().cpu().clone())
        chunk_gt_log.append(comp._target_at(knot_t).detach().cpu().clone())
        tr = comp.in_transition()
        trans_log.append(
            torch.zeros_like(t_now, dtype=torch.bool).cpu()
            if tr is None
            else tr.detach().cpu().clone()
        )
        time_log.append(t_now.detach().cpu().clone())

        done_ids = done.nonzero(as_tuple=False).flatten()
        if len(done_ids) > 0:
            num_resets += len(done_ids)
            env.reset(done_ids)
            obs = env.get_obs()
        td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))

    out = {k: torch.stack(v).numpy() for k, v in logs.items()}
    out["chunk"] = torch.stack(chunk_log).numpy()
    out["chunk_t0"] = torch.stack(chunk_t0_log).numpy()
    out["chunk_gt"] = torch.stack(chunk_gt_log).numpy()
    out["in_transition"] = torch.stack(trans_log).numpy()
    out["episode_time"] = torch.stack(time_log).numpy()
    out["dt"] = np.float32(env.dt)
    out["hlp_dt"] = np.float32(cfg.hlp_dt)
    out["num_resets"] = np.int64(num_resets)
    path = out_dir / f"{cond}.npz"
    np.savez(path, **out)
    print(f"[{cond}] resets={num_resets} → {path}")


# ---------------------------------------------------------------------------
# 집계
# ---------------------------------------------------------------------------
def summarize(d, mask):
    """조건 요약. mask (T, N) = 유효 스텝(전환·초기 블렌드 제외)."""
    dt = float(d["dt"])
    m = mask[..., None]
    power = np.abs(d["tau_agent"] * d["dof_vel"]).sum(-1)
    err = np.abs(d["dof_pos"] - d["theta_g"]).mean(-1)
    sel = mask
    return {
        "human_power_mean": float(power[sel].mean()),
        "human_work_per_episode": float(power[sel].mean() * dt * power.shape[0]),
        "tracking_err_mean_deg": float(math.degrees(err[sel].mean())),
        "tracking_err_p95_deg": float(math.degrees(np.percentile(err[sel], 95))),
        "assist_torque_rms": float(np.sqrt((d["tau_assist"] ** 2 * m)[sel].mean())),
        "assist_power_mean": float(
            np.abs(d["tau_assist"] * d["dof_vel"]).sum(-1)[sel].mean()
        ),
        "num_resets": int(d["num_resets"]),
        "valid_frac": float(sel.mean()),
    }


def hlp_error_per_lead(d, mask):
    """chunk knot vs 그 시각의 실제 의도 — lead별 MAE [deg]."""
    chunk, gt = d["chunk"], d["chunk_gt"]  # (T, N, K, 2)
    hlp_dt = float(d["hlp_dt"])
    out = []
    for k in range(chunk.shape[2]):
        e = np.abs(chunk[:, :, k] - gt[:, :, k]).mean(-1)  # (T, N)
        out.append(
            {
                "lead_ms": round(k * hlp_dt * 1000),
                "mae_deg": float(math.degrees(e[mask].mean())),
            }
        )
    return out


def valid_mask(d, warmup_s: float):
    """초기 블렌드·클립 전환 구간과 리셋 직후를 제외한 유효 스텝."""
    t = d["episode_time"]
    return (t >= warmup_s) & (~d["in_transition"].astype(bool))


def report(out_dir: Path, conds, args):
    data = {c: dict(np.load(out_dir / f"{c}.npz")) for c in conds}
    masks = {c: valid_mask(d, args.warmup) for c, d in data.items()}
    summaries = {c: summarize(data[c], masks[c]) for c in conds}

    # 페어링 확인: θ_g가 조건 간 동일해야 한다
    pairing = {}
    if len(conds) > 1:
        base = data[conds[0]]["theta_g"]
        for c in conds[1:]:
            pairing[c] = float(np.abs(data[c]["theta_g"] - base).max())

    lines = ["# A/B/C paired 평가", ""]
    lines.append(f"- checkpoint: `{args.checkpoint}`")
    lines.append(f"- zarr: `{args.zarr}` (split={args.split})")
    lines.append(
        f"- envs={args.num_envs} steps={args.num_steps} seed={args.seed} "
        f"motion_seed={args.motion_seed} warmup={args.warmup}s"
    )
    if pairing:
        worst = max(pairing.values())
        lines.append(
            f"- θ_g 페어링: 최대 차 {worst:.2e} rad "
            f"({'동일' if worst < 1e-6 else '⚠️ 어긋남 — 리셋 횟수 확인'})"
        )
    lines += ["", "## 지표", "", "| 지표 | " + " | ".join(
        f"{c} ({CONDITIONS[c]})" for c in conds
    ) + " |", "|---|" + "---|" * len(conds)]
    for key in summaries[conds[0]]:
        row = " | ".join(f"{summaries[c][key]:.4g}" for c in conds)
        lines.append(f"| {key} | {row} |")

    if "A" in conds:
        lines += ["", "## 사람 파워 절감 (A 대비)", ""]
        pa = summaries["A"]["human_power_mean"]
        for c in conds:
            if c == "A":
                continue
            red = 100.0 * (1.0 - summaries[c]["human_power_mean"] / pa)
            lines.append(f"- **{c}** ({CONDITIONS[c]}): {red:.1f} %")

    lines += ["", "## 온라인 HLP 오차 (chunk knot vs 실제 의도)", ""]
    leads = {c: hlp_error_per_lead(data[c], masks[c]) for c in conds if c != "A"}
    if leads:
        cols = [c for c in conds if c != "A"]
        lines.append("| lead | " + " | ".join(cols) + " |")
        lines.append("|---|" + "---|" * len(cols))
        for i in range(len(leads[cols[0]])):
            ms = leads[cols[0]][i]["lead_ms"]
            row = " | ".join(f"{leads[c][i]['mae_deg']:.2f}°" for c in cols)
            lines.append(f"| {ms} ms | {row} |")

    text = "\n".join(lines) + "\n"
    (out_dir / "report.md").write_text(text)
    (out_dir / "report.json").write_text(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "zarr": args.zarr,
                "split": args.split,
                "summaries": summaries,
                "pairing_max_theta_g_diff": pairing,
                "hlp_error_per_lead": leads,
            },
            indent=2,
        )
    )
    print("\n" + text)
    print(f"saved: {out_dir / 'report.md'}")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--zarr", required=True, help="의도 궤적 zarr (flexion_trunk)")
    p.add_argument("--split", default="val", choices=["all", "train", "val"])
    p.add_argument("--metadata", default=None, help="SEED metadata parquet")
    p.add_argument("--type-filter", default=None, help="예: walking,jogging")
    p.add_argument(
        "--align-ref-zarr",
        default=None,
        help="이 zarr(=SEED) 규약으로 영점 정렬 (suit14 평가에 필요)",
    )
    p.add_argument("--conditions", default="A,B,C")
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--num-steps", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--motion-seed", type=int, default=0)
    p.add_argument("--refresh-steps", type=int, default=0, help="0=config 값 유지")
    p.add_argument(
        "--warmup", type=float, default=1.0, help="에피소드 앞부분 제외 [s]"
    )
    p.add_argument("--out", required=True)
    p.add_argument("--_condition", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    channel_offset = None
    if args.align_ref_zarr:
        from protomotions.envs.control.motion_intent import channel_mean_offset

        cache = out_dir / "channel_offset.json"
        if cache.exists():
            channel_offset = json.loads(cache.read_text())
        else:
            channel_offset = list(channel_mean_offset(args.align_ref_zarr, args.zarr))
            cache.write_text(json.dumps(channel_offset))
        print(
            "channel offset (deg): "
            + ", ".join(f"{math.degrees(v):+.2f}" for v in channel_offset)
        )

    if args._condition:
        run_condition(args._condition, args, channel_offset, out_dir)
        return 0

    conds = [c.strip().upper() for c in args.conditions.split(",") if c.strip()]
    for c in conds:
        assert c in CONDITIONS, f"unknown condition {c}"
        print(f"\n===== 조건 {c}: {CONDITIONS[c]} =====")
        cmd = [sys.executable, os.path.abspath(__file__), "--_condition", c]
        for k, v in vars(args).items():
            if k in ("_condition",) or v is None or v is False:
                continue
            cmd += [f"--{k.replace('_', '-')}", str(v)]
        rc = subprocess.call(cmd, env=os.environ.copy())
        if rc != 0:
            raise SystemExit(f"조건 {c} 실패 (rc={rc})")

    report(out_dir, conds, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
