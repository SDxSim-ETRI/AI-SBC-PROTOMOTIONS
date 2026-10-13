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
"""run_system.py 기록(traces)의 지표·그림·리포트.

시간 정렬 규약 (hook 호출 i = LLP step i, 시각 t_i):
  * hip_pos/hip_vel/trunk/ref_hip[i] — t_i의 상태
  * tau_assist[i]                    — 창 [t_i, t_{i+1})에 주입된 assist 토크
  * tau_pd_prev[i] + tau_ff_prev[i]  — 창 [t_{i-1}, t_i)의 착용자 토크 (서브스텝 평균)
따라서 착용자 파워는 τ_agent[i] · ½(ω[i−1] + ω[i]), assist 파워는
τ_assist[i−1] · ½(ω[i−1] + ω[i])로 같은 창에서 계산한다.

지표 (hip flexion 좌/우, warmup 이후):
  * 착용자 토크 RMS, |파워| 평균 → A 대비 B 감소율
  * 추종 오차 (관절각 − 레퍼런스) 평균/p95
  * assist 토크 RMS, 파워, 착용자 토크와의 상관, 총토크 보존 오차
  * HLP 온라인 오차: chunk 갱신 시각의 knot(k·25 ms 앞) vs 실제 관절각
"""

import json
import math
from pathlib import Path

import numpy as np

A, B = 0, 1
SIDES = ("R", "L")


def assist_output_limit(meta):
    """Torque bound after the runner's output scale, in the trace's N·m units."""
    limit = float(meta["torque_limit"])
    if meta.get("oracle") in ("beta", "phase"):
        if meta.get("oracle_limit") is not None:
            limit = float(meta["oracle_limit"])
    else:
        limit = min(limit, float(meta.get("policy_torque_limit", limit)))
    return abs(float(meta.get("assist_scale", 1.0))) * limit


def assist_saturation_pct(torque, meta):
    """Fraction above 95% of the scaled bound; disabled assistance is not saturated."""
    limit = assist_output_limit(meta)
    return 100.0 * float(np.mean(np.abs(torque) > 0.95 * limit)) if limit > 0 else 0.0


def _rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else float("nan")


def _corr(x, y):
    if x.size < 2 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def hlp_lead_errors(tr, env, t_min, hlp_dt, num_knots):
    """chunk 갱신 시각(age==0)의 knot vs 실제 관절각 → lead별 MAE [deg], (K,)."""
    t = tr["t"]
    age = tr["chunk_age"][:, env]
    fresh = np.where((age < 1e-6) & (t >= t_min))[0]
    pos = tr["hip_pos"][:, env]  # (N, 2)
    errs = [[] for _ in range(num_knots)]
    for i in fresh:
        t0 = t[i]
        for k in range(num_knots):
            tk = t0 + k * hlp_dt
            if tk > t[-1]:
                break
            actual = np.stack([np.interp(tk, t, pos[:, d]) for d in range(2)])
            errs[k].append(np.abs(tr["chunk"][i, env, k] - actual))
    return np.array([math.degrees(np.mean(e)) if e else np.nan for e in errs])


def compute_metrics(tr, meta):
    t = tr["t"]
    dt = meta["llp_dt"]
    mask = t >= meta["warmup"]
    idx = np.where(mask)[0]
    idx = idx[idx >= 1]  # 이전 창이 필요

    vel = tr["hip_vel"]  # (N, 2, 2)
    omega_mid = 0.5 * (vel[idx - 1] + vel[idx])  # (n, 2env, 2dof)
    tau_agent = tr["tau_pd_prev"][idx] + tr["tau_ff_prev"][idx]  # 창 [i-1, i)
    tau_assist_prev = tr["tau_assist"][idx - 1]
    err = tr["hip_pos"][idx] - tr["ref_hip"][idx]

    per_env = {}
    for env, name in ((A, "A"), (B, "B")):
        p_agent = np.abs(tau_agent[:, env] * omega_mid[:, env])
        p_assist = np.abs(tau_assist_prev[:, env] * omega_mid[:, env])
        per_env[name] = {
            "tau_agent_rms_Nm": [_rms(tau_agent[:, env, d]) for d in range(2)],
            "tau_agent_rms_Nm_mean": _rms(tau_agent[:, env]),
            "human_power_W": [float(p_agent[:, d].mean()) for d in range(2)],
            "human_power_W_sum": float(p_agent.sum(axis=1).mean()),
            "tracking_err_deg_mean": float(np.degrees(np.abs(err[:, env]).mean())),
            "tracking_err_deg_p95": float(np.degrees(np.percentile(np.abs(err[:, env]), 95))),
            "assist_rms_Nm": [_rms(tau_assist_prev[:, env, d]) for d in range(2)],
            "assist_power_W_sum": float(p_assist.sum(axis=1).mean()),
        }
    a, b = per_env["A"], per_env["B"]
    conservation = tau_agent[:, B] + tau_assist_prev[:, B] - tau_agent[:, A]
    summary = {
        "human_power_W_A": a["human_power_W_sum"],
        "human_power_W_B": b["human_power_W_sum"],
        "human_power_reduction_pct": 100.0 * (1.0 - b["human_power_W_sum"] / max(a["human_power_W_sum"], 1e-9)),
        "tau_agent_rms_A": a["tau_agent_rms_Nm_mean"],
        "tau_agent_rms_B": b["tau_agent_rms_Nm_mean"],
        "tau_agent_rms_reduction_pct": 100.0 * (1.0 - b["tau_agent_rms_Nm_mean"] / max(a["tau_agent_rms_Nm_mean"], 1e-9)),
        "tracking_err_deg_A": a["tracking_err_deg_mean"],
        "tracking_err_deg_B": b["tracking_err_deg_mean"],
        "assist_rms_Nm": _rms(tau_assist_prev[:, B]),
        "assist_power_W": b["assist_power_W_sum"],
        "assist_vs_agentA_corr": float(np.nanmean([
            _corr(tau_assist_prev[:, B, d], tau_agent[:, A, d]) for d in range(2)
        ])),
        "total_torque_conservation_rms_Nm": _rms(conservation),
        # 토크가 얼마나 거칠게 변하는가 (구동기·착용감 관점)
        "assist_slew_rms_Nm_s": float(_rms(np.diff(tau_assist_prev[:, B], axis=0)) / dt),
        "assist_output_limit_Nm": assist_output_limit(meta),
        "assist_saturation_pct": assist_saturation_pct(tau_assist_prev[:, B], meta),
        "assist_sign_flips_per_s": float(
            np.mean(np.sum(np.diff(np.sign(tau_assist_prev[:, B]), axis=0) != 0, axis=0))
            / max(dt * (len(idx) - 1), 1e-9)
        ),
        "survived_s": meta.get("survived_s"),
        "hlp_ms_mean": meta["hlp_ms"]["mean"],
        "llp_ms_mean": meta["llp_ms"]["mean"],
    }
    summary.update(_task_metrics(tr, meta, idx))
    lead = hlp_lead_errors(tr, B, meta["warmup"], meta["hlp_dt"], meta["num_knots"])
    return {
        "summary": summary,
        "per_env": per_env,
        "phase_bins": phase_table(tr, meta),
        "hlp_lead_mae_deg": {f"{int(round(k*meta['hlp_dt']*1e3))}ms": (None if np.isnan(v) else float(v)) for k, v in enumerate(lead)},
        "n_samples": int(len(idx)),
    }


def _task_metrics(tr, meta, idx):
    """보조가 **과제 자체**를 바꿨는지 — 파워 절감이 "덜 걸어서" 온 것인지 가른다.

    파워 |τ·ω|는 다리가 덜 흔들리기만 해도 내려간다. 그래서 같은 거리를
    같은 시간에 갔는지(전진 속도), 착용자 몸체가 레퍼런스에서 얼마나 벗어났는지를
    A/B로 같이 본다. 둘 다 유지되면서 파워가 내려가야 진짜 절감이다.
    """
    out = {}
    if "root_xy" in tr and np.abs(tr["root_xy"]).max() > 0:
        xy = tr["root_xy"]
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=-1)[idx[0] - 1 : idx[-1]]  # (n, 2env)
        span = float(tr["t"][idx[-1]] - tr["t"][idx[0] - 1])
        net = np.linalg.norm(xy[idx[-1]] - xy[idx[0] - 1], axis=-1)  # (2env,)
        for env, name in ((A, "A"), (B, "B")):
            out[f"path_len_m_{name}"] = float(seg[:, env].sum())
            out[f"net_disp_m_{name}"] = float(net[env])
            out[f"walk_speed_mps_{name}"] = float(net[env] / max(span, 1e-9))
        out["walk_speed_change_pct"] = 100.0 * (
            out["walk_speed_mps_B"] / max(out["walk_speed_mps_A"], 1e-9) - 1.0
        )
    if "body_err" in tr:
        be, bt = tr["body_err"], tr["body_err_t"]
        m = bt >= meta["warmup"]
        for env, name in ((A, "A"), (B, "B")):
            out[f"body_err_m_{name}"] = float(be[m, env].mean())
            out[f"body_err_p95_m_{name}"] = float(np.percentile(be[m, env], 95))
        out["body_err_change_pct"] = 100.0 * (
            out["body_err_m_B"] / max(out["body_err_m_A"], 1e-9) - 1.0
        )
    return out


def phase_table(tr, meta, bins=10):
    """보행 위상 10 % 구간별 A/B 파워·토크 (오라클 창이 실제로 뭘 바꿨는지 읽는 표)."""
    if "gait_phase" not in tr or not np.isfinite(tr["gait_phase"]).any():
        return None
    t = tr["t"]
    idx = np.where(t >= meta["warmup"])[0]
    idx = idx[idx >= 1]
    vel = tr["hip_vel"]
    w = 0.5 * (vel[idx - 1] + vel[idx])
    tau = tr["tau_pd_prev"][idx] + tr["tau_ff_prev"][idx]
    assist = tr["tau_assist"][idx - 1]
    ph = tr["gait_phase"][idx]  # (n, 2legs)
    rows = []
    for b in range(bins):
        sel = (ph >= b / bins) & (ph < (b + 1) / bins)  # (n, 2legs)
        if not sel.any():
            continue
        pa = np.abs(tau[:, A] * w[:, A])[sel]
        pb = np.abs(tau[:, B] * w[:, B])[sel]
        rows.append({
            "phase_pct": [100 * b / bins, 100 * (b + 1) / bins],
            "power_A": float(pa.mean()), "power_B": float(pb.mean()),
            "tau_A": float(tau[:, A][sel].mean()), "tau_B": float(tau[:, B][sel].mean()),
            "assist": float(assist[:, B][sel].mean()),
            "omega_A": float(np.abs(w[:, A])[sel].mean()), "omega_B": float(np.abs(w[:, B])[sel].mean()),
        })
    return rows


def save_plots(tr, meta, out_path: Path, window_s=(2.0, 8.0)):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = tr["t"]
    sel = (t >= window_s[0]) & (t <= window_s[1])
    if sel.sum() < 10:
        sel = np.ones_like(t, dtype=bool)
    tt = t[sel]
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
    tau_agent = tr["tau_pd_prev"] + tr["tau_ff_prev"]
    lead_idx = tr["theta_d_future"].shape[2] - 1
    for d, side in enumerate(SIDES):
        ax = axes[0, d]
        ax.plot(tt, np.degrees(tr["ref_hip"][sel, B, d]), "k--", lw=1, label="reference θ_g")
        ax.plot(tt, np.degrees(tr["hip_pos"][sel, A, d]), color="tab:gray", lw=1.2, label="A human only")
        ax.plot(tt, np.degrees(tr["hip_pos"][sel, B, d]), color="tab:blue", lw=1.2, label="B assisted")
        ax.plot(tt, np.degrees(tr["theta_d_future"][sel, B, lead_idx, d]), color="tab:orange", lw=0.9,
                alpha=0.8, label=f"HLP pred (+{int(round(lead_idx*meta['obs_future_dt']*1e3))} ms)")
        ax.set_ylabel("hip flexion [deg]")
        ax.set_title(f"hip_flexion_{side.lower()}")
        if d == 0:
            ax.legend(loc="upper right", fontsize=8)

        ax = axes[1, d]
        ax.plot(tt, tau_agent[sel, A, d], color="tab:gray", lw=1.2, label="τ_agent A")
        ax.plot(tt, tau_agent[sel, B, d], color="tab:blue", lw=1.2, label="τ_agent B")
        ax.plot(tt, tr["tau_assist"][sel, B, d], color="tab:green", lw=1.2, label="τ_assist B")
        ax.set_ylabel("torque [N·m]")
        if d == 0:
            ax.legend(loc="upper right", fontsize=8)

        ax = axes[2, d]
        w = tr["hip_vel"][:, :, d]
        ax.plot(tt, np.abs(tau_agent[sel, A, d] * w[sel, A]), color="tab:gray", lw=1.0, label="|P_agent| A")
        ax.plot(tt, np.abs(tau_agent[sel, B, d] * w[sel, B]), color="tab:blue", lw=1.0, label="|P_agent| B")
        ax.set_ylabel("power [W]")
        if d == 0:
            ax.legend(loc="upper right", fontsize=8)

        ax = axes[3, d]
        ax.plot(tt, np.degrees(tr["hip_pos"][sel, B, d] - tr["ref_hip"][sel, B, d]), color="tab:blue", lw=1.0, label="tracking err B")
        ax.plot(tt, np.degrees(tr["hip_pos"][sel, A, d] - tr["ref_hip"][sel, A, d]), color="tab:gray", lw=1.0, label="tracking err A")
        ax.set_ylabel("error [deg]")
        ax.set_xlabel("time [s]")
        if d == 0:
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"HLP+LLP on skeleton — human={meta['human']}  (A: no assist, B: assist)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def write_report(metrics, meta, out_path: Path):
    s = metrics["summary"]
    lines = [
        f"# HLP+LLP 통합 시스템 — 스켈레톤 착용자 ({meta['human']})",
        "",
        f"- 착용자 대역: `{meta['human']}`" + (f" (골반 고정 +{meta['fixed_base_z']} m, 피드포워드 {meta['feedforward']})" if meta["human"] == "mocap_pd" else f" (tracker `{meta['tracker_checkpoint']}`)"),
        f"- 모션: `{meta['motion_file']}` id={meta['motion_id']} start={meta['start_time']} s, {meta['seconds']} s 실행 (warmup {meta['warmup']} s 제외)",
        f"- LLP: `{meta['llp_checkpoint']}` (epoch {meta['llp_epoch']}), assist 한계 {meta['torque_limit']} N·m, 배율 {meta['assist_scale']}",
        f"- HLP: `{meta['hlp_checkpoint']}` denoise {meta['denoise']}, knots {meta['num_knots']} × {meta['hlp_dt']*1e3:.0f} ms",
        f"- HLP 영점 오프셋 (deg): {None if meta['calibration_deg'] is None else [round(v, 2) for v in meta['calibration_deg']]}",
        f"- 시간: 물리 {meta['physics_fps']} Hz / env {meta['env_dt']*1e3:.0f} ms / LLP {meta['llp_dt']*1e3:.0f} ms",
        f"- 추론 시간 (batch 2): HLP {s['hlp_ms_mean']:.2f} ms, LLP {s['llp_ms_mean']:.2f} ms",
        "",
        "## hip flexion 지표 (A: 보조 없음, B: 보조)",
        "",
        "| 지표 | A | B | 변화 |",
        "|---|---|---|---|",
        f"| 착용자 파워 |τ·ω| 합 [W] | {s['human_power_W_A']:.2f} | {s['human_power_W_B']:.2f} | **{s['human_power_reduction_pct']:+.1f} % 절감** |",
        f"| 착용자 토크 RMS [N·m] | {s['tau_agent_rms_A']:.2f} | {s['tau_agent_rms_B']:.2f} | {s['tau_agent_rms_reduction_pct']:+.1f} % 절감 |",
        f"| 추종 오차 평균 [deg] | {s['tracking_err_deg_A']:.2f} | {s['tracking_err_deg_B']:.2f} | |",
        f"| assist 토크 RMS [N·m] | 0 | {s['assist_rms_Nm']:.2f} | |",
        f"| assist 토크 변화율 RMS [N·m/s] | 0 | {s['assist_slew_rms_Nm_s']:.0f} | 낮을수록 부드럽다 |",
        f"| assist 한계 도달 시간 비율 [%] | 0 | {s['assist_saturation_pct']:.1f} | |",
        f"| assist 부호 전환 [회/s] | 0 | {s['assist_sign_flips_per_s']:.2f} | |",
        f"| assist 파워 [W] | 0 | {s['assist_power_W']:.2f} | |",
        f"| assist ↔ τ_agent(A) 상관 | | {s['assist_vs_agentA_corr']:.2f} | |",
        f"| 총토크 보존 오차 RMS [N·m] | | {s['total_torque_conservation_rms_Nm']:.2f} | |",
        "",
        "## HLP 온라인 오차 (chunk knot vs 실제, B)",
        "",
        "| lead | " + " | ".join(metrics["hlp_lead_mae_deg"].keys()) + " |",
        "|---|" + "---|" * len(metrics["hlp_lead_mae_deg"]),
        "| MAE [deg] | " + " | ".join("-" if v is None else f"{v:.2f}" for v in metrics["hlp_lead_mae_deg"].values()) + " |",
        "",
    ]
    if meta.get("survived_s") is not None:
        lines.append(f"⚠️ 낙상 감지: {meta['survived_s']:.2f} s에서 중단")
    lines += ["", "![plots](plots.png)", ""]
    out_path.write_text("\n".join(lines))


def analyze(traces, meta, out_dir: Path):
    out_dir = Path(out_dir)
    metrics = compute_metrics(traces, meta)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    save_plots(traces, meta, out_dir / "plots.png")
    write_report(metrics, meta, out_dir / "report.md")
    return metrics
