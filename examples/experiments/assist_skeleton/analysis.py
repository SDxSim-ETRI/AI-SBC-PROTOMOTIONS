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
"""run_system.py 기록(traces.npz + meta.json)의 지표 계산 — numpy만 쓰는 순수 분석 모음.

시간 정렬 규약 (hook 호출 i = LLP step i, 시각 t_i):
  * hip_pos/hip_vel/trunk/ref_hip[i] — t_i의 상태
  * tau_assist[i]                    — 창 [t_i, t_{i+1})에 주입된 assist 토크
  * tau_pd_prev[i] + tau_ff_prev[i]  — 창 [t_{i-1}, t_i)의 착용자 토크 (서브스텝 평균)
따라서 착용자 파워는 τ_agent[i] · ½(ω[i−1] + ω[i]), assist 파워는 τ_assist[i−1] · ½(ω[i−1] + ω[i])로
같은 창에서 계산한다. 보행 위상 0 %는 참조 mocap의 그 다리 최대 굴곡이며 발 착지 검출이 아니다.

모듈 함수 `analyze/compute_metrics/...`는 run_system이 실행 직후 쓰는 기본 지표·그림·report.md다.
그 위에 기록을 다시 읽어 비교하는 계산을 class로 모았다:
  `RunRecord`             기록 하나 (traces, meta, warmup 이후 색인)
  `ObjectiveComparison`   목적함수를 바꿔 가며 A/B 재측정 (τ², |τ|, 일, 속도, 몸체 오차, 위상별 τ²) — compare_runs
  `WearerStats`           착용자(tracker)가 얼마나 사람 같은가 (스트라이드 피크, 피크/RMS) — eval_wearer
  `TorqueAlignment`       같은 이전 창에 정렬한 토크와 위상 프로파일(RMS 우선) — report.py finetune
2026-11-04 L1 계보 정리로 L1 기준 비교·감속 구간·유각 게이트 분석 class는 각 연구의 `results/<연구>/scripts/`로 옮겼다.
"""

import glob
import json
import math
from pathlib import Path

import numpy as np

A, B = 0, 1
SIDES = ("R", "L")
ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# 공통 helper
# ---------------------------------------------------------------------------
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


def stride_boundaries(phase):
    """위상 열(0→1 톱니)의 되감김 위치 = 스트라이드 시작 색인."""
    return np.flatnonzero(np.diff(phase) < -0.5) + 1


def circular_phase_midpoint(before, after):
    """두 이웃 창의 위상 중점 — 99 % → 1 %의 되감김을 50 %로 만들지 않는다."""
    return (before + .5 * ((after - before + .5) % 1 - .5)) % 1


def latest_run(pattern):
    """glob 패턴에 맞는 실행 폴더 중 이름순 마지막 (타임스탬프 폴더 = 최신)."""
    hits = sorted(glob.glob(str(pattern)))
    assert hits, f"no run for {pattern}"
    return Path(hits[-1])


def read_json(path):
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------
# run_system 직후의 기본 지표·그림·report.md
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 기록 하나
# ---------------------------------------------------------------------------
class RunRecord:
    """run_system.py 출력 폴더 하나: traces.npz + meta.json."""

    def __init__(self, run_dir):
        self.run = Path(run_dir)
        self.traces = dict(np.load(self.run / "traces.npz"))
        self.meta = json.loads((self.run / "meta.json").read_text())

    def index(self, min_index=1, offset=0.0):
        """warmup(+offset) 이후이고 이전 창이 있는 색인. 분석마다 다른 세 규약을 매개변수로 둔다."""
        t = self.traces["t"]
        idx = np.flatnonzero(t >= self.meta["warmup"] + offset)
        return idx[idx >= min_index]

    @classmethod
    def load(cls, run):
        """(traces, meta, idx) — warmup 이후·이전 창 필요 규약."""
        r = cls(run)
        return r.traces, r.meta, r.index()


# ---------------------------------------------------------------------------
# 목적함수를 바꿔 가며 A/B 재측정
# ---------------------------------------------------------------------------
class ObjectiveComparison:
    """파워 |τ·ω|는 다리가 덜 흔들리기만 해도 내려간다. 같은 기록에서 τ²(근활성 프록시)·|τ|·양/음의 일·
    전진 속도·몸체 추종 오차를 함께 뽑아 "장치가 부하를 받았는가"와 "동작이 바뀌었는가"를 갈라 본다.
    `metrics`의 유니코드 키("τ²", "|ω|", ...)는 comparison.json에 저장된 계약이라 바꾸지 않는다."""

    @staticmethod
    def metrics(run: Path):
        tr, meta, idx = RunRecord.load(run)
        w = 0.5 * (tr["hip_vel"][idx - 1] + tr["hip_vel"][idx])
        tau = (tr["tau_pd_prev"] + tr["tau_ff_prev"])[idx]
        assist = tr["tau_assist"][idx - 1]
        out = {"run": Path(run).name.split("_tracker_")[-1]}
        obj = {
            "P|τω|": lambda e: np.abs(tau[:, e] * w[:, e]).sum(1),
            "τ²": lambda e: (tau[:, e] ** 2).sum(1),
            "|τ|": lambda e: np.abs(tau[:, e]).sum(1),
            "W+": lambda e: np.clip(tau[:, e] * w[:, e], 0, None).sum(1),
            "W−": lambda e: np.clip(tau[:, e] * w[:, e], None, 0).sum(1),
            "|ω|": lambda e: np.abs(w[:, e]).sum(1),
        }
        for name, fn in obj.items():
            a, b = float(fn(A).mean()), float(fn(B).mean())
            out[name] = 100.0 * (b / a - 1.0) if abs(a) > 1e-9 else float("nan")
        out["P_A"] = float(np.abs(tau[:, A] * w[:, A]).sum(1).mean())
        out["P_B"] = float(np.abs(tau[:, B] * w[:, B]).sum(1).mean())
        out["assist_rms"] = float(np.sqrt((assist[:, B] ** 2).mean()))
        if "root_xy" in tr and np.abs(tr["root_xy"]).max() > 0:
            xy = tr["root_xy"]
            net = np.linalg.norm(xy[idx[-1]] - xy[idx[0] - 1], axis=-1)
            out["speed"] = 100.0 * (net[B] / max(net[A], 1e-9) - 1.0)
            out["dist_A"] = float(net[A])
        if "body_err" in tr:
            be, bt = tr["body_err"], tr["body_err_t"]
            m = bt >= meta["warmup"]
            out["body_A"] = float(be[m, A].mean())
            out["body_B"] = float(be[m, B].mean())
            out["body"] = 100.0 * (out["body_B"] / max(out["body_A"], 1e-9) - 1.0)
        out["fell"] = meta.get("survived_s")
        return out

    @staticmethod
    def phase_tau2(run: Path, bins=10):
        """위상 구간별 τ² 변화 [%] — 어디서 벌고 어디서 잃는지."""
        tr, meta, idx = RunRecord.load(run)
        if "gait_phase" not in tr or not np.isfinite(tr["gait_phase"]).any():
            return None
        tau = (tr["tau_pd_prev"] + tr["tau_ff_prev"])[idx]
        ph = tr["gait_phase"][idx]
        out = []
        for b in range(bins):
            sel = (ph >= b / bins) & (ph < (b + 1) / bins)
            a = (tau[:, A][sel] ** 2).mean()
            out.append(100.0 * ((tau[:, B][sel] ** 2).mean() / max(a, 1e-9) - 1.0))
        return np.array(out)

    @staticmethod
    def html_metrics(run: Path):
        """HTML 비교표용 (ASCII 키; 상관·스윙 진폭 포함) — report_html.metrics."""
        tr, meta, idx = RunRecord.load(run)
        w = 0.5 * (tr["hip_vel"][idx - 1] + tr["hip_vel"][idx])
        tau = (tr["tau_pd_prev"] + tr["tau_ff_prev"])[idx]
        assist = tr["tau_assist"][idx - 1]
        pct = lambda f: 100.0 * (float(f(B).mean()) / max(float(f(A).mean()), 1e-9) - 1.0)  # noqa: E731
        out = {
            "P": pct(lambda e: np.abs(tau[:, e] * w[:, e]).sum(1)),
            "tau2": pct(lambda e: (tau[:, e] ** 2).sum(1)),
            "abs_tau": pct(lambda e: np.abs(tau[:, e]).sum(1)),
            "omega": pct(lambda e: np.abs(w[:, e]).sum(1)),
            "P_A": float(np.abs(tau[:, A] * w[:, A]).sum(1).mean()),
            "P_B": float(np.abs(tau[:, B] * w[:, B]).sum(1).mean()),
            "assist_rms": float(np.sqrt((assist[:, B] ** 2).mean())),
            "corr": float(np.mean([np.corrcoef(assist[:, B, d], tau[:, A, d])[0, 1] for d in range(2)])),
            "survived": meta.get("survived_s"),
        }
        try:
            from scipy.signal import find_peaks
            dt = meta["llp_dt"]

            def amp(x):
                pk, _ = find_peaks(x, distance=int(0.7 / dt)); tr_, _ = find_peaks(-x, distance=int(0.7 / dt))
                return float(np.degrees(x[pk].mean() - x[tr_].mean()))
            q, ref = tr["hip_pos"][idx], tr["ref_hip"][idx]
            out["amp_A"] = np.mean([amp(q[:, A, d]) for d in range(2)])
            out["amp_B"] = np.mean([amp(q[:, B, d]) for d in range(2)])
            out["amp_ref"] = np.mean([amp(ref[:, A, d]) for d in range(2)])
        except Exception:
            pass
        if "root_xy" in tr and np.abs(tr["root_xy"]).max() > 0:
            xy = tr["root_xy"]
            net = np.linalg.norm(xy[idx[-1]] - xy[idx[0] - 1], axis=-1)
            out["speed"] = 100.0 * (net[B] / max(net[A], 1e-9) - 1.0)
        if "body_err" in tr:
            be, bt = tr["body_err"], tr["body_err_t"]
            m = bt >= meta["warmup"]
            out["body_A"], out["body_B"] = float(be[m, A].mean()), float(be[m, B].mean())
            out["body"] = 100.0 * (out["body_B"] / max(out["body_A"], 1e-9) - 1.0)
        return out

    @staticmethod
    def phase_rows(run: Path, bins=10):
        """위상 10 % 구간별 τ² 절대 기여·토크·보조·|ω| — report_html.phase_rows."""
        tr, meta, idx = RunRecord.load(run)
        if "gait_phase" not in tr or not np.isfinite(tr["gait_phase"]).any():
            return None
        tau = (tr["tau_pd_prev"] + tr["tau_ff_prev"])[idx]
        ph = tr["gait_phase"][idx]
        assist = tr["tau_assist"][idx - 1]
        rows = []
        for b in range(bins):
            sel = (ph >= b / bins) & (ph < (b + 1) / bins)
            a2 = (tau[:, A][sel] ** 2).mean()
            b2 = (tau[:, B][sel] ** 2).mean()
            w = 0.5 * (tr["hip_vel"][idx - 1] + tr["hip_vel"][idx])
            rows.append({
                "contrib": float((b2 - a2) * sel.mean() * 2),   # 전체 τ² 평균에 대한 절대 기여 (N·m)²
                "tauA": float(tau[:, A][sel].mean()), "tauB": float(tau[:, B][sel].mean()),
                "assist": float(assist[:, B][sel].mean()),
                "omegaA": float(np.abs(w[:, A])[sel].mean()), "omegaB": float(np.abs(w[:, B])[sel].mean()),
            })
        return rows

    @staticmethod
    def extra_metrics(run: Path):
        """html_metrics()에 없는 것: 포화 비율, 부호 일치, 변화율, 스트라이드 주기."""
        tr, meta, idx = RunRecord.load(run)
        tau = (tr["tau_pd_prev"] + tr["tau_ff_prev"])[idx]
        assist = tr["tau_assist"][idx - 1]
        lim = assist_output_limit(meta)
        dt = float(meta["llp_dt"])
        sat = assist_saturation_pct(assist[:, B], meta)
        big = np.abs(tau[:, A]) > 5.0
        agree = 100.0 * float(np.mean((np.sign(assist[:, B]) == np.sign(tau[:, A]))[big]))
        slew = float(np.sqrt(np.mean(np.diff(assist[:, B], axis=0) ** 2)) / dt)
        ph = tr["gait_phase"][idx, 0]
        wraps = np.where(np.diff(ph) < -0.5)[0]
        stride = float(np.median(np.diff(tr["t"][idx][wraps]))) if len(wraps) > 2 else float("nan")
        return {"sat": sat, "agree": agree, "slew": slew, "stride": stride, "limit": lim}

    @staticmethod
    def cycle_profile(run: Path, bins=50):
        """위상 bin별 착용자 토크 A·B 평균, 보조 토크 평균·표준편차 (양쪽 다리 합침)."""
        tr, meta, idx = RunRecord.load(run)
        tau = (tr["tau_pd_prev"] + tr["tau_ff_prev"])[idx]
        assist = tr["tau_assist"][idx - 1]
        ph = tr["gait_phase"][idx]  # (n, 2 legs)
        edges = np.linspace(0, 1, bins + 1)
        out = {"phase": 100.0 * 0.5 * (edges[:-1] + edges[1:]), "tauA": [], "tauB": [], "assist": [], "assist_sd": []}
        for b in range(bins):
            sel = (ph >= edges[b]) & (ph < edges[b + 1])
            out["tauA"].append(float(tau[:, A][sel].mean()))
            out["tauB"].append(float(tau[:, B][sel].mean()))
            a = assist[:, B][sel]
            out["assist"].append(float(a.mean()))
            out["assist_sd"].append(float(a.std()))
        return {k: np.asarray(v) for k, v in out.items()}

    @staticmethod
    def time_window(run: Path, strides=3, t_from=10.0):
        """오른다리 위상 0 % 경계에서 시작하는 `strides` 스트라이드 창."""
        tr, meta, idx = RunRecord.load(run)
        t = tr["t"]
        ph = tr["gait_phase"][:, 0]
        wraps = stride_boundaries(ph)
        wraps = wraps[t[wraps] > t_from]
        i0 = int(wraps[0])
        i1 = int(wraps[strides]) if len(wraps) > strides else len(t) - 1
        sl = slice(i0, i1 + 1)
        tau = tr["tau_pd_prev"] + tr["tau_ff_prev"]
        return {
            "t": t[sl] - t[i0],
            "marks": t[wraps[: strides + 1]] - t[i0],
            "angA": np.degrees(tr["hip_pos"][sl, A, 0]),
            "angB": np.degrees(tr["hip_pos"][sl, B, 0]),
            "ref": np.degrees(tr["ref_hip"][sl, A, 0]),
            "tauA": tau[sl, A, 0],
            "tauB": tau[sl, B, 0],
            "assist": np.concatenate([[0.0], tr["tau_assist"][i0 : i1, B, 0]]),
            "limit": assist_output_limit(meta),
        }

    @staticmethod
    def print_table(rows):
        """터미널 표 (compare_runs CLI)."""
        cols = ["P|τω|", "τ²", "|τ|", "W+", "W−", "|ω|", "speed", "body"]
        head = f"{'run':14s} {'P_A':>6s} {'P_B':>6s} " + " ".join(f"{c:>7s}" for c in cols) + f" {'보조RMS':>7s} {'몸체 A→B [m]':>14s}"
        print("\n무보조 A 대비 보조 B의 변화 [%] — 음수가 좋다 (W−는 브레이킹 일, 음수가 덜 브레이킹)")
        print(head)
        print("-" * len(head))
        for r in rows:
            line = f"{r['run']:14s} {r['P_A']:6.0f} {r['P_B']:6.0f} " + " ".join(
                f"{r.get(c, float('nan')):+7.1f}" for c in cols
            )
            line += f" {r['assist_rms']:7.1f}"
            if "body_A" in r:
                line += f"   {r['body_A']:.3f}→{r['body_B']:.3f}"
            if r.get("fell"):
                line += f"  낙상 {r['fell']:.1f}s"
            print(line)
        print("\nP|τω| = 현재 학습 목적 (속도로 내려감) | τ² = 근활성 프록시 (속도로 못 내림)")
        print("speed = 전진 거리 변화, body = 착용자 몸체 추종 오차 변화")


# ---------------------------------------------------------------------------
# 착용자(tracker)가 얼마나 사람 같은가
# ---------------------------------------------------------------------------
class WearerStats:
    """무보조 env(A)로 잰 착용자 자체의 사람다움 (W1 판정용).

    | 지표 | 뜻 | v32 (참고) | W1 목표 |
    |---|---|---|---|
    | hip 스윙 진폭 A / mocap | 각도 진폭이 mocap에 맞는가 | 54.5° / 46° | mocap ±3° |
    | hip 토크 스트라이드 피크 (N·m, N·m/kg) | 착지 슬램 크기 | −332 (3.7) | ≤ 200 (2.2) |
    | 주기 평균 토크 피크 | 문헌 ≈ 1 N·m/kg | −157 (1.7) | ≈ 1 |
    | 피크/RMS | bang-bang 정도 | 4.3 | < 3 |
    | 몸체 오차 A [m] | mocap 몸체 추종 | 0.069 | ≤ 0.08 |
    """

    MASS_KG = 90.7

    @staticmethod
    def compute(run: Path):
        tr, meta, idx = RunRecord.load(run)
        tau = (tr["tau_pd_prev"] + tr["tau_ff_prev"])[idx]  # (n, env, dof)
        ph = tr["gait_phase"][idx]                             # (n, leg)
        m = ObjectiveComparison.html_metrics(run)
        out = {"run": Path(run).name, "survived": meta.get("survived_s"), "body_A": m.get("body_A"), "tau2_B": m["tau2"],
               "omega_B": m["omega"], "amp_A": m.get("amp_A"), "amp_B": m.get("amp_B"), "amp_ref": m.get("amp_ref"),
               "assist_rms": m["assist_rms"], "corr": m["corr"]}
        tA = tau[:, A]
        out["rms_A"] = float(np.sqrt((tA ** 2).mean()))
        # 스트라이드별 피크 (신전 = 음수): 위상 0 % 경계로 스트라이드를 자른다 (오른다리, dof 0)
        wraps = stride_boundaries(ph[:, 0])
        ext, flex = [], []
        for a, b in zip(wraps[:-1], wraps[1:]):
            seg = tA[a:b, 0]
            if len(seg) > 10:
                ext.append(seg.min()); flex.append(seg.max())
        out["stride_ext_peak"] = float(np.median(ext)) if ext else float("nan")
        out["stride_ext_p95"] = float(np.percentile(ext, 5)) if ext else float("nan")   # 가장 센 5 %
        out["stride_flex_peak"] = float(np.median(flex)) if flex else float("nan")
        out["peak_over_rms"] = float(np.abs(np.median(ext)) / out["rms_A"]) if ext else float("nan")
        # 주기 평균 프로파일 피크 (양쪽 다리)
        bins = 50
        prof = np.array([tA[(ph >= k / bins) & (ph < (k + 1) / bins)].mean() for k in range(bins)])
        out["cycle_ext_peak"], out["cycle_flex_peak"] = float(prof.min()), float(prof.max())
        out["n_strides"] = len(ext)
        return out

    @classmethod
    def print_table(cls, runs, mass=None):
        mass = cls.MASS_KG if mass is None else mass
        hdr = f"{'run':38s} {'생존':>8s} {'몸체A[m]':>8s} {'스윙A/B/mocap[°]':>18s} {'RMS_A':>6s} {'스트라이드 신전피크':>16s} {'(N·m/kg)':>8s} {'p5':>6s} {'주기평균 신전/굴곡':>16s} {'피크/RMS':>8s} {'B τ²':>7s} {'보조RMS':>7s} {'상관':>6s}"
        print(hdr)
        for r in runs:
            s = cls.compute(r)
            surv = f"{s['survived']:.1f}s 낙상" if s["survived"] else "완주"
            amp = f"{s['amp_A']:.1f}/{s['amp_B']:.1f}/{s['amp_ref']:.1f}" if s.get("amp_A") is not None else "—"
            print(f"{s['run'][:38]:38s} {surv:>8s} {s['body_A']:8.3f} {amp:>18s} {s['rms_A']:6.1f} {s['stride_ext_peak']:16.0f} "
                  f"{abs(s['stride_ext_peak'])/mass:8.2f} {s['stride_ext_p95']:6.0f} {s['cycle_ext_peak']:8.0f}/{s['cycle_flex_peak']:<7.0f} "
                  f"{s['peak_over_rms']:8.2f} {s['tau2_B']:+7.1f} {s['assist_rms']:7.1f} {s['corr']:+6.2f}")
        print("스트라이드 신전피크 = 오른쪽 hip 스트라이드별 최소 토크의 중앙값(p5 = 가장 센 5 %), 주기평균 = 위상 50 bin 평균 프로파일의 극값, B τ² = 보조 켠 env의 착용자 τ² 변화 [%]")


# ---------------------------------------------------------------------------
# 같은 이전 창에 정렬한 토크 · 위상 프로파일 · 후보 순위
# ---------------------------------------------------------------------------
class TorqueAlignment:
    @staticmethod
    def aligned_torques(tr, meta):
        """Align all torques to the SAME previous control window, including sample 0."""
        idx = np.flatnonzero((tr["t"] >= meta["warmup"]) & (np.arange(len(tr["t"])) >= 1))
        agent = (tr["tau_pd_prev"] + tr["tau_ff_prev"])[idx]
        assist = tr["tau_assist"][idx - 1]
        phase = circular_phase_midpoint(tr["gait_phase"][idx - 1], tr["gait_phase"][idx])
        return {"t": .5 * (tr["t"][idx - 1] + tr["t"][idx]), "phase": phase,
                "agent_A": agent[:, 0], "agent_B": agent[:, 1],
                "assist_A": assist[:, 0], "assist_B": assist[:, 1],
                "total_B": agent[:, 1] + assist[:, 1]}

    @staticmethod
    def phase_profile(data, bins):
        """Pool the two hips by each hip's reference phase; RMS precedes averaging."""
        out = []
        total_samples = data["phase"].size
        for i in range(bins):
            mask = (data["phase"] >= i / bins) & (data["phase"] < (i + 1) / bins)
            row = {"phase_start_pct": i * 100 / bins, "phase_end_pct": (i + 1) * 100 / bins,
                   "phase_pct": (i + .5) * 100 / bins, "samples": int(mask.sum())}
            for key in ("agent_A", "agent_B", "assist_A", "assist_B", "total_B"):
                v = data[key][mask]
                row[key] = float(v.mean()) if v.size else None
                row[key + "_rms"] = float(np.sqrt(np.mean(v ** 2))) if v.size else None
            if mask.any():
                a2 = float(np.mean(data["agent_A"][mask] ** 2))
                b2 = float(np.mean(data["agent_B"][mask] ** 2))
                row["rms_reduction_Nm"] = row["agent_A_rms"] - row["agent_B_rms"]
                row["tau2_change_pct"] = 100 * (b2 / a2 - 1) if a2 > 1e-9 else None
                row["tau2_contribution_Nm2"] = (b2 - a2) * 2 * mask.sum() / total_samples
            else:
                row.update(rms_reduction_Nm=None, tau2_change_pct=None, tau2_contribution_Nm2=None)
            out.append(row)
        return out


# 테스트·외부 모듈이 쓰는 이름 (모듈 수준 별칭)
aligned_torques = TorqueAlignment.aligned_torques
phase_profile = TorqueAlignment.phase_profile
