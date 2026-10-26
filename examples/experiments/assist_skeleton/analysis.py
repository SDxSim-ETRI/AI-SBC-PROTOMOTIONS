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
  `TorqueAlignment`       같은 이전 창에 정렬한 토크, 위상 프로파일(RMS 우선), 후보 순위 — report_torque_html
  `PhaseScreen`           L1 ×0.5 대비 개발 단계 통과 기준 (일반화 증거 아님) — phase_screen
  `DecelerationMasks`     무보조 기록으로 정한 저부담 감속 구간 마스크와 구간별 지표 — report_output_scale
  `NeedBins`              무보조 착용자 |토크| 구간별 보조 크기·방향 일치 — analyze_need_bins
  `WalkingPhaseDigest`    4 % 위상 구간 요약 + 저부담 구간 + 각속도 구간 + 다른 관절 — report_general_phase·report_generalization
  `SwingGateBins`         유각 브레이크 게이트 분석 (4 % 구간, 게이트 창 위치) — report_swing_gate·run_wide_sequence
"""

import glob
import json
import math
import re
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

    @staticmethod
    def time_rows(data):
        t, ph = data["t"], data["phase"][:, 0]
        wraps = stride_boundaries(ph)
        eligible = wraps[t[wraps] >= 10]
        # A short/fallen run still gets a useful window without inventing samples.
        if len(eligible) < 4:
            eligible = wraps
        start = int(eligible[0]) if len(eligible) else 0
        end = int(eligible[3]) if len(eligible) > 3 else len(t) - 1
        rows = [{"time_s": float(t[i]), **{k: float(data[k][i, 0]) for k in ("agent_A", "agent_B", "assist_A", "assist_B", "total_B")}}
                for i in range(start, end + 1)]
        marks = [float(t[i]) for i in wraps if start <= i <= end]
        return rows, marks

    @staticmethod
    def label(key):
        if key.startswith("reference"):
            return "L1 ×0.5 · 기존 기준선"
        version = re.search(r"(?:^|_)(L[0-9]+)(?:_|$)", key)
        epoch = re.search(r"_e([0-9]+)(?:_|$)", key)
        return f'{version.group(1) if version else key} ×1 · {epoch.group(1) if epoch else "?"} epoch'

    @classmethod
    def rank_candidates(cls, records, top_k=3):
        """Use both motions, reject falls/global regressions, then rank worst phase harm.

        This selects a display shortlist, NOT validated winners. Keep all source runs.
        """
        groups, diagnostics = {}, {}
        for key, m in records.items():
            run = Path(m["run_dir"])
            meta = json.loads((run / "meta.json").read_text())
            with np.load(run / "traces.npz") as f:
                tr = {k: f[k] for k in ("t", "gait_phase", "tau_pd_prev", "tau_ff_prev", "tau_assist")}
            data = cls.aligned_torques(tr, meta)
            if len(data["t"]) < 2:
                diagnostics[key] = {"invalid": "insufficient samples after warmup"}
                continue
            rows = cls.phase_profile(data, 10)
            increases = [max(0., -(r["rms_reduction_Nm"] or 0.)) for r in rows]
            diagnostics[key] = {"worst_phase_rms_increase_Nm": max(increases),
                                "mean_phase_rms_increase_Nm": float(np.mean(increases)),
                                "phase": rows, "intervals": {}}
            for lo, hi in ((20, 40), (70, 90)):
                sel = (data["phase"] >= lo/100) & (data["phase"] < hi/100)
                if not sel.any():
                    diagnostics[key]["intervals"][f"{lo}-{hi}"] = {"samples": 0}
                    continue
                a, b, u = (data[k][sel] for k in ("agent_A", "agent_B", "assist_B"))
                diagnostics[key]["intervals"][f"{lo}-{hi}"] = {
                    "agent_A_rms_Nm": float(np.sqrt(np.mean(a*a))),
                    "agent_B_rms_Nm": float(np.sqrt(np.mean(b*b))),
                    "assist_rms_Nm": float(np.sqrt(np.mean(u*u))),
                    "opposition_mean_Nm2": float(np.maximum(-b*u, 0).mean()),
                    "opposing_fraction_pct": float(100*np.mean(b*u < 0)),
                }
            if key.startswith("reference"):
                continue
            group = key.rsplit("_m", 1)[0]
            finished = meta["survived_s"] is None and len(tr["t"]) >= round(meta["seconds"]/meta["llp_dt"]) and meta["seconds"] >= 40 and all(r["samples"] > 0 for r in rows)
            groups.setdefault(group, {})[meta["motion_id"]] = (key, m, finished)
        ranking = []
        for group, motions in groups.items():
            if set(motions) != {0, 1}:
                continue
            if not all(done and m.get("τ²", float("inf")) < 0 for _, m, done in motions.values()):
                continue
            keys = [motions[i][0] for i in (0, 1)]
            worst = max(diagnostics[k]["worst_phase_rms_increase_Nm"] for k in keys)
            mean_harm = float(np.mean([diagnostics[k]["mean_phase_rms_increase_Nm"] for k in keys]))
            global_worst = max(motions[i][1]["τ²"] for i in (0, 1))
            ranking.append((worst, mean_harm, global_worst, group, keys))
        ranking.sort()
        selected = [k for k in records if k.startswith("reference")]
        selected += [k for row in ranking[:top_k] for k in row[-1]]
        return {k: records[k] for k in selected}, {"top_k": top_k,
                "eligible_count": len(ranking), "selected_groups": [r[-2] for r in ranking[:top_k]],
                "ranking_rule": "Both 40s motions complete and tau2 improves in both; sort by worst phase RMS increase, mean positive phase RMS increase, worst global tau2 change.",
                "evaluations": diagnostics}


# ---------------------------------------------------------------------------
# 개발 단계 통과 기준 (L1 ×0.5 대비) — 일반화 증거 아님
# ---------------------------------------------------------------------------
class PhaseScreen:
    """Thresholds are declared before the L8 results: each 10% phase may regress at most 1 Nm in RMS change,
    but worst phase and the two suspect intervals must not regress. Body tracking tolerance is 1 mm above the
    run's unassisted control. Angular-speed change magnitude may exceed the L1 change by at most two
    percentage points."""

    THRESHOLDS = {"phase_regression_tolerance_Nm": 1.0, "body_error_increase_limit_m": .001,
                  "omega_extra_change_limit_pp": 2.0, "speed_change_limit_pct": 2.0,
                  "duration_s": 40.0}

    @classmethod
    def screen_motion(cls, row, ref, diag, ref_diag, completed, gain, limit):
        """Keep individual failures: good phases/metrics cannot cancel bad ones."""
        SCREEN = cls.THRESHOLDS
        checks = []

        def add(name, value, maximum):
            finite = value is not None and maximum is not None and math.isfinite(value) and math.isfinite(maximum)
            checks.append({"name": name, "value": value if value is not None and math.isfinite(value) else None,
                           "maximum": maximum if maximum is not None and math.isfinite(maximum) else None,
                           "pass": bool(finite and value <= maximum)})
        checks.append({"name": "40s_complete", "pass": bool(completed)})
        checks.append({"name": "gain1_limit100", "pass": gain == 1.0 and limit == 100.0})
        add("global_tau2_change_pct", row.get("τ²"), ref.get("τ²"))
        if "phase" not in diag or "phase" not in ref_diag:
            checks.append({"name": "phase_data_available", "pass": False})
            return {"pass": False, "checks": checks}
        checks.append({"name": "ten_phase_bins", "pass": len(diag["phase"]) == len(ref_diag["phase"]) == 10})
        for i, (phase, base) in enumerate(zip(diag["phase"], ref_diag["phase"])):
            if not phase["samples"] or not base["samples"]:
                checks.append({"name": f"phase_{i*10}_{(i+1)*10}_samples", "pass": False})
                continue
            # New (B-A) minus baseline (B-A), not signed torque averages.
            regression = base["rms_reduction_Nm"] - phase["rms_reduction_Nm"]
            add(f"phase_{i*10}_{(i+1)*10}_RMS_regression_Nm", regression, SCREEN["phase_regression_tolerance_Nm"])
        add("worst_phase_RMS_increase_Nm", diag.get("worst_phase_rms_increase_Nm"), ref_diag.get("worst_phase_rms_increase_Nm"))
        for interval in ("20-40", "70-90"):
            current, base = diag["intervals"].get(interval, {}), ref_diag["intervals"].get(interval, {})
            if "agent_B_rms_Nm" not in current or "agent_B_rms_Nm" not in base:
                checks.append({"name": f"interval_{interval}_available", "pass": False})
                continue
            add(f"interval_{interval}_RMS_increase_Nm",
                current["agent_B_rms_Nm"] - current["agent_A_rms_Nm"],
                base["agent_B_rms_Nm"] - base["agent_A_rms_Nm"])
        add("body_error_increase_m", row.get("body_B", math.inf) - row.get("body_A", 0), SCREEN["body_error_increase_limit_m"])
        add("absolute_omega_change_pct", abs(row.get("|ω|", math.inf)), abs(ref.get("|ω|", 0)) + SCREEN["omega_extra_change_limit_pp"])
        add("absolute_speed_change_pct", abs(row.get("speed", math.inf)), SCREEN["speed_change_limit_pct"])
        return {"pass": all(c["pass"] for c in checks), "checks": checks}

    @classmethod
    def screen_records(cls, records, diagnostics):
        SCREEN = cls.THRESHOLDS
        refs = {int(k.rsplit("_m", 1)[1]): k for k in records if k.startswith("reference_L1_")}
        result, groups = {}, {}
        for key, row in records.items():
            if key.startswith("reference"):
                continue
            run = Path(row["run_dir"])
            meta = json.loads((run / "meta.json").read_text())
            with np.load(run / "traces.npz") as tr:
                t = tr["t"]
            completed = (meta["survived_s"] is None and meta["seconds"] >= SCREEN["duration_s"]
                         and len(t) >= round(meta["seconds"] / meta["llp_dt"]))
            motion = meta["motion_id"]
            ref_key = refs.get(motion)
            if ref_key is None:
                result[key] = {"pass": False, "checks": [{"name": "matching_L1_reference", "pass": False}]}
            else:
                effective_limit = abs(meta["assist_scale"])*min(meta["torque_limit"], meta.get("policy_torque_limit", meta["torque_limit"]))
                result[key] = cls.screen_motion(row, records[ref_key], diagnostics[key], diagnostics[ref_key],
                                                completed, meta["assist_scale"], effective_limit)
            groups.setdefault(key.rsplit("_m", 1)[0], {})[motion] = result[key]["pass"]
        return {"scope": "Development screen only; multiple seeds and held-out conditions still required.",
                "thresholds": SCREEN, "evaluations": result,
                "candidate_pass": {k: set(m) == {0, 1} and all(m.values()) for k, m in groups.items()}}


# ---------------------------------------------------------------------------
# 저부담 감속 구간 마스크 (무보조 기록으로 정함) — 출력 크기 비교의 공통 잣대
# ---------------------------------------------------------------------------
class DecelerationMasks:
    """무보조 A의 50 ms 토크 RMS ≤ 10 N·m(또는 5)이면서 고관절이 감속 중인 시점을 한 번 정해 모든 모델·
    출력 크기에 같은 시각을 쓴다. 토크는 i−1 창의 보조를 i의 착용자 반응과 짝짓는다 (색인 ≥ 2, 1.5 s 이후)."""

    @staticmethod
    def smooth(values, width=5):
        """Centered offline moving average; analysis masks exclude boundary padding."""
        return np.stack([np.convolve(values[:, j], np.ones(width) / width, mode="same")
                         for j in range(values.shape[1])], axis=1)

    @staticmethod
    def align(traces):
        """Associate the torque at i-1 with wearer response measured at i."""
        index = np.arange(2, len(traces["t"]))
        return dict(t=traces["t"][index], index=index,
                    A=(traces["tau_pd_prev"] + traces["tau_ff_prev"])[index, 0],
                    B=(traces["tau_pd_prev"] + traces["tau_ff_prev"])[index, 1],
                    assist=traces["tau_assist"][index - 1, 1],
                    velocity=.5 * (traces["hip_vel"][index - 1] + traces["hip_vel"][index]))

    @classmethod
    def masks_from_reference(cls, traces):
        aligned = cls.align(traces)
        t = aligned["t"]
        if not np.allclose(np.diff(t), .01, atol=2e-5):
            raise ValueError("Expected uninterrupted 10 ms samples")
        velocity = cls.smooth(aligned["velocity"][:, 0])
        acceleration = np.gradient(velocity, .01, axis=0)
        torque_rms = np.sqrt(cls.smooth(np.square(aligned["A"])))
        valid = (t >= 1.5) & (t <= t[-1] - .05)
        deceleration = (velocity * acceleration < 0) & (np.abs(velocity) > .3) & (np.abs(acceleration) > 1.)
        masks = {"deceleration": valid[:, None] & deceleration}
        for threshold in (5, 10):
            masks[f"low_decel_{threshold}"] = masks["deceleration"] & (torque_rms <= threshold)
        masks["high_decel_20"] = masks["deceleration"] & (torque_rms > 20.)
        return dict(t=t, masks=masks, torque_rms=torque_rms, velocity=velocity,
                    acceleration=acceleration, A=aligned["A"])

    @staticmethod
    def selected_metrics(aligned, mask):
        count = int(mask.sum())
        result = dict(joint_samples=count, joint_seconds=count * .01,
                      samples_right=int(mask[:, 0].sum()), samples_left=int(mask[:, 1].sum()))
        if not count:
            return result
        a, b, assist = (aligned[key][mask].astype(np.float64) for key in ("A", "B", "assist"))
        velocity = aligned["velocity"][:, 1][mask]
        sq_a, sq_b = float(np.mean(a*a)), float(np.mean(b*b))
        result.update(A_tau_sq=sq_a, B_tau_sq=sq_b, tau_sq_increase=sq_b-sq_a,
                      A_rms=float(np.sqrt(sq_a)), B_rms=float(np.sqrt(sq_b)),
                      rms_increase=float(np.sqrt(sq_b)-np.sqrt(sq_a)),
                      assist_rms=float(np.sqrt(np.mean(assist*assist))),
                      assist_mean=float(np.mean(assist)),
                      braking_sample_pct=float(100*np.mean(assist*velocity < 0)),
                      absorbed_power_per_joint_W=float(np.maximum(-assist*velocity, 0).mean()),
                      absolute_assist_power_per_joint_W=float(np.abs(assist*velocity).mean()),
                      absorbed_work_selected_J=float(np.maximum(-assist*velocity, 0).sum()*.01),
                      absolute_assist_work_selected_J=float(np.abs(assist*velocity).sum()*.01))
        return result

    @classmethod
    def analyze_traces(cls, traces, reference):
        aligned = cls.align(traces)
        n = len(aligned["t"])
        if n > len(reference["t"]) or not np.allclose(aligned["t"], reference["t"][:n], atol=1e-6):
            raise ValueError("Cannot apply reference masks to different control times")
        overall_mask = np.broadcast_to((aligned["t"] >= 1.5)[:, None], aligned["A"].shape)
        result = {"overall": cls.selected_metrics(aligned, overall_mask)}
        for name, mask in reference["masks"].items():
            selected = mask[:n] & overall_mask
            result[name] = cls.selected_metrics(aligned, selected)
            result[name]["by_side"] = {}
            for side, label in enumerate(("right", "left")):
                side_mask = selected.copy()
                side_mask[:, 1-side] = False
                result[name]["by_side"][label] = cls.selected_metrics(aligned, side_mask)
            if selected.any():
                result[name]["overall_tau_sq_contribution"] = float(
                    ((aligned["B"]**2-aligned["A"]**2)*selected).sum()/overall_mask.sum())
        result["reference_A_difference_rms_Nm"] = float(np.sqrt(
            np.mean((aligned["A"][overall_mask]-reference["A"][:n][overall_mask])**2)))
        local_masks = cls.masks_from_reference(traces)["masks"]
        for name in ("low_decel_5", "low_decel_10"):
            fixed, local = reference["masks"][name][:n], local_masks[name]
            union = int((fixed | local).sum())
            result[name]["fixed_vs_local_mask_jaccard"] = float((fixed & local).sum()/union) if union else None
        return result

    @staticmethod
    def read_traces(record):
        with np.load(ROOT / record["path"] / "traces.npz") as file:
            return dict(file)


# ---------------------------------------------------------------------------
# 무보조 착용자 |토크| 구간별 보조 크기·방향 일치
# ---------------------------------------------------------------------------
class NeedBins:
    """저장 기록만 읽는다: 무보조 착용자 토크 |τ_A|로 시간을 나눠 각 구간의 보조 RMS·포화·방향 일치·B τ²."""

    BINS = [0, 5, 10, 20, 40, 80, 400]

    @classmethod
    def binned(cls, abs_need, assist, tau_a, tau_b, limit):
        rows = []
        for lo, hi in zip(cls.BINS[:-1], cls.BINS[1:]):
            s = (abs_need >= lo) & (abs_need < hi)
            if not s.any():
                continue
            rows.append(dict(need_lo_Nm=lo, need_hi_Nm=hi, sample_fraction=float(s.mean()),
                             assist_rms_Nm=float(np.sqrt(np.mean(assist[s] ** 2))),
                             saturation_fraction=float(np.mean(np.abs(assist[s]) > .95 * limit)),
                             sign_agreement=float(np.mean(np.sign(assist[s]) == np.sign(tau_a[s]))),
                             A_tau_sq_Nm2=float(np.mean(tau_a[s] ** 2)), B_tau_sq_Nm2=float(np.mean(tau_b[s] ** 2))))
        return rows

    @classmethod
    def walking(cls, path, limit):
        z = np.load(path)
        t = z["t"]; idx = np.arange(2, len(t))
        wearer = z["tau_pd_prev"] + z["tau_ff_prev"]
        a, b = wearer[idx, 0], wearer[idx, 1]
        assist = z["tau_assist"][idx - 1, 1]
        m = t[idx] >= 1.5
        a, b, assist = a[m], b[m], assist[m]
        return dict(source=str(Path(path).relative_to(ROOT)), joint_samples=int(a.size),
                    A_rms_Nm=float(np.sqrt(np.mean(a ** 2))), B_rms_Nm=float(np.sqrt(np.mean(b ** 2))),
                    assist_rms_Nm=float(np.sqrt(np.mean(assist ** 2))), saturation_fraction=float(np.mean(np.abs(assist) > .95 * limit)),
                    corr_assist_vs_unassisted_wearer=float(np.corrcoef(assist.ravel(), a.ravel())[0, 1]),
                    sign_agreement=float(np.mean(np.sign(assist) == np.sign(a))),
                    bins=cls.binned(np.abs(a), assist, a, b, limit))


# ---------------------------------------------------------------------------
# 4 % 위상 구간 요약 (W1 기록 한 건 → 그래프·표에 쓰는 digest)
# ---------------------------------------------------------------------------
class WalkingPhaseDigest:
    """기록 한 건을 4 % 위상 구간(각 다리의 참조 최대 굴곡 = 0 %)으로 묶고, 저부담 감속 구간·각속도 구간·
    다른 관절(전신 기록이 있으면)까지 한 dict로 만든다. 색인 ≥ 2, warmup + 0.5 s 이후."""

    BINS = 25
    OMEGA = [0, 1, 2, 3, 4, 5, 6, 8]

    @staticmethod
    def scalar(x):
        return float(x) if np.isfinite(x) else None

    @staticmethod
    def rms(x):
        return float(np.sqrt(np.mean(np.square(x))))

    @classmethod
    def digest_run(cls, record):
        BINS, rms = cls.BINS, cls.rms
        directory = ROOT / record['path']
        meta = json.loads((directory / 'meta.json').read_text())
        with np.load(directory / 'traces.npz') as f:
            tr = dict(f)
        idx = np.flatnonzero((tr['t'] >= meta['warmup'] + .5) & (np.arange(len(tr['t'])) >= 2))
        if len(idx) < 2:
            return dict(record=record, meta=meta, error='1.5초 이후 분석 가능한 창이 없음')
        tau = (tr['tau_pd_prev'] + tr['tau_ff_prev'])[idx]
        assist = tr['tau_assist'][idx - 1, 1]
        phase = circular_phase_midpoint(tr['gait_phase'][idx - 1], tr['gait_phase'][idx])
        assert np.isfinite(phase).all()
        limit = abs(meta['assist_scale']) * min(meta['torque_limit'], meta['policy_torque_limit'])
        assert np.max(np.abs(assist)) <= limit + 1e-4
        rows = []
        for side in (0, 1):
            values = []
            for k in range(BINS):
                sel = (phase[:, side] >= k / BINS) & (phase[:, side] < (k + 1) / BINS)
                a, b, s = tau[sel, 0, side], tau[sel, 1, side], assist[sel, side]
                row = dict(lo=k*100/BINS, hi=(k+1)*100/BINS, n=int(sel.sum()))
                if sel.any():
                    row.update(A_mean=float(a.mean()), B_mean=float(b.mean()), assist_mean=float(s.mean()),
                               A_rms=rms(a), B_rms=rms(b), assist_rms=rms(s), reduction=rms(a)-rms(b),
                               assist_min=float(s.min()), assist_max=float(s.max()),
                               saturation_pct=float(100*np.mean(np.abs(s) > .95*limit)))
                values.append(row)
            rows.append(values)
        omega = .5*(tr['hip_vel'][idx-1] + tr['hip_vel'][idx])
        err = np.degrees(tr['hip_pos'][idx] - tr['ref_hip'][idx])
        summary = dict(samples=len(idx), t_start=float(tr['t'][idx[0]]), t_end=float(tr['t'][idx[-1]]),
                       limit=limit, assist_rms=rms(assist), assist_max=float(np.abs(assist).max()),
                       saturation_pct=float(100*np.mean(np.abs(assist) > .95*limit)))
        for j, name in enumerate(('A', 'B')):
            summary[name] = dict(tau_rms=rms(tau[:, j]), tau_sq=float(np.square(tau[:, j]).mean()),
                                work=float(np.abs(tau[:, j]*omega[:, j]).sum(axis=1).mean()),
                                tracking_mae=float(np.abs(err[:, j]).mean()))
            if 'root_xy' in tr:
                summary[name]['speed'] = float(np.linalg.norm(tr['root_xy'][idx[-1], j] - tr['root_xy'][idx[0]-1, j]) /
                                              (tr['t'][idx[-1]] - tr['t'][idx[0]-1]))
            if 'body_err' in tr:
                summary[name]['body_err'] = float(tr['body_err'][tr['body_err_t'] >= meta['warmup']+.5, j].mean())
        other = []
        if (directory / 'fullbody.npz').exists():
            fm = json.loads((directory / 'fullbody_meta.json').read_text())
            with np.load(directory / 'fullbody.npz') as f:
                fulltau = (f['control_tau_pd_prev'] + f['control_tau_ff_prev'])[idx]
            names = fm['dof_names_common']
            hip_indices = [names.index('hip_flexion_' + side) for side in ('r', 'l')]
            assert np.allclose(fulltau[:, :, hip_indices], tau, atol=1e-4), 'Full-body torque alignment mismatch'
            # Suit slide DOFs carry linear forces [N], not joint torques [N m].
            for group, indices in [('전체 비보조 회전관절', [i for i, n in enumerate(names) if n not in ('hip_flexion_r', 'hip_flexion_l') and not n.startswith('slide')]),
                                   ('무릎', [i for i, n in enumerate(names) if 'knee' in n]),
                                   ('발목', [i for i, n in enumerate(names) if 'ankle' in n]),
                                   ('몸통', [i for i, n in enumerate(names) if any(s in n for s in ('lumbar', 'torso', 'abdomen'))])]:
                if indices:
                    other.append(dict(group=group, dofs=[names[i] for i in indices],
                                      A=float(np.square(fulltau[:, 0, indices]).mean()),
                                      B=float(np.square(fulltau[:, 1, indices]).mean())))
        return dict(record=record, meta=meta, rows=rows, summary=summary, other_joints=other)

    @classmethod
    def omega_bins(cls, speed_a, speed_b, assist):
        """Share of unassisted time and applied assist RMS per hip angular speed bin [rad/s]."""
        rows = []
        for lo, hi in zip(cls.OMEGA[:-1], cls.OMEGA[1:]):
            share = (speed_a >= lo) & (speed_a < hi)
            sel = (speed_b >= lo) & (speed_b < hi)
            rows.append(dict(lo=lo, hi=hi, share=float(share.mean()),
                             assist_rms=float(np.sqrt(np.mean(assist[sel] ** 2))) if sel.sum() > 50 else None))
        return rows

    @staticmethod
    def curve(xs, ys, x):
        """Piecewise-linear L1 curve through the origin; None outside the measured range."""
        xs, ys = np.array([0.] + list(xs)), np.array([0.] + list(ys))
        order = np.argsort(xs)
        xs, ys = xs[order], ys[order]
        if x > xs[-1] + 1e-9:
            return None
        return float(np.interp(max(x, 0.), xs, ys))

    @classmethod
    def digest(cls, record, reference):
        """기록 한 건의 비교용 요약: 이득(τ² 감소), 저부담 손해, 최악 위상, 좌우, 몸체·속도, need bin, 각속도 bin."""
        run = cls.digest_run(record)
        traces = DecelerationMasks.read_traces(record)
        seg = DecelerationMasks.analyze_traces(traces, reference)
        a = DecelerationMasks.align(traces)
        keep = a['t'] >= 1.5
        side_benefit = (np.square(a['A'][keep]) - np.square(a['B'][keep])).mean(axis=0)
        worst = []
        for side in (0, 1):
            rows = [r for r in run['rows'][side] if r.get('n')]
            w = max(rows, key=lambda r: -r['reduction'])
            worst.append(dict(value=-w['reduction'], lo=w['lo'], hi=w['hi']))
        phase = traces['gait_phase']
        period = (traces['t'][-1] - traces['t'][0]) / max(int(np.sum(np.diff(phase[:, 0]) < -.5)), 1)
        s = run['summary']
        other = {o['group']: o['A'] - o['B'] for o in run['other_joints']}
        return dict(
            benefit=-seg['overall']['tau_sq_increase'], A_tau_sq=seg['overall']['A_tau_sq'],
            assist_rms=s['assist_rms'], saturation=s['saturation_pct'],
            low10=seg['low_decel_10'].get('rms_increase'), low10_s=seg['low_decel_10']['joint_seconds'],
            low5=seg['low_decel_5'].get('rms_increase'), low5_s=seg['low_decel_5']['joint_seconds'],
            low10_side=[seg['low_decel_10']['by_side'][k].get('rms_increase') for k in ('right', 'left')],
            benefit_side=side_benefit.tolist(), worst_side=worst, worst=max(w['value'] for w in worst),
            mask_A_diff=seg['reference_A_difference_rms_Nm'],
            body_err_mm=[1000 * s['A']['body_err'], 1000 * s['B']['body_err']], speed=[s['A']['speed'], s['B']['speed']],
            tracking=[s['A']['tracking_mae'], s['B']['tracking_mae']], fall=run['meta']['survived_s'],
            other=other, rows=run['rows'], period_s=float(period),
            need=NeedBins.walking(ROOT / record['path'] / 'traces.npz', s['limit'])['bins'], path=record['path'],
            omega=cls.omega_bins(np.abs(a['velocity'][keep, 0]), np.abs(a['velocity'][keep, 1]), a['assist'][keep]))


# ---------------------------------------------------------------------------
# 유각 브레이크 게이트 분석 (4 % 구간)
# ---------------------------------------------------------------------------
class SwingGateBins:
    BINS = 25  # 4 % 구간

    @staticmethod
    def find_run(folder, tag):
        hits = sorted(p for p in Path(folder).glob(f"*_tracker_{tag}") if (p / "traces.npz").exists())
        return hits[-1] if hits else None

    @staticmethod
    def load(run_dir):
        """토크를 같은 10 ms 창에 정렬한 뒤 위상(각 다리의 참조 최대굴곡 = 0 %)으로 묶을 재료를 만든다."""
        meta = json.loads((Path(run_dir) / "meta.json").read_text())
        tr = dict(np.load(Path(run_dir) / "traces.npz"))
        idx = np.flatnonzero((tr["t"] >= meta["warmup"] + 0.5) & (np.arange(len(tr["t"])) >= 2))
        agent = (tr["tau_pd_prev"] + tr["tau_ff_prev"])[idx]
        assist = tr["tau_assist"][idx - 1]
        phase = circular_phase_midpoint(tr["gait_phase"][idx - 1], tr["gait_phase"][idx])
        dt = float(np.median(np.diff(tr["t"])))
        hv = tr["hip_vel"]
        acc = np.gradient(hv, dt, axis=0)
        data = {
            "phase": phase, "agent_A": agent[:, 0], "agent_B": agent[:, 1], "assist": assist[:, 1],
            "pos_A": tr["hip_pos"][idx, 0], "pos_B": tr["hip_pos"][idx, 1],
            "vel_A": hv[idx, 0], "vel_B": hv[idx, 1], "acc_A": acc[idx, 0],
        }
        if "swing_gate" in tr:
            data["gate"] = tr["swing_gate"][idx - 1, 1]
        return meta, data, tr

    @classmethod
    def bins(cls, d, n=None):
        n = cls.BINS if n is None else n
        rows = []
        for i in range(n):
            sel = [(d["phase"][:, L] >= i / n) & (d["phase"][:, L] < (i + 1) / n) for L in (0, 1)]
            pick = lambda key: np.concatenate([d[key][m, L] for m, L in zip(sel, (0, 1))])  # noqa: E731
            a, b, s = pick("agent_A"), pick("agent_B"), pick("assist")
            r = dict(lo=100 * i / n, hi=100 * (i + 1) / n, A=a.mean(), B=b.mean(), S=s.mean(),
                     A_rms=np.sqrt((a ** 2).mean()), B_rms=np.sqrt((b ** 2).mean()), S_rms=np.sqrt((s ** 2).mean()),
                     dAB=(b - a).mean(), c=(-(b - a).mean() / s.mean()) if abs(s.mean()) > 2 else np.nan,
                     oppA=100 * np.mean(s * a < 0), pos=np.degrees(pick("pos_A").mean()),
                     vel=pick("vel_A").mean(), acc=pick("acc_A").mean(),
                     dpos=np.degrees((pick("pos_B") - pick("pos_A")).mean()),
                     contrib=((b ** 2).mean() - (a ** 2).mean()) / n)
            if "gate" in d:
                r["gate_pct"] = 100 * pick("gate").mean()
            rows.append(r)
        return rows

    @staticmethod
    def window_contribution(rows, lo, hi):
        """구간 [lo, hi) %의 τ² 평균 기여 [(N·m)²] — 양수면 부담 증가."""
        return sum(r["contrib"] for r in rows if r["lo"] >= lo and r["hi"] <= hi)

    @staticmethod
    def gate_accuracy(d):
        """게이트 창이 참조 위상 어디에 걸렸는지 (관절각 추정 vs 참조 위상표)."""
        if "gate" not in d:
            return None
        ph, g = d["phase"], d["gate"] > 0.5
        on = ph[g]
        out = {"on_pct": 100 * g.mean(), "on_phase_mean": 100 * float(np.angle(np.exp(2j * np.pi * on).mean()) / (2 * np.pi) % 1)}
        out["in_70_90_pct"] = 100 * float(np.mean((on >= .70) & (on < .90)))
        out["in_66_92_pct"] = 100 * float(np.mean((on >= .66) & (on < .92)))
        starts, ends = [], []
        for L in (0, 1):
            gl = g[:, L].astype(int)
            s = np.flatnonzero(np.diff(gl) == 1) + 1
            e = np.flatnonzero(np.diff(gl) == -1) + 1
            starts += list(ph[s, L]); ends += list(ph[e, L])
        out["start_phase"] = (100 * np.mean(starts), 100 * np.std(starts)) if starts else (np.nan, np.nan)
        out["end_phase"] = (100 * np.mean(ends), 100 * np.std(ends)) if ends else (np.nan, np.nan)
        return out

    @staticmethod
    def run_summary(run_dir):
        """실행 하나의 전체 수치 (τ² 변화, 보조 RMS, 몸체 오차, 생존, 포화) — 2026-10 시퀀스 보고서의 comparison.json 행 규약."""
        run_dir = Path(run_dir)
        meta = json.loads((run_dir / "meta.json").read_text())
        with np.load(run_dir / "traces.npz") as f:
            tr = {key: f[key] for key in f.files}
        valid = tr["t"] >= meta["warmup"]
        # Early falls are valid outcomes; never retry them as execution failures.
        out = ObjectiveComparison.metrics(run_dir) if valid.sum() > 1 else {"valid_after_warmup": False}
        out["survived_s"] = meta["survived_s"]
        out["run_dir"] = str(run_dir)
        out["assist_scale"] = meta["assist_scale"]
        if valid.any():
            if "tau_requested" in tr:
                request = tr["tau_requested"][valid, 1]
                out["requested_peak_Nm"] = float(np.abs(request).max())
                out["request_clipped_pct"] = float(100 * (np.abs(request) > meta["policy_torque_limit"]).mean())
            out["assist_sat95_pct"] = assist_saturation_pct(tr["tau_assist"][valid, 1], meta)
        return out


# 테스트·외부 모듈이 쓰는 이름 (모듈 수준 별칭)
aligned_torques = TorqueAlignment.aligned_torques
phase_profile = TorqueAlignment.phase_profile
rank_candidates = TorqueAlignment.rank_candidates
screen_motion = PhaseScreen.screen_motion
screen_records = PhaseScreen.screen_records
align = DecelerationMasks.align
analyze_traces = DecelerationMasks.analyze_traces
masks_from_reference = DecelerationMasks.masks_from_reference
read_traces = DecelerationMasks.read_traces
digest_run = WalkingPhaseDigest.digest_run
