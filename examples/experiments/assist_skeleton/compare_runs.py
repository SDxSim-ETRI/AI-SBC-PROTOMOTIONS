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
"""run_system.py 실행 여러 개를 한 표로 — 목적함수를 바꿔 가며 A/B를 다시 잰다.

파워 |τ·ω|는 다리가 덜 흔들리기만 해도 내려간다. 그래서 같은 기록에서
τ²(근활성 프록시)·|τ|·양/음의 일·전진 속도·몸체 추종 오차를 함께 뽑아,
"장치가 부하를 받았는가"와 "동작이 바뀌었는가"를 갈라 본다.

실행 (sbc env):
    python examples/experiments/assist_skeleton/compare_runs.py results/oracle/*_tracker_*
    python examples/experiments/assist_skeleton/compare_runs.py --plot docs/oracle.png results/oracle/*
"""

import json
import sys
from pathlib import Path

import numpy as np

A, B = 0, 1


def metrics(run: Path):
    tr = np.load(run / "traces.npz")
    meta = json.loads((run / "meta.json").read_text())
    t = tr["t"]
    idx = np.where(t >= meta["warmup"])[0]
    idx = idx[idx >= 1]
    w = 0.5 * (tr["hip_vel"][idx - 1] + tr["hip_vel"][idx])
    tau = (tr["tau_pd_prev"] + tr["tau_ff_prev"])[idx]
    assist = tr["tau_assist"][idx - 1]
    out = {"run": run.name.split("_tracker_")[-1]}
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


def phase_tau2(run: Path, bins=10):
    """위상 구간별 τ² 변화 [%] — 어디서 벌고 어디서 잃는지."""
    tr = np.load(run / "traces.npz")
    meta = json.loads((run / "meta.json").read_text())
    if "gait_phase" not in tr or not np.isfinite(tr["gait_phase"]).any():
        return None
    t = tr["t"]
    idx = np.where(t >= meta["warmup"])[0]
    idx = idx[idx >= 1]
    tau = (tr["tau_pd_prev"] + tr["tau_ff_prev"])[idx]
    ph = tr["gait_phase"][idx]
    out = []
    for b in range(bins):
        sel = (ph >= b / bins) & (ph < (b + 1) / bins)
        a = (tau[:, A][sel] ** 2).mean()
        out.append(100.0 * ((tau[:, B][sel] ** 2).mean() / max(a, 1e-9) - 1.0))
    return np.array(out)


def save_plot(rows, runs, out_png: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
    matplotlib.rcParams["axes.unicode_minus"] = False
    INK, TEAL, ORANGE, GREY = "#141a21", "#0e7a74", "#c2410c", "#9aa8b6"

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.0), gridspec_kw={"width_ratios": [1, 1.25]})
    names = [r["run"] for r in rows]
    vals = [r["τ²"] for r in rows]
    colors = [ORANGE if v > 0 else TEAL for v in vals]
    ax[0].barh(range(len(rows)), vals, color=colors)
    ax[0].set_yticks(range(len(rows)), names, fontsize=9)
    ax[0].invert_yaxis()
    ax[0].axvline(0, color=INK, lw=1)
    ax[0].set_xlabel("착용자 τ² 변화 [%] — 음수가 좋다")
    ax[0].set_title("근활성 프록시로 다시 재면 순위가 뒤집힌다", fontsize=11, loc="left")
    lo, hi = min(vals + [0]), max(vals + [0])
    pad = 0.28 * max(hi - lo, 1.0)
    ax[0].set_xlim(lo - pad, hi + pad)
    for i, v in enumerate(vals):
        ax[0].text(v + (1.2 if v > 0 else -1.2), i, f"{v:+.1f}", va="center",
                   ha="left" if v > 0 else "right", fontsize=9, color=INK)
    ax[0].grid(axis="x", alpha=0.25)

    x = np.arange(10) * 10 + 5
    for run, row in zip(runs, rows):
        pt = phase_tau2(run)
        if pt is None:
            continue
        style = dict(lw=2.4) if row["run"] in ("base", "beta05") else dict(lw=1.3, ls="--")
        ax[1].plot(x, pt, marker="o", ms=3.5, label=row["run"], **style)
    ax[1].axhline(0, color=INK, lw=1)
    ax[1].axvspan(0, 20, color="#fde68a", alpha=0.45, zorder=0)
    top = 60.0
    ax[1].set_ylim(-95, top)
    for run, row in zip(runs, rows):  # 잘린 점은 값을 적어 둔다
        pt = phase_tau2(run)
        if pt is None:
            continue
        for xi, v in zip(x, pt):
            if v > top:
                ax[1].annotate(f"{row['run']} {v:+.0f} %", xy=(xi, top), xytext=(xi + 3, top - 9),
                               fontsize=9, color=ORANGE,
                               arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.1))
    ax[1].text(10, top * 0.80, "착지·하중 수용", ha="center", fontsize=9, color=INK)
    ax[1].set_xlabel("보행 주기 [%] — 0 % = 최대 굴곡")
    ax[1].set_ylabel("착용자 τ² 변화 [%]")
    ax[1].set_title("v10은 착지 구간에서만 손해를 본다", fontsize=11, loc="left")
    ax[1].legend(fontsize=8.5, ncol=2)
    ax[1].grid(alpha=0.25)

    fig.suptitle("오라클 진단 — 보조가 부하를 받았는가, 아니면 동작을 바꿨는가", fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=135)
    print("saved", out_png)


def main():
    argv = sys.argv[1:]
    out_png = None
    if argv and argv[0] == "--plot":
        out_png, argv = argv[1], argv[2:]
    runs = [Path(p) for p in argv]
    rows = [metrics(r) for r in runs if (r / "traces.npz").exists()]
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
    if out_png:
        save_plot(rows, [r for r in runs if (r / "traces.npz").exists()], out_png)


if __name__ == "__main__":
    main()
