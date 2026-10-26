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
"""matplotlib PNG 그림 모음 (문서용). import만으로는 아무것도 그리지 않는다.

  `GaitSymmetryPlot`     (`gait-symmetry`)  착용자 좌우 고관절 궤적·가동범위 비교 (무보조 A env)
  `TimingDiagram`        (`timing`)         HLP(느린 예측기) + LLP(빠른 제어기) 타이밍 그림 — 예측 범위 부등식
  `TorquePhasePlot`      (`torque-phase`)   보행 한 주기로 접은 토크·파워 — 보조 토크가 언제 나오고 언제 쉬는지
  `OracleComparisonPlot` (`compare`)        실행 여러 개의 τ² 변화 막대 + 위상별 τ² 곡선 (오라클 진단 그림)

    python examples/experiments/assist_skeleton/plots.py timing
    python examples/experiments/assist_skeleton/plots.py torque-phase <run_dir>
    python examples/experiments/assist_skeleton/plots.py compare --plot docs/ai_sbc/oracle.png results/oracle/*
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis import ObjectiveComparison  # noqa: E402


class Style:
    INK, SOFT, TEAL, ORANGE, GREY = "#141a21", "#5d6a78", "#0e7a74", "#c2410c", "#9aa8b6"

    @staticmethod
    def pyplot():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
        matplotlib.rcParams["axes.unicode_minus"] = False
        return plt


class GaitSymmetryPlot:
    """`run_system.py`가 남긴 traces.npz 두 개의 왼쪽·오른쪽 고관절 굴곡 각도와 가동범위 (관절각 추종 항 추가 전/후)."""

    RUNS = [("기존 v31", "results/gait_eval/*_v6final_m0"),
            ("재학습 v32", "results/sym_check/*_final_m0")]

    @classmethod
    def render(cls, runs=None, out="results/sym_check/symmetry_compare.png"):
        plt = Style.pyplot()
        runs = cls.RUNS if runs is None else runs
        fig, axes = plt.subplots(2, 2, figsize=(13, 6.4), gridspec_kw={"width_ratios": [3, 1]})
        for row, (tag, pat) in enumerate(runs):
            d = sorted(glob.glob(pat))[-1]
            tr = np.load(d + "/traces.npz")
            t = tr["t"]; m = (t >= 4.0) & (t <= 9.0)
            hp = np.degrees(tr["hip_pos"][m, 0]); ref = np.degrees(tr["ref_hip"][m, 0]); tt = t[m]
            ax = axes[row, 0]
            ax.plot(tt, ref[:, 0], color="0.6", lw=1.0, ls="--", label="mocap 오른쪽")
            ax.plot(tt, ref[:, 1], color="0.6", lw=1.0, ls=":", label="mocap 왼쪽")
            ax.plot(tt, hp[:, 0], color=Style.ORANGE, lw=1.8, label="착용자 오른쪽")
            ax.plot(tt, hp[:, 1], color=Style.TEAL, lw=1.8, label="착용자 왼쪽")
            ax.set_title(f"{tag} — 고관절 굴곡 각도", loc="left", fontsize=11)
            ax.set_ylabel("각도 [deg]"); ax.set_ylim(-45, 60); ax.grid(alpha=.25)
            if row == 1:
                ax.set_xlabel("시간 [s]")
            ax.legend(ncol=4, fontsize=8, loc="upper right", framealpha=.9)

            # 가동범위 막대
            m2 = t >= 2.0
            hp2 = np.degrees(tr["hip_pos"][m2, 0]); ref2 = np.degrees(tr["ref_hip"][m2, 0])
            rom = [hp2[:, i].max() - hp2[:, i].min() for i in range(2)]
            rr = [ref2[:, i].max() - ref2[:, i].min() for i in range(2)]
            axb = axes[row, 1]
            x = np.arange(2)
            axb.bar(x - 0.2, rr, 0.36, color="0.75", label="mocap")
            axb.bar(x + 0.2, rom, 0.36, color=[Style.ORANGE, Style.TEAL], label="착용자")
            axb.set_xticks(x); axb.set_xticklabels(["오른쪽", "왼쪽"])
            axb.set_ylim(0, 80); axb.set_ylabel("가동범위 [deg]"); axb.grid(alpha=.25, axis="y")
            axb.set_title(f"좌우 차이 {abs(rom[0]-rom[1]):.1f}°  (mocap {abs(rr[0]-rr[1]):.1f}°)",
                          loc="left", fontsize=10)
            for xi, v in zip(x + 0.2, rom):
                axb.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=9)

        fig.suptitle("보조 없는 상태의 착용자 보행 — 관절각 추종 항 추가 전/후", fontsize=13, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=130)
        print("saved", out)


class TimingDiagram:
    """예측 범위 ≥ 갱신 주기 + 연산 시간 + 원하는 선행 — 그림으로."""

    @staticmethod
    def render(out="docs/ai_sbc/hlp_llp_timing.png"):
        plt = Style.pyplot()
        from matplotlib.patches import Rectangle

        INK, SOFT, TEAL, ORANGE, GREY = Style.INK, Style.SOFT, Style.TEAL, Style.ORANGE, Style.GREY
        T0, T1 = 85, 185
        fig, ax = plt.subplots(figsize=(13.5, 6.6))
        ax.set_xlim(T0, T1); ax.set_ylim(0, 10.4); ax.axis("off")

        # 시간축
        ax.annotate("", xy=(T1, 0.7), xytext=(T0, 0.7), arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
        for t in range(90, 181, 10):
            ax.plot([t, t], [0.55, 0.85], color=INK, lw=1)
            ax.text(t, 0.15, f"{t}", ha="center", fontsize=9, color=SOFT)
        ax.text(T1, 0.15, "ms", ha="right", fontsize=9, color=SOFT)

        def lane(y, label, sub=""):
            ax.text(T0-1.0, y, label, ha="right", va="center", fontsize=10.5, color=INK, fontweight="bold")
            if sub:
                ax.text(T0-1.0, y-0.42, sub, ha="right", va="center", fontsize=8.5, color=SOFT)

        # --- 1. 물리/센서 (2.5 ms) ---
        lane(9.4, "물리 substep", "2.5 ms")
        for t in [T0 + 2.5*i for i in range(int((T1-T0)/2.5)+1)]:
            ax.plot([t, t], [9.15, 9.65], color=GREY, lw=0.8)

        # --- 2. LLP (10 ms) ---
        lane(8.0, "LLP 추론", "10 ms · 0.25 ms 연산")
        for t in range(90, 181, 10):
            ax.add_patch(Rectangle((t, 7.7), 1.2, 0.6, facecolor=TEAL, edgecolor="none"))
            ax.text(t+0.6, 8.55, "τ", ha="center", fontsize=8, color=TEAL)

        # --- 3. HLP 주기 A/B ---
        lane(6.0, "HLP 연산", "10 ms마다 · 5 ms")
        for t0, col in ((100, ORANGE), (110, "#7c3aed"), (120, "#0369a1")):
            ax.add_patch(Rectangle((t0, 5.7), 5, 0.6, facecolor=col, edgecolor="none", alpha=.85))
            ax.text(t0+2.5, 6.55, f"입력 마감 {t0}", ha="center", fontsize=8, color=col)
            ax.plot([t0, t0], [0.7, 5.7], color=col, lw=0.7, ls=":", alpha=.6)
            ax.annotate("", xy=(t0+5, 5.35), xytext=(t0+5, 5.75),
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.4))
            ax.text(t0+5.4, 5.2, f"도착 {t0+5}", fontsize=8, color=col, va="center")

        # --- 4. chunk 유효 구간 ---
        lane(3.6, "chunk 유효 범위", "기준시각 +0…+50 ms")
        for i, (t0, col, lbl) in enumerate(((100, ORANGE, "A"), (110, "#7c3aed", "B"), (120, "#0369a1", "C"))):
            y = 3.9 - i*0.75
            ax.add_patch(Rectangle((t0, y-0.22), 50, 0.44, facecolor=col, alpha=.18, edgecolor=col, lw=1))
            for k in range(6):
                ax.plot([t0+10*k], [y], marker="|", color=col, ms=9, mew=1.6)
            ax.text(t0+50+1.0, y, f"chunk {lbl}", fontsize=9, color=col, va="center")
            ax.plot([t0, t0+5], [y, y], color=col, lw=6, alpha=.35)   # 아직 도착 전 구간
            ax.text(t0+2.5, y+0.30, "연산중", fontsize=7, color=col, ha="center")

        # --- 5. 워스트 케이스 표시 ---
        ax.annotate("", xy=(130, 1.55), xytext=(110, 1.55),
                    arrowprops=dict(arrowstyle="<->", color="#b45309", lw=1.3))
        ax.text(120, 1.70, "가장 불리한 순간: t=110에서 chunk A로 130까지 봐야 한다 (B는 115에 도착)",
                ha="center", fontsize=9, color="#b45309")

        ax.text(T0, 10.1,
                "예측 범위 ≥ 갱신 주기(10) + 연산 시간(5) + 원하는 선행(20) = 35 ms  →  10 ms 격자 6점 = 50 ms (여유 15 ms)",
                fontsize=11.5, color=INK, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out, dpi=140, bbox_inches="tight")
        print("saved", out)


class TorquePhasePlot:
    """최대 굴곡을 위상 0 %로 잡고 주기 평균 — 보조가 파워(|τ·ω|)를 목적으로 하므로 착용자 토크가 가장 큰
    순간이 아니라 파워가 큰 순간에 토크를 쓴다는 것을 보여 준다."""

    N = 100

    @classmethod
    def cycle_average(cls, x, peaks, dt):
        out = []
        for a, b in zip(peaks[:-1], peaks[1:]):
            if not (0.7 < (b - a) * dt < 1.6):
                continue
            out.append(np.interp(np.linspace(0, 1, cls.N), np.linspace(0, 1, b - a), x[a:b]))
        return np.array(out).mean(0)

    @classmethod
    def render(cls, run=None, out="docs/ai_sbc/torque_phase.png"):
        plt = Style.pyplot()
        from scipy.signal import find_peaks

        N = cls.N
        run = run or sorted(glob.glob("results/gait_eval/*_v10_m0"))[-1]
        tr = np.load(str(run) + "/traces.npz")
        t = tr["t"]
        dt = float(t[1] - t[0])
        m = t >= 2.0
        q = tr["hip_pos"][m, :, 0]
        w = tr["hip_vel"][m, :, 0]
        tau = (tr["tau_pd_prev"] + tr["tau_ff_prev"])[m, :, 0]
        assist = tr["tau_assist"][m, 1, 0]
        pk, _ = find_peaks(q[:, 0], distance=int(0.7 / dt))

        Q = cls.cycle_average(np.degrees(q[:, 0]), pk, dt)
        W = cls.cycle_average(w[:, 0], pk, dt)
        TA = cls.cycle_average(tau[:, 0], pk, dt)
        TB = cls.cycle_average(tau[:, 1], pk, dt)
        S = cls.cycle_average(assist, pk, dt)
        PA = cls.cycle_average(np.abs(tau[:, 0] * w[:, 0]), pk, dt)
        x = np.arange(N)

        fig, ax = plt.subplots(3, 1, figsize=(11.5, 8.2), sharex=True,
                               gridspec_kw={"height_ratios": [1, 1.5, 1]})
        INK, TEAL, ORANGE, GREY = Style.INK, Style.TEAL, Style.ORANGE, Style.GREY

        ax[0].plot(x, Q, color=INK, lw=2, label="고관절 각도 (+굴곡)")
        ax[0].plot(x, np.degrees(W) / 10, color=GREY, lw=1.4, ls="--", label="각속도 ÷10 [deg/s]")
        ax[0].axhline(0, color=GREY, lw=.6)
        ax[0].set_ylabel("각도 [deg]")
        ax[0].legend(fontsize=8.5, ncol=2, loc="upper right")

        ax[1].plot(x, TA, color=ORANGE, lw=2, label="착용자 토크 (보조 없음)")
        ax[1].plot(x, TB, color=ORANGE, lw=1.2, ls=":", label="착용자 토크 (보조 켬)")
        ax[1].plot(x, S, color=TEAL, lw=2.2, label="보조 토크")
        ax[1].axhline(0, color=GREY, lw=.6)
        ax[1].axhline(40, color=TEAL, lw=.8, ls="--")
        ax[1].axhline(-40, color=TEAL, lw=.8, ls="--")
        ax[1].text(99, 41, "보조 한계 ±40 N·m", ha="right", fontsize=8, color=TEAL)
        ax[1].set_ylabel("토크 [N·m]")
        ax[1].legend(fontsize=8.5, ncol=3, loc="lower right")

        ax[2].fill_between(x, 0, PA, color=ORANGE, alpha=.25)
        ax[2].plot(x, PA, color=ORANGE, lw=1.8, label="착용자 파워 |τ·ω| (보조 없음)")
        ax[2].plot(x, np.abs(S) * 2, color=TEAL, lw=1.4, ls="--", label="|보조 토크| ×2 [N·m]")
        ax[2].set_ylabel("파워 [W]")
        ax[2].set_xlabel("보행 주기 [%] — 0 % = 최대 굴곡")
        ax[2].legend(fontsize=8.5, loc="upper right")

        i = TA.argmin()
        for a in ax:
            a.axvspan(max(i - 4, 0), i + 4, color="#fde68a", alpha=.45, zorder=0)
            a.grid(alpha=.25)
        ax[1].annotate(f"착용자 토크 음의 피크 {TA[i]:.0f} N·m\n하지만 ω≈{W[i]:+.1f} rad/s → 파워 {PA[i]:.0f} W\n보조는 {S[i]:+.0f} N·m",
                       xy=(i, TA[i]), xytext=(i + 6, TA[i] - 20), fontsize=9, color=INK,
                       arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))

        fig.suptitle("보조 토크는 착용자 토크가 큰 곳이 아니라 파워가 큰 곳에서 나온다", fontsize=13, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out, dpi=135)
        print("saved", out)


class OracleComparisonPlot:
    """compare의 τ² 변화 막대 + 위상별 τ² 곡선 (2026-09-15 오라클 진단 그림)."""

    @staticmethod
    def render(rows, runs, out_png: str):
        plt = Style.pyplot()
        INK, TEAL, ORANGE = Style.INK, Style.TEAL, Style.ORANGE

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
            pt = ObjectiveComparison.phase_tau2(run)
            if pt is None:
                continue
            style = dict(lw=2.4) if row["run"] in ("base", "beta05") else dict(lw=1.3, ls="--")
            ax[1].plot(x, pt, marker="o", ms=3.5, label=row["run"], **style)
        ax[1].axhline(0, color=INK, lw=1)
        ax[1].axvspan(0, 20, color="#fde68a", alpha=0.45, zorder=0)
        top = 60.0
        ax[1].set_ylim(-95, top)
        for run, row in zip(runs, rows):  # 잘린 점은 값을 적어 둔다
            pt = ObjectiveComparison.phase_tau2(run)
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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("gait-symmetry")
    s.add_argument("--out", default="results/sym_check/symmetry_compare.png")
    s = sub.add_parser("timing")
    s.add_argument("--out", default="docs/ai_sbc/hlp_llp_timing.png")
    s = sub.add_parser("torque-phase")
    s.add_argument("run", nargs="?", default=None)
    s.add_argument("--out", default="docs/ai_sbc/torque_phase.png")
    s = sub.add_parser("compare", help="실행 여러 개의 A/B 표 (compare_runs) + 선택적 그림")
    s.add_argument("--plot", default=None, help="그림 PNG 경로")
    s.add_argument("runs", nargs="+")
    args = p.parse_args(argv)
    if args.command == "gait-symmetry":
        GaitSymmetryPlot.render(out=args.out)
    elif args.command == "timing":
        TimingDiagram.render(args.out)
    elif args.command == "torque-phase":
        TorquePhasePlot.render(args.run, args.out)
    else:
        runs = [Path(r) for r in args.runs if (Path(r) / "traces.npz").exists()]
        rows = [ObjectiveComparison.metrics(r) for r in runs]
        ObjectiveComparison.print_table(rows)
        if args.plot:
            OracleComparisonPlot.render(rows, runs, args.plot)


if __name__ == "__main__":
    main()
