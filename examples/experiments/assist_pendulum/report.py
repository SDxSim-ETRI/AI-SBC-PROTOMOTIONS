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
"""그래프 위주 보고서: LLP 20 N·m · 단일 관절 학습·양다리 가중치 공유 · 실제 동작 의도와 A·Δ 2×2 비교 (`MotionLegReport`).

저장된 기록만 읽는다 (`study.py`의 연구 폴더). W1 지표는 2026-10-21 일반화 보고서와 같은 코드
(스켈레톤 `report_generalization.digest`)로 계산하고, 저부담 구간은 같은 조건·평가 난수의 L1 ±20 무보조
기록으로 정한다. 생성 뒤 `python docs/ai_sbc/organize_reports.py`를 실행한다.

    python examples/experiments/assist_pendulum/report.py --date 261022 [--study results/llp20_motion_20261021]
"""

import argparse
from datetime import datetime, timezone
import glob
from html import escape
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/experiments/assist_pendulum"))
from study import DEFAULT_STUDY  # noqa: E402


def _load_skeleton(name):
    """스켈레톤 쪽 모듈을 파일로 읽는다 — 이 폴더에도 report.py/evaluate.py가 있어 이름으로 import하면 자기 자신이 잡힌다."""
    import importlib.util

    path = ROOT / "examples/experiments/assist_skeleton" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"skeleton_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"skeleton_{name}"] = module
    spec.loader.exec_module(module)
    return module


_analysis = _load_skeleton("analysis")
_report = _load_skeleton("report")
DecelerationMasks, WalkingPhaseDigest, read = _analysis.DecelerationMasks, _analysis.WalkingPhaseDigest, _analysis.read_json
Figures, HtmlShell, WalkingRecords = _report.Figures, _report.HtmlShell, _report.WalkingRecords
CSS = HtmlShell.STUDY_CSS
curve, digest = WalkingPhaseDigest.curve, WalkingPhaseDigest.digest
masks_from_reference, read_traces = DecelerationMasks.masks_from_reference, DecelerationMasks.read_traces
svg = Figures.embed
plt = Figures.pyplot()


class MotionLegReport:
    SEEDS = (42, 7)
    W1_CONDITIONS = ("walk", "walk_seg400", "run")
    SHORT = {"walk": "기존 걷기", "walk_seg400": "같은 걷기 다른 구간", "run": "달리기 2.59 m/s"}
    NEW = ("A0D0", "A1D1", "A1D0", "A0D1")
    BASE = ("L1_20", "L1_15", "L1_10", "R0_20", "R0_15")
    NAMES = {"L1_20": "L1 ±20", "L1_15": "L1 ±15", "L1_10": "L1 ±10", "R0_20": "R0 ±20", "R0_15": "R0 ±15",
             "A0D0": "새 · A·Δ 없음", "A1D1": "새 · A·Δ 모두", "A1D0": "새 · A만", "A0D1": "새 · Δ만"}
    COLORS = {"L1_20": "#2f3640", "L1_15": "#7b8794", "L1_10": "#a7b1bc", "R0_20": "#245aa0", "R0_15": "#6f9bd1",
              "A0D0": "#c2410c", "A1D1": "#1a8b67", "A1D0": "#8b70b2", "A0D1": "#b8860b"}
    PHASE_BINS = 25
    NEED = ["0–5", "5–10", "10–20", "20–40", "40–80", "80+"]
    OMEGA = [0, 1, 2, 3, 4, 5, 6, 8]
    SEED_GROUPS = ("running", "jogging", "jumping", "walking", "deep", "other")
    SEED_GROUP_NAMES = {"running": "달리기", "jogging": "조깅", "jumping": "점프", "walking": "걷기",
                        "deep": "앉기·일어서기·무릎 꿇기", "other": "기타"}
    ETRI_GROUPS = ("run", "walk", "squat", "lunge", "highknee", "stepinplace", "onestep", "hop")
    ETRI_NAMES = {"run": "달리기", "walk": "걷기", "squat": "스쿼트", "lunge": "런지", "highknee": "무릎 높이 들기",
                  "stepinplace": "제자리 걷기", "onestep": "한 걸음", "hop": "한 발 뛰기"}
    STORED_PENDULUM = {"L1_20": "results/pendulum_general23_20261015_all/L1_test_H1_{c}/summary.json",
                       "R0_20": "results/llp_relative_20261020/pendulum/R0_s0_H1_{c}/summary.json"}
    SYNTHETIC = ("legacy", "phase", "hold50")
    SYNTH_NAMES = {"legacy": "합성 궤적", "phase": "합성 + 위상 외력", "hold50": "합성 + 50 ms 명령 유지"}

    def __init__(self, study_dir: str = DEFAULT_STUDY):
        self.study = (ROOT / study_dir).resolve()

    # ---------------------------------------------------------------------------
    # 기록 모으기
    # ---------------------------------------------------------------------------
    def w1_records(self):
        """(조건, 모델, seed) → 기록, 저역통과 민감도 기록 따로."""
        out, lpf = {}, {}
        for (c, m, lim, s), r in WalkingRecords.gather().items():
            if c in self.W1_CONDITIONS:
                out[c, f"{m}_{lim}", s] = r
        runs = read(self.study / "eval_runs.json") if (self.study / "eval_runs.json").exists() else {}
        for identity, r in runs.items():
            if r["kind"] != "w1":
                continue
            # w1_<조건>[_lpf10]_<M_조건이름 | L1 | R0>_seed<seed>
            body, seed = identity[len("w1_"):].rsplit("_seed", 1)
            if "_M_" in body:
                body, model = body.rsplit("_M_", 1)
            else:
                body, model = body.rsplit("_", 1)
                model = f"{model}_20"
            is_lpf = body.endswith("_lpf10")
            cond = body[: -len("_lpf10")] if is_lpf else body
            (lpf if is_lpf else out)[cond, model, int(seed)] = r
        return out, lpf

    @staticmethod
    def command_stats(record):
        """W1 보조 B의 명령 변화량과 10 Hz 초과 성분 비율."""
        with np.load(ROOT / record["path"] / "traces.npz") as z:
            t, tau = z["t"], z["tau_assist"][:, 1]
        keep = t >= 1.5
        x = tau[keep]
        dt = float(np.median(np.diff(t)))
        change = np.abs(np.diff(x, axis=0))
        centred = x - x.mean(axis=0, keepdims=True)
        power = np.abs(np.fft.rfft(centred, axis=0)) ** 2
        freq = np.fft.rfftfreq(len(x), d=dt)
        return dict(change_mean=float(change.mean()), change_p99=float(np.percentile(change, 99)),
                    high_freq=float(power[freq > 10].sum() / power[1:].sum()))

    def omega_bins(self, record):
        with np.load(ROOT / record["path"] / "traces.npz") as z:
            t, vel, tau = z["t"], z["hip_vel"], z["tau_assist"][:, 1]
        idx = np.arange(1, len(t))
        keep = t[idx] >= 1.5
        sa = np.abs(.5 * (vel[idx - 1, 0] + vel[idx, 0]))[keep]
        sb = np.abs(.5 * (vel[idx - 1, 1] + vel[idx, 1]))[keep]
        assist = tau[idx - 1][keep]
        rows = []
        for lo, hi in zip(self.OMEGA[:-1], self.OMEGA[1:]):
            sel = (sb >= lo) & (sb < hi)
            rows.append(dict(lo=lo, hi=hi, share=float(((sa >= lo) & (sa < hi)).mean()),
                             assist_rms=float(np.sqrt(np.mean(assist[sel] ** 2))) if sel.sum() > 50 else None))
        return rows

    def collect_w1(self):
        raw, raw_lpf = self.w1_records()
        runs, lpf = {}, {}
        for c in self.W1_CONDITIONS:
            for s in self.SEEDS:
                if (c, "L1_20", s) not in raw:
                    continue
                reference = masks_from_reference(read_traces(raw[c, "L1_20", s]))
                for m in self.BASE + self.NEW:
                    if (c, m, s) in raw:
                        d = digest(raw[c, m, s], reference)
                        d.update(command=self.command_stats(raw[c, m, s]), omega_new=self.omega_bins(raw[c, m, s]))
                        runs[c, m, s] = d
                    if (c, m, s) in raw_lpf:
                        d = digest(raw_lpf[c, m, s], reference)
                        d.update(command=self.command_stats(raw_lpf[c, m, s]))
                        lpf[c, m, s] = d
        return runs, lpf

    def matched(self, runs, c, m, s):
        l1 = [runs[c, f"L1_{lim}", s] for lim in (10, 15, 20)]
        r = runs[c, m, s]
        at_rms = curve([x["assist_rms"] for x in l1], [x["benefit"] for x in l1], r["assist_rms"])
        at_harm = None
        if r["low10"] is not None and all(x["low10"] is not None for x in l1):
            at_harm = curve([x["low10"] for x in l1], [x["benefit"] for x in l1], r["low10"])
        top = runs[c, "L1_20", s]
        return dict(rms_diff=None if at_rms is None else r["benefit"] - at_rms,
                    rms_pct=None if not at_rms else 100 * (r["benefit"] - at_rms) / at_rms,
                    harm_diff=None if at_harm is None else r["benefit"] - at_harm,
                    harm_pct=None if not at_harm else 100 * (r["benefit"] - at_harm) / at_harm,
                    # L1 곡선 범위(±20까지) 밖이면 L1 ±20과 직접 비교한다
                    over_rms=r["assist_rms"] - top["assist_rms"], vs_top=r["benefit"] - top["benefit"],
                    vs_top_pct=100 * (r["benefit"] - top["benefit"]) / top["benefit"])

    @staticmethod
    def matched_text(rows):
        """같은 보조 RMS의 L1 곡선 대비 [%]; 범위 밖이면 L1 ±20 직접 비교 문구."""
        pct = [x["rms_pct"] for x in rows]
        if all(v is not None for v in pct):
            return f"{np.mean(pct):+.0f} %"
        return (f"범위 밖: L1 ±20보다 보조 RMS {np.mean([x['over_rms'] for x in rows]):+.1f} N·m, "
                f"이득 {np.mean([x['vs_top_pct'] for x in rows]):+.0f} %")

    @staticmethod
    def avg(values):
        values = [v for v in values if v is not None]
        return (float(np.mean(values)), float(min(values)), float(max(values))) if values else (None, None, None)

    def w1_mean(self, runs, c, m, key, idx=None):
        vals = []
        for s in self.SEEDS:
            if (c, m, s) in runs:
                v = runs[c, m, s][key]
                vals.append(v if idx is None else v[idx])
        return self.avg(vals) if len(vals) == len(self.SEEDS) else (None, None, None)

    def w1_mean_command(self, runs, c, m):
        if not all((c, m, s) in runs for s in self.SEEDS):
            return None, None
        return (float(np.mean([runs[c, m, s]["command"]["change_mean"] for s in self.SEEDS])),
                float(np.mean([runs[c, m, s]["command"]["high_freq"] for s in self.SEEDS])))

    def lpf_benefit(self, runs, lpf, c, m):
        """(원래 이득, 10 Hz 저역통과 뒤 이득) — 평가 난수 평균."""
        if not all((c, m, s) in lpf and (c, m, s) in runs for s in self.SEEDS):
            return self.w1_mean(runs, c, m, "benefit")[0], None
        return (float(np.mean([runs[c, m, s]["benefit"] for s in self.SEEDS])),
                float(np.mean([lpf[c, m, s]["benefit"] for s in self.SEEDS])))

    def pendulum_summaries(self):
        out = {}
        for path in glob.glob(str(self.study / "pendulum/*/summary.json")):
            name = Path(path).parent.name  # pendulum_<cond>_<model>
            body = name[len("pendulum_"):]
            for cond in (*self.SYNTHETIC, "seed_val", "etri"):
                if body.startswith(cond + "_"):
                    model = body[len(cond) + 1:]
                    model = model[2:] if model.startswith("M_") else model
                    model = {"L1": "L1_20", "R0": "R0_20"}.get(model, model)
                    out[cond, model] = read(path)
        for model, pattern in self.STORED_PENDULUM.items():
            for c in self.SYNTHETIC:
                out[c, model] = read(ROOT / pattern.format(c=c))
        return out

    def training_curves(self):
        """조건·다리별 epoch 곡선 (보조 RMS, 착용자 τ², 명령 변화)."""
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        date = self.study.name.rsplit("_", 1)[-1]
        out = {}
        for c in self.NEW:
            for leg, seed in (("right", 0),):  # 왼쪽 줄은 사용자 결정으로 중지 (decision_weight_sharing.json)
                files = sorted(glob.glob(str(ROOT / f"results/llp20_motion_{c}_{leg}_s{seed}_{date}/lightning_logs/*/events*")))
                if not files:
                    continue
                series = {}
                for f in files:
                    ea = EventAccumulator(f, size_guidance={"scalars": 0})
                    ea.Reload()
                    for tag in ("env/raw_r/assist_effort_mean", "env/raw_r/human_torque_sq_mean",
                                "env/raw_r/action_rate_mean", "env/raw_r/tracking_mean"):
                        if tag in ea.Tags()["scalars"]:
                            for e in ea.Scalars(tag):
                                series.setdefault(tag, {})[e.step] = e.value
                out[c, leg] = {k: (np.array(sorted(v)), np.array([v[x] for x in sorted(v)])) for k, v in series.items()}
        return out

    def load_diagnostics(self):
        diag = {}
        for f in glob.glob(str(self.study / "diagnostics/chunk_*.json")):
            name, source = Path(f).stem[len("chunk_"):].rsplit("_", 1)
            diag[{"L1": "L1_20", "R0": "R0_20"}.get(name, name), source] = read(f)["pools"]
        return diag

    # ---------------------------------------------------------------------------
    # 그림
    # ---------------------------------------------------------------------------
    def fig_phase_core(self, runs, models):
        """보행 위상별 착용자 고관절 토크 RMS 감소량 (무보조 A − 보조 B, 양수 = 도움). 위 행부터 걷기·다른 구간·달리기."""
        SEEDS, PHASE_BINS, COLORS, NAMES = self.SEEDS, self.PHASE_BINS, self.COLORS, self.NAMES
        conds = [c for c in self.W1_CONDITIONS if all((c, m, sd) in runs for m in models for sd in SEEDS)]
        x = (np.arange(PHASE_BINS) + .5) * 100 / PHASE_BINS
        fig, axes = plt.subplots(len(conds), 2, figsize=(12, 3.5 * len(conds)), squeeze=False, sharex=True)
        for i, c in enumerate(conds):
            series = {m: [np.array([np.mean([runs[c, m, sd]["rows"][side][k].get("reduction", np.nan) for sd in SEEDS])
                                    for k in range(PHASE_BINS)]) for side in (0, 1)] for m in models}
            high = max(np.nanmax(v) for vs in series.values() for v in vs) + 2
            low = min(np.nanmin(v) for vs in series.values() for v in vs) - 2
            for side in (0, 1):
                ax = axes[i, side]
                ax.axhspan(low, 0, color="#c64448", alpha=.07)
                ax.axhline(0, color="#64748b", lw=.8)
                for m in models:
                    ax.plot(x, series[m][side], "--" if m == "L1_20" else "-", lw=2.2 if m in ("L1_20", "R0_20") else 1.6,
                            color=COLORS[m], label=NAMES[m])
                ax.set_title(f"{self.SHORT[c]} · {('오른쪽', '왼쪽')[side]} 다리", fontsize=10)
                ax.set_xlim(0, 100)
                ax.set_xticks(np.arange(0, 101, 20))
                ax.set_ylim(low, high)
                ax.grid(alpha=.15)
                if side == 0:
                    ax.set_ylabel("착용자 토크 RMS 감소량 A − B [N·m]\n위 = 도움 · 아래(빨강) = 방해", fontsize=9)
        for ax in axes[-1]:
            ax.set_xlabel("참조 보행 위상 [%] · 최대 굴곡 = 0 %")
        axes[0, 1].legend(fontsize=8, frameon=False, loc="best")
        fig.tight_layout()
        return svg(fig, "phase_core")

    def fig_curves(self, runs, models, ykey, xkey, ylabel, xlabel, identity):
        SEEDS, COLORS, NAMES = self.SEEDS, self.COLORS, self.NAMES
        conds = [c for c in self.W1_CONDITIONS if all((c, f"L1_{lim}", s) in runs for lim in (10, 15, 20) for s in SEEDS)]
        fig, axes = plt.subplots(1, len(conds), figsize=(4.3 * len(conds), 3.6), squeeze=False)
        for ax, c in zip(axes[0], conds):
            pts = [(self.w1_mean(runs, c, f"L1_{lim}", xkey)[0], self.w1_mean(runs, c, f"L1_{lim}", ykey)[0]) for lim in (10, 15, 20)]
            ax.plot([0] + [p[0] for p in pts], [0] + [p[1] for p in pts], "--", color=COLORS["L1_20"], lw=1.2, label="L1 출력만 줄인 곡선")
            for m in ("L1_10", "L1_15", "L1_20", "R0_20") + tuple(models):
                x, y = self.w1_mean(runs, c, m, xkey), self.w1_mean(runs, c, m, ykey)
                if x[0] is None or y[0] is None:
                    continue
                ax.errorbar(x[0], y[0], xerr=[[x[0] - x[1]], [x[2] - x[0]]], yerr=[[y[0] - y[1]], [y[2] - y[0]]],
                            fmt="o" if m.startswith("L1") else ("s" if m.startswith("R0") else "D"), color=COLORS[m],
                            ms=6, capsize=2, lw=1, label=NAMES[m] if m not in ("L1_10", "L1_15") else None)
            ax.set_title(self.SHORT[c], fontsize=10)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=.25)
            ax.axhline(0, color="#888", lw=.6)
        axes[0, -1].legend(fontsize=7.5, frameon=False, loc="best")
        fig.tight_layout()
        return svg(fig, identity)

    def fig_need(self, runs, models):
        SEEDS, COLORS, NAMES = self.SEEDS, self.COLORS, self.NAMES
        conds = [c for c in ("walk", "run") if all((c, m, s) in runs for m in models for s in SEEDS)]
        fig, axes = plt.subplots(2, len(conds), figsize=(5.2 * len(conds), 6), squeeze=False, sharey="row")
        for j, c in enumerate(conds):
            for m in models:
                per = [runs[c, m, s]["need"] for s in SEEDS]
                rms = np.mean([[b["assist_rms_Nm"] for b in p] for p in per], axis=0)
                sign = np.mean([[100 * b["sign_agreement"] for b in p] for p in per], axis=0)
                x = np.arange(len(rms))
                axes[0, j].plot(x, rms, "o-", color=COLORS[m], lw=1.4, ms=4, label=NAMES[m])
                axes[1, j].plot(x, sign, "o-", color=COLORS[m], lw=1.4, ms=4)
            for i in (0, 1):
                axes[i, j].set_xticks(np.arange(6), self.NEED, fontsize=8)
                axes[i, j].grid(alpha=.25)
            axes[1, j].axhline(50, color="#c2410c", lw=.8, ls=":")
            axes[0, j].set_title(self.SHORT[c], fontsize=10)
            axes[1, j].set_xlabel("무보조 착용자 |토크| [N·m]", fontsize=9)
        axes[0, 0].set_ylabel("보조 RMS [N·m]")
        axes[1, 0].set_ylabel("보조 방향 일치 [%]")
        axes[0, 0].legend(fontsize=8, frameon=False)
        fig.tight_layout()
        return svg(fig, "need_bins")

    def fig_omega(self, runs, models):
        SEEDS, COLORS, NAMES, OMEGA = self.SEEDS, self.COLORS, self.NAMES, self.OMEGA
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
        x = np.arange(len(OMEGA) - 1)
        labels = [f"{lo}–{hi}" for lo, hi in zip(OMEGA[:-1], OMEGA[1:])]
        for i, c in enumerate(("walk", "run")):
            for m in models:
                if not all((c, m, s) in runs for s in SEEDS):
                    continue
                vals = np.array([[np.nan if r["assist_rms"] is None else r["assist_rms"] for r in runs[c, m, s]["omega_new"]]
                                 for s in SEEDS], dtype=float).mean(axis=0)
                ax[i].plot(x, vals, "o-", color=COLORS[m], lw=1.4, ms=4, label=NAMES[m])
            share = np.mean([[r["share"] for r in runs[c, "L1_20", s]["omega_new"]] for s in SEEDS], axis=0)
            twin = ax[i].twinx()
            twin.bar(x, 100 * share, color="#d9dee5", alpha=.6, zorder=0)
            twin.set_ylabel("무보조 시간 [%]", fontsize=8, color="#7b8794")
            ax[i].set_zorder(twin.get_zorder() + 1)
            ax[i].patch.set_visible(False)
            ax[i].set_xticks(x, labels)
            ax[i].set_xlabel("고관절 각속도 |ω| [rad/s]")
            ax[i].set_title(self.SHORT[c], fontsize=10)
            ax[i].grid(alpha=.25)
        ax[0].set_ylabel("보조 RMS [N·m]")
        ax[1].legend(fontsize=8, frameon=False)
        fig.tight_layout()
        return svg(fig, "omega")

    def fig_pendulum_groups(self, pend, models):
        COLORS, NAMES = self.COLORS, self.NAMES
        panels = (("seed_val", self.SEED_GROUPS, self.SEED_GROUP_NAMES, "SEED 미학습 동작"),
                  ("etri", self.ETRI_GROUPS, self.ETRI_NAMES, "ETRI 슈트 13동작"))
        fig, axes = plt.subplots(2, 1, figsize=(11, 7.4))
        for ax, (cond, groups, names, title) in zip(axes, panels):
            present = [m for m in models if (cond, m) in pend]
            width = .8 / max(len(present), 1)
            for k, m in enumerate(present):
                g = pend[cond, m]["extended"]["groups"]
                vals = [np.nan if g.get(n) is None else -g[n]["torque_sq_change_percent"] for n in groups]
                ax.bar(np.arange(len(groups)) + (k - (len(present) - 1) / 2) * width, vals, width, color=COLORS[m], label=NAMES[m])
            ax.set_xticks(np.arange(len(groups)), [names[n] for n in groups], fontsize=9)
            ax.set_ylabel("착용자 τ² 감소 [%]")
            ax.set_title(title, fontsize=10)
            ax.grid(axis="y", alpha=.25)
            ax.axhline(0, color="#444", lw=.7)
        axes[0].legend(fontsize=8, frameon=False, ncol=3)
        fig.tight_layout()
        return svg(fig, "pendulum_groups")

    def fig_training(self, curves):
        COLORS, NAMES = self.COLORS, self.NAMES
        tags = (("env/raw_r/assist_effort_mean", "보조 RMS [N·m]", lambda v: np.sqrt(np.maximum(v, 0))),
                ("env/raw_r/human_torque_sq_mean", "착용자 τ² [(N·m)²]", lambda v: v),
                ("env/raw_r/action_rate_mean", "명령 변화 |Δτ| [N·m / 10 ms]", lambda v: v))
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
        for ax, (tag, label, fn) in zip(axes, tags):
            for (c, leg), series in sorted(curves.items()):
                if tag not in series:
                    continue
                x, y = series[tag]
                k = max(1, len(y) // 200)
                ys = np.convolve(fn(y), np.ones(25) / 25, mode="valid") if len(y) > 50 else fn(y)
                xs = x[len(x) - len(ys):]
                ax.plot(xs[::k], ys[::k], "-", color=COLORS[c], lw=1.2, label=NAMES[c])
            ax.set_xlabel("epoch")
            ax.set_ylabel(label)
            ax.grid(alpha=.25)
        axes[0].legend(fontsize=7, frameon=False, ncol=1)
        fig.tight_layout()
        return svg(fig, "training")

    # ---------------------------------------------------------------------------
    # HTML
    # ---------------------------------------------------------------------------
    @staticmethod
    def fmt(v, spec=".1f"):
        return "—" if v is None else format(v, spec)

    @staticmethod
    def cell(t, spec=".1f"):
        m, lo, hi = t
        return "—" if m is None else f"<b>{format(m, spec)}</b><br><small>{format(lo, spec)} ~ {format(hi, spec)}</small>"

    def render(self, date: str) -> Path:
        STUDY, SEEDS, NEW, NAMES, SHORT = self.study, self.SEEDS, self.NEW, self.NAMES, self.SHORT
        W1_CONDITIONS, SYNTHETIC, SYNTH_NAMES = self.W1_CONDITIONS, self.SYNTHETIC, self.SYNTH_NAMES
        fmt, cell, avg, w1_mean = self.fmt, self.cell, self.avg, self.w1_mean
        out = ROOT / "docs" / "ai_sbc" / f"{date}_llp20_motion_shared" / "report.html"
        manifest = read(STUDY / "manifest.json")
        runs, lpf = self.collect_w1()
        pend = self.pendulum_summaries()
        curves = self.training_curves()
        models = [m for m in NEW if all((c, m, s) in runs for c in W1_CONDITIONS for s in SEEDS)]
        shown = ["L1_20", "R0_20"] + models
        match = {(c, m): [self.matched(runs, c, m, s) for s in SEEDS] for c in W1_CONDITIONS for m in ["R0_20"] + models}
        executions = {}
        for leg in ("right", "left"):
            f = STUDY / f"execution_{leg}.json"
            if f.exists():
                executions.update(read(f))
        audit = read(STUDY / "training_distribution_audit.json") if (STUDY / "training_distribution_audit.json").exists() else None
        etri_info = read(STUDY / "eval_data/etri_eval_data.json")

        def benefit_pair(c, m):
            return w1_mean(runs, c, m, "benefit")[0]

        body = ['<nav><a href="../reports.html">보고서 목록</a><a href="../experiment_summary.html">전체 요약</a>'
                '<a href="../261021_r0_generalization/report.html">R0 일반화</a><a href="../261020_relative/phase_report.html">R0 기존 걷기</a></nav>',
                '<h1>LLP 20 N·m · 단일 관절 학습·양다리 공유 · 실제 동작 궤적 — 보조 토크² 벌점(A)·명령 변화 벌점(Δ) 비교</h1>',
                '<p>R0 레시피를 세 가지만 바꿔 진자에서 다시 학습했다. (1) 출력 20·tanh, 한계 20 N·m, 평가 배율 1. '
                '(2) 사람의 의도 궤적에 SEED 실제 고관절 동작(걷기·조깅·달리기·점프·앉기/일어서기류)을 넣고, 달리기를 따라갈 수 있는 가벼운 다리 집단을 따로 둠. '
                '(3) 고관절 하나의 진자로 정책 하나를 학습하고, 평가·배포에서는 같은 가중치를 왼쪽·오른쪽 다리에 각각 적용(가중치 공유). 각 다리는 자기 다리의 관측만 본다. '
                '그 위에서 보조 토크² 벌점 A와 명령 변화 벌점 Δ를 켜고 끈 네 경우를 같은 학습 seed(0)로 학습했다. '
                'W1·H1·L1·R0는 그대로이며, L1·R0는 출력 ×0.5(±20 N·m)로 비교한다.</p>']

        # ---------- 결론 ----------
        diag = self.load_diagnostics()
        lines = []
        if models:
            for c in W1_CONDITIONS:
                vals = ", ".join(f"{NAMES[m]} {fmt(benefit_pair(c, m), '.0f')}" for m in ["L1_20", "R0_20"] + models)
                lines.append(f"<b>{SHORT[c]}</b> 고관절 τ² 감소 [(N·m)²]: {vals}.")
            if set(models) == set(NEW):
                sat = [w1_mean(runs, "walk", m, "saturation")[0] for m in NEW]
                lines.append(f"<b>새 모델 넷 모두 W1에서 L1 ±20보다 덜 줄인다.</b> A를 켠 두 모델은 걷기에서 착용자 부담을 오히려 늘렸다. "
                             f"새 모델은 보조를 거의 늘 한계까지 낸다: W1 걷기 포화 {min(sat):.0f}–{max(sat):.0f} % "
                             f"(L1 ±20 {w1_mean(runs, 'walk', 'L1_20', 'saturation')[0]:.0f} %, R0 ±20 {w1_mean(runs, 'walk', 'R0_20', 'saturation')[0]:.1f} %). "
                             f"달리기에서는 A·Δ 없음만 R0 ±20보다 크다.")
            if all((m, src) in diag for m in ["L1_20", "R0_20"] + list(NEW) for src in ("emulated", "h1")):
                def red(m, src):
                    return -diag[m, src]["all"]["torque_sq_change_percent"]

                def cor(m, src):
                    return diag[m, src]["all"]["assist_vs_unassisted_torque_corr"]
                em = [red(m, "emulated") for m in NEW]
                h1 = [red(m, "h1") for m in NEW]
                lines.append(f"<b>원인은 학습 때의 미래 목표 신호로 보인다.</b> 학습 분포 진자에서 학습과 같은 에뮬레이션 예측이면 새 모델이 착용자 τ²를 "
                             f"{min(em):.0f}–{max(em):.0f} % 줄여 R0 ×0.5({red('R0_20', 'emulated'):.0f} %)·L1 ×0.5({red('L1_20', 'emulated'):.0f} %)보다 크지만, "
                             f"예측만 실제 H1로 바꾸면 {min(h1):.0f}–{max(h1):.0f} %로 떨어진다(R0 {red('R0_20', 'h1'):.0f} %, L1 {red('L1_20', 'h1'):.0f} %). "
                             f"보조와 필요 토크의 상관은 새 모델 {min(cor(m, 'emulated') for m in NEW):.2f}–{max(cor(m, 'emulated') for m in NEW):.2f} → "
                             f"{min(cor(m, 'h1') for m in NEW):.2f}–{max(cor(m, 'h1') for m in NEW):.2f}, R0 {cor('R0_20', 'emulated'):.2f} → {cor('R0_20', 'h1'):.2f}. "
                             "에뮬레이션 예측은 의도 궤적에 고정돼 착용자의 추종 오차(필요 토크의 단서)를 담지만, 실제 H1은 실제 각도의 미래를 예측해 그 단서가 없다.")
            if set(models) == set(NEW) and all(("seed_val", m) in pend for m in NEW):
                walk = {m: w1_mean(runs, "walk", m, "benefit")[0] for m in NEW}
                seed = {m: -pend["seed_val", m]["extended"]["all"]["torque_sq_change_percent"] for m in NEW}
                cmd = {m: pend["legacy", m]["extended"]["command_change"]["mean_Nm_per_step"] for m in NEW if ("legacy", m) in pend}
                run_cmd = {m: self.w1_mean_command(runs, "run", m)[0] for m in NEW}
                lines.append("<b>A·Δ의 효과는 평가마다 방향이 다르다.</b> "
                             f"W1 걷기는 A를 끈 두 모델이 낫고(A·Δ 없음 {walk['A0D0']:.0f}, Δ만 {walk['A0D1']:.0f} vs A·Δ 모두 {walk['A1D1']:.0f}, A만 {walk['A1D0']:.0f}), "
                             f"진자 SEED 동작은 A를 켠 두 모델이 낫다({seed['A1D0']:.1f}, {seed['A1D1']:.1f} vs {seed['A0D0']:.1f}, {seed['A0D1']:.1f} %). "
                             f"A를 켜도 보조 크기는 거의 줄지 않았다(W1 걷기 보조 RMS {min(w1_mean(runs, 'walk', m, 'assist_rms')[0] for m in NEW):.1f}–{max(w1_mean(runs, 'walk', m, 'assist_rms')[0] for m in NEW):.1f} N·m). "
                             + (f"Δ는 진자 합성 궤적에서 10 ms당 명령 변화를 {np.mean([cmd['A0D0'], cmd['A1D0']]):.2f} → {np.mean([cmd['A1D1'], cmd['A0D1']]):.2f} N·m로 줄였지만, "
                                f"W1 달리기에서는 줄이지 못했다(Δ 끔 {run_cmd['A0D0']:.2f}·{run_cmd['A1D0']:.2f}, Δ 켬 {run_cmd['A1D1']:.2f}·{run_cmd['A0D1']:.2f}). " if len(cmd) == 4 else "")
                             + "학습 seed가 하나라 이 차이를 A·Δ의 효과로 단정하지 않는다.")
            if all(self.lpf_benefit(runs, lpf, "walk", m)[1] is not None for m in ["L1_20"] + models):
                pairs = ", ".join(f"{NAMES[m]} {self.lpf_benefit(runs, lpf, 'walk', m)[0]:.0f} → {self.lpf_benefit(runs, lpf, 'walk', m)[1]:.0f}"
                                  for m in ["L1_20", "R0_20"] + models)
                lines.append(f"<b>출력 10 Hz 저역통과는 모든 모델의 W1 이득을 늘렸다</b> (기존 걷기: {pairs}). Δ를 빼서 생긴 고주파 문제는 보이지 않았다.")
        else:
            lines.append("아직 W1 평가가 끝난 새 모델이 없다.")
        body.append('<div class="status">' + "<br>".join(lines) + "</div>")

        if models:
            body.append("<h2>보행 위상별 착용자 토크 감소량 (핵심 그래프)</h2>"
                        "<p>착용자 고관절 토크 RMS를 참조 보행 위상(그 다리의 최대 굴곡 = 0 %)의 4 % 구간마다 비교했다. "
                        "값은 무보조 A − 보조 B이므로 <b>위로 갈수록 보조가 착용자 토크를 더 줄인 구간</b>이고, 0 아래(빨간 영역)는 보조 때문에 착용자 토크가 늘어난 방해 구간이다. "
                        "평가 난수 42·7 평균이며, L1·R0는 ±20 N·m(×0.5), 새 모델은 배율 1이다. 이전 보고서(R0/R1)의 같은 그림은 부호가 반대(증가 = 위)였다.</p>")
            body.append(self.fig_phase_core(runs, shown))

        # ---------- 1. 학습 설정 ----------
        pools = manifest["recipe"]["intent_pools"]
        prow = []
        for name, frac, src, lo, hi, weights in pools:
            inertia = f"{0.2863 + 0.1316 * lo:.2f}–{0.2863 + 0.1316 * hi:.2f}"
            mix = "기존 합성 궤적" if src == "synthetic" else ", ".join(f"{w[0]} {100 * w[1]:.0f} %" for w in weights)
            realized = ""
            if audit:
                pa = next(x for x in audit["pools"] if x["name"] == name)
                realized = f"{100 * pa['fast_time_fraction']:.1f} %"
            prow.append(f"<tr><td>{escape(name)}</td><td>{100 * frac:.0f} %</td><td>{inertia}</td><td>{escape(mix)}</td><td>{realized}</td></tr>")
        body.append('<h2>1. 무엇을 바꿨나</h2>'
                    '<p>환경은 세 집단에 고정 배정된다. 각 다리가 따라갈 수 없는 동작(클립 각가속도 99 % 값 × 다리 관성 > 0.8 × 착용자 토크 한계 150 N·m)은 그 다리에서 뽑지 않는다. '
                    'R0의 관성 분포만으로는 달리기를 따라갈 수 있는 다리가 2 % 남짓이라 가벼운 다리 집단을 따로 두었다. 비율은 쓰이는 시간 기준이다.</p>'
                    '<div class="scroll"><table><tr><th>집단</th><th>env 비율</th><th>hip 축 관성 [kg·m²]</th><th>동작 비율 (시간)</th><th>의도 각속도 4 rad/s 초과 시간 (실측)</th></tr>'
                    + "".join(prow) + "</table></div>")
        if audit:
            body.append(f'<p class="note">학습 환경 점검(무작위 행동, {audit["num_envs"]} env × {audit["steps"]} step): 의도 각속도 4 rad/s 초과 시간 '
                        f'{100 * audit["overall_fast_time_fraction"]:.1f} % (기존 합성 궤적만 쓴 R0는 약 3 %), 의도 각속도 99 % 값 '
                        f'{audit["intent_speed_percentiles_rad_s"]["99"]:.1f} rad/s. 실현 불가능한 클립을 쓴 시간 {100 * audit["infeasible_current_clip_fraction"]:.1f} %, '
                        f'담당하지 않는 다리의 보조 최대 {audit["other_joint_assist_max_Nm"]:.1f} N·m, 보상 원시값과 담당 다리만의 직접 계산 차이 최대 '
                        f'{max(audit["reward_raw_max_abs_error"].values()):.1e}.</p>')
        crow = []
        for c in NEW:
            e = executions.get(f"{c}_right")
            state = f"완료 {e['elapsed_s'] / 3600:.1f} h" if e else "미완료"
            crow.append(f"<tr><td>{NAMES[c]}</td><td>{'−2e-5' if c[1] == '1' else '0'}</td><td>{'−0.005' if c[3] == '1' else '0'}</td>"
                        f"<td>{state}</td></tr>")
        body.append('<div class="scroll"><table><tr><th>경우</th><th>A 가중치</th><th>Δ 가중치</th><th>학습 (학습 seed 0)</th></tr>'
                    + "".join(crow) + "</table></div>"
                    '<p class="note">A = 보조 토크² 합 벌점, Δ = 10 ms 사이 보조 명령 변화 크기 벌점. 다른 보상(추종, 착용자 |τ·ω|, 착용자 τ²)은 R0와 같다. '
                    '학습 순서는 A·Δ 없음 → A·Δ 모두 → A만 → Δ만이며, 모든 실행은 6,000 epoch 마지막 모델을 쓴다. '
                    '학습 환경에는 고관절 두 개가 있지만 보조·보상·종료·관측은 오른쪽 관절 하나만 쓰므로(왼쪽 진자는 학습에 영향 없는 별도 진자) 단일 진자 학습과 같다. '
                    'SEED에는 좌우를 뒤집은 클립이 함께 있어 오른쪽 채널에 왼쪽 동작도 들어간다. 처음 함께 띄운 왼쪽 관절 학습(seed 1)은 사용자 결정으로 중지했다.</p>')
        if curves:
            body.append("<h3>학습 중 보조 크기·착용자 부담·명령 변화</h3><p>학습 환경(학습 관절)의 epoch별 평균이다.</p>")
            body.append(self.fig_training(curves))

        # ---------- 2. W1 ----------
        if models:
            body.append('<h2>2. W1 착용자: 같은 보조 크기에서 더 줄이는가</h2>'
                        '<p>점선은 L1의 출력만 ±10/15/20 N·m로 줄인 결과를 원점과 이은 곡선이다. 그보다 위의 점은 L1 출력을 줄여서는 얻을 수 없는 이득이다. '
                        '오차 막대는 평가 난수 42·7 두 번의 범위다.</p>')
            body.append(self.fig_curves(runs, models, "benefit", "assist_rms", "고관절 τ² 감소 [(N·m)²]", "보조 RMS [N·m]", "benefit_vs_rms"))
            body.append('<p>가로축을 저부담 감속 구간의 착용자 RMS 증가로 바꾼 그림이다. 왼쪽 위일수록 같은 방해로 더 많이 돕는다. 달리기는 이 구간이 1초도 안 되어 최악 위상 증가를 함께 본다.</p>')
            body.append(self.fig_curves(runs, models, "benefit", "low10", "고관절 τ² 감소 [(N·m)²]", "저부담 감속 RMS 증가 [N·m]", "benefit_vs_harm"))
            trows = []
            for c in W1_CONDITIONS:
                for i, m in enumerate(shown):
                    mt = match.get((c, m))
                    rms_txt = self.matched_text(mt) if mt else "—"
                    harm_pct = avg([x["harm_pct"] for x in mt])[0] if mt else None
                    falls = sum(runs[c, m, s]["fall"] is not None for s in SEEDS)
                    label = f'<td rowspan="{len(shown)}">{escape(SHORT[c])}</td>' if i == 0 else ""
                    trows.append(f"<tr>{label}<td>{NAMES[m]}</td><td>{cell(w1_mean(runs, c, m, 'benefit'), '.0f')}</td>"
                                 f"<td>{cell(w1_mean(runs, c, m, 'assist_rms'), '.1f')}</td><td>{cell(w1_mean(runs, c, m, 'saturation'))}</td>"
                                 f"<td>{cell(w1_mean(runs, c, m, 'low10'), '.2f')}</td><td>{cell(w1_mean(runs, c, m, 'worst'), '.2f')}</td>"
                                 f"<td>{rms_txt}</td><td>{fmt(harm_pct, '+.0f')}{'' if harm_pct is None else ' %'}</td>"
                                 f"<td>{falls}/2</td></tr>")
            body.append('<div class="scroll"><table><tr><th>조건</th><th>모델</th><th>고관절 τ² 감소 [(N·m)²]</th><th>보조 RMS [N·m]</th><th>포화 [%]</th>'
                        '<th>저부담 감속 RMS 증가 [N·m]</th><th>최악 위상 RMS 증가 [N·m]</th><th>같은 보조 RMS의 L1 곡선 대비</th><th>같은 손해의 L1 곡선 대비</th><th>낙상</th></tr>'
                        + "".join(trows) + "</table></div>"
                        '<p class="note">굵은 값은 평가 난수 두 번의 평균, 아래는 범위. τ² 감소 = 무보조 A와 보조 B의 착용자 고관절 토크 제곱 평균 차이(좌우 평균, 1.5초 이후). '
                        '포화 = 보조가 상한의 95 % 이상인 시간. L1·R0 기록은 2026-10-16/20/21 저장본을 같은 코드로 다시 계산했다. 토크²는 시뮬레이션 부담 지표다.</p>')

            body.append("<h2>3. 필요한 만큼, 빠를 때도 돕는가</h2>"
                        "<p>무보조 기록에서 착용자가 낸 토크 크기로 시간을 나눠 각 구간의 보조 RMS(위)와 보조 방향이 무보조 착용자 토크 방향과 같은 비율(아래)을 봤다. "
                        "R0는 달리기에서 필요 토크가 큰 구간의 보조가 오히려 작았다(2026-10-21).</p>")
            body.append(self.fig_need(runs, shown))
            body.append("<p>보조 착용자의 고관절 각속도별 보조 RMS다. 회색 막대는 그 각속도에 머문 무보조 시간 비율이다.</p>")
            body.append(self.fig_omega(runs, shown))

        # ---------- 4. 진자 ----------
        pmodels = [m for m in ["L1_20", "R0_20"] + list(NEW) if (("seed_val", m) in pend)]
        if pmodels:
            body.append("<h2>4. 진자: 학습하지 않은 실제 동작과 스쿼트·런지</h2>"
                        "<p>무보조 진자와 보조 진자를 같은 의도·관성·외란으로 나란히 돌리고, 보조 진자는 동결 H1의 실제 예측을 본다. "
                        "SEED는 H1·새 LLP 모두 학습하지 않은 클립이고, ETRI 13동작은 W1 평가와 같은 출처라 학습에 쓰지 않았다. "
                        "두 조건 모두 가벼운 다리 50 %·일반 50 % 집단, 같은 실현 가능성 규칙을 쓴다. 128 env × 10 s × 2회.</p>")
            body.append(self.fig_pendulum_groups(pend, pmodels))
            prow = []
            for cond, label in (("seed_val", "SEED 미학습 동작"), ("etri", "ETRI 13동작")):
                for i, m in enumerate(pmodels):
                    if (cond, m) not in pend:
                        continue
                    s = pend[cond, m]
                    e = s["extended"]
                    lab = f'<td rowspan="{len(pmodels)}">{label}<br><small>무보조 τ² {e["all"]["reference_tau_sq_Nm2"]:.0f}</small></td>' if i == 0 else ""
                    prow.append(f"<tr>{lab}<td>{NAMES[m]}</td><td>{-e['all']['torque_sq_change_percent']:.1f}</td>"
                                f"<td>{-e['fast_over_4rad_s']['torque_sq_change_percent']:.1f}</td>"
                                f"<td>{e['all']['assist_rms_Nm']:.1f}</td><td>{100 * e['all']['saturation_fraction']:.0f}</td>"
                                f"<td>{100 * e['need_bins'][0]['sign_agreement']:.0f}</td>"
                                f"<td>{e['all']['reference_tracking_rmse_deg']:.2f} → {e['all']['assisted_tracking_rmse_deg']:.2f}</td>"
                                f"<td>{e['command_change']['mean_Nm_per_step']:.2f}</td><td>{100 * e.get('assist_power_above_10Hz_fraction', float('nan')):.1f}</td>"
                                f"<td>{s['failures']['reference']} / {s['failures']['assisted']}</td></tr>")
            body.append('<div class="scroll"><table><tr><th>조건</th><th>모델</th><th>착용자 τ² 감소 [%]</th><th>각속도 4 rad/s 초과 구간 τ² 감소 [%]</th>'
                        '<th>보조 RMS [N·m]</th><th>포화 [%]</th><th>필요 토크 5 N·m 미만 방향 일치 [%]</th><th>추종 오차 무보조 → 보조 [°]</th>'
                        '<th>명령 변화 [N·m / 10 ms]</th><th>10 Hz 초과 성분 [%]</th><th>실패 무보조 / 보조</th></tr>' + "".join(prow) + "</table></div>")
            clamp = ", ".join(f"{Path(k).stem} {100 * v['clamped_frame_fraction']:.0f} %" for k, v in etri_info["motions"].items()
                              if v["clamped_frame_fraction"] > 0.005)
            body.append(f'<p class="note">ETRI 기록은 SEED 영점으로 맞추고 고관절을 진자 범위 ±86°로 잘랐다. 잘린 프레임: {clamp}. '
                        '실패는 무보조 진자의 추종 실패(오차 > 1 rad)로, 주로 가벼운 다리가 달리기 중 입각 하중까지 받을 때 착용자 토크가 한계에 걸려 생긴다. 지표는 두 진자가 모두 살아 있는 구간만 쓴다.</p>')
            srow = []
            for c in SYNTHETIC:
                for m in ["L1_20", "R0_20"] + list(NEW):
                    if (c, m) not in pend:
                        continue
                    a = pend[c, m]["metrics"]["all"]
                    srow.append(f"<tr><td>{SYNTH_NAMES[c]}</td><td>{NAMES[m]}</td><td>{a['reference_tau_sq_Nm2']:.1f}</td>"
                                f"<td>{-a['torque_sq_change_percent']:.1f}</td><td>{a['assist_rms_Nm']:.1f}</td>"
                                f"<td>{a['assisted_tracking_rmse_deg']:.2f}</td><td>{pend[c, m]['failures']['assisted']}</td></tr>")
            body.append('<h3>기존 합성 궤적 평가 (2026-10-20과 같은 seed)</h3><div class="scroll"><table><tr><th>조건</th><th>모델</th><th>무보조 τ² (프로토콜 일치 확인)</th>'
                        '<th>착용자 τ² 감소 [%]</th><th>보조 RMS [N·m]</th><th>보조 추종 오차 [°]</th><th>보조 실패</th></tr>' + "".join(srow) + "</table></div>"
                        '<p class="note">L1·R0는 저장 기록이다. 무보조 τ²가 같으면 같은 진자·궤적이다.</p>')

        # ---------- 4b. 예측 신호 진단 ----------
        dmodels = [m for m in ["L1_20", "R0_20"] + list(NEW) if (m, "emulated") in diag and (m, "h1") in diag]
        if dmodels:
            drow = []
            for m in dmodels:
                cells = []
                for pool in ("all", "light_fast", "general", "synthetic"):
                    e, h = diag[m, "emulated"][pool], diag[m, "h1"][pool]
                    cells.append(f"<td>{-e['torque_sq_change_percent']:.1f} → {-h['torque_sq_change_percent']:.1f}</td>")
                e, h = diag[m, "emulated"]["all"], diag[m, "h1"]["all"]
                drow.append(f"<tr><td>{NAMES[m]}</td>{''.join(cells)}<td>{e['assist_vs_unassisted_torque_corr']:.2f} → {h['assist_vs_unassisted_torque_corr']:.2f}</td>"
                            f"<td>{e['assist_rms_Nm']:.1f} → {h['assist_rms_Nm']:.1f}</td></tr>")
            body.append("<h2>5. 왜 실제 동작에서 나빠지나: 학습 때의 예측 신호와 실제 H1의 차이</h2>"
                        "<p>새 레시피의 학습 환경(같은 집단·관성·하중, H1 학습분할 동작)을 그대로 짝 진자로 돌리고, 보조 관절 하나의 착용자 τ² 감소를 봤다. "
                        "미래 목표 관측만 바꿨다: 학습과 같은 에뮬레이션 예측(의도 궤적의 미래 + 잡음·지연) → 동결 H1이 실제 각도 이력으로 낸 예측. "
                        "에뮬레이션 예측은 의도에 고정돼 있어 예측과 현재 각도의 차이에 착용자의 추종 오차(= 필요한 토크의 단서)가 섞이지만, 실제 H1에는 그 단서가 없다. "
                        "1024 env × 10 s, L1·R0는 ×0.5.</p>"
                        '<div class="scroll"><table><tr><th>모델</th><th>전체: 착용자 τ² 감소 [%] 에뮬레이션 → H1</th><th>가벼운 다리</th><th>일반</th><th>합성</th>'
                        "<th>보조와 무보조 착용자 토크의 상관</th><th>보조 RMS [N·m]</th></tr>" + "".join(drow) + "</table></div>")

        # ---------- 6. A·Δ ----------
        if models:
            arow = []
            for m in ["L1_20", "R0_20"] + models:
                lw, lr = self.lpf_benefit(runs, lpf, "walk", m), self.lpf_benefit(runs, lpf, "run", m)
                pv = [-pend[c, m]["extended"]["all"]["torque_sq_change_percent"] if (c, m) in pend and "extended" in pend[c, m] else None
                      for c in ("seed_val", "etri")]
                leg = -pend["legacy", m]["metrics"]["all"]["torque_sq_change_percent"] if ("legacy", m) in pend else None
                dg = [-diag[m, src]["all"]["torque_sq_change_percent"] if (m, src) in diag else None for src in ("emulated", "h1")]
                pcmd = pend["legacy", m]["extended"]["command_change"]["mean_Nm_per_step"] if ("legacy", m) in pend and "extended" in pend["legacy", m] else None
                wcmd = self.w1_mean_command(runs, "run", m)[0]
                weights = {"L1_20": "−2e-5 / −0.005 (40 N·m)", "R0_20": "−2e-5 / −0.005 (40 N·m)"}.get(
                    m, f"{'−2e-5' if m[1] == '1' else '0'} / {'−0.005' if m[3] == '1' else '0'}")
                arow.append(f"<tr><td>{NAMES[m]}</td><td>{weights}</td>"
                            f"<td>{fmt(lw[0], '.0f')} → {fmt(lw[1], '.0f')}</td><td>{fmt(lr[0], '.0f')} → {fmt(lr[1], '.0f')}</td>"
                            f"<td>{fmt(w1_mean(runs, 'walk', m, 'assist_rms')[0])} / {fmt(w1_mean(runs, 'walk', m, 'saturation')[0], '.0f')}</td>"
                            f"<td>{fmt(pv[0])}</td><td>{fmt(pv[1])}</td><td>{fmt(leg)}</td><td>{fmt(dg[0])} → {fmt(dg[1])}</td>"
                            f"<td>{fmt(pcmd, '.2f')}</td><td>{fmt(wcmd, '.2f')}</td></tr>")
            body.append("<h2>6. A와 Δ는 필요한가</h2>"
                        "<p>네 경우는 같은 학습 seed·환경 난수에서 A·Δ 가중치만 다르다. 모터가 10 ms마다 토크를 크게 바꾸지 못한다는 점을 보려고, "
                        "W1 출력에 1차 저역통과 10 Hz를 건 평가를 더 했다(원래 → 저역통과 뒤). L1·R0는 비교용이다.</p>"
                        '<div class="scroll"><table><tr><th>모델</th><th>A / Δ 가중치</th><th>W1 걷기 τ² 감소 (원래 → 10 Hz)</th><th>W1 달리기 (원래 → 10 Hz)</th>'
                        "<th>W1 걷기 보조 RMS [N·m] / 포화 [%]</th><th>진자 SEED [%]</th><th>진자 ETRI [%]</th><th>진자 합성 [%]</th>"
                        "<th>학습 분포 진자 에뮬레이션 → H1 [%]</th><th>명령 변화 진자 합성 [N·m / 10 ms]</th><th>명령 변화 W1 달리기</th></tr>"
                        + "".join(arow) + "</table></div>"
                        '<p class="note">진자 [%]는 착용자 τ² 감소율이다. 진자 합성은 2026-10-20과 같은 seed의 합성 궤적 평가이며 L1·R0 값은 저장 기록이다.</p>')

        # ---------- 7. 좌우·안정성 ----------
        if models:
            lrow = []
            for c in W1_CONDITIONS:
                for m in shown:
                    b = [w1_mean(runs, c, m, "benefit_side", k)[0] for k in (0, 1)]
                    r = [runs[c, m, s] for s in SEEDS]
                    lrow.append(f"<tr><td>{escape(SHORT[c])}</td><td>{NAMES[m]}</td><td>{fmt(b[0], '.0f')} / {fmt(b[1], '.0f')}</td>"
                                f"<td>{np.mean([x['body_err_mm'][0] for x in r]):.1f} → {np.mean([x['body_err_mm'][1] for x in r]):.1f}</td>"
                                f"<td>{np.mean([x['speed'][1] - x['speed'][0] for x in r]):+.4f}</td>"
                                + "".join(f"<td>{-np.mean([x['other'].get(g, np.nan) for x in r]):+.0f}</td>" for g in ("무릎", "발목", "몸통")) + "</tr>")
            body.append("<h2>7. 좌우와 보행 안정성</h2><p>새 모델은 같은 가중치를 양쪽 다리에 쓰므로, 좌우 차이는 원본 동작과 W1의 비대칭에서 온다. 다른 관절 열은 보조 B − 무보조 A의 토크 제곱 평균 변화이며 음수면 부담이 줄었다.</p>"
                        '<div class="scroll"><table><tr><th>조건</th><th>모델</th><th>τ² 감소 오른쪽 / 왼쪽</th><th>몸체 오차 A → B [mm]</th><th>속도 변화 [m/s]</th>'
                        "<th>무릎</th><th>발목</th><th>몸통</th></tr>" + "".join(lrow) + "</table></div>")

        body.append('<details><summary>해석 범위</summary><ul>'
                    '<li>세 가지 공통 변경(20 N·m, 실제 동작 궤적, 단일 관절 학습·가중치 공유)을 한꺼번에 적용했으므로, R0와의 차이를 그중 하나의 효과로 나눌 수 없다. 분리되는 것은 A와 Δ뿐이다.</li>'
                    '<li>각 경우는 학습 실행 하나(seed 0)다. A·Δ 차이가 학습 seed에 따른 흔들림보다 큰지는 이 실험만으로 확인할 수 없다.</li>'
                    '<li>가상 착용자 W1 하나, 걷기·달리기 기록 하나씩이다. W1이 학습하지 않은 동작(스쿼트·런지 등)은 W1에서 넘어져 진자에서만 평가했다.</li>'
                    '<li>모터 응답은 시뮬레이터에 없다. 10 Hz 저역통과는 민감도 점검이며 실제 구동기 모델이 아니다.</li>'
                    '<li>진자의 다리 관성·하중은 단순화된 모델이며, 가벼운 다리 집단은 달리기를 따라갈 수 있게 하려고 둔 학습 분포 선택이다.</li></ul></details>')
        body.append('<details><summary>재현 경로</summary><ul>'
                    '<li>학습: <code>python examples/experiments/assist_pendulum/study.py --study ' + escape(str(STUDY.relative_to(ROOT))) + ' train --lane right</code> '
                    '(레시피 <code>mlp.py --recipe motion_leg</code>)</li>'
                    '<li>평가: <code>python examples/experiments/assist_pendulum/study.py --study ... eval</code> '
                    '(진자 <code>evaluate.py paired --share-legs</code>, W1 <code>run_system.py --llp-share-legs</code>)</li>'
                    '<li>이 페이지: <code>python examples/experiments/assist_pendulum/report.py --date ' + date + '</code></li>'
                    f'<li>원본: <code>{STUDY.relative_to(ROOT)}/</code> (manifest, execution_*, eval_runs, pendulum/, walking/). '
                    '2026-10-21 연구는 통합 전 스크립트(<code>scripts/</code>)로 수행했다.</li></ul></details>')
        body.append('<p class="note">생성 UTC ' + datetime.now(timezone.utc).isoformat(timespec="seconds") + "</p>")
        evidence = dict(models=models, matched={f"{c}/{m}": v for (c, m), v in match.items()},
                        w1={"/".join(map(str, k)): {kk: vv for kk, vv in v.items() if kk not in ("rows",)} for k, v in runs.items()},
                        w1_lowpass={"/".join(map(str, k)): {kk: vv for kk, vv in v.items() if kk not in ("rows",)} for k, v in lpf.items()})
        (STUDY / "comparison.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False, default=float))
        html = ('<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>LLP 20 N·m 공유 동작 학습</title><style>' + CSS + "</style></head><body><main>" + "".join(body) + "</main></body></html>")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html)
        print(f"wrote {out.relative_to(ROOT)} ({len(html) / 1e6:.1f} MB); models={models}")
        return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", default=datetime.now().strftime("%y%m%d"), help="docs/ai_sbc/<date>_llp20_motion_shared/")
    p.add_argument("--study", default=DEFAULT_STUDY)
    args = p.parse_args(argv)
    MotionLegReport(args.study).render(args.date)


if __name__ == "__main__":
    main()
