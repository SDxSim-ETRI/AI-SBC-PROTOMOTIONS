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
"""걷기 70–90 % 구간 분석과 유각 브레이크 게이트 보고서 (2026-10-14).

run_system.py 기록(traces.npz + meta.json)만 읽어 HTML 하나를 만든다. 새 시뮬레이션은 돌리지 않는다.

    python examples/experiments/assist_skeleton/report_swing_gate.py --out docs/ai_sbc/walking_swing_gate.html
"""

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/experiments/assist_pendulum"))
from run_wide_sequence import summarize  # noqa: E402

BINS = 25  # 4 % 구간


# ---------------------------------------------------------------------------
# 기록 읽기
# ---------------------------------------------------------------------------
def find_run(folder, tag):
    hits = sorted(p for p in Path(folder).glob(f"*_tracker_{tag}") if (p / "traces.npz").exists())
    return hits[-1] if hits else None


def load(run_dir):
    """토크를 같은 10 ms 창에 정렬한 뒤 위상(각 다리의 참조 최대굴곡 = 0 %)으로 묶을 재료를 만든다."""
    meta = json.loads((Path(run_dir) / "meta.json").read_text())
    tr = dict(np.load(Path(run_dir) / "traces.npz"))
    idx = np.flatnonzero((tr["t"] >= meta["warmup"] + 0.5) & (np.arange(len(tr["t"])) >= 2))
    agent = (tr["tau_pd_prev"] + tr["tau_ff_prev"])[idx]
    assist = tr["tau_assist"][idx - 1]
    before, after = tr["gait_phase"][idx - 1], tr["gait_phase"][idx]
    phase = (before + .5 * ((after - before + .5) % 1 - .5)) % 1
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


def bins(d, n=BINS):
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


def window_contribution(rows, lo, hi):
    """구간 [lo, hi) %의 τ² 평균 기여 [(N·m)²] — 양수면 부담 증가."""
    return sum(r["contrib"] for r in rows if r["lo"] >= lo and r["hi"] <= hi)


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


# ---------------------------------------------------------------------------
# HTML 조각
# ---------------------------------------------------------------------------
def f1(v, signed=False):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, "+.1f" if signed else ".1f")


def line_svg(series, ylabel, ylim, shade=None, h=250, w=900):
    """series: [(label, color, y[25], dash)] — x = 4 % 구간 중심."""
    ml, mr, mt, mb = 60, 15, 18, 40
    x = lambda p: ml + p / 100 * (w - ml - mr)  # noqa: E731
    y = lambda v: mt + (ylim[1] - v) / (ylim[1] - ylim[0]) * (h - mt - mb)  # noqa: E731
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(ylabel)}">']
    if shade:
        out.append(f'<rect x="{x(shade[0])}" y="{mt}" width="{x(shade[1]) - x(shade[0])}" height="{h - mt - mb}" fill="#f3d9c6" opacity=".6"/>')
    step = 25 if ylim[1] - ylim[0] > 80 else 10
    v = np.ceil(ylim[0] / step) * step
    while v <= ylim[1]:
        out.append(f'<line x1="{ml}" x2="{w - mr}" y1="{y(v)}" y2="{y(v)}" stroke="#e3e7ec"/>'
                   f'<text x="{ml - 6}" y="{y(v) + 4}" text-anchor="end" font-size="11" fill="#555">{v:+.0f}</text>')
        v += step
    for p in range(0, 101, 10):
        out.append(f'<line x1="{x(p)}" x2="{x(p)}" y1="{mt}" y2="{h - mb}" stroke="#eef1f4"/>'
                   f'<text x="{x(p)}" y="{h - mb + 14}" text-anchor="middle" font-size="11" fill="#555">{p}</text>')
    out.append(f'<line x1="{ml}" x2="{w - mr}" y1="{y(0)}" y2="{y(0)}" stroke="#999"/>')
    for label, color, ys, dash in series:
        pts = " ".join(f"{x((i + .5) * 100 / BINS):.1f},{y(v):.1f}" for i, v in enumerate(ys) if np.isfinite(v))
        dash_attr = 'stroke-dasharray="6 4"' if dash else ""
        out.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" {dash_attr} points="{pts}"><title>{html.escape(label)}</title></polyline>')
    out.append(f'<text x="{ml}" y="12" font-size="12" fill="#333">{html.escape(ylabel)}</text>'
               f'<text x="{w - mr}" y="{h - 4}" text-anchor="end" font-size="12" fill="#333">보행 위상 [%] · 0 = 참조 최대 굴곡</text></svg>')
    legend = " ".join(f'<span style="display:inline-block;margin-right:14px"><span style="display:inline-block;width:22px;border-top:3px {"dashed" if dash else "solid"} {color};vertical-align:middle"></span> {html.escape(label)}</span>'
                      for label, color, _, dash in series)
    return f'<div class="legend">{legend}</div>' + "".join(out)


def bars_svg(rows_list, labels, colors, h=260, w=900):
    """구간별 착용자 RMS 변화 (B − A) 막대, 실행 여러 개 나란히. 양수 = 부담 증가."""
    ml, mr, mt, mb = 60, 15, 18, 40
    vals = [[r["B_rms"] - r["A_rms"] for r in rows] for rows in rows_list]
    bound = max(5.0, max(abs(v) for vs in vals for v in vs)) * 1.15
    y = lambda v: mt + (bound - v) / (2 * bound) * (h - mt - mb)  # noqa: E731
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="구간별 착용자 토크 RMS 변화">']
    for v in (-bound / 2, 0, bound / 2):
        out.append(f'<line x1="{ml}" x2="{w - mr}" y1="{y(v)}" y2="{y(v)}" stroke="#ddd"/><text x="{ml - 6}" y="{y(v) + 4}" text-anchor="end" font-size="11" fill="#555">{v:+.1f}</text>')
    n = len(rows_list[0]); dx = (w - ml - mr) / n; k = len(rows_list); bw = dx * .8 / k
    for i in range(n):
        for j in range(k):
            v = vals[j][i]
            xx = ml + dx * (i + .1) + j * bw
            out.append(f'<rect x="{xx:.1f}" y="{min(y(v), y(0)):.1f}" width="{bw:.1f}" height="{max(1, abs(y(v) - y(0))):.1f}" fill="{colors[j]}"><title>{html.escape(labels[j])} {i * 4}–{(i + 1) * 4} %: {v:+.1f} N·m</title></rect>')
        if i % 2 == 0:
            out.append(f'<text x="{ml + dx * (i + .5):.1f}" y="{h - mb + 14}" text-anchor="middle" font-size="11" fill="#555">{i * 4}</text>')
    out.append(f'<text x="{ml}" y="12" font-size="12" fill="#333">착용자 토크 RMS 변화 B − A [N·m] · 양수 = 부담 증가</text>'
               f'<text x="{w - mr}" y="{h - 4}" text-anchor="end" font-size="12" fill="#333">보행 위상 [%]</text></svg>')
    legend = " ".join(f'<span style="display:inline-block;margin-right:14px"><span style="display:inline-block;width:14px;height:12px;background:{c};vertical-align:middle"></span> {html.escape(l)}</span>' for l, c in zip(labels, colors))
    return f'<div class="legend">{legend}</div>' + "".join(out)


def phase_table(rows, lo, hi, gate=False):
    hdr = ["위상", "착용자 A 평균", "착용자 B 평균", "보조 평균", "B − A", "상쇄비 c", "A와 반대 %", "A RMS", "B RMS", "τ² 기여", "hip 각 A", "각속도 A", "각가속도 A", "B − A 각"]
    if gate:
        hdr.append("게이트 %")
    body = []
    for r in rows:
        if r["lo"] < lo or r["hi"] > hi:
            continue
        cells = [f'{r["lo"]:.0f}–{r["hi"]:.0f}', f1(r["A"], 1), f1(r["B"], 1), f1(r["S"], 1), f1(r["dAB"], 1), f1(r["c"], 1) if np.isfinite(r["c"]) else "—",
                 f'{r["oppA"]:.0f}', f1(r["A_rms"]), f1(r["B_rms"]), f1(r["contrib"], 1), f1(r["pos"], 1) + "°", f1(r["vel"], 1), f1(r["acc"], 1), f1(r["dpos"], 1) + "°"]
        if gate:
            cells.append(f'{r.get("gate_pct", 0):.0f}')
        body.append("<tr><th>" + cells[0] + "</th>" + "".join(f"<td>{c}</td>" for c in cells[1:]) + "</tr>")
    return '<div class="scroll"><table><thead><tr>' + "".join(f"<th>{h}</th>" for h in hdr) + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="docs/ai_sbc/walking_swing_gate.html")
    p.add_argument("--baseline-dir", default="results/pendulum_reeval_20261013/evaluations")
    p.add_argument("--probe-dir", default="results/walk_phase_analysis_20261014/evaluations")
    args = p.parse_args()

    runs = {
        "L1 ×0.5 · 걷기": find_run(args.baseline_dir, "reference_L1_x0.5_m0"),
        "L1 ×0.5 · 달리기": find_run(args.baseline_dir, "reference_L1_x0.5_m1"),
        "β 오라클 0.5 · 걷기": find_run(args.probe_dir, "beta05_m0"),
        "L1 ×0.5 + 참조위상 70–90 % 차단 · 걷기": find_run(args.probe_dir, "L1s05_mask70_90_m0"),
        "L1 ×0.5 + 참조위상 70–90 % 차단 · 달리기": find_run(args.probe_dir, "L1s05_mask70_90_m1"),
        "L1 ×0.5 + 참조위상 20–30 % 차단 · 걷기": find_run(args.probe_dir, "L1s05_mask20_30_m0"),
        "L1 ×0.5 + 참조위상 20–40 % 차단 · 걷기": find_run(args.probe_dir, "L1s05_mask20_40_m0"),
        "L1 ×0.5 + 유각 게이트 0.20 · 걷기": find_run(args.probe_dir, "L1s05_gate020_m0"),
        "L1 ×0.5 + 유각 게이트 0.20 · 달리기": find_run(args.probe_dir, "L1s05_gate020_m1"),
        "L1 ×0.5 + 유각 게이트 0.15 · 걷기": find_run(args.probe_dir, "L1s05_gate015_m0"),
        "L1 ×0.5 + 유각 게이트 0.25 · 걷기": find_run(args.probe_dir, "L1s05_gate025_m0"),
        "L1 ×0.5 · 걷기 · seed 7": find_run(args.probe_dir, "L1s05_seed7_m0"),
        "L1 ×0.5 + 유각 게이트 0.20 · 걷기 · seed 7": find_run(args.probe_dir, "L1s05_gate020_seed7_m0"),
        "L1 ×0.5 · 달리기 · seed 7": find_run(args.probe_dir, "L1s05_seed7_m1"),
        "L1 ×0.5 + 유각 게이트 0.20 · 달리기 · seed 7": find_run(args.probe_dir, "L1s05_gate020_seed7_m1"),
    }
    runs = {k: v for k, v in runs.items() if v is not None}
    rec = {}
    for name, run in runs.items():
        meta, d, tr = load(run)
        s = summarize(run)
        rows = bins(d)
        rec[name] = dict(meta=meta, d=d, s=s, rows=rows, run=run,
                         harm_68_92=window_contribution(rows, 68, 92), harm_20_40=window_contribution(rows, 20, 40),
                         worst=max(r["B_rms"] - r["A_rms"] for r in rows), gate=gate_accuracy(d))

    base, beta, mask, gate = (rec.get(k) for k in ("L1 ×0.5 · 걷기", "β 오라클 0.5 · 걷기", "L1 ×0.5 + 참조위상 70–90 % 차단 · 걷기", "L1 ×0.5 + 유각 게이트 0.20 · 걷기"))
    base_r, gate_r, mask_r = (rec.get(k) for k in ("L1 ×0.5 · 달리기", "L1 ×0.5 + 유각 게이트 0.20 · 달리기", "L1 ×0.5 + 참조위상 70–90 % 차단 · 달리기"))

    # ---- 표: 전체 수치
    head = ["조건", "착용자 τ² 변화", "보조 RMS [N·m]", "몸체 오차 변화", "40초 유지", "가장 나빠진 4 % 구간 [N·m]", "68–92 % τ² 기여", "20–40 % τ² 기여"]
    body = []
    for name, r in rec.items():
        s = r["s"]
        body.append("<tr><th>" + html.escape(name) + "</th>" + "".join(f"<td>{c}</td>" for c in (
            f'{s["τ²"]:+.1f} %', f'{s["assist_rms"]:.1f}', f'{s["body"]:+.1f} %', "예" if s["survived_s"] is None else f'{s["survived_s"]:.1f} s 낙상',
            f'{r["worst"]:+.1f}', f'{r["harm_68_92"]:+.1f}', f'{r["harm_20_40"]:+.1f}')) + "</tr>")
    summary_table = '<div class="scroll"><table><thead><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"

    def ys(r, key):
        return [row[key] for row in r["rows"]]

    plots = ""
    if base and gate and beta:
        plots += "<h3>착용자가 낸 토크 · 걷기 (부호 있는 평균, 좌우 hip을 각자 위상으로 모음)</h3>"
        plots += line_svg([("무보조 A", "#77766f", ys(base, "A"), False), ("보조 B · L1 ×0.5", "#d55b26", ys(base, "B"), False),
                           ("보조 B · 게이트 0.20", "#1f7a4d", ys(gate, "B"), False), ("보조 B · β 오라클", "#8b55ac", ys(beta, "B"), True)],
                          "τ_agent [N·m] · +굴곡 / −신전", (-100, 50), shade=(68, 92))
        plots += "<h3>장치가 낸 토크 · 걷기</h3>"
        plots += line_svg([("L1 ×0.5", "#d55b26", ys(base, "S"), False), ("게이트 0.20", "#1f7a4d", ys(gate, "S"), False), ("β 오라클", "#8b55ac", ys(beta, "S"), True)],
                          "τ_assist [N·m]", (-40, 40), shade=(68, 92))
        plots += "<h3>hip 각속도·각가속도 · 무보조 A · 걷기</h3>"
        plots += line_svg([("각속도 ×10 [rad/s]", "#2275ce", [10 * v for v in ys(base, "vel")], False), ("각가속도 [rad/s²]", "#c9a227", ys(base, "acc"), False)],
                          "각속도 ×10, 각가속도", (-60, 60), shade=(68, 92))
        plots += "<h3>구간별 착용자 토크 RMS 변화 · 걷기</h3>"
        plots += bars_svg([base["rows"], mask["rows"], gate["rows"]] if mask else [base["rows"], gate["rows"]],
                          ["L1 ×0.5", "참조위상 70–90 % 차단", "게이트 0.20"] if mask else ["L1 ×0.5", "게이트 0.20"],
                          ["#d55b26", "#999", "#1f7a4d"] if mask else ["#d55b26", "#1f7a4d"])

    def pct(r):
        return f'{r["s"]["τ²"]:+.1f} %' if r else "—"

    ga = gate["gate"] if gate else None
    gate_text = ""
    if ga:
        gate_text = (f"게이트가 켜진 표본은 전체의 {ga['on_pct']:.1f} %이고, 그 표본의 참조 위상은 평균 {ga['on_phase_mean']:.0f} %, "
                     f"66–92 % 안에 {ga['in_66_92_pct']:.0f} %, 70–90 % 안에 {ga['in_70_90_pct']:.0f} %가 들어간다. "
                     f"창 시작은 참조 위상 {ga['start_phase'][0]:.1f} ± {ga['start_phase'][1]:.1f} %, 끝은 {ga['end_phase'][0]:.1f} ± {ga['end_phase'][1]:.1f} %다.")

    conclusion = f"""
<p><strong>걷기에서 L1 ×0.5가 착용자를 방해하는 구간은 68–92 %(유각 중·후반) 하나다.</strong> 이 구간에서 착용자 hip 토크는 거의 0인데
(다리 감속은 무릎 결합·중력이 낸다) 정책은 −15~−20 N·m 신전 브레이크를 걸고, 착용자는 같은 크기의 굴곡으로 맞선다(상쇄비 c ≈ 1).
20–40 %는 손해 구간이 아니다 — 보조를 끊어도 전체 수치가 변하지 않는다({pct(rec.get('L1 ×0.5 + 참조위상 20–30 % 차단 · 걷기'))}, {pct(rec.get('L1 ×0.5 + 참조위상 20–40 % 차단 · 걷기'))}).</p>
<p><strong>관절각만 보는 게이트로 이 브레이크를 막으면 걷기 {pct(base)} → {pct(gate)}</strong>
(참조 위상을 아는 차단 {pct(mask)}, β 오라클 상한 {pct(beta)}). 달리기는 {pct(base_r)} → {pct(gate_r)} (참조 위상 차단 {pct(mask_r)}).</p>
<p>다른 난수 초기값(seed 7)으로 다시 돌려도 같다: 걷기 {pct(rec.get('L1 ×0.5 · 걷기 · seed 7'))} → {pct(rec.get('L1 ×0.5 + 유각 게이트 0.20 · 걷기 · seed 7'))},
달리기 {pct(rec.get('L1 ×0.5 · 달리기 · seed 7'))} → {pct(rec.get('L1 ×0.5 + 유각 게이트 0.20 · 달리기 · seed 7'))}.
게이트 비율 0.15 / 0.20 / 0.25는 걷기 {pct(rec.get('L1 ×0.5 + 유각 게이트 0.15 · 걷기'))} / {pct(gate)} / {pct(rec.get('L1 ×0.5 + 유각 게이트 0.25 · 걷기'))} —
0.20과 0.25의 차이는 재실행 산포 안이다.</p>
<p class="report-limit">가상 착용자 W1 한 모델, 걷기·달리기 각 한 클립의 결과다. 게이트 비율 0.20은 W1 걷기의 위상표에서 정한 값이며 여러 착용자·속도에서 검증하지 않았다.</p>
"""

    cause = ""
    if base:
        cause += "<h3>걷기 · L1 ×0.5 · 60–100 %</h3>" + phase_table(base["rows"], 60, 100)
        cause += "<h3>걷기 · L1 ×0.5 · 12–40 %</h3>" + phase_table(base["rows"], 12, 40)
    if gate:
        cause += "<h3>걷기 · 게이트 0.20 · 60–100 % (마지막 열 = 게이트가 켜진 표본 비율)</h3>" + phase_table(gate["rows"], 60, 100, gate=True)
    if base_r and gate_r:
        cause += "<h3>달리기 · L1 ×0.5 · 76–100 %</h3>" + phase_table(base_r["rows"], 76, 100)
        cause += "<h3>달리기 · 게이트 0.20 · 76–100 %</h3>" + phase_table(gate_r["rows"], 76, 100, gate=True)

    sources = "".join(f"<li>{html.escape(k)} — <code>{html.escape(str(v['run']))}</code></li>" for k, v in rec.items())

    page = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>걷기 유각 브레이크 분석과 관절각 게이트</title>
<style>body{{font:16px/1.7 system-ui,sans-serif;margin:30px auto;max-width:1180px;padding:0 20px;color:#1c2d3d}}table{{border-collapse:collapse;font-size:13px}}th,td{{border:1px solid #d8e0e7;padding:4px 8px;text-align:right;white-space:nowrap}}th:first-child{{text-align:left}}thead th{{background:#eef2f6}}.scroll{{overflow-x:auto}}.legend{{font-size:13px;margin:6px 0}}svg{{width:100%;height:auto;background:white;border:1px solid #e3e7ec;margin:6px 0 18px}}h2{{margin-top:34px}}code{{font-size:12px}}.report-limit{{color:#526474;font-size:14px}}</style></head><body>
<p>AI-SBC · 원인 진단과 수정 · 2026-10-14</p>
<h1>걷기 유각 브레이크 분석과 관절각 게이트</h1>
<p>L1 ×0.5가 걷기에서 착용자를 방해하는 구간을 4 % 단위로 나눠 원인을 찾고, 관절각만으로 그 구간의 신전 보조를 끊는 게이트를 만들어 같은 조건에서 비교했다.</p>
<h2>결론</h2>{conclusion}
<h2>조건</h2>
<p>가상 착용자 W1 + 각도 예측기 H1 + L1(출력 0.5배, 한계 ±20 N·m), 10 ms 제어, 40초, seed 42, 영점은 착용자 기준. 무보조 A와 보조 B를 같은 초기 상태에서 함께 돌린다.
τ_agent는 A·B 모두 착용자(W1)가 낸 토크(tau_pd + tau_ff), τ_assist는 장치가 보탠 토크다. 위상 0 %는 참조 mocap의 그 다리 최대 굴곡이며 발 착지 검출이 아니다.
"참조위상 차단"은 mocap 위상표를 아는 특권 진단이고, "유각 게이트"는 제어기 안에서 hip 관절각만으로 창을 잡는 배포 가능한 규칙이다
(관절각이 지수이동평균 중간값을 굴곡 방향으로 지나는 순간부터, 같은 hip의 직전 교차 간격 × 비율 동안 신전 보조를 0으로).
상쇄비 c = −(B − A)/보조: 1이면 착용자가 보조를 전부 되받아 총 토크가 무보조와 같다.</p>
<h2>전체 수치</h2>{summary_table}
<p>τ² 기여 = 그 구간이 전체 τ² 평균 변화에 보탠 양 [(N·m)²]. 걷기 무보조 τ² 평균은 약 1,490, 달리기는 약 3,130이다.</p>
<h2>위상별 그래프 · 걷기</h2>{plots}
<p>음영 = 68–92 %. 무보조에서 이 구간의 각가속도는 −25 rad/s² 안팎인데 착용자 토크는 ±10 N·m 안이다 — 감속을 hip이 내지 않는다는 직접 증거다.</p>
<h2>구간별 수치</h2>{cause}
<h2>게이트 창이 실제로 걸린 위치</h2><p>{gate_text or "게이트 기록 없음."}</p>
<p>리셋 뒤 주기가 잡히기까지(약 2초) 게이트는 꺼져 있다. 창 시작은 관절각 중간값 교차(≈ 69 %), 끝은 직전 주기 × 0.20 뒤다.</p>
<h2>한계</h2>
<ul><li>한 착용자·한 클립·한 seed. 재실행 산포는 L1 ×0.5 기준으로 걷기 τ² ±0.7 %p 정도였다(2026-10-13 재평가).</li>
<li>게이트는 걷기의 "자유 감속" 구간을 주기 비율로 자른다. 보행 종류·속도가 바뀌면 비율이 달라질 수 있다. 달리기 결과는 같은 0.20으로만 확인했다.</li>
<li>20–40 %의 추가 이득은 β 오라클도 작다(비례 보조여도 그 구간 토크가 작다). 남은 격차는 착지 0–12 %의 크기(비례성)다.</li></ul>
<h2>원본 기록</h2><ul>{sources}</ul>
</body></html>"""
    Path(args.out).write_text(page, encoding="utf-8")
    print(f"saved {args.out} ({len(rec)} runs)")
    for k, r in rec.items():
        print(f"  {k}: τ² {r['s']['τ²']:+.1f} %, 68–92 {r['harm_68_92']:+.1f}, 20–40 {r['harm_20_40']:+.1f}, worst {r['worst']:+.1f}")


if __name__ == "__main__":
    main()
