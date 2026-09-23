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
"""run_system.py 실행 묶음을 HTML 보고서 한 장으로 — 비교표 + 위상 그림 + 위상표.

`compare_runs.py`가 터미널에 찍는 표와 같은 수치를 브라우저용으로 낸다. 위상 그림은
보행 주기 10 % 구간별 착용자 τ² 변화(절대 (N·m)², 무보조 대비)를 실행별 막대로 그린다 —
"어느 구간에서 벌고 어느 구간에서 잃는가"가 한눈에 보이게.

실행 (sbc env):
    python examples/experiments/assist_skeleton/report_html.py --out docs/report.html \\
        --walk base=results/oracle/*_tracker_base v12@1000=results/oracle/*_tracker_v12e1000_m0 \\
        --run  v10=results/oracle/*_tracker_v10_m1 --chart base,v12@1000
"""

import argparse
import glob
import html
import json
from pathlib import Path

import numpy as np

A, B = 0, 1
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]        # 검증된 범주 팔레트 (light)
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500"]


def load(run: Path):
    tr = np.load(run / "traces.npz")
    meta = json.loads((run / "meta.json").read_text())
    t = tr["t"]
    idx = np.where(t >= meta["warmup"])[0]
    idx = idx[idx >= 1]
    return tr, meta, idx


def metrics(run: Path):
    tr, meta, idx = load(run)
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


def phase_rows(run: Path, bins=10):
    tr, meta, idx = load(run)
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
        rows.append({
            "contrib": float((b2 - a2) * sel.mean() * 2),   # 전체 τ² 평균에 대한 절대 기여 (N·m)²
            "tauA": float(tau[:, A][sel].mean()), "tauB": float(tau[:, B][sel].mean()),
            "assist": float(assist[:, B][sel].mean()),
        })
    return rows


def fmt(v, digits=1, sign=True):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:+.{digits}f}" if sign else f"{v:.{digits}f}"


def table_html(rows: dict, title: str, note: str, running=False):
    cols = [("P", "\\|τ·ω\\| 파워"), ("tau2", "τ² 근활성"), ("abs_tau", "\\|τ\\|"), ("omega", "\\|ω\\| 각속도"),
            ("speed", "전진 거리"), ("body", "몸체 오차")]
    cols = [(k, n.replace("\\|", "|")) for k, n in cols]
    best = {k: min((m[k] for m in rows.values() if k in m and np.isfinite(m[k])), default=None) for k, _ in cols}
    head = "".join(f"<th>{html.escape(n)}</th>" for _, n in cols)
    body = []
    for name, m in rows.items():
        cells = []
        for k, _ in cols:
            v = m.get(k)
            cls = ' class="best"' if v is not None and best[k] is not None and abs(v - best[k]) < 1e-9 else ""
            cells.append(f"<td{cls}>{fmt(v)}</td>")
        surv = f"{m['survived']:.1f} s 낙상" if m.get("survived") else "40 s 완주"
        extra = f"<td>{m['assist_rms']:.1f}</td><td>{fmt(m['corr'], 2)}</td><td class=\"txt\">{surv}</td>"
        body.append(f"<tr><th scope=\"row\">{html.escape(name)}</th>{''.join(cells)}{extra}</tr>")
    return f"""
<section>
  <h2>{html.escape(title)}</h2>
  <p class="note">{note}</p>
  <div class="scroll"><table>
    <thead><tr><th>실행</th>{head}<th>보조 RMS [N·m]</th><th>보조↔착용자 토크 상관</th><th>결과</th></tr></thead>
    <tbody>{''.join(body)}</tbody>
  </table></div>
</section>"""


def phase_chart_svg(series: dict, colors_css: list):
    """series: name -> 10 contrib values. 얇은 묶음 막대, 0선 기준, 착지 0~20 % 띠."""
    names = list(series)
    n = len(names)
    W, H, ml, mr, mt, mb = 860, 360, 62, 16, 28, 44
    pw, ph = W - ml - mr, H - mt - mb
    allv = [v for vs in series.values() for v in vs]
    lo, hi = min(min(allv), 0.0), max(max(allv), 0.0)
    span = (hi - lo) or 1.0
    lo -= 0.08 * span; hi += 0.08 * span
    y = lambda v: mt + (hi - v) / (hi - lo) * ph  # noqa: E731
    gw = pw / 10
    bw = max((gw - 12) / n - 2, 3)
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="보행 위상 구간별 착용자 τ² 변화" style="max-width:100%;height:auto;display:block">']
    parts.append(f'<rect x="{ml}" y="{mt}" width="{gw*2}" height="{ph}" class="band"/>')
    parts.append(f'<text x="{ml+gw}" y="{mt+14}" class="bandlab" text-anchor="middle">착지·하중 수용</text>')
    # 눈금
    step = _nice((hi - lo) / 5)
    tick = np.floor(lo / step) * step
    while tick <= hi:
        yy = y(tick)
        parts.append(f'<line x1="{ml}" x2="{W-mr}" y1="{yy:.1f}" y2="{yy:.1f}" class="grid"/>')
        parts.append(f'<text x="{ml-8}" y="{yy+4:.1f}" class="tick" text-anchor="end">{tick:+,.0f}</text>')
        tick += step
    parts.append(f'<line x1="{ml}" x2="{W-mr}" y1="{y(0):.1f}" y2="{y(0):.1f}" class="zero"/>')
    for b in range(10):
        x0 = ml + b * gw + 6
        parts.append(f'<text x="{ml+b*gw+gw/2:.1f}" y="{H-mb+18}" class="tick" text-anchor="middle">{10*b}–{10*b+10}</text>')
        for i, name in enumerate(names):
            v = series[name][b]
            x = x0 + i * (bw + 2)
            top, bot = (y(v), y(0)) if v >= 0 else (y(0), y(v))
            h = max(bot - top, 0.5)
            parts.append(
                f'<rect x="{x:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{h:.1f}" rx="2" fill="{colors_css[i]}" '
                f'class="bar" data-run="{html.escape(name)}" data-bin="{10*b}–{10*b+10} %" data-v="{v:+.0f}"/>'
            )
    parts.append(f'<text x="{ml-8}" y="{mt-10}" class="axis" text-anchor="end">착용자 τ² 기여 [(N·m)²]</text>')
    parts.append(f'<text x="{W-mr}" y="{H-6}" class="axis" text-anchor="end">보행 주기 [%] — 0 % = 최대 굴곡. 음수 = 무보조보다 덜 쓴다</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _nice(x):
    e = 10 ** np.floor(np.log10(max(x, 1e-9)))
    for m in (1, 2, 2.5, 5, 10):
        if m * e >= x:
            return m * e
    return 10 * e


def phase_table_html(series_rows: dict):
    names = list(series_rows)
    head = "".join(f"<th>{10*b}–{10*b+10}</th>" for b in range(10))
    body = []
    for name in names:
        rows = series_rows[name]
        cells = "".join(f"<td>{r['contrib']:+.0f}</td>" for r in rows)
        land = sum(r["contrib"] for r in rows[:2]); swing = sum(r["contrib"] for r in rows[3:7]); tot = sum(r["contrib"] for r in rows)
        body.append(f"<tr><th scope=\"row\">{html.escape(name)}</th>{cells}<td class=\"sum\">{land:+.0f}</td><td class=\"sum\">{swing:+.0f}</td><td class=\"sum\">{tot:+.0f}</td></tr>")
    body2 = []
    for name in names:
        rows = series_rows[name]
        body2.append(f"<tr><th scope=\"row\">{html.escape(name)} · 보조 τ</th>" + "".join(f"<td>{r['assist']:+.0f}</td>" for r in rows) + "<td></td><td></td><td></td></tr>")
    base_row = ""
    first = series_rows[names[0]]
    base_row = "<tr class=\"ref\"><th scope=\"row\">무보조 착용자 τ (참고)</th>" + "".join(f"<td>{r['tauA']:+.0f}</td>" for r in first) + "<td></td><td></td><td></td></tr>"
    return f"""
<div class="scroll"><table class="phase">
  <thead><tr><th>위상 [%]</th>{head}<th>착지 0–20</th><th>유각 30–70</th><th>합계</th></tr></thead>
  <tbody>{''.join(body)}{base_row}{''.join(body2)}</tbody>
</table></div>"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--walk", nargs="+", required=True, help="이름=경로glob (걷기 실행)")
    p.add_argument("--run", nargs="*", default=[], help="이름=경로glob (달리기 실행)")
    p.add_argument("--chart", default=None, help="위상 그림에 올릴 걷기 실행 이름들 (';' 구분, 최대 4; 이름에 쉼표 가능)")
    p.add_argument("--title", default="진자 v4 → 스켈레톤 전이 시험")
    p.add_argument("--verdict", default="", help="맨 위 한 줄 판정 (HTML 가능)")
    p.add_argument("--notes", default="", help="맨 아래 각주 (HTML 가능)")
    p.add_argument("--out", default="docs/report.html")
    args = p.parse_args()

    def resolve(items):
        out = {}
        for it in items:
            name, pat = it.split("=", 1)
            hits = sorted(glob.glob(pat))
            assert hits, f"no run for {pat}"
            out[name] = Path(hits[-1])
        return out

    walk = resolve(args.walk)
    run = resolve(args.run)
    walk_m = {n: metrics(r) for n, r in walk.items()}
    run_m = {n: metrics(r) for n, r in run.items()}
    chart_names = [s.strip() for s in (args.chart.split(";") if args.chart else list(walk)[:4])][:4]
    phase = {n: phase_rows(walk[n]) for n in walk}
    chart_series = {n: [r["contrib"] for r in phase[n]] for n in chart_names if phase.get(n)}
    missing = [n for n in chart_names if n not in walk]
    assert chart_series, f"--chart 이름이 --walk 이름과 맞지 않음: {missing}"
    colors = [f"var(--s{i+1})" for i in range(len(chart_series))]

    legend = "".join(
        f'<span class="lg"><i style="background:var(--s{i+1})"></i>{html.escape(n)}</span>' for i, n in enumerate(chart_series)
    )
    walk_tbl = table_html(walk_m, "걷기 (motion 0, 40 s, 착용자 v32)",
                          "무보조 A 대비 보조 B의 변화 [%]. <b>음수가 좋다.</b> τ²는 다리를 덜 움직여서는 못 내리는 지표라 기준 지표로 쓴다. 굵은 칸 = 열에서 가장 낮은 값.")
    run_tbl = table_html(run_m, "달리기 (motion 1, 40 s)",
                         "같은 잣대. 스켈레톤에서 학습한 버전(v10·v12)은 달리기에서 착용자 근토크(τ²)를 오히려 늘렸다 — "
                         "|τ·ω| 파워가 내려간 것은 각속도 감소분이다. 진자에서 학습한 v4 새 형식이 τ²를 줄인 첫 버전.") if run_m else ""

    page = f"""<title>{html.escape(args.title)}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--rule:#e4e2dc;--band:#f3efe2;--zero:#8a877f;--grid:#ecebe6;
  --s1:{SERIES[0]};--s2:{SERIES[1]};--s3:{SERIES[2]};--s4:{SERIES[3]};--best:#eef4fc}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--rule:#35342f;--band:#2a2820;--zero:#9a978e;--grid:#2a2926;
  --s1:{SERIES_DARK[0]};--s2:{SERIES_DARK[1]};--s3:{SERIES_DARK[2]};--s4:{SERIES_DARK[3]};--best:#22304a}}}}
:root[data-theme="dark"]{{--bg:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--rule:#35342f;--band:#2a2820;--zero:#9a978e;--grid:#2a2926;
  --s1:{SERIES_DARK[0]};--s2:{SERIES_DARK[1]};--s3:{SERIES_DARK[2]};--s4:{SERIES_DARK[3]};--best:#22304a}}
body{{background:var(--bg);color:var(--ink);font:15px/1.55 "IBM Plex Sans KR",system-ui,sans-serif;padding-block:32px 56px;padding-inline:20px}}
main{{max-width:900px;margin:0 auto;display:grid;gap:36px}}
h1{{font-size:26px;font-weight:600;margin:0 0 6px;text-wrap:balance;letter-spacing:-.01em}}
h2{{font-size:17px;font-weight:600;margin:0 0 6px}}
.eyebrow{{font:500 11px/1 "IBM Plex Mono",monospace;letter-spacing:.08em;text-transform:uppercase;color:var(--ink2);margin-bottom:10px}}
.verdict{{font-size:16px;max-width:66ch;margin:0}}
.note{{color:var(--ink2);font-size:13.5px;max-width:70ch;margin:0 0 12px}}
.scroll{{overflow-x:auto}}
table{{border-collapse:collapse;font-variant-numeric:tabular-nums;font-family:"IBM Plex Mono",monospace;font-size:13px;min-width:640px}}
th,td{{padding:7px 10px;text-align:right;border-bottom:1px solid var(--rule);white-space:nowrap}}
thead th{{font:500 11.5px/1.3 "IBM Plex Sans KR",sans-serif;color:var(--ink2);text-align:right;vertical-align:bottom}}
tbody th{{text-align:left;font-family:"IBM Plex Sans KR",sans-serif;font-weight:600}}
td.best{{background:var(--best);font-weight:500}}
td.txt{{text-align:left;font-family:"IBM Plex Sans KR",sans-serif}}
td.sum{{font-weight:500;border-left:1px solid var(--rule)}}
tr.ref td,tr.ref th{{color:var(--ink2);font-weight:400}}
.legend{{display:flex;flex-wrap:wrap;gap:14px;font-size:13px;margin:0 0 8px}}
.lg i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}}
svg .grid{{stroke:var(--grid);stroke-width:1}} svg .zero{{stroke:var(--zero);stroke-width:1.2}}
svg .band{{fill:var(--band)}} svg .bandlab{{fill:var(--ink2);font:12px "IBM Plex Sans KR",sans-serif}}
svg .tick{{fill:var(--ink2);font:11px "IBM Plex Mono",monospace}} svg .axis{{fill:var(--ink2);font:11.5px "IBM Plex Sans KR",sans-serif}}
svg .bar:hover{{opacity:.75}}
.tip{{position:fixed;pointer-events:none;background:var(--ink);color:var(--bg);font:12.5px "IBM Plex Mono",monospace;padding:6px 9px;border-radius:4px;transform:translate(-50%,-130%);white-space:nowrap}}
.foot{{color:var(--ink2);font-size:13px;max-width:70ch;border-top:1px solid var(--rule);padding-top:14px}}
@media (prefers-reduced-motion: no-preference){{svg .bar{{transition:opacity .12s}}}}
</style>
<main>
  <header>
    <div class="eyebrow">AI-SBC · LLP 전이 시험 · 2026-09-23</div>
    <h1>{html.escape(args.title)}</h1>
    <p class="verdict">{args.verdict}</p>
  </header>
  {walk_tbl}
  {run_tbl}
  <section>
    <h2>어느 구간에서 벌고 어느 구간에서 잃는가 — 걷기 위상 그림</h2>
    <p class="note">보행 주기를 10 % 구간으로 나눠, 각 구간이 착용자 τ² 평균 변화에 기여한 양(절대 (N·m)²). 0 % = 최대 굴곡 ≈ 착지. 막대에 마우스를 올리면 값이 보인다.</p>
    <div class="legend">{legend}</div>
    {phase_chart_svg(chart_series, colors)}
    <h2 style="margin-top:18px">위상표 (같은 수치 + 구간별 보조 토크 평균 [N·m])</h2>
    {phase_table_html({n: phase[n] for n in walk if phase.get(n)})}
  </section>
  <p class="foot">{args.notes}</p>
</main>
<div id="tip" class="tip" hidden></div>
<script>
(function(){{
  var tip=document.getElementById('tip');
  document.querySelectorAll('svg .bar').forEach(function(b){{
    b.addEventListener('mousemove',function(e){{tip.hidden=false;tip.textContent=b.dataset.run+' · '+b.dataset.bin+' · '+b.dataset.v+' (N·m)²';tip.style.left=e.clientX+'px';tip.style.top=e.clientY+'px';}});
    b.addEventListener('mouseleave',function(){{tip.hidden=true;}});
  }});
}})();
</script>
"""
    Path(args.out).write_text(page)
    print("saved", args.out, f"({len(page)//1024} KB)")


if __name__ == "__main__":
    main()
