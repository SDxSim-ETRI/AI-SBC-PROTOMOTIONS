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
"""LLP 체크포인트 하나를 깊게 보는 HTML 보고서 — 비교표 + 위상 그림 + 시계열 + 주기 평균 + 학습 궤적.

`report_html.py`가 여러 실행을 한 표에 올리는 "비교" 보고서라면, 이 스크립트는 **보관할 체크포인트
하나**(예: v13 = 진자 v4 epoch 6,000)가 스켈레톤에서 실제로 무엇을 하는지 파형 수준까지 보여 준다.

섹션:
1. 핵심 숫자 (걷기·달리기: τ², |τ·ω|, 몸체 오차, 보조 RMS, 포화, 상관, 완주)
2. 비교표 — 기준 실행(v10, β 오라클)과 같은 잣대 (`report_html.table_html` 재사용)
3. 걷기·달리기 위상별 τ² 기여 그림·표, |ω| 표
4. 시계열 3 스트라이드 — hip 각도(A·B·mocap), 착용자 토크(A·B), 보조 토크(B)
5. 주기 평균 프로파일 — 착용자 토크 A·B, 보조 토크 평균 ± 표준편차
6. 학습 궤적 — epoch별 스켈레톤 전이 지표 + 진자 안 보상 항 (tensorboard)

실행 (sbc env):
    python examples/experiments/assist_skeleton/report_checkpoint_html.py --out docs/llp_v13_report.html \\
        --name "v13" --walk "results/oracle/*_tracker_v4new6k_m0" --run "results/oracle/*_tracker_v4new6k_m1" \\
        --ref-walk "v10=results/oracle/*_tracker_base" "β 오라클=results/oracle/*_tracker_beta05" \\
        --ref-run "v10=results/oracle/2026-09-16_19-48-34_tracker_v10_m1" \\
        --traj "2000=results/oracle/*_tracker_v4new2k" "6000=results/oracle/*_tracker_v4new6k" ... \\
        --tb-dir results/assist_pendulum_v4/lightning_logs
"""

import argparse
import glob
import html
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis import assist_output_limit, assist_saturation_pct  # noqa: E402
from report_html import (  # noqa: E402
    A,
    B,
    SERIES,
    SERIES_DARK,
    fmt,
    load,
    metrics,
    omega_table_html,
    phase_chart_svg,
    phase_rows,
    phase_table_html,
    table_html,
)


# ----------------------------------------------------------------------------
# 수치
# ----------------------------------------------------------------------------
def extra_metrics(run: Path):
    """report_html.metrics()에 없는 것: 포화 비율, 부호 일치, 변화율, 스트라이드 주기."""
    tr, meta, idx = load(run)
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


def cycle_profile(run: Path, bins=50):
    """위상 bin별 착용자 토크 A·B 평균, 보조 토크 평균·표준편차 (양쪽 다리 합침)."""
    tr, meta, idx = load(run)
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


def time_window(run: Path, strides=3, t_from=10.0):
    """오른다리 위상 0 % 경계에서 시작하는 `strides` 스트라이드 창."""
    tr, meta, idx = load(run)
    t = tr["t"]
    ph = tr["gait_phase"][:, 0]
    wraps = np.where(np.diff(ph) < -0.5)[0] + 1
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


def tb_curves(tb_dir: str, tags: dict, max_epoch: int, smooth: int = 100):
    """lightning_logs/*/events.* 를 epoch 기준으로 합친다 (재시작 구간은 나중 파일이 덮는다)."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return None
    files = sorted(glob.glob(os.path.join(tb_dir, "*", "events.out.tfevents.*")), key=os.path.getmtime)
    if not files:
        return None
    data = {k: {} for k in tags}
    for f in files:
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        avail = set(ea.Tags()["scalars"])
        first = None
        for k, tag in tags.items():
            if tag not in avail:
                continue
            ev = ea.Scalars(tag)
            if first is None:
                first = ev[0].step if ev else None
            for e in ev:
                if e.step <= max_epoch:
                    data[k][e.step] = e.value
        if first is not None and first > max_epoch:
            break
    out = {}
    for k, d in data.items():
        if not d:
            continue
        steps = np.array(sorted(d))
        vals = np.array([d[s] for s in steps])
        if smooth > 1 and len(vals) > smooth:
            kern = np.ones(smooth) / smooth
            vals = np.convolve(vals, kern, mode="same")
            steps, vals = steps[smooth // 2 : -smooth // 2], vals[smooth // 2 : -smooth // 2]
        out[k] = (steps, vals)
    return out


# ----------------------------------------------------------------------------
# SVG
# ----------------------------------------------------------------------------
def _nice(x):
    e = 10 ** np.floor(np.log10(max(x, 1e-9)))
    for m in (1, 2, 2.5, 5, 10):
        if m * e >= x:
            return m * e
    return 10 * e


def _path(xs, ys, fx, fy):
    pts = [f"{fx(x):.1f},{fy(y):.1f}" for x, y in zip(xs, ys) if np.isfinite(y)]
    return "M" + " L".join(pts) if pts else ""


def line_svg(x, series, *, ylabel, xlabel=None, ylim=None, xlim=None, vlines=(), hlines=(), band=None,
             H=210, W=860, xtick=None, label=""):
    """series: list of dict(y, color, width, dash). band: (lo, hi, color)."""
    ml, mr, mt, mb = 58, 14, 14, 34 if xlabel else 18
    pw, ph = W - ml - mr, H - mt - mb
    xlo, xhi = xlim if xlim else (float(np.min(x)), float(np.max(x)))
    if ylim is None:
        ys = np.concatenate([np.asarray(s["y"], dtype=float) for s in series] + ([band[0], band[1]] if band else []))
        ys = ys[np.isfinite(ys)]
        lo, hi = float(ys.min()), float(ys.max())
        for h in hlines:
            lo, hi = min(lo, h), max(hi, h)
        pad = 0.08 * (hi - lo or 1.0)
        ylim = (lo - pad, hi + pad)
    ylo, yhi = ylim
    fx = lambda v: ml + (v - xlo) / (xhi - xlo or 1.0) * pw  # noqa: E731
    fy = lambda v: mt + (yhi - v) / (yhi - ylo or 1.0) * ph  # noqa: E731
    p = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(label or ylabel)}" style="max-width:100%;height:auto;display:block">']
    # 세로 표시선 (위상 0 %)
    for v in vlines:
        p.append(f'<line x1="{fx(v):.1f}" x2="{fx(v):.1f}" y1="{mt}" y2="{mt+ph}" class="mark"/>')
    # 가로 눈금
    step = _nice((yhi - ylo) / 4)
    tick = np.ceil(ylo / step) * step
    while tick <= yhi + 1e-9:
        yy = fy(tick)
        p.append(f'<line x1="{ml}" x2="{W-mr}" y1="{yy:.1f}" y2="{yy:.1f}" class="grid"/>')
        p.append(f'<text x="{ml-8}" y="{yy+4:.1f}" class="tick" text-anchor="end">{tick:g}</text>')
        tick += step
    if ylo < 0 < yhi:
        p.append(f'<line x1="{ml}" x2="{W-mr}" y1="{fy(0):.1f}" y2="{fy(0):.1f}" class="zero"/>')
    for h in hlines:
        p.append(f'<line x1="{ml}" x2="{W-mr}" y1="{fy(h):.1f}" y2="{fy(h):.1f}" class="limit"/>')
    # 가로 눈금
    xt = xtick or _nice((xhi - xlo) / 6)
    tick = np.ceil(xlo / xt) * xt
    while tick <= xhi + 1e-9:
        xx = fx(tick)
        p.append(f'<text x="{xx:.1f}" y="{H-mb+16}" class="tick" text-anchor="middle">{tick:g}</text>')
        tick += xt
    if band:
        lo, hi, col = band
        up = _path(x, hi, fx, fy)
        down = _path(x[::-1], np.asarray(lo)[::-1], fx, fy)
        if up and down:
            p.append(f'<path d="{up} {down.replace("M", "L", 1)} Z" fill="{col}" opacity="0.18" stroke="none"/>')
    for s in series:
        d = _path(x, s["y"], fx, fy)
        dash = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
        p.append(f'<path d="{d}" fill="none" stroke="{s["color"]}" stroke-width="{s.get("width", 1.6)}"{dash} stroke-linejoin="round"/>')
    p.append(f'<text x="{ml-44}" y="{mt+ph/2:.0f}" class="axis" text-anchor="middle" transform="rotate(-90 {ml-44} {mt+ph/2:.0f})">{html.escape(ylabel)}</text>')
    if xlabel:
        p.append(f'<text x="{W-mr}" y="{H-4}" class="axis" text-anchor="end">{html.escape(xlabel)}</text>')
    p.append("</svg>")
    return "\n".join(p)


def legend_html(items):
    return '<div class="legend">' + "".join(
        f'<span class="lg"><i style="background:{c}{";height:2px" if dash else ""}"></i>{html.escape(n)}</span>' for n, c, dash in items
    ) + "</div>"


# ----------------------------------------------------------------------------
# 섹션
# ----------------------------------------------------------------------------
def key_numbers_html(name, wm, wx, rm, rx):
    def card(title, m, x):
        surv = f"{m['survived']:.1f} s 낙상" if m.get("survived") else "40 s 완주"
        rows = [
            ("착용자 τ² (근활성)", fmt(m["tau2"]) + " %", "hero"),
            ("착용자 |τ·ω| 파워", fmt(m["P"]) + " %", ""),
            ("몸체 추종 오차", fmt(m.get("body")) + " %", ""),
            ("hip 각속도 |ω|", fmt(m["omega"]) + " %", ""),
            ("보조 토크 RMS", f"{m['assist_rms']:.1f} N·m", ""),
            ("한계 도달 시간", f"{x['sat']:.0f} %", ""),
            ("착용자 토크와 상관 / 부호 일치", f"{m['corr']:+.2f} / {x['agree']:.0f} %", ""),
            ("결과", surv, ""),
        ]
        cells = "".join(
            f'<div class="kv{(" " + cls) if cls else ""}"><dt>{html.escape(k)}</dt><dd>{html.escape(v)}</dd></div>' for k, v, cls in rows
        )
        return f'<div class="card"><h3>{html.escape(title)}</h3><dl>{cells}</dl></div>'

    return f"""
<section>
  <h2>핵심 숫자 — {html.escape(name)}, 무보조 A 대비 보조 B (40 s 짝비교)</h2>
  <p class="note">τ²는 착용자 hip 토크 제곱의 좌우 합을 시간 평균한 기준 지표이며 실제 근활성 측정값은 아니다. |τ·ω|와 달리 각속도를 직접 곱하지 않지만, 동작이 바뀌면 τ²도 달라질 수 있어 전진 거리·몸체 오차와 함께 판단한다.</p>
  <div class="cards">{card("걷기 (motion 0)", wm, wx)}{card("달리기 (motion 1)", rm, rx)}</div>
</section>"""


def timeseries_html(name, win, gait, stride):
    t = win["t"]
    marks = win["marks"]
    lim = win["limit"]
    xlim = (0.0, float(t[-1]))
    s_ang = [
        {"y": win["ref"], "color": "var(--s3)", "width": 1.3, "dash": "4 3"},
        {"y": win["angA"], "color": "var(--a)", "width": 1.6},
        {"y": win["angB"], "color": "var(--s2)", "width": 1.8},
    ]
    s_tau = [
        {"y": win["tauA"], "color": "var(--a)", "width": 1.6},
        {"y": win["tauB"], "color": "var(--s2)", "width": 1.8},
    ]
    s_as = [{"y": win["assist"], "color": "var(--s1)", "width": 1.8}]
    return f"""
<section>
  <h2>{html.escape(gait)} 시계열 — 오른쪽 hip, 3 스트라이드 (주기 {stride:.2f} s)</h2>
  <p class="note">세로 점선 = 위상 0 % (그 다리의 최대 굴곡 ≈ 착지 직전). A(무보조)와 B(보조)는 같은 mocap 클립·같은 초기 상태에서 나란히 돌린 별개 시뮬이라 시간축이 같다.</p>
  {legend_html([("mocap 기준 각도", "var(--s3)", True), ("A 무보조", "var(--a)", False), ("B 보조 " + name, "var(--s2)", False)])}
  {line_svg(t, s_ang, ylabel="hip 굴곡 [°]", xlim=xlim, vlines=marks, H=190, xtick=0.5)}
  {line_svg(t, s_tau, ylabel="착용자 hip 토크 [N·m]", xlim=xlim, vlines=marks, H=230, xtick=0.5)}
  {legend_html([("보조 토크 (B)", "var(--s1)", False)])}
  {line_svg(t, s_as, ylabel="보조 토크 [N·m]", xlabel="시간 [s]", xlim=xlim, ylim=(-lim*1.15, lim*1.15), vlines=marks, hlines=(-lim, lim), H=170, xtick=0.5)}
</section>"""


def profile_html(name, prof, gait):
    ph = prof["phase"]
    s_tau = [
        {"y": prof["tauA"], "color": "var(--a)", "width": 1.6},
        {"y": prof["tauB"], "color": "var(--s2)", "width": 1.8},
    ]
    s_as = [{"y": prof["assist"], "color": "var(--s1)", "width": 1.8}]
    band = (prof["assist"] - prof["assist_sd"], prof["assist"] + prof["assist_sd"], SERIES[0])
    return f"""
<section>
  <h2>{html.escape(gait)} 주기 평균 — 위상별 착용자 토크와 보조 토크 (양쪽 다리, 39 s)</h2>
  <p class="note">0 % = 최대 굴곡. 착지 직후(0~20 %) 착용자는 큰 신전(−) 토크를 내고, 유각(30~70 %)에는 굴곡(+) 토크를 낸다. 보조가 착용자 토크와 같은 부호로 나오는 구간이 "덜어 주는" 구간이다. 띠 = 보조 토크 ±1 표준편차(스트라이드 간 흔들림).</p>
  {legend_html([("A 무보조 착용자 토크", "var(--a)", False), ("B 보조 시 착용자 토크", "var(--s2)", False)])}
  {line_svg(ph, s_tau, ylabel="착용자 hip 토크 [N·m]", xlim=(0, 100), H=220, xtick=10)}
  {legend_html([("보조 토크 평균 ± 1σ (B)", "var(--s1)", False)])}
  {line_svg(ph, s_as, ylabel="보조 토크 [N·m]", xlabel="보행 주기 [%]", xlim=(0, 100), band=band, H=180, xtick=10)}
</section>"""


def trajectory_html(traj, curves, pick_epoch):
    rows = []
    for ep, (wm, wx, rm, rx) in traj.items():
        cls = ' class="pick"' if ep == pick_epoch else ""
        rows.append(
            f"<tr{cls}><th scope=\"row\">{ep:,}</th><td>{fmt(wm['tau2'])}</td><td>{fmt(rm['tau2']) if rm else '—'}</td>"
            f"<td>{fmt(wm.get('body'))}</td><td>{wm['assist_rms']:.1f}</td><td>{wx['sat']:.0f}</td><td>{fmt(wm['corr'],2)}</td>"
            f"<td>{fmt(rm['corr'],2) if rm else '—'}</td></tr>"
        )
    tbl = f"""
<div class="scroll"><table>
  <thead><tr><th>epoch</th><th>걷기 τ² [%]</th><th>달리기 τ² [%]</th><th>걷기 몸체 오차 [%]</th><th>보조 RMS [N·m]</th><th>포화 [%]</th><th>상관 (걷기)</th><th>상관 (달리기)</th></tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table></div>"""
    charts = ""
    if curves:
        def one(key, ylabel, color):
            if key not in curves:
                return ""
            st, va = curves[key]
            return line_svg(st, [{"y": va, "color": color, "width": 1.8}], ylabel=ylabel, xlabel="epoch",
                            xlim=(0, float(st.max())), vlines=list(traj.keys()), H=200, W=420, xtick=5000)
        charts = f"""
<div class="grid2">
  <div>{one("tau2", "진자 안 착용자 τ² (raw)", "var(--s2)")}</div>
  <div>{one("effort", "보조 토크 절약 항 Στ²_assist (raw)", "var(--s1)")}</div>
</div>"""
    return f"""
<section>
  <h2>왜 epoch 6,000인가 — 학습 궤적</h2>
  <p class="note">같은 run의 체크포인트를 스켈레톤에 차례로 올린 결과(위 표)와 진자 안 보상 항(아래 그림, 100 epoch 이동평균, 세로선 = 평가한 epoch). 진자 안 착용자 τ²는 5,000 이후 제자리인데 보조 토크 절약 항은 계속 내려간다 — 후기 학습은 "덜 밀기"를 배우고, 하중이 3배인 스켈레톤에서는 그것이 덜 돕는 것으로 나타난다.</p>
  {tbl}
  {charts}
</section>"""


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="v13")
    p.add_argument("--walk", required=True, help="이 체크포인트의 걷기 실행 glob")
    p.add_argument("--run", required=True, help="이 체크포인트의 달리기 실행 glob")
    p.add_argument("--ref-walk", nargs="*", default=[], help="이름=glob (걷기 기준 실행)")
    p.add_argument("--ref-run", nargs="*", default=[], help="이름=glob (달리기 기준 실행)")
    p.add_argument("--traj", nargs="*", default=[], help="epoch=glob접두 (뒤에 _m0/_m1이 붙는 실행 쌍)")
    p.add_argument("--tb-dir", default=None)
    p.add_argument("--tb-max-epoch", type=int, default=20000)
    p.add_argument("--title", default="LLP v13 스켈레톤 시험")
    p.add_argument("--eyebrow", default="AI-SBC · LLP 체크포인트 보고서")
    p.add_argument("--verdict", default="")
    p.add_argument("--notes", default="")
    p.add_argument("--out", default="docs/report_ckpt.html")
    args = p.parse_args()

    def one(pat):
        hits = sorted(glob.glob(pat))
        assert hits, f"no run for {pat}"
        return Path(hits[-1])

    def many(items):
        return {n: one(g) for n, g in (it.split("=", 1) for it in items)}

    walk, run = one(args.walk), one(args.run)
    ref_w, ref_r = many(args.ref_walk), many(args.ref_run)
    wm, rm = metrics(walk), metrics(run)
    wx, rx = extra_metrics(walk), extra_metrics(run)

    walk_rows = {**{n: metrics(r) for n, r in ref_w.items()}, args.name: wm}
    run_rows = {**{n: metrics(r) for n, r in ref_r.items()}, args.name: rm}
    walk_tbl = table_html(walk_rows, "걷기 — 기준 실행과 같은 잣대",
                          "무보조 A 대비 보조 B의 변화 [%]. τ²와 함께 전진 거리·몸체 오차·완주 여부를 확인한다. β 오라클은 착용자 토크를 사용하는 특권 참고값이며 배포 정책이 아니다. 굵은 칸 = 열에서 가장 낮은 값.")
    run_tbl = table_html(run_rows, "달리기 — 기준 실행과 같은 잣대",
                         "같은 지표로 비교한다. 보조 배율·착용자·제어 주기 등 실행 조건과 원본 경로는 아래 설정을 확인한다.")

    phase_w = {n: phase_rows(r) for n, r in {**ref_w, args.name: walk}.items()}
    phase_r = {n: phase_rows(r) for n, r in {**ref_r, args.name: run}.items()}
    chart_w = {n: [q["contrib"] for q in v] for n, v in phase_w.items() if v}
    chart_r = {n: [q["contrib"] for q in v] for n, v in phase_r.items() if v}
    colors = [f"var(--s{i+1})" for i in range(4)]
    leg = lambda names: "".join(f'<span class="lg"><i style="background:var(--s{i+1})"></i>{html.escape(n)}</span>' for i, n in enumerate(names))  # noqa: E731

    traj = {}
    for it in args.traj:
        ep, pre = it.split("=", 1)
        w = sorted(glob.glob(pre + "_m0"))
        r = sorted(glob.glob(pre + "_m1"))
        if not w:
            continue
        w = Path(w[-1])
        r = Path(r[-1]) if r else None
        traj[int(ep)] = (metrics(w), extra_metrics(w), metrics(r) if r else None, extra_metrics(r) if r else None)
    curves = tb_curves(args.tb_dir, {"tau2": "env/raw_r/human_torque_sq_mean", "effort": "env/raw_r/assist_effort_mean"},
                       args.tb_max_epoch) if args.tb_dir else None
    pick = int(load(walk)[1].get("llp_epoch") or 0)

    page = f"""<meta charset="utf-8">
<title>{html.escape(args.title)}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--rule:#e4e2dc;--band:#f3efe2;--zero:#8a877f;--grid:#ecebe6;--a:#8a877f;--limit:#c9c5b8;
  --s1:{SERIES[0]};--s2:{SERIES[1]};--s3:{SERIES[2]};--s4:{SERIES[3]};--best:#eef4fc;--card:#f6f5f1;--pick:#fbf1e9}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--rule:#35342f;--band:#2a2820;--zero:#9a978e;--grid:#2a2926;--a:#a3a096;--limit:#4a483f;
  --s1:{SERIES_DARK[0]};--s2:{SERIES_DARK[1]};--s3:{SERIES_DARK[2]};--s4:{SERIES_DARK[3]};--best:#22304a;--card:#232320;--pick:#3a2a20}}}}
:root[data-theme="dark"]{{--bg:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--rule:#35342f;--band:#2a2820;--zero:#9a978e;--grid:#2a2926;--a:#a3a096;--limit:#4a483f;
  --s1:{SERIES_DARK[0]};--s2:{SERIES_DARK[1]};--s3:{SERIES_DARK[2]};--s4:{SERIES_DARK[3]};--best:#22304a;--card:#232320;--pick:#3a2a20}}
body{{background:var(--bg);color:var(--ink);font:15px/1.55 "IBM Plex Sans KR",system-ui,sans-serif;padding-block:32px 56px;padding-inline:20px}}
main{{max-width:900px;margin:0 auto;display:grid;gap:40px}}
main > *{{min-width:0}}
h1{{font-size:26px;font-weight:600;margin:0 0 6px;text-wrap:balance;letter-spacing:-.01em}}
h2{{font-size:17px;font-weight:600;margin:0 0 6px;text-wrap:balance}}
h3{{font-size:14px;font-weight:600;margin:0 0 10px;color:var(--ink2)}}
.eyebrow{{font:500 11px/1 "IBM Plex Mono",monospace;letter-spacing:.08em;text-transform:uppercase;color:var(--ink2);margin-bottom:10px}}
.verdict{{font-size:16px;max-width:66ch;margin:0}}
.note{{color:var(--ink2);font-size:13.5px;max-width:72ch;margin:0 0 12px}}
.scroll{{overflow-x:auto}}
table{{border-collapse:collapse;font-variant-numeric:tabular-nums;font-family:"IBM Plex Mono",monospace;font-size:13px;min-width:640px}}
th,td{{padding:7px 10px;text-align:right;border-bottom:1px solid var(--rule);white-space:nowrap}}
thead th{{font:500 11.5px/1.3 "IBM Plex Sans KR",sans-serif;color:var(--ink2);text-align:right;vertical-align:bottom}}
tbody th{{text-align:left;font-family:"IBM Plex Sans KR",sans-serif;font-weight:600}}
td.best{{background:var(--best);font-weight:500}}
td.txt{{text-align:left;font-family:"IBM Plex Sans KR",sans-serif}}
td.sum{{font-weight:500;border-left:1px solid var(--rule)}}
tr.ref td,tr.ref th{{color:var(--ink2);font-weight:400}}
tr.pick td,tr.pick th{{background:var(--pick);font-weight:500}}
.dim{{color:var(--ink2)}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}}
.card{{background:var(--card);border-radius:6px;padding:16px 18px}}
.card dl{{margin:0;display:grid;gap:6px}}
.kv{{display:flex;justify-content:space-between;gap:12px;align-items:baseline;border-bottom:1px solid var(--rule);padding-bottom:5px}}
.kv dt{{color:var(--ink2);font-size:13px}}
.kv dd{{margin:0;font:500 14px "IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}}
.kv.hero dd{{font-size:22px;color:var(--s2)}}
.legend{{display:flex;flex-wrap:wrap;gap:14px;font-size:13px;margin:0 0 6px}}
.lg i{{display:inline-block;width:12px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}}
.grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px;margin-top:14px}}
svg .grid{{stroke:var(--grid);stroke-width:1}} svg .zero{{stroke:var(--zero);stroke-width:1.2}}
svg .mark{{stroke:var(--zero);stroke-width:1;stroke-dasharray:3 3}} svg .limit{{stroke:var(--limit);stroke-width:1.2;stroke-dasharray:6 3}}
svg .band{{fill:var(--band)}} svg .bandlab{{fill:var(--ink2);font:12px "IBM Plex Sans KR",sans-serif}}
svg .tick{{fill:var(--ink2);font:11px "IBM Plex Mono",monospace}} svg .axis{{fill:var(--ink2);font:11.5px "IBM Plex Sans KR",sans-serif}}
svg .bar:hover{{opacity:.75}}
.tip{{position:fixed;pointer-events:none;background:var(--ink);color:var(--bg);font:12.5px "IBM Plex Mono",monospace;padding:6px 9px;border-radius:4px;transform:translate(-50%,-130%);white-space:nowrap}}
.foot{{color:var(--ink2);font-size:13px;max-width:72ch;border-top:1px solid var(--rule);padding-top:14px}}
.foot p{{margin:0 0 8px}}
@media (prefers-reduced-motion: no-preference){{svg .bar{{transition:opacity .12s}}}}
</style>
<main>
  <header>
    <div class="eyebrow">{html.escape(args.eyebrow)}</div>
    <h1>{html.escape(args.title)}</h1>
    <p class="verdict">{args.verdict}</p>
  </header>
  {key_numbers_html(args.name, wm, wx, rm, rx)}
  {walk_tbl}
  {run_tbl}
  <section>
    <h2>어느 구간에서 벌고 어느 구간에서 잃는가 — 걷기 위상 그림</h2>
    <p class="note">보행 주기를 10 % 구간으로 나눠, 각 구간이 착용자 τ² 평균 변화에 기여한 양(절대 (N·m)²). 0 % = 최대 굴곡 ≈ 착지. 막대에 마우스를 올리면 값이 보인다.</p>
    <div class="legend">{leg(chart_w)}</div>
    {phase_chart_svg(chart_w, colors)}
    <h2 style="margin-top:18px">걷기 위상표 (같은 수치 + 구간별 보조 토크 평균 [N·m])</h2>
    {phase_table_html(phase_w)}
    <h2 style="margin-top:22px">걷기 위상별 hip 각속도 |ω| 변화 (A → B, %)</h2>
    <p class="note">보조가 착용자의 다리를 어디서 덜(또는 더) 움직이게 하는가. 음수 = 보조 켤 때 그 구간에서 다리가 느리다.</p>
    {omega_table_html(phase_w)}
  </section>
  <section>
    <h2>달리기 위상 그림·표</h2>
    <p class="note">같은 방식. 달리기는 주기가 짧고 착지 하중이 커서 착지 구간(0~20 %)의 기여가 걷기보다 크다.</p>
    <div class="legend">{leg(chart_r)}</div>
    {phase_chart_svg(chart_r, colors)}
    {phase_table_html(phase_r)}
  </section>
  {timeseries_html(args.name, time_window(walk), "걷기", wx["stride"])}
  {timeseries_html(args.name, time_window(run), "달리기", rx["stride"])}
  {profile_html(args.name, cycle_profile(walk), "걷기")}
  {profile_html(args.name, cycle_profile(run), "달리기")}
  {trajectory_html(traj, curves, pick) if traj else ""}
  <div class="foot">{args.notes}</div>
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
    print(f"walk: tau2 {wm['tau2']:+.1f} %  P {wm['P']:+.1f} %  body {wm.get('body', float('nan')):+.1f} %  rms {wm['assist_rms']:.1f}  sat {wx['sat']:.1f} %  corr {wm['corr']:+.2f}  agree {wx['agree']:.0f} %  stride {wx['stride']:.3f} s")
    print(f"run : tau2 {rm['tau2']:+.1f} %  P {rm['P']:+.1f} %  body {rm.get('body', float('nan')):+.1f} %  rms {rm['assist_rms']:.1f}  sat {rx['sat']:.1f} %  corr {rm['corr']:+.2f}  agree {rx['agree']:.0f} %  stride {rx['stride']:.3f} s")
    for ep, (a, b, c, d) in traj.items():
        print(f"traj {ep}: walk tau2 {a['tau2']:+.1f}  run tau2 {(c['tau2'] if c else float('nan')):+.1f}  rms {a['assist_rms']:.1f}  sat {b['sat']:.0f}")


if __name__ == "__main__":
    main()
