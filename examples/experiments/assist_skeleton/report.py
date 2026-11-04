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
"""run_system.py 기록의 HTML 보고서 — 공통 틀(HtmlShell·Svg·Fmt·Figure)과 보고서 세 종류.

  `ComparisonReport` (`comparison`)  실행 여러 개를 한 표로: 비교표 + 위상 그림 + 위상표 (옛 report_html)
  `CheckpointReport` (`checkpoint`)  체크포인트 하나를 깊게: 핵심 숫자·비교표·위상·시계열·주기 평균·학습 궤적 (옛 report_checkpoint_html)
  `FinetuneReport`   (`finetune`)    추가학습 전후 비교: 보행 진행률 10 % 구간별 토크 실효값 감소량 + 전체 지표
수치 계산은 전부 `analysis.py`에 있고 여기서는 HTML·SVG만 만든다. 생성 뒤 `python docs/ai_sbc/organize_reports.py`를 실행한다.
2026-11-04 L1 계보 정리로 `torque`·`swing-gate` 보고서는 `results/llp_wide100_sequence/scripts/`,
`results/walk_phase_analysis_20261014/scripts/`로 옮겼다.

    python examples/experiments/assist_skeleton/report.py comparison --out <html> --walk base=results/oracle/*_tracker_base ...
    python examples/experiments/assist_skeleton/report.py checkpoint --out <html> --name v13 --walk <glob> --run <glob> ...
"""

import argparse
import glob
import html
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis import (  # noqa: E402
    A,
    B,
    ObjectiveComparison,
    RunRecord,
    TorqueAlignment,
    latest_run,
)


# ---------------------------------------------------------------------------
# 공통 틀
# ---------------------------------------------------------------------------
class HtmlShell:
    """세 가지 시각 스타일 — plex(비교·체크포인트), torque(시퀀스 보고서 템플릿), study(연구 보고서)."""

    SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]        # 검증된 범주 팔레트 (light)
    SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500"]
    PLEX_FONTS = '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">'

    @classmethod
    def plex_css(cls):
        S, D = cls.SERIES, cls.SERIES_DARK
        return f"""
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--rule:#e4e2dc;--band:#f3efe2;--zero:#8a877f;--grid:#ecebe6;--a:#8a877f;--limit:#c9c5b8;
  --s1:{S[0]};--s2:{S[1]};--s3:{S[2]};--s4:{S[3]};--best:#eef4fc;--card:#f6f5f1;--pick:#fbf1e9}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--rule:#35342f;--band:#2a2820;--zero:#9a978e;--grid:#2a2926;--a:#a3a096;--limit:#4a483f;
  --s1:{D[0]};--s2:{D[1]};--s3:{D[2]};--s4:{D[3]};--best:#22304a;--card:#232320;--pick:#3a2a20}}}}
:root[data-theme="dark"]{{--bg:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--rule:#35342f;--band:#2a2820;--zero:#9a978e;--grid:#2a2926;--a:#a3a096;--limit:#4a483f;
  --s1:{D[0]};--s2:{D[1]};--s3:{D[2]};--s4:{D[3]};--best:#22304a;--card:#232320;--pick:#3a2a20}}
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
svg .leg{{fill:var(--ink);font:12px "IBM Plex Sans KR",sans-serif}} svg .legbg{{fill:var(--bg);opacity:.92}}
svg .bar:hover{{opacity:.75}}
.tip{{position:fixed;pointer-events:none;background:var(--ink);color:var(--bg);font:12.5px "IBM Plex Mono",monospace;padding:6px 9px;border-radius:4px;transform:translate(-50%,-130%);white-space:nowrap}}
.foot{{color:var(--ink2);font-size:13px;max-width:72ch;border-top:1px solid var(--rule);padding-top:14px}}
.foot p{{margin:0 0 8px}}
@media (prefers-reduced-motion: no-preference){{svg .bar{{transition:opacity .12s}}}}
"""

    TOOLTIP_SCRIPT = """<div id="tip" class="tip" hidden></div>
<script>
(function(){
  var tip=document.getElementById('tip');
  document.querySelectorAll('svg .bar').forEach(function(b){
    b.addEventListener('mousemove',function(e){tip.hidden=false;tip.textContent=b.dataset.run+' · '+b.dataset.bin+' · '+b.dataset.v+' (N·m)²';tip.style.left=e.clientX+'px';tip.style.top=e.clientY+'px';});
    b.addEventListener('mouseleave',function(){tip.hidden=true;});
  });
})();
</script>
"""


class Fmt:
    @staticmethod
    def signed(v, digits=1, sign=True):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "—"
        return f"{v:+.{digits}f}" if sign else f"{v:.{digits}f}"


    @staticmethod
    def cell(t, spec=".1f"):
        m, lo, hi = t
        return "—" if m is None else f"<b>{format(m, spec)}</b><br><small>{format(lo, spec)} ~ {format(hi, spec)}</small>"


class Svg:
    """손으로 그리는 SVG — 막대·선·범례. present = True면 발표 자료용(Figure)으로, 큰 글자(축 이름 20 px > 숫자 15 px)에 맞춰 여백을 넓힌다."""

    present = False
    GAP = 6  # 범례 상자와 그래프 틀 사이 여백
    LEG = {True: (14, 16, 18, 6), False: (11, 12, 14, 5)}  # 발표용/보고서: (글자 크기, 견본 폭, 줄 간격, 안쪽 여백)

    @classmethod
    def frame(cls, big=None, xlabel=True):
        """(ml, mr, mt, mb, tk, tx, yl, xl): 여백, 세로 눈금 x 간격, 가로 눈금 y 간격, 세로축 이름 x, 가로축 이름 y. big = 발표 자료용(Svg.present)."""
        big = cls.present if big is None else big
        if big:
            return 92, 22, 20, (64 if xlabel else 30), 12, 24, 68, 8
        return 58, 14, 14, (34 if xlabel else 18), 8, 16, 44, 4

    @classmethod
    def legend_dims(cls, items, big=None):
        """[(이름, 색, 점선)] 범례 상자의 (폭, 높이). 글자 폭은 CJK 1.0·그 외 0.58 em으로 어림한다."""
        fs, sw, row, pad = cls.LEG[cls.present if big is None else big]
        est = lambda t: sum(fs * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in t)  # noqa: E731
        return pad + sw + 6 + max(est(n) for n, _, _ in items) + pad, 2 * pad + (len(items) - 1) * row + fs

    @classmethod
    def legend_box(cls, items, x, y, big=None):
        """그래프 안 작은 범례 — 한 줄에 하나씩, (x, y) = 왼쪽 위 모서리. 점선 항목은 점선 견본."""
        big = cls.present if big is None else big
        fs, sw, row, pad = cls.LEG[big]
        w, h = cls.legend_dims(items, big)
        sh = round(fs * 0.75)
        out = [f'<rect class="legbg" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="3"/>']
        for i, (name, color, dash) in enumerate(items):
            yy = y + pad + i * row
            if dash:
                out.append(f'<line x1="{x+pad:.1f}" x2="{x+pad+sw:.1f}" y1="{yy+fs/2:.1f}" y2="{yy+fs/2:.1f}" stroke="{color}" stroke-width="{2.5 if big else 2}" stroke-dasharray="5 3"/>')
            else:
                out.append(f'<rect x="{x+pad:.1f}" y="{yy+(fs-sh)/2:.1f}" width="{sw}" height="{sh}" rx="2" fill="{color}"/>')
            out.append(f'<text class="leg" x="{x+pad+sw+6:.1f}" y="{yy+fs*0.85:.1f}">{html.escape(name)}</text>')
        return "".join(out)

    @classmethod
    def legend_auto(cls, items, pts, ml, mt, pw, ph, big=None, soft=()):
        """범례를 그래프 안 왼쪽 아래에 둔다(사용자 지시 2026-10-31: 그 자리가 비어 있으므로; 그래프 범위는 바꾸지 않는다).
        pts(데이터의 픽셀 좌표)가 거기 있으면 왼쪽 위 → 오른쪽 아래 → 오른쪽 위 순으로 빈 모서리를 고른다.
        soft = 가려도 되는 보조선(예: 그래프 폭 전체에 걸친 기준 점선)의 좌표: 네 모서리가 다 막혔을 때는 soft만 있는 모서리를,
        그것도 없으면 pts가 가장 적게 걸리는 모서리를 고른다 (2026-11-02: 기준선이 위 두 모서리를 항상 막아 데이터 점을 가리던 문제)."""
        big = cls.present if big is None else big
        w, h = cls.legend_dims(items, big)
        g = cls.GAP
        corners = [(ml + g, mt + ph - g - h), (ml + g, mt + g), (ml + pw - g - w, mt + ph - g - h), (ml + pw - g - w, mt + g)]
        hits = lambda x, y, ps: sum(x - 2 <= px <= x + w + 2 and y - 2 <= py <= y + h + 2 for px, py in ps)  # noqa: E731
        for x, y in corners:
            if not hits(x, y, pts) and not hits(x, y, soft):
                break
        else:
            free = [c for c in corners if not hits(c[0], c[1], pts)]
            x, y = free[0] if free else min(corners, key=lambda c: hits(c[0], c[1], pts))
        return cls.legend_box(items, x, y, big)

    @staticmethod
    def densify(xs, ys, n=10):
        """꺾은선의 각 구간을 n개 점으로 보간한 (x, y) 목록 — 범례 상자와 선분이 겹치는지 볼 때 쓴다(NaN 구간은 건너뛴다)."""
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        pts = []
        for i in range(len(xs) - 1):
            if np.isfinite(ys[i]) and np.isfinite(ys[i + 1]):
                t = np.linspace(0, 1, n, endpoint=False)
                pts += list(zip(xs[i] + t * (xs[i + 1] - xs[i]), ys[i] + t * (ys[i + 1] - ys[i])))
        if len(xs) and np.isfinite(ys[-1]):
            pts.append((float(xs[-1]), float(ys[-1])))
        return pts

    @staticmethod
    def nice(x):
        e = 10 ** np.floor(np.log10(max(x, 1e-9)))
        for m in (1, 2, 2.5, 5, 10):
            if m * e >= x:
                return m * e
        return 10 * e

    @staticmethod
    def path(xs, ys, fx, fy):
        pts = [f"{fx(x):.1f},{fy(y):.1f}" for x, y in zip(xs, ys) if np.isfinite(y)]
        return "M" + " L".join(pts) if pts else ""

    @classmethod
    def line(cls, x, series, *, ylabel, xlabel=None, ylim=None, xlim=None, vlines=(), hlines=(), band=None,
             H=210, W=860, xtick=None, label="", legend=None):
        """series: list of dict(y, color, width, dash). band: (lo, hi, color). legend: [(이름, 색, 점선)] — 그래프 안 왼쪽 아래."""
        big = cls.present
        ml, mr, mt, mb, tk, tx, yl, xl = cls.frame(big, bool(xlabel))
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
        leg = cls.legend_auto(legend, [(fx(px), fy(py)) for s in series for px, py in cls.densify(x, s["y"])], ml, mt, pw, ph, big) if legend else ""
        p = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(label or ylabel)}" style="max-width:100%;height:auto;display:block">']
        # 세로 표시선 (위상 0 %)
        for v in vlines:
            p.append(f'<line x1="{fx(v):.1f}" x2="{fx(v):.1f}" y1="{mt}" y2="{mt+ph}" class="mark"/>')
        # 가로 눈금
        step = cls.nice((yhi - ylo) / (6 if big else 4))
        tick = np.ceil(ylo / step) * step
        while tick <= yhi + 1e-9:
            yy = fy(tick)
            p.append(f'<line x1="{ml}" x2="{W-mr}" y1="{yy:.1f}" y2="{yy:.1f}" class="grid"/>')
            p.append(f'<text x="{ml-tk}" y="{yy+4:.1f}" class="tick" text-anchor="end">{tick + 0.0:g}</text>')
            tick += step
        if ylo < 0 < yhi:
            p.append(f'<line x1="{ml}" x2="{W-mr}" y1="{fy(0):.1f}" y2="{fy(0):.1f}" class="zero"/>')
        for h in hlines:
            p.append(f'<line x1="{ml}" x2="{W-mr}" y1="{fy(h):.1f}" y2="{fy(h):.1f}" class="limit"/>')
        # 가로 눈금
        xt = xtick or cls.nice((xhi - xlo) / 6)
        tick = np.ceil(xlo / xt) * xt
        while tick <= xhi + 1e-9:
            xx = fx(tick)
            p.append(f'<text x="{xx:.1f}" y="{H-mb+tx}" class="tick" text-anchor="middle">{tick:g}</text>')
            tick += xt
        if band:
            lo, hi, col = band
            up = cls.path(x, hi, fx, fy)
            down = cls.path(x[::-1], np.asarray(lo)[::-1], fx, fy)
            if up and down:
                p.append(f'<path d="{up} {down.replace("M", "L", 1)} Z" fill="{col}" opacity="0.18" stroke="none"/>')
        for s in series:
            d = cls.path(x, s["y"], fx, fy)
            dash = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
            p.append(f'<path d="{d}" fill="none" stroke="{s["color"]}" stroke-width="{s.get("width", 1.6)}"{dash} stroke-linejoin="round"/>')
        p.append(leg)
        p.append(f'<text x="{ml-yl}" y="{mt+ph/2:.0f}" class="axis" text-anchor="middle" transform="rotate(-90 {ml-yl} {mt+ph/2:.0f})">{html.escape(ylabel)}</text>')
        if xlabel:
            p.append(f'<text x="{W-mr}" y="{H-xl}" class="axis" text-anchor="end">{html.escape(xlabel)}</text>')
        p.append("</svg>")
        return "\n".join(p)

    @staticmethod
    def legend(items):
        return '<div class="legend">' + "".join(
            f'<span class="lg"><i style="background:{c}{";height:2px" if dash else ""}"></i>{html.escape(n)}</span>' for n, c, dash in items
        ) + "</div>"

    @classmethod
    def phase_bars(cls, series: dict, colors_css: list):
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
        step = cls.nice((hi - lo) / 5)
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


    @staticmethod
    def _bins_frame(series, h, w):
        """막대·꺾은선 공통: (ml, mt, pw, ph, dx, y, 눈금 SVG, 축 이름 SVG). 세로 범위는 ±(최대 절대값 × 1.18)."""
        big = Svg.present
        ml, mr, mt, mb, tk, tx, yl, xl = Svg.frame(big)
        pw, ph = w - ml - mr, h - mt - mb
        vals = [v for d in series.values() for v in d["mean"] if v is not None]
        bound = max(5.0, max(abs(v) for v in vals)) * 1.18
        y = lambda v: mt + (bound - v) / (2 * bound) * ph  # noqa: E731
        dx = pw / 10
        out = []
        step = Svg.nice(bound / 2.5)
        tick = -np.floor(bound / step) * step
        while tick <= bound:
            out.append(f'<line class="grid" x1="{ml}" x2="{w-mr}" y1="{y(tick):.1f}" y2="{y(tick):.1f}"/>'
                       f'<text class="tick" x="{ml-tk}" y="{y(tick)+4:.1f}" text-anchor="end">{tick + 0.0:+.0f}</text>')
            tick += step
        out.append(f'<line class="zero" x1="{ml}" x2="{w-mr}" y1="{y(0):.1f}" y2="{y(0):.1f}"/>')
        for b in range(10):
            out.append(f'<text class="tick" x="{ml+dx*(b+.5):.1f}" y="{h-mb+tx}" text-anchor="middle">{b*10}–{(b+1)*10}</text>')
        axes = (f'<text class="axis" x="{ml-yl}" y="{mt+ph/2:.0f}" text-anchor="middle" transform="rotate(-90 {ml-yl} {mt+ph/2:.0f})">토크 실효값 감소량 [N·m]</text>'
                f'<text class="axis" x="{w-mr}" y="{h-xl}" text-anchor="end">보행 진행률 [%]</text></svg>')
        return ml, mt, pw, ph, dx, y, "".join(out), axes

    @staticmethod
    def reduction_bars_grouped(series: dict, colors: list, h=300, w=900, legend=None):
        """series: 이름 -> {"mean": [10], "runs": [[10], ...], "labels": [...]} — 보행 진행률 10 % 구간별
        착용자 토크 실효값 감소량(무보조 A − 보조 B)을 모델별로 나란히. 양수 = 부담 감소(설명은 HTML caption에 둔다).
        legend: [(이름, 색, 점선)] — 그래프 안 왼쪽 아래(막대가 있으면 빈 모서리)."""
        names = list(series)
        ml, mt, pw, ph, dx, y, head, axes = Svg._bins_frame(series, h, w)
        out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="보행 진행률별 착용자 토크 실효값 감소량" style="max-width:100%;height:auto;display:block">', head]
        n = len(names); bw = dx * .8 / n
        pts = []
        for b in range(10):
            for j, name in enumerate(names):
                v = series[name]["mean"][b]
                if v is None:
                    continue
                xx = ml + dx * (b + .1) + j * bw
                per = ", ".join(f"{l} {r[b]:+.1f}" for l, r in zip(series[name]["labels"], series[name]["runs"]) if r[b] is not None)
                out.append(f'<rect class="bar" x="{xx:.1f}" y="{min(y(v), y(0)):.1f}" width="{bw:.1f}" height="{max(1, abs(y(v)-y(0))):.1f}" rx="1.5" fill="{colors[j]}">'
                           f'<title>{html.escape(name)} · {b*10}–{(b+1)*10} %: {v:+.2f} N·m ({html.escape(per)})</title></rect>')
                pts += [(xx + k * bw / 2, yy) for k in (0, 1, 2) for yy in np.linspace(y(0), y(v), 6)]
        if legend:
            out.append(Svg.legend_auto(legend, pts, ml, mt, pw, ph))
        out.append(axes)
        return "".join(out)

    @staticmethod
    def reduction_lines_grouped(series: dict, colors: list, h=300, w=900, legend=None):
        """reduction_bars_grouped와 같은 입력을 꺾은선으로 — 구간 중심(5, 15, …, 95 %)을 점으로 잇는다.
        seed별 값은 옅은 가는 선, 평균은 굵은 선(설명은 HTML caption에 둔다)."""
        names = list(series)
        ml, mt, pw, ph, dx, y, head, axes = Svg._bins_frame(series, h, w)
        x = lambda b: ml + dx * (b + .5)  # noqa: E731
        out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="보행 진행률별 착용자 토크 실효값 감소량 (꺾은선)" style="max-width:100%;height:auto;display:block">', head]
        pts = []
        for j, name in enumerate(names):
            c = colors[j]
            for lab, run in zip(series[name]["labels"], series[name]["runs"]):
                pp = [(x(b), y(v)) for b, v in enumerate(run) if v is not None]
                out.append(f'<polyline fill="none" stroke="{c}" stroke-width="1" opacity=".35" points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pp)}"><title>{html.escape(name)} · {html.escape(lab)}</title></polyline>')
                pts += Svg.densify([x(b) for b in range(10)], [np.nan if v is None else y(v) for v in run])
            mean = series[name]["mean"]
            pp = [(x(b), y(v)) for b, v in enumerate(mean) if v is not None]
            out.append(f'<polyline fill="none" stroke="{c}" stroke-width="2.2" stroke-linejoin="round" points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pp)}"/>')
            pts += Svg.densify([x(b) for b in range(10)], [np.nan if v is None else y(v) for v in mean])
            for b, v in enumerate(mean):
                if v is None:
                    continue
                per = ", ".join(f"{l} {r[b]:+.1f}" for l, r in zip(series[name]["labels"], series[name]["runs"]) if r[b] is not None)
                out.append(f'<circle cx="{x(b):.1f}" cy="{y(v):.1f}" r="3.2" fill="{c}"><title>{html.escape(name)} · {b*10}–{(b+1)*10} %: {v:+.2f} N·m ({html.escape(per)})</title></circle>')
        if legend:
            out.append(Svg.legend_auto(legend, pts, ml, mt, pw, ph))
        out.append(axes)
        return "".join(out)


# ---------------------------------------------------------------------------
# 비교 보고서 — 실행 여러 개를 한 표로
# ---------------------------------------------------------------------------
class ComparisonReport:
    @staticmethod
    def table_html(rows: dict, title: str, note: str, running=False):
        fmt = Fmt.signed
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
            ampc = (f"<td class=\"txt\">{m['amp_A']:.1f}° → {m['amp_B']:.1f}° <span class=\"dim\">(mocap {m['amp_ref']:.1f}°)</span></td>"
                    if "amp_A" in m else "<td>—</td>")
            extra = f"{ampc}<td>{m['assist_rms']:.1f}</td><td>{fmt(m['corr'], 2)}</td><td class=\"txt\">{surv}</td>"
            body.append(f"<tr><th scope=\"row\">{html.escape(name)}</th>{''.join(cells)}{extra}</tr>")
        return f"""
<section>
  <h2>{html.escape(title)}</h2>
  <p class="note">{note}</p>
  <div class="scroll"><table>
    <thead><tr><th>실행</th>{head}<th>hip 스윙 진폭 A → B</th><th>보조 RMS [N·m]</th><th>보조↔착용자 토크 상관</th><th>결과</th></tr></thead>
    <tbody>{''.join(body)}</tbody>
  </table></div>
</section>"""

    @staticmethod
    def phase_table_html(series_rows: dict):
        names = list(series_rows)
        head = "".join(f"<th>{10*b}–{10*b+10}</th>" for b in range(10))
        body = []
        for name in names:
            rows = series_rows[name]
            cells = "".join(f"<td>{r['contrib']:+.0f}</td>" for r in rows)
            land = sum(r["contrib"] for r in rows[:2]); swing = sum(r["contrib"] for r in rows[3:7]); tot = sum(r["contrib"] for r in rows)
            body.append(f"<tr><th scope=\"row\">착용자 τ² 변화 [(N·m)²] — {html.escape(name)}</th>{cells}<td class=\"sum\">{land:+.0f}</td><td class=\"sum\">{swing:+.0f}</td><td class=\"sum\">{tot:+.0f}</td></tr>")
        body2 = []
        for name in names:
            rows = series_rows[name]
            body2.append(f"<tr><th scope=\"row\">보조 토크 평균 [N·m] — {html.escape(name)}</th>" + "".join(f"<td>{r['assist']:+.0f}</td>" for r in rows) + "<td></td><td></td><td></td></tr>")
        first = series_rows[names[0]]
        base_row = "<tr class=\"ref\"><th scope=\"row\">무보조 착용자 토크 평균 [N·m] (참고)</th>" + "".join(f"<td>{r['tauA']:+.0f}</td>" for r in first) + "<td></td><td></td><td></td></tr>"
        return f"""
<div class="scroll"><table class="phase">
  <thead><tr><th>행 ＼ 보행 위상 [%]</th>{head}<th>착지 0–20 [(N·m)²]</th><th>유각 30–70 [(N·m)²]</th><th>합계 [(N·m)²]</th></tr></thead>
  <tbody>{''.join(body)}{base_row}{''.join(body2)}</tbody>
</table></div>"""

    @staticmethod
    def omega_table_html(series_rows: dict):
        """위상 구간별 |ω| A → B 변화 [%] — 보조가 다리를 덜 움직이게 하는 구간이 어디인가."""
        names = list(series_rows)
        head = "".join(f"<th>{10*b}–{10*b+10}</th>" for b in range(10))
        body = []
        first = series_rows[names[0]]
        body.append("<tr class=\"ref\"><th scope=\"row\">무보조 hip 각속도 |ω| 평균 [rad/s] (참고)</th>" + "".join(f"<td>{r['omegaA']:.2f}</td>" for r in first) + "<td></td></tr>")
        for name in names:
            rows = series_rows[name]
            cells = "".join(f"<td>{100*(r['omegaB']/max(r['omegaA'],1e-9)-1):+.0f}</td>" for r in rows)
            tot = 100 * (sum(r["omegaB"] for r in rows) / max(sum(r["omegaA"] for r in rows), 1e-9) - 1)
            body.append(f"<tr><th scope=\"row\">|ω| 변화 A → B [%] — {html.escape(name)}</th>{cells}<td class=\"sum\">{tot:+.0f}</td></tr>")
        return f"""
<div class="scroll"><table class="phase">
  <thead><tr><th>행 ＼ 보행 위상 [%]</th>{head}<th>전체 [%]</th></tr></thead>
  <tbody>{''.join(body)}</tbody>
</table></div>"""

    @staticmethod
    def resolve(items):
        out = {}
        for it in items:
            name, pat = it.split("=", 1)
            out[name] = latest_run(pat)
        return out

    @classmethod
    def build(cls, args):
        walk = cls.resolve(args.walk)
        run = cls.resolve(args.run)
        walk_m = {n: ObjectiveComparison.html_metrics(r) for n, r in walk.items()}
        run_m = {n: ObjectiveComparison.html_metrics(r) for n, r in run.items()}
        chart_names = [s.strip() for s in (args.chart.split(";") if args.chart else list(walk)[:4])][:4]
        phase = {n: ObjectiveComparison.phase_rows(walk[n]) for n in walk}
        chart_series = {n: [r["contrib"] for r in phase[n]] for n in chart_names if phase.get(n)}
        missing = [n for n in chart_names if n not in walk]
        assert chart_series, f"--chart 이름이 --walk 이름과 맞지 않음: {missing}"
        colors = [f"var(--s{i+1})" for i in range(len(chart_series))]
        legend = "".join(
            f'<span class="lg"><i style="background:var(--s{i+1})"></i>{html.escape(n)}</span>' for i, n in enumerate(chart_series)
        )
        walk_tbl = cls.table_html(walk_m, "걷기 (motion 0, 40 s, 착용자 v32)",
                                  "무보조 A 대비 보조 B의 변화 [%]. <b>음수가 좋다.</b> τ²는 다리를 덜 움직여서는 못 내리는 지표라 기준 지표로 쓴다. 굵은 칸 = 열에서 가장 낮은 값.")
        run_tbl = cls.table_html(run_m, "달리기 (motion 1, 40 s)",
                                 "같은 잣대. 스켈레톤에서 학습한 버전(v10·v12)은 달리기에서 착용자 근토크(τ²)를 오히려 늘렸다 — "
                                 "|τ·ω| 파워가 내려간 것은 각속도 감소분이다. 진자에서 학습한 v4 새 형식이 τ²를 줄인 첫 버전.") if run_m else ""
        page = f"""<title>{html.escape(args.title)}</title>
{HtmlShell.PLEX_FONTS}
<style>{HtmlShell.plex_css()}</style>
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
    {Svg.phase_bars(chart_series, colors)}
    <h2 style="margin-top:18px">위상표 (같은 수치 + 구간별 보조 토크 평균 [N·m])</h2>
    {cls.phase_table_html({n: phase[n] for n in walk if phase.get(n)})}
    <h2 style="margin-top:22px">위상별 hip 각속도 |ω| 변화 (A → B, %)</h2>
    <p class="note">보조가 착용자의 다리를 어디서 덜(또는 더) 움직이게 하는가. 음수 = 보조 켤 때 그 구간에서 다리가 느리다. β 오라클(착용자 토크와 같은 방향으로만 미는 상한)은 거의 0이다.</p>
    {cls.omega_table_html({n: phase[n] for n in walk if phase.get(n)})}
  </section>
  <p class="foot">{args.notes}</p>
</main>
{HtmlShell.TOOLTIP_SCRIPT}"""
        Path(args.out).write_text(page)
        print("saved", args.out, f"({len(page)//1024} KB)")


# ---------------------------------------------------------------------------
# 체크포인트 보고서 — 하나를 깊게
# ---------------------------------------------------------------------------
class CheckpointReport:
    @staticmethod
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

    @staticmethod
    def key_numbers_html(name, wm, wx, rm, rx):
        fmt = Fmt.signed

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

    @staticmethod
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
  {Svg.legend([("mocap 기준 각도", "var(--s3)", True), ("A 무보조", "var(--a)", False), ("B 보조 " + name, "var(--s2)", False)])}
  {Svg.line(t, s_ang, ylabel="hip 굴곡 [°]", xlim=xlim, vlines=marks, H=190, xtick=0.5)}
  {Svg.line(t, s_tau, ylabel="착용자 hip 토크 [N·m]", xlim=xlim, vlines=marks, H=230, xtick=0.5)}
  {Svg.legend([("보조 토크 (B)", "var(--s1)", False)])}
  {Svg.line(t, s_as, ylabel="보조 토크 [N·m]", xlabel="시간 [s]", xlim=xlim, ylim=(-lim*1.15, lim*1.15), vlines=marks, hlines=(-lim, lim), H=170, xtick=0.5)}
</section>"""

    @staticmethod
    def profile_html(name, prof, gait):
        ph = prof["phase"]
        s_tau = [
            {"y": prof["tauA"], "color": "var(--a)", "width": 1.6},
            {"y": prof["tauB"], "color": "var(--s2)", "width": 1.8},
        ]
        s_as = [{"y": prof["assist"], "color": "var(--s1)", "width": 1.8}]
        band = (prof["assist"] - prof["assist_sd"], prof["assist"] + prof["assist_sd"], HtmlShell.SERIES[0])
        return f"""
<section>
  <h2>{html.escape(gait)} 주기 평균 — 위상별 착용자 토크와 보조 토크 (양쪽 다리, 39 s)</h2>
  <p class="note">0 % = 최대 굴곡. 착지 직후(0~20 %) 착용자는 큰 신전(−) 토크를 내고, 유각(30~70 %)에는 굴곡(+) 토크를 낸다. 보조가 착용자 토크와 같은 부호로 나오는 구간이 "덜어 주는" 구간이다. 띠 = 보조 토크 ±1 표준편차(스트라이드 간 흔들림).</p>
  {Svg.legend([("A 무보조 착용자 토크", "var(--a)", False), ("B 보조 시 착용자 토크", "var(--s2)", False)])}
  {Svg.line(ph, s_tau, ylabel="착용자 hip 토크 [N·m]", xlim=(0, 100), H=220, xtick=10)}
  {Svg.legend([("보조 토크 평균 ± 1σ (B)", "var(--s1)", False)])}
  {Svg.line(ph, s_as, ylabel="보조 토크 [N·m]", xlabel="보행 주기 [%]", xlim=(0, 100), band=band, H=180, xtick=10)}
</section>"""

    @staticmethod
    def trajectory_html(traj, curves, pick_epoch):
        fmt = Fmt.signed
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
  <thead><tr><th>epoch</th><th>걷기 τ² [%]</th><th>달리기 τ² [%]</th><th>걷기 몸체 오차 [%]</th><th>보조 RMS [N·m]</th><th>최대출력 95% 이상 사용 시간비율 [%]</th><th>상관 (걷기)</th><th>상관 (달리기)</th></tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table></div>"""
        charts = ""
        if curves:
            def one(key, ylabel, color):
                if key not in curves:
                    return ""
                st, va = curves[key]
                return Svg.line(st, [{"y": va, "color": color, "width": 1.8}], ylabel=ylabel, xlabel="epoch",
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

    @classmethod
    def build(cls, args):
        def many(items):
            return {n: latest_run(g) for n, g in (it.split("=", 1) for it in items)}

        M = ObjectiveComparison
        walk, run = latest_run(args.walk), latest_run(args.run)
        ref_w, ref_r = many(args.ref_walk), many(args.ref_run)
        wm, rm = M.html_metrics(walk), M.html_metrics(run)
        wx, rx = M.extra_metrics(walk), M.extra_metrics(run)

        walk_rows = {**{n: M.html_metrics(r) for n, r in ref_w.items()}, args.name: wm}
        run_rows = {**{n: M.html_metrics(r) for n, r in ref_r.items()}, args.name: rm}
        walk_tbl = ComparisonReport.table_html(walk_rows, "걷기 — 기준 실행과 같은 잣대",
                                               "무보조 A 대비 보조 B의 변화 [%]. τ²와 함께 전진 거리·몸체 오차·완주 여부를 확인한다. β 오라클은 착용자 토크를 사용하는 특권 참고값이며 배포 정책이 아니다. 굵은 칸 = 열에서 가장 낮은 값.")
        run_tbl = ComparisonReport.table_html(run_rows, "달리기 — 기준 실행과 같은 잣대",
                                              "같은 지표로 비교한다. 보조 배율·착용자·제어 주기 등 실행 조건과 원본 경로는 아래 설정을 확인한다.")

        phase_w = {n: M.phase_rows(r) for n, r in {**ref_w, args.name: walk}.items()}
        phase_r = {n: M.phase_rows(r) for n, r in {**ref_r, args.name: run}.items()}
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
            traj[int(ep)] = (M.html_metrics(w), M.extra_metrics(w), M.html_metrics(r) if r else None, M.extra_metrics(r) if r else None)
        curves = cls.tb_curves(args.tb_dir, {"tau2": "env/raw_r/human_torque_sq_mean", "effort": "env/raw_r/assist_effort_mean"},
                               args.tb_max_epoch) if args.tb_dir else None
        pick = int(RunRecord(walk).meta.get("llp_epoch") or 0)

        page = f"""<meta charset="utf-8">
<title>{html.escape(args.title)}</title>
{HtmlShell.PLEX_FONTS}
<style>{HtmlShell.plex_css()}</style>
<main>
  <header>
    <div class="eyebrow">{html.escape(args.eyebrow)}</div>
    <h1>{html.escape(args.title)}</h1>
    <p class="verdict">{args.verdict}</p>
  </header>
  {cls.key_numbers_html(args.name, wm, wx, rm, rx)}
  {walk_tbl}
  {run_tbl}
  <section>
    <h2>어느 구간에서 벌고 어느 구간에서 잃는가 — 걷기 위상 그림</h2>
    <p class="note">보행 주기를 10 % 구간으로 나눠, 각 구간이 착용자 τ² 평균 변화에 기여한 양(절대 (N·m)²). 0 % = 최대 굴곡 ≈ 착지. 막대에 마우스를 올리면 값이 보인다.</p>
    <div class="legend">{leg(chart_w)}</div>
    {Svg.phase_bars(chart_w, colors)}
    <h2 style="margin-top:18px">걷기 위상표 (같은 수치 + 구간별 보조 토크 평균 [N·m])</h2>
    {ComparisonReport.phase_table_html(phase_w)}
    <h2 style="margin-top:22px">걷기 위상별 hip 각속도 |ω| 변화 (A → B, %)</h2>
    <p class="note">보조가 착용자의 다리를 어디서 덜(또는 더) 움직이게 하는가. 음수 = 보조 켤 때 그 구간에서 다리가 느리다.</p>
    {ComparisonReport.omega_table_html(phase_w)}
  </section>
  <section>
    <h2>달리기 위상 그림·표</h2>
    <p class="note">같은 방식. 달리기는 주기가 짧고 착지 하중이 커서 착지 구간(0~20 %)의 기여가 걷기보다 크다.</p>
    <div class="legend">{leg(chart_r)}</div>
    {Svg.phase_bars(chart_r, colors)}
    {ComparisonReport.phase_table_html(phase_r)}
  </section>
  {cls.timeseries_html(args.name, M.time_window(walk), "걷기", wx["stride"])}
  {cls.timeseries_html(args.name, M.time_window(run), "달리기", rx["stride"])}
  {cls.profile_html(args.name, M.cycle_profile(walk), "걷기")}
  {cls.profile_html(args.name, M.cycle_profile(run), "달리기")}
  {cls.trajectory_html(traj, curves, pick) if traj else ""}
  <div class="foot">{args.notes}</div>
</main>
{HtmlShell.TOOLTIP_SCRIPT}"""
        Path(args.out).write_text(page)
        print("saved", args.out, f"({len(page)//1024} KB)")
        print(f"walk: tau2 {wm['tau2']:+.1f} %  P {wm['P']:+.1f} %  body {wm.get('body', float('nan')):+.1f} %  rms {wm['assist_rms']:.1f}  sat {wx['sat']:.1f} %  corr {wm['corr']:+.2f}  agree {wx['agree']:.0f} %  stride {wx['stride']:.3f} s")
        print(f"run : tau2 {rm['tau2']:+.1f} %  P {rm['P']:+.1f} %  body {rm.get('body', float('nan')):+.1f} %  rms {rm['assist_rms']:.1f}  sat {rx['sat']:.1f} %  corr {rm['corr']:+.2f}  agree {rx['agree']:.0f} %  stride {rx['stride']:.3f} s")
        for ep, (a, b, c, d) in traj.items():
            print(f"traj {ep}: walk tau2 {a['tau2']:+.1f}  run tau2 {(c['tau2'] if c else float('nan')):+.1f}  rms {a['assist_rms']:.1f}  sat {b['sat']:.0f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 추가학습 전후 보고서 — 보행 진행률별 토크 실효값 감소량
# ---------------------------------------------------------------------------
class Figure:
    """발표 자료용 그림. 보고서와 같은 SVG(class="tick|axis|leg|…")를 글자 크기·색을 속성으로 박은 독립 SVG로 바꾸고
    Chrome headless로 PNG(기본 3배 해상도)도 저장한다. 글자는 모두 검정, 축 이름(20 px 굵게)이 축 숫자(15 px)보다 크다
    (사용자 지시 2026-10-30: 보고서 그래프는 발표 자료에서 축 이름·숫자가 너무 작고 흐리다). Svg.present = True로 그린 SVG(넓은 여백, 범례는 그래프 안)를 넘긴다."""

    CHROME = "/usr/bin/google-chrome"
    FONT = "Noto Sans CJK KR, Noto Sans KR, sans-serif"
    TICK, AXIS, LEGEND = 15, 20, 17
    ATTRS = {  # class → 속성 (PowerPoint는 SVG 안의 CSS class를 읽지 못하므로 속성으로)
        "tick": f'font-size="{TICK}" fill="#000"',
        "axis": f'font-size="{AXIS}" font-weight="700" fill="#000"',
        "leg": f'font-size="{LEGEND}" fill="#000"',
        "legbg": 'fill="#fff"',
        "bandlab": f'font-size="{TICK}" fill="#000"',
        "grid": 'stroke="#d8d8d8" stroke-width="1"',
        "zero": 'stroke="#000" stroke-width="1.4"',
        "mark": 'stroke="#000" stroke-width="1" stroke-dasharray="4 3"',
        "limit": 'stroke="#000" stroke-width="1.4" stroke-dasharray="7 4"',
        "band": 'fill="#f3efe2"',
        "bar": "",
    }

    @classmethod
    def save(cls, svg: str, stem, scale=3):
        """svg: Svg.*가 돌려준 문자열(첫 태그에 viewBox). stem: 확장자 없는 경로 → .svg와 .png."""
        m = re.search(r'viewBox="0 0 (\d+) (\d+)"', svg)
        W, H = int(m.group(1)), int(m.group(2))
        body = re.sub(r"^<svg[^>]*>", "", svg.strip(), count=1)
        body = re.sub(r"</svg>\s*$", "", body, count=1)
        for k, v in cls.ATTRS.items():
            body = body.replace(f'class="{k}"', v)
        out = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{cls.FONT}">'
               f'<rect width="100%" height="100%" fill="#fff"/>{body}</svg>')
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        stem.with_suffix(".svg").write_text(out)
        with tempfile.TemporaryDirectory() as td:
            page = Path(td) / "fig.html"
            page.write_text(f'<!doctype html><html><head><meta charset="utf-8"></head><body style="margin:0;background:#fff">{out}</body></html>')
            subprocess.run([cls.CHROME, "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars", f"--user-data-dir={td}/profile",
                            f"--force-device-scale-factor={scale}", f"--window-size={W},{H}", f"--screenshot={stem.with_suffix('.png')}", page.as_uri()],
                           check=True, capture_output=True)
        print("saved", stem.with_suffix(".svg"), stem.with_suffix(".png"))


class FinetuneReport:
    """추가학습 전후(및 기준선)를 걷기·달리기별로 비교한다. 그래프는 보행 진행률 10 % 구간별 착용자 토크
    실효값 감소량(무보조 A − 보조 B, N·m)이고, 같은 이름의 실행(평가 seed 여러 개)은 구간별 평균을 그린다."""

    @staticmethod
    def resolve(items):
        """이름=glob → {이름: [run dirs]}; glob은 run_system 출력 폴더(traces.npz) 여러 개에 맞을 수 있다."""
        out = {}
        for it in items:
            name, pat = it.split("=", 1)
            hits = sorted(h for h in glob.glob(pat) if (Path(h) / "traces.npz").exists())
            assert hits, f"no run for {pat}"
            out[name] = [Path(h) for h in hits]
        return out

    @staticmethod
    def digest(run: Path):
        rec = RunRecord(run)
        meta = rec.meta
        data = TorqueAlignment.aligned_torques(rec.traces, meta)
        rows = TorqueAlignment.phase_profile(data, 10)
        limit = abs(meta["assist_scale"]) * min(meta["torque_limit"], meta.get("policy_torque_limit", meta["torque_limit"]))
        red = [r["rms_reduction_Nm"] for r in rows]
        m = re.search(r"_s(\d+)$", run.name)  # meta.json에 seed가 없으면 평가 폴더 이름의 _s<seed> 접미에서 읽는다
        seed = meta.get("seed", int(m.group(1)) if m else None)
        return dict(run=str(run), seed=seed, reduction=red, metrics=ObjectiveComparison.html_metrics(run), limit=limit,
                    saturation=float(100 * np.mean(np.abs(data["assist_B"]) > .95 * limit)),
                    worst=max(-(v or 0) for v in red), bad_bins=sum((v or 0) < 0 for v in red),
                    survived=meta.get("survived_s"), seconds=meta.get("seconds"))

    @classmethod
    def series(cls, runs: dict):
        out = {}
        for name, paths in runs.items():
            ds = [cls.digest(p) for p in paths]
            mean = [None if any(d["reduction"][b] is None for d in ds) else float(np.mean([d["reduction"][b] for d in ds])) for b in range(10)]
            out[name] = dict(mean=mean, runs=[d["reduction"] for d in ds], labels=[f"seed {d['seed']}" for d in ds], digests=ds)
        return out

    @staticmethod
    def table_html(series: dict):
        def cell(values, spec="+.1f", unit=""):
            vals = [v for v in values if v is not None]
            if not vals:
                return "<td>–</td>"
            per = ", ".join(format(v, spec) for v in vals)
            return f'<td>{np.mean(vals):{spec}}{unit} <span class="dim">({per})</span></td>'
        head = ["모델", "착용자 τ² 변화 [%]", "보조 RMS [N·m]", "최대출력 95% 이상 사용 시간비율 [%]", "최악 구간 RMS 증가 [N·m]", "부담 증가 구간 / 10", "몸체 오차 변화 [%]", "완주"]
        body = []
        for name, d in series.items():
            ds = d["digests"]
            done = ", ".join(f'{x["seconds"]:g} s' if x["survived"] is None else f'{x["survived"]:.1f} s 낙상' for x in ds)
            body.append("<tr><th>" + html.escape(name) + "</th>" + cell([x["metrics"]["tau2"] for x in ds])
                        + cell([x["metrics"]["assist_rms"] for x in ds], ".1f") + cell([x["saturation"] for x in ds], ".1f")
                        + cell([x["worst"] for x in ds], "+.1f") + cell([x["bad_bins"] for x in ds], ".0f")
                        + cell([x["metrics"].get("body") for x in ds]) + f'<td class="txt">{done}</td></tr>')
        return ('<div class="scroll"><table><thead><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>"
                + "".join(body) + "</tbody></table></div>")

    @staticmethod
    def bins_table_html(series: dict):
        names = list(series)
        head = "<tr><th>구간 [%]</th>" + "".join(f"<th>{html.escape(n)}</th>" for n in names) + "</tr>"
        rows = []
        for b in range(10):
            best = max((series[n]["mean"][b] for n in names if series[n]["mean"][b] is not None), default=None)
            cells = "".join(f'<td class="{"best" if series[n]["mean"][b] == best else ""}">{series[n]["mean"][b]:+.2f}</td>' if series[n]["mean"][b] is not None else "<td>–</td>" for n in names)
            rows.append(f"<tr><th>{b*10}–{(b+1)*10}</th>{cells}</tr>")
        return f'<div class="scroll"><table><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>'

    PALETTE = HtmlShell.SERIES + ["#8e5bd6", "#5b8f3a", "#b5473f", "#6b6b6b"]

    @classmethod
    def chart(cls, series: dict, spec=None, chart_colors=None):
        """그래프에 올릴 부분집합: spec = ["run 이름" 또는 "run 이름=범례 이름", ...] (없으면 전부). 색은 spec과 같은 순서.
        → (부분집합 dict, 범례 이름 list, 색 list). 없는 run 이름은 건너뛴다."""
        spec = spec or list(series)
        palette = chart_colors or cls.PALETTE
        rows = []
        for i, item in enumerate(spec):
            name, _, label = item.partition("=")
            if name in series:
                rows.append((name, label or name, palette[i % len(palette)]))
        assert rows, f"--chart 이름이 실행 이름과 맞지 않음: {spec}"
        return {n: series[n] for n, _, _ in rows}, [l for _, l, _ in rows], [c for _, _, c in rows]

    BARS_CAPTION = ("막대 = 보행 진행률 10 % 구간별 착용자 고관절 토크 실효값 감소량(무보조 − 보조), seed 평균. 양수 = 부담 감소. "
                    "가로축 0 % = 참조 최대 굴곡(좌우 고관절을 각 다리의 참조 위상으로 모아 계산). 막대에 마우스를 올리면 checkpoint와 seed별 값이 보인다.")
    LINES_CAPTION = "같은 값을 꺾은선으로. 점은 각 10 % 구간의 중심, 굵은 선 = seed 평균, 가는 선 = seed별."

    @classmethod
    def section(cls, title, series: dict, note: str, spec=None, chart_colors=None):
        chart, labels, colors = cls.chart(series, spec, chart_colors)
        legend = [(l, c, False) for l, c in zip(labels, colors)]
        paths = "".join(f"<li><code>{html.escape(str(p))}</code></li>" for d in series.values() for p in [x["run"] for x in d["digests"]])
        return f"""<section>
  <h2>{html.escape(title)}</h2>
  <p class="note">{note}</p>
  {Svg.reduction_bars_grouped(chart, colors, legend=legend)}
  <p class="note">{cls.BARS_CAPTION}</p>
  {Svg.reduction_lines_grouped(chart, colors, legend=legend)}
  <p class="note">{cls.LINES_CAPTION}</p>
  <h3 style="margin-top:16px">전체 지표 (평가 seed 평균, 괄호 안은 seed별)</h3>
  {cls.table_html(series)}
  <h3 style="margin-top:16px">구간별 감소량 [N·m] (seed 평균, 굵은 칸 = 구간에서 가장 큰 감소)</h3>
  {cls.bins_table_html(series)}
  <details><summary>원본 기록</summary><ul>{paths}</ul></details>
</section>"""

    @classmethod
    def build(cls, args):
        walk, run = cls.resolve(args.walk), cls.resolve(args.run)
        note = ("감소량 = 무보조 A의 착용자 고관절 토크 실효값 − 보조 B의 실효값, 구간 안에서 좌우 고관절을 각 다리의 참조 위상으로 모아 계산. "
                "양수가 부담 감소다. 그래프는 평가 동작 기준 best checkpoint끼리, 표는 전부.")
        split = lambda v: [x.strip() for x in v.split(";")] if v else None  # noqa: E731
        spec_w, spec_r = split(args.chart_walk or args.chart), split(args.chart_run or args.chart)
        chart_colors = split(args.chart_colors)
        walk_s, run_s = cls.series(walk), cls.series(run)
        sections = cls.section("걷기 (motion 0, 40 s, W1)", walk_s, note, spec_w, chart_colors)
        if run:
            sections += cls.section("달리기 (motion 1, 40 s, W1)", run_s, note, spec_r, chart_colors)
        if args.figures:  # 발표 자료용 그림 (큰 검은 글자)
            Svg.present = True
            for motion, series, spec in (("걷기", walk_s, spec_w), ("달리기", run_s, spec_r)):
                if not series:
                    continue
                sub, labels, colors = cls.chart(series, spec, chart_colors)
                legend = [(l, c, False) for l, c in zip(labels, colors)]
                Figure.save(Svg.reduction_bars_grouped(sub, colors, h=420, legend=legend), f"{args.figures}/{motion}_모델비교_구간별감소량_막대")
                Figure.save(Svg.reduction_lines_grouped(sub, colors, h=420, legend=legend), f"{args.figures}/{motion}_모델비교_구간별감소량_꺾은선")
            Svg.present = False
        page = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(args.title)}</title>
{HtmlShell.PLEX_FONTS}
<style>{HtmlShell.plex_css()}</style></head><body>
<main>
  <header>
    <div class="eyebrow">{html.escape(args.eyebrow)}</div>
    <h1>{html.escape(args.title)}</h1>
    <p class="verdict">{args.verdict}</p>
  </header>
  {Path(args.prelude).read_text() if args.prelude else ""}
  {sections}
  <div class="foot">{args.notes}</div>
</main></body></html>"""
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(page)
        print("saved", args.out, f"({len(page)//1024} KB)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("comparison", help="실행 여러 개를 한 표로")
    s.add_argument("--walk", nargs="+", required=True, help="이름=경로glob (걷기 실행)")
    s.add_argument("--run", nargs="*", default=[], help="이름=경로glob (달리기 실행)")
    s.add_argument("--chart", default=None, help="위상 그림에 올릴 걷기 실행 이름들 (';' 구분, 최대 4; 이름에 쉼표 가능)")
    s.add_argument("--title", default="진자 v4 → 스켈레톤 전이 시험")
    s.add_argument("--verdict", default="", help="맨 위 한 줄 판정 (HTML 가능)")
    s.add_argument("--notes", default="", help="맨 아래 각주 (HTML 가능)")
    s.add_argument("--out", default="docs/report.html")

    s = sub.add_parser("checkpoint", help="체크포인트 하나를 깊게")
    s.add_argument("--name", default="v13")
    s.add_argument("--walk", required=True, help="이 체크포인트의 걷기 실행 glob")
    s.add_argument("--run", required=True, help="이 체크포인트의 달리기 실행 glob")
    s.add_argument("--ref-walk", nargs="*", default=[], help="이름=glob (걷기 기준 실행)")
    s.add_argument("--ref-run", nargs="*", default=[], help="이름=glob (달리기 기준 실행)")
    s.add_argument("--traj", nargs="*", default=[], help="epoch=glob접두 (뒤에 _m0/_m1이 붙는 실행 쌍)")
    s.add_argument("--tb-dir", default=None)
    s.add_argument("--tb-max-epoch", type=int, default=20000)
    s.add_argument("--title", default="LLP v13 스켈레톤 시험")
    s.add_argument("--eyebrow", default="AI-SBC · LLP 체크포인트 보고서")
    s.add_argument("--verdict", default="")
    s.add_argument("--notes", default="")
    s.add_argument("--out", default="docs/report_ckpt.html")

    s = sub.add_parser("finetune", help="추가학습 전후 비교: 보행 진행률 10 % 구간별 토크 실효값 감소량 + 전체 지표")
    s.add_argument("--walk", nargs="+", required=True, help="이름=glob (걷기 실행; glob이 seed 여러 개에 맞으면 평균)")
    s.add_argument("--run", nargs="*", default=[], help="이름=glob (달리기 실행)")
    s.add_argument("--chart", default=None, help="그래프에 올릴 항목 'run 이름' 또는 'run 이름=범례 이름' (';' 구분, 기본 전부; 표에는 전부 나온다)")
    s.add_argument("--chart-walk", default=None, help="걷기 절의 그래프 항목 (없으면 --chart)")
    s.add_argument("--chart-run", default=None, help="달리기 절의 그래프 항목 (없으면 --chart)")
    s.add_argument("--chart-colors", default=None, help="그래프 항목과 같은 순서의 색 (';' 구분, 기본 PALETTE)")
    s.add_argument("--title", default="LLP 추가학습 전후 비교")
    s.add_argument("--eyebrow", default="AI-SBC · LLP 추가학습")
    s.add_argument("--verdict", default="", help="맨 위 한 줄 판정 (HTML 가능)")
    s.add_argument("--notes", default="", help="맨 아래 각주 (HTML 가능)")
    s.add_argument("--prelude", default=None, help="걷기·달리기 절 앞에 넣을 HTML 조각 파일 (예: 예측기 비교 표)")
    s.add_argument("--figures", default=None, help="발표 자료용 그림 폴더: 걷기·달리기 구간별 감소량 막대·꺾은선을 큰 검은 글자의 .svg/.png로 저장")
    s.add_argument("--out", default="docs/ai_sbc/finetune_report.html")

    args = p.parse_args(argv)
    if args.command == "comparison":
        ComparisonReport.build(args)
    elif args.command == "checkpoint":
        CheckpointReport.build(args)
    else:
        FinetuneReport.build(args)


if __name__ == "__main__":
    main()
