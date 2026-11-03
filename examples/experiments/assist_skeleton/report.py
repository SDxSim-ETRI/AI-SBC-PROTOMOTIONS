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
"""run_system.py 기록의 HTML 보고서 — 공통 틀(HtmlShell·Svg·Fmt·Figures)과 보고서 네 종류.

  `ComparisonReport` (`comparison`)  실행 여러 개를 한 표로: 비교표 + 위상 그림 + 위상표 (옛 report_html)
  `CheckpointReport` (`checkpoint`)  체크포인트 하나를 깊게: 핵심 숫자·비교표·위상·시계열·주기 평균·학습 궤적 (옛 report_checkpoint_html)
  `TorqueReport`     (`torque`)      시퀀스 comparison.json의 phase/time 토크 보고서 + 개발 단계 통과 판정 (옛 report_torque_html)
  `SwingGateReport`  (`swing-gate`)  걷기 68–92 % 구간 분석과 관절각 유각 게이트 (옛 report_swing_gate)
`WalkingRecords.gather()`는 보관된 W1 기준 기록(L1 ±10/15/20, R0 ±15/20)의 색인이다 (진자 `report.py`가 쓴다).
수치 계산은 전부 `analysis.py`에 있고 여기서는 HTML·SVG만 만든다. 생성 뒤 `python docs/ai_sbc/organize_reports.py`를 실행한다.

    python examples/experiments/assist_skeleton/report.py swing-gate --out docs/ai_sbc/walking_swing_gate.html
    python examples/experiments/assist_skeleton/report.py torque --manifest results/<seq>/comparison.json --out <html>
    python examples/experiments/assist_skeleton/report.py comparison --out <html> --walk base=results/oracle/*_tracker_base ...
    python examples/experiments/assist_skeleton/report.py checkpoint --out <html> --name v13 --walk <glob> --run <glob> ...
"""

import argparse
import glob
import html
import io
import json
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
    ROOT,
    ObjectiveComparison,
    PhaseScreen,
    RunRecord,
    SwingGateBins,
    TorqueAlignment,
    assist_output_limit,
    latest_run,
    read_json,
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

    STUDY_CSS = ("html{scroll-behavior:smooth}body{margin:0;background:#f4f6f8;color:#1c2d3d;font:16px/1.8 system-ui,sans-serif}"
                 "main{max-width:1180px;margin:auto;padding:28px 26px 70px}h1{font-size:30px;line-height:1.4}h2{font-size:23px;margin:38px 0 12px}"
                 "a{color:#245aa0}table{width:100%;border-collapse:collapse;background:white;font-size:14px;line-height:1.6}th{background:#e5edf5;text-align:left}"
                 "td,th{padding:8px 11px;border:1px solid #dce3eb;vertical-align:top}strong{color:#185b4a}code{font-size:.88em;overflow-wrap:anywhere}"
                 "nav{display:flex;gap:18px;flex-wrap:wrap;border-bottom:1px solid #d8e1ea;padding-bottom:15px}"
                 "details{background:white;border:1px solid #d8e1ea;border-radius:8px;padding:15px 20px;margin:20px 0}summary{cursor:pointer;font-weight:650}"
                 ".note{color:#62748a;font-size:14px}.plot{background:white;border:1px solid #dce3eb;border-radius:10px;padding:10px;margin:12px 0}"
                 ".plot svg{width:100%;height:auto;display:block}.status{padding:16px 18px;background:#e5eef9;border-radius:8px;border-left:5px solid #245aa0}"
                 ".ok{color:#185b4a;font-weight:650}.part{color:#9a6700;font-weight:650}.no{color:#b42318;font-weight:650}small{color:#62748a}"
                 ".scroll{overflow-x:auto}@media(max-width:750px){main{padding:20px 16px}h1{font-size:25px}table{display:block;overflow-x:auto}}")

    TORQUE_TEMPLATE = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLP | 보행 phase와 토크 분담</title><style>
:root{--ink:#15232d;--ink2:#59656b;--grid:#e8e7e2;--zero:#929995;--limit:#97afc4}
*{box-sizing:border-box}body{margin:0;background:#fafaf7;color:var(--ink);font:15px/1.65 system-ui,sans-serif}main{max-width:1080px;margin:auto;padding:36px 24px 70px}h1{font-size:30px;margin:6px 0 12px}h2{font-size:21px;margin:26px 0 8px}h3{font-size:16px;margin:0 0 8px}.eyebrow{color:#2275ce;font-size:12px;letter-spacing:.12em;font-weight:700}.note{color:var(--ink2);font-size:13px;margin:8px 0 16px}.explain{background:#edf3f4;border-left:3px solid #2275ce;padding:15px 20px;margin:22px 0}.controls{position:sticky;top:0;background:#fafaf7f5;z-index:2;border-bottom:1px solid #dadeda;padding:16px 0;display:flex;gap:12px;align-items:end;flex-wrap:wrap}.controls label{font-size:12px;color:var(--ink2);display:grid;gap:3px}select,button{font:inherit;padding:8px 12px;border:1px solid #cbd2d1;border-radius:6px;background:white;color:var(--ink);cursor:pointer}select:focus,button:focus{outline:2px solid #2275ce}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:18px 0}.cards>div{background:#f0f0e9;padding:14px;border-radius:8px}.cards span{display:block;font-size:12px;color:var(--ink2)}.cards strong{display:block;font-size:23px;font-weight:600}.plot{padding:18px 18px 6px;background:white;border:1px solid #e4e6e1;border-radius:8px;margin:14px 0}.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:12px}.lg i{display:inline-block;width:16px;height:3px;margin-right:6px;vertical-align:middle}svg{width:100%;height:auto;display:block}svg .grid{stroke:var(--grid);stroke-width:1}svg .zero{stroke:var(--zero)}svg .mark{stroke:var(--zero);stroke-dasharray:4 4}svg .limit{stroke:var(--limit);stroke-dasharray:6 4}svg .tick{fill:var(--ink2);font:11px system-ui}svg .axis{fill:var(--ink2);font:12px system-ui}.scroll{overflow-x:auto;max-width:100%}table{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}td,th{padding:9px 10px;border-bottom:1px solid #e0e3dc;text-align:right;white-space:nowrap}th:first-child{text-align:left}thead{background:#f0f0e9}button.pick{border:0;padding:0;background:none;color:#1763a5;text-align:left}details{margin:25px 0;color:var(--ink2);font-size:13px}summary{cursor:pointer;font-weight:600}.path{overflow-wrap:anywhere}.download{margin:12px 0}[hidden]{display:none!important}.footer{margin-top:40px;border-top:1px solid #ddd;padding-top:16px;color:var(--ink2);font-size:13px}
@media(max-width:650px){main{padding:20px 12px}h1{font-size:24px}.cards{grid-template-columns:1fr 1fr}.cards strong{font-size:20px}.plot{padding:12px 4px}.controls{gap:8px}.controls select{max-width:94vw;font-size:13px}}
</style></head><body><main>
<header><div class="eyebrow">AI-SBC / TORQUE ANALYSIS</div><h1>보행 phase와 토크 분담</h1><p>L1 ×0.5 기준선과 최대 3개 후보만 비교합니다. 전체 평균과 phase별 악화를 함께 봅니다.</p></header>
<div class="explain"><b>A = 무보조, B = 보조 사용.</b> τ_agent는 두 경우 모두 <b>착용자만 낸 토크</b>입니다.<br>
τ_assist는 장치가 보탠 토크이며, τ_total = τ_agent + τ_assist입니다.<br>
보조에 따라 움직임과 착용자의 반응이 달라지므로, B의 합계가 A와 반드시 같지는 않습니다.</div>
<details><summary>선정 후보 비교표 열기 · 행을 누르면 해당 그래프로 이동</summary><div class="scroll"><table><thead><tr><th>모델 / 동작</th><th>착용자 τ² 변화</th><th>몸체 오차 변화</th><th>보조 RMS [N·m]</th><th>결과</th></tr></thead><tbody>__SUMMARY__</tbody></table></div></details>
<p class="note">선정 기준: 걷기·달리기 모두 40초 완주하고 전체 τ²가 모두 감소한 후보 중, 최악 phase의 RMS 증가가 작은 순입니다. 동률은 phase 악화 평균·전체 τ² 순으로 비교합니다. 유효 후보가 3개 미만이면 그 수만 표시합니다. 선정은 합격이나 일반화 입증을 뜻하지 않습니다.</p>
<details><summary>개발 단계 통과 기준 · L8 결과 이전에 고정</summary><p>걷기·달리기 모두 40초 완주, 추론 배율 1과 ±100 N·m. 전체 τ², 최악 phase, 20–40% 및 70–90% 구간의 RMS 증가가 L1×0.5보다 나빠지면 미달입니다. 각 10% 구간의 L1 대비 RMS 변화 악화는 1 N·m 이내, 몸체 오차 증가는 무보조 대비 1 mm 이내, 각속도 변화 크기는 L1보다 2%p 이내 추가 변화, 전진 속도 변화는 ±2% 이내로 제한합니다. 이는 개발용 선별 기준이며 여러 seed·미사용 조건 검증을 대신하지 않습니다.</p></details>
<div class="controls"><label>모델 / 학습 시점<select id="model">__OPTIONS__</select></label><label>동작<select id="motion"><option value="0">걷기</option><option value="1">달리기</option></select></label><label>그래프<select id="view"><option value="phase">보행 phase별</option><option value="time">시간별 · 오른쪽 hip</option></select></label></div>
<p class="note">phase 0% = 기준 mocap에서 해당 다리의 최대 hip 굴곡. 실제 발 접촉 검출 기반 phase가 아닙니다. A/B에 동일한 기준 phase를 사용합니다.</p>
__PANELS__
<p id="empty" hidden>해당 조건의 평가 기록이 아직 없습니다.</p>
<div class="footer">시뮬레이션 개발 평가 · 학습 seed 0 / 평가 seed 42. τ²는 토크 부담의 대리지표이며 실제 근활성이나 대사량이 아닙니다. 모델마다 별도의 A/B 실행을 사용했습니다. 동작 변화·추종 오차·낙상과 함께 판단해야 합니다.</div>
</main><script id="data" type="application/json">__PAYLOAD__</script><script>
const data=JSON.parse(document.getElementById('data').textContent),model=document.getElementById('model'),motion=document.getElementById('motion'),view=document.getElementById('view');
function update(){let found=false;document.querySelectorAll('.panel').forEach(p=>{const active=p.dataset.model===model.value&&p.dataset.motion===motion.value;p.hidden=!active;if(active)found=true;p.querySelector('.phase-view').hidden=view.value!=='phase';p.querySelector('.time-view').hidden=view.value!=='time';});document.getElementById('empty').hidden=found;}
[model,motion,view].forEach(x=>x.addEventListener('change',update));
document.querySelectorAll('.pick').forEach(b=>b.addEventListener('click',()=>{model.value=b.dataset.model;motion.value=b.dataset.motion;update();document.querySelector('.controls').scrollIntoView({behavior:'smooth'});}));
document.querySelectorAll('.download').forEach(b=>b.addEventListener('click',()=>{const rows=data[b.dataset.key][b.dataset.kind],keys=Object.keys(rows[0]);const csv='\uFEFF'+keys.join(',')+'\n'+rows.map(r=>keys.map(k=>r[k]===null?'':r[k]).join(',')).join('\n');const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download=b.dataset.key+'_'+b.dataset.kind+'.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}));update();
</script></body></html>"""


class Fmt:
    @staticmethod
    def signed(v, digits=1, sign=True):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "—"
        return f"{v:+.{digits}f}" if sign else f"{v:.{digits}f}"

    @staticmethod
    def number(v, signed=False):
        return "—" if v is None else format(v, "+.1f" if signed else ".1f")

    @staticmethod
    def f1(v, signed=False):
        return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, "+.1f" if signed else ".1f")

    @staticmethod
    def plain(v, spec=".1f"):
        return "—" if v is None else format(v, spec)

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
    def reduction_bars(rows):
        w, h, ml, mr, mt, mb = 900, 230, 65, 15, 20, 45
        bound = max(5, max(abs(r["rms_reduction_Nm"] or 0) for r in rows)) * 1.18
        y = lambda v: mt + (bound - v) / (2 * bound) * (h - mt - mb)  # noqa: E731
        out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="phase별 착용자 토크 RMS 감소량">']
        for v in (-bound / 2, 0, bound / 2):
            out.append(f'<line class="grid" x1="{ml}" x2="{w-mr}" y1="{y(v)}" y2="{y(v)}"/><text class="tick" x="{ml-8}" y="{y(v)+4}" text-anchor="end">{v:+.1f}</text>')
        dx = (w - ml - mr) / len(rows)
        for i, r in enumerate(rows):
            v = r["rms_reduction_Nm"]
            if v is None:
                continue
            x = ml + dx * (i + .15)
            color = "#178463" if v >= 0 else "#cf543b"
            out.append(f'<rect x="{x}" y="{min(y(v),y(0))}" width="{dx*.7}" height="{max(1,abs(y(v)-y(0)))}" fill="{color}"><title>{i*10}–{(i+1)*10}%: {v:+.2f} N·m 감소</title></rect>')
            out.append(f'<text class="tick" x="{x+dx*.35}" y="{y(v)-6 if v>=0 else y(v)+14}" text-anchor="middle">{v:+.1f}</text>')
            out.append(f'<text class="tick" x="{x+dx*.35}" y="{h-22}" text-anchor="middle">{i*10}–{(i+1)*10}</text>')
        return "".join(out) + f'<text class="axis" x="{ml}" y="12">RMS 감소량 [N·m] · 양수 = 부담 감소</text><text class="axis" x="{w-mr}" y="{h-3}" text-anchor="end">보행 phase [%]</text></svg>'

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

    @staticmethod
    def bins_line(series, ylabel, ylim, shade=None, h=250, w=900, bins=25):
        """series: [(label, color, y[bins], dash)] — x = 구간 중심 (유각 게이트 보고서)."""
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
            pts = " ".join(f"{x((i + .5) * 100 / bins):.1f},{y(v):.1f}" for i, v in enumerate(ys) if np.isfinite(v))
            dash_attr = 'stroke-dasharray="6 4"' if dash else ""
            out.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" {dash_attr} points="{pts}"><title>{html.escape(label)}</title></polyline>')
        out.append(f'<text x="{ml}" y="12" font-size="12" fill="#333">{html.escape(ylabel)}</text>'
                   f'<text x="{w - mr}" y="{h - 4}" text-anchor="end" font-size="12" fill="#333">보행 위상 [%] · 0 = 참조 최대 굴곡</text></svg>')
        legend = " ".join(f'<span style="display:inline-block;margin-right:14px"><span style="display:inline-block;width:22px;border-top:3px {"dashed" if dash else "solid"} {color};vertical-align:middle"></span> {html.escape(label)}</span>'
                          for label, color, _, dash in series)
        return f'<div class="legend">{legend}</div>' + "".join(out)

    @staticmethod
    def grouped_bars(rows_list, labels, colors, h=260, w=900):
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


class Figures:
    """matplotlib 그림 → 인라인 SVG (한글 폰트·Agg 설정은 한 번만)."""

    FONT_FILE = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    PREVIEW = Path(os.environ["GEN_REPORT_PNG"]) if os.environ.get("GEN_REPORT_PNG") else None
    _plt = None

    @classmethod
    def pyplot(cls):
        if cls._plt is None:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.font_manager import FontProperties

            font = FontProperties(fname=cls.FONT_FILE)
            plt.rcParams.update({"font.family": font.get_name(), "font.size": 10, "axes.unicode_minus": False, "svg.fonttype": "path",
                                 "axes.spines.top": False, "axes.spines.right": False})
            cls._plt = plt
        return cls._plt

    @classmethod
    def embed(cls, fig, identity, preview_dir=None):
        plt = cls.pyplot()
        preview = cls.PREVIEW if preview_dir is None else Path(preview_dir)
        if preview:
            preview.mkdir(parents=True, exist_ok=True)
            fig.savefig(preview / f"{identity}.png", dpi=110, bbox_inches="tight")
        plt.rcParams["svg.hashsalt"] = identity
        buf = io.StringIO()
        fig.savefig(buf, format="svg", bbox_inches="tight")
        plt.close(fig)
        text = buf.getvalue()
        return '<div class="plot">' + text[text.index("<svg"):] + "</div>"


class WalkingRecords:
    """보관된 W1 기준 기록의 색인: L1 ±10/15/20 (2026-10-16), R0 ±15/20 (2026-10-20), R0 일반화 조건 (2026-10-21)."""

    OUTPUT_SCALE = ROOT / "results/llp_output_scale_20261016"
    RELATIVE = ROOT / "results/llp_relative_20261020/walking"
    GENERALIZATION = ROOT / "results/llp_r0_generalization_20261021"
    SEEDS = (42, 7)
    LIMITS = {"L1": (10, 15, 20), "R0": (15, 20)}

    @classmethod
    def gather(cls):
        """(condition, model, limit, seed) -> record with a repository-relative path."""
        out = {}
        old = read_json(cls.OUTPUT_SCALE / "runs.json")
        relative = read_json(cls.RELATIVE / "runs.json")
        for seed in cls.SEEDS:
            for limit in cls.LIMITS["L1"]:
                out["walk", "L1", limit, seed] = old[f"L1_{limit}Nm_seed{seed}"]
            for limit in cls.LIMITS["R0"]:
                out["walk", "R0", limit, seed] = relative[f"R0_s0_{limit}Nm_seed{seed}"]
        if (cls.GENERALIZATION / "runs.json").exists():
            for r in read_json(cls.GENERALIZATION / "runs.json").values():
                out[r["condition"], r["model"], r["limit_Nm"], r["seed"]] = r
        return out


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
# 시퀀스 토크 보고서 (comparison.json → phase/time, 개발 단계 통과 판정)
# ---------------------------------------------------------------------------
class TorqueReport:
    COLORS = {"a": "#77766f", "b": "#d55b26", "assist": "#2275ce", "total": "#8b55ac"}

    @classmethod
    def series(cls, data, key, color, dash=None):
        return {"y": [np.nan if r[key] is None else r[key] for r in data],
                "color": cls.COLORS[color], "width": 2, "dash": dash}

    @classmethod
    def torque_plots(cls, x, rows, xlabel, xlim, limit, marks=()):
        plots = []
        # A shared physical torque scale makes separate agent/assist panels comparable.
        largest = max([limit] + [abs(r[k]) for r in rows for k in ("agent_A", "agent_B", "assist_B", "total_B") if r[k] is not None])
        ylim = (-np.ceil(largest / 25) * 25 * 1.08, np.ceil(largest / 25) * 25 * 1.08)
        specs = [
            ("착용자가 낸 토크 · τ_agent", [("agent_A", "a", "무보조 A · τ_agent"), ("agent_B", "b", "보조 B · τ_agent")]),
            ("장치가 낸 토크 · τ_assist", [("assist_A", "a", "무보조 A · τ_assist = 0"), ("assist_B", "assist", "보조 B · τ_assist")]),
            ("착용자 + 장치 · τ_total", [("agent_A", "a", "무보조 A · τ_total = τ_agent"), ("total_B", "total", "보조 B · τ_total = τ_agent + τ_assist")]),
        ]
        for title, lines in specs:
            leg = Svg.legend([(label, cls.COLORS[color], False) for key, color, label in lines])
            svg = Svg.line(x, [cls.series(rows, key, color) for key, color, label in lines],
                           ylabel="토크 [N·m]", xlabel=xlabel, xlim=xlim, ylim=ylim, H=240,
                           xtick=10 if xlim == (0, 100) else None, vlines=marks,
                           hlines=(-limit, limit) if "τ_assist" in title else (), label=title)
            plots.append(f'<div class="plot"><h3>{title}</h3>{leg}{svg}</div>')
        return "".join(plots)

    @staticmethod
    def phase_table(rows):
        number = Fmt.number
        headers = ["phase", "τ_agent A RMS", "τ_agent B RMS", "RMS 감소량", "τ² 변화", "τ_assist B 평균", "τ_assist B RMS"]
        body = []
        for r in rows:
            body.append('<tr><th>' + f'{r["phase_start_pct"]:.0f}–{r["phase_end_pct"]:.0f}%' + '</th>' +
                        ''.join(f'<td>{number(r[k], k in ("rms_reduction_Nm", "tau2_change_pct", "assist_B"))}{"%" if k == "tau2_change_pct" else ""}</td>'
                                for k in ("agent_A_rms", "agent_B_rms", "rms_reduction_Nm", "tau2_change_pct", "assist_B", "assist_B_rms")) + '</tr>')
        return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(body) + '</tbody></table></div>'

    @classmethod
    def build(cls, manifest, output):
        T = TorqueAlignment
        manifest, output = Path(manifest), Path(output)
        all_records = json.loads(manifest.read_text())
        records, ranking = T.rank_candidates(all_records)
        screen = PhaseScreen.screen_records(all_records, ranking["evaluations"])
        manifest.with_name("phase_screen.json").write_text(json.dumps(screen, indent=2, ensure_ascii=False, allow_nan=False))
        manifest.with_name("phase_diagnostics.json").write_text(json.dumps(ranking, indent=2, ensure_ascii=False, allow_nan=False))
        panels, summary, payload, options = [], [], {}, {}
        for key, m in records.items():
            run = Path(m["run_dir"])
            meta = json.loads((run / "meta.json").read_text())
            with np.load(run / "traces.npz") as f:
                tr = {k: f[k] for k in ("t", "gait_phase", "tau_pd_prev", "tau_ff_prev", "tau_assist")}
            data = T.aligned_torques(tr, meta)
            fine, coarse = T.phase_profile(data, 50), T.phase_profile(data, 10)
            times, marks = T.time_rows(data)
            motion = int(meta["motion_id"])
            name = T.label(key)
            model_id = key.rsplit("_m", 1)[0]
            options[model_id] = name
            duration = float(meta["seconds"])
            status = f'{duration:g}초 완주' if meta["survived_s"] is None else f'{meta["survived_s"]:.2f}초 낙상'
            title = f'{name} · {"걷기" if motion == 0 else "달리기"}'
            limit = abs(meta["assist_scale"]) * min(meta["torque_limit"], meta.get("policy_torque_limit", meta["torque_limit"]))
            summary.append(f'<tr><th><button class="pick" data-model="{model_id}" data-motion="{motion}">{html.escape(title)}</button></th><td>{m["τ²"]:+.1f}%</td><td>{m["body"]:+.1f}%</td><td>{m["assist_rms"]:.1f}</td><td>{status}</td></tr>')
            phaseplots = cls.torque_plots([r["phase_pct"] for r in fine], fine, "보행 phase [%]", (0, 100), limit)
            timeplots = cls.torque_plots([r["time_s"] for r in times], times, "실험 시작 후 시간 [s]", (times[0]["time_s"], times[-1]["time_s"]), limit, marks)
            payload[key] = {"phase": coarse, "time": times}
            worst_harm = ranking["evaluations"][key]["worst_phase_rms_increase_Nm"]
            bad_bins = sum((r["rms_reduction_Nm"] or 0) < 0 for r in coarse)
            cards = ''.join(f'<div><span>{k}</span><strong>{v}</strong></div>' for k, v in [
                ("착용자 τ² 변화", f'{m["τ²"]:+.1f}%'), ("최악 phase RMS 증가", f'+{worst_harm:.1f} N·m'),
                ("부담이 늘어난 phase", f'{bad_bins} / 10 구간'), ("보조 RMS", f'{m["assist_rms"]:.1f} N·m'),
                ("몸체 추종 오차 변화", f'{m["body"]:+.1f}%'), ("평가 결과", status)])
            screening = "기준선" if key.startswith("reference") else ("개발 기준 통과 · 일반화 미검증" if screen["candidate_pass"].get(model_id) else "개발 기준 미달")
            panels.append(f'''
<article class="panel" data-model="{model_id}" data-motion="{motion}" hidden>
<h2>{html.escape(title)}</h2><p><strong>{screening}</strong> · 후보 순위와 통과 판정은 별도입니다.</p><div class="cards">{cards}</div>
<p class="note">추론 배율 {meta['assist_scale']:g} · 출력 제한 ±{limit:g} N·m · warmup {meta['warmup']:g}초 제외.
{'낙상 전 기록만 포함하므로 완주 모델과 동등한 비교가 아닙니다.' if meta['survived_s'] is not None else '40초 기록 중 warmup 이후를 분석했습니다.'}</p>
<section class="phase-view"><h2>어느 phase에서 착용자 부담이 줄었나?</h2>
<p class="note">RMS 감소량 = 무보조 A − 보조 B. 초록/양수는 감소, 빨강/음수는 증가입니다. RMS는 구간 안에서 토크를 제곱·평균한 뒤 제곱근을 취합니다.</p>
{Svg.reduction_bars(coarse)}{cls.phase_table(coarse)}
<p class="note">토크 단위 N·m. τ² 변화는 B/A − 1로, 음수일수록 감소입니다. 좌·우 hip을 각 다리의 기준 phase로 모았습니다.</p>
<h2>Phase별 토크를 역할별로 보기</h2><p class="note">아래 곡선은 부호를 가진 평균 토크입니다. +굴곡 / −신전. 세 그래프의 토크 축 범위는 같습니다.</p>
{phaseplots}
<button class="download" data-key="{key}" data-kind="phase">phase 수치 CSV 저장</button></section>
<section class="time-view" hidden><h2>실제 시간에 따른 토크 · 오른쪽 hip</h2>
<p class="note">위상 0% 경계부터 최대 세 주기를 표시합니다. 세로 점선은 기준 phase 0%. 모든 토크는 동일한 물리 시간 구간으로 정렬했습니다.</p>
{timeplots}<button class="download" data-key="{key}" data-kind="time">시계열 CSV 저장</button></section>
<details><summary>원본 기록과 해석 기준</summary><p class="path">{html.escape(str(run))}</p>
<p>τ_agent = tau_pd_prev + tau_ff_prev, τ_assist = 실제 주입 명령. τ_agent[i]와 τ_assist[i−1]을 같은 창으로 맞추고 그 창의 중앙 시각·중앙 phase로 표시했습니다.
τ_total은 이 두 구동 토크의 합이며 관절의 모든 수동·접촉 힘을 포함한 역동역학 토크가 아닙니다.</p></details></article>
''')
        default = ranking["selected_groups"][0] if ranking["selected_groups"] else next(iter(options))
        option_html = ''.join(f'<option value="{k}" {"selected" if k==default else ""}>{html.escape(v)}</option>' for k, v in options.items())
        page = HtmlShell.TORQUE_TEMPLATE.replace("__OPTIONS__", option_html).replace("__SUMMARY__", ''.join(summary)).replace("__PANELS__", ''.join(panels)).replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False, allow_nan=False))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(page)
        print(f"saved {output} ({len(records)} evaluations)")


# ---------------------------------------------------------------------------
# 유각 브레이크 게이트 보고서
# ---------------------------------------------------------------------------
class SwingGateReport:
    """run_system.py 기록만 읽어 HTML 하나를 만든다 (2026-10-14). 새 시뮬레이션은 돌리지 않는다."""

    RUNS = {
        "L1 ×0.5 · 걷기": ("baseline", "reference_L1_x0.5_m0"),
        "L1 ×0.5 · 달리기": ("baseline", "reference_L1_x0.5_m1"),
        "β 오라클 0.5 · 걷기": ("probe", "beta05_m0"),
        "L1 ×0.5 + 참조위상 70–90 % 차단 · 걷기": ("probe", "L1s05_mask70_90_m0"),
        "L1 ×0.5 + 참조위상 70–90 % 차단 · 달리기": ("probe", "L1s05_mask70_90_m1"),
        "L1 ×0.5 + 참조위상 20–30 % 차단 · 걷기": ("probe", "L1s05_mask20_30_m0"),
        "L1 ×0.5 + 참조위상 20–40 % 차단 · 걷기": ("probe", "L1s05_mask20_40_m0"),
        "L1 ×0.5 + 유각 게이트 0.20 · 걷기": ("probe", "L1s05_gate020_m0"),
        "L1 ×0.5 + 유각 게이트 0.20 · 달리기": ("probe", "L1s05_gate020_m1"),
        "L1 ×0.5 + 유각 게이트 0.15 · 걷기": ("probe", "L1s05_gate015_m0"),
        "L1 ×0.5 + 유각 게이트 0.25 · 걷기": ("probe", "L1s05_gate025_m0"),
        "L1 ×0.5 · 걷기 · seed 7": ("probe", "L1s05_seed7_m0"),
        "L1 ×0.5 + 유각 게이트 0.20 · 걷기 · seed 7": ("probe", "L1s05_gate020_seed7_m0"),
        "L1 ×0.5 · 달리기 · seed 7": ("probe", "L1s05_seed7_m1"),
        "L1 ×0.5 + 유각 게이트 0.20 · 달리기 · seed 7": ("probe", "L1s05_gate020_seed7_m1"),
    }

    @staticmethod
    def phase_table(rows, lo, hi, gate=False):
        f1 = Fmt.f1
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

    @classmethod
    def build(cls, out, baseline_dir, probe_dir):
        G = SwingGateBins
        folders = {"baseline": baseline_dir, "probe": probe_dir}
        runs = {name: G.find_run(folders[kind], tag) for name, (kind, tag) in cls.RUNS.items()}
        runs = {k: v for k, v in runs.items() if v is not None}
        rec = {}
        for name, run in runs.items():
            meta, d, tr = G.load(run)
            s = G.run_summary(run)
            rows = G.bins(d)
            rec[name] = dict(meta=meta, d=d, s=s, rows=rows, run=run,
                             harm_68_92=G.window_contribution(rows, 68, 92), harm_20_40=G.window_contribution(rows, 20, 40),
                             worst=max(r["B_rms"] - r["A_rms"] for r in rows), gate=G.gate_accuracy(d))

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
            plots += Svg.bins_line([("무보조 A", "#77766f", ys(base, "A"), False), ("보조 B · L1 ×0.5", "#d55b26", ys(base, "B"), False),
                                    ("보조 B · 게이트 0.20", "#1f7a4d", ys(gate, "B"), False), ("보조 B · β 오라클", "#8b55ac", ys(beta, "B"), True)],
                                   "τ_agent [N·m] · +굴곡 / −신전", (-100, 50), shade=(68, 92))
            plots += "<h3>장치가 낸 토크 · 걷기</h3>"
            plots += Svg.bins_line([("L1 ×0.5", "#d55b26", ys(base, "S"), False), ("게이트 0.20", "#1f7a4d", ys(gate, "S"), False), ("β 오라클", "#8b55ac", ys(beta, "S"), True)],
                                   "τ_assist [N·m]", (-40, 40), shade=(68, 92))
            plots += "<h3>hip 각속도·각가속도 · 무보조 A · 걷기</h3>"
            plots += Svg.bins_line([("각속도 ×10 [rad/s]", "#2275ce", [10 * v for v in ys(base, "vel")], False), ("각가속도 [rad/s²]", "#c9a227", ys(base, "acc"), False)],
                                   "각속도 ×10, 각가속도", (-60, 60), shade=(68, 92))
            plots += "<h3>구간별 착용자 토크 RMS 변화 · 걷기</h3>"
            plots += Svg.grouped_bars([base["rows"], mask["rows"], gate["rows"]] if mask else [base["rows"], gate["rows"]],
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
            cause += "<h3>걷기 · L1 ×0.5 · 60–100 %</h3>" + cls.phase_table(base["rows"], 60, 100)
            cause += "<h3>걷기 · L1 ×0.5 · 12–40 %</h3>" + cls.phase_table(base["rows"], 12, 40)
        if gate:
            cause += "<h3>걷기 · 게이트 0.20 · 60–100 % (마지막 열 = 게이트가 켜진 표본 비율)</h3>" + cls.phase_table(gate["rows"], 60, 100, gate=True)
        if base_r and gate_r:
            cause += "<h3>달리기 · L1 ×0.5 · 76–100 %</h3>" + cls.phase_table(base_r["rows"], 76, 100)
            cause += "<h3>달리기 · 게이트 0.20 · 76–100 %</h3>" + cls.phase_table(gate_r["rows"], 76, 100, gate=True)

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
        Path(out).write_text(page, encoding="utf-8")
        print(f"saved {out} ({len(rec)} runs)")
        for k, r in rec.items():
            print(f"  {k}: τ² {r['s']['τ²']:+.1f} %, 68–92 {r['harm_68_92']:+.1f}, 20–40 {r['harm_20_40']:+.1f}, worst {r['worst']:+.1f}")


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

    s = sub.add_parser("torque", help="시퀀스 comparison.json의 phase/time 토크 보고서")
    s.add_argument("--manifest", default="results/llp_wide100_sequence/comparison.json")
    s.add_argument("--out", default="docs/ai_sbc/l6_l7_torque_report.html")

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

    s = sub.add_parser("swing-gate", help="걷기 유각 브레이크 분석과 관절각 게이트")
    s.add_argument("--out", default="docs/ai_sbc/walking_swing_gate.html")
    s.add_argument("--baseline-dir", default="results/pendulum_reeval_20261013/evaluations")
    s.add_argument("--probe-dir", default="results/walk_phase_analysis_20261014/evaluations")

    args = p.parse_args(argv)
    if args.command == "comparison":
        ComparisonReport.build(args)
    elif args.command == "checkpoint":
        CheckpointReport.build(args)
    elif args.command == "torque":
        TorqueReport.build(args.manifest, args.out)
    elif args.command == "finetune":
        FinetuneReport.build(args)
    else:
        SwingGateReport.build(args.out, args.baseline_dir, args.probe_dir)


if __name__ == "__main__":
    main()
