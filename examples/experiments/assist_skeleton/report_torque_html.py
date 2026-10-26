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
"""Standalone phase/time torque report from a wide-sequence comparison manifest."""

import argparse
import html
import json
import re
from pathlib import Path

import numpy as np

from report_checkpoint_html import legend_html, line_svg

COLORS = {"a": "#77766f", "b": "#d55b26", "assist": "#2275ce", "total": "#8b55ac"}


def aligned_torques(tr, meta):
    """Align all torques to the SAME previous control window, including sample 0."""
    idx = np.flatnonzero((tr["t"] >= meta["warmup"]) & (np.arange(len(tr["t"])) >= 1))
    agent = (tr["tau_pd_prev"] + tr["tau_ff_prev"])[idx]
    assist = tr["tau_assist"][idx - 1]
    # Circular midpoint avoids turning a phase wrap 99% -> 1% into 50%.
    before, after = tr["gait_phase"][idx - 1], tr["gait_phase"][idx]
    phase = (before + .5 * ((after - before + .5) % 1 - .5)) % 1
    return {"t": .5 * (tr["t"][idx - 1] + tr["t"][idx]), "phase": phase,
            "agent_A": agent[:, 0], "agent_B": agent[:, 1],
            "assist_A": assist[:, 0], "assist_B": assist[:, 1],
            "total_B": agent[:, 1] + assist[:, 1]}


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


def number(v, signed=False):
    return "—" if v is None else format(v, "+.1f" if signed else ".1f")


def series(data, key, color, dash=None):
    return {"y": [np.nan if r[key] is None else r[key] for r in data],
            "color": COLORS[color], "width": 2, "dash": dash}


def torque_plots(x, rows, xlabel, xlim, limit, marks=()):
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
        leg = legend_html([(label, COLORS[color], False) for key, color, label in lines])
        svg = line_svg(x, [series(rows, key, color) for key, color, label in lines],
                       ylabel="토크 [N·m]", xlabel=xlabel, xlim=xlim, ylim=ylim, H=240,
                       xtick=10 if xlim == (0, 100) else None, vlines=marks,
                       hlines=(-limit, limit) if "τ_assist" in title else (), label=title)
        plots.append(f'<div class="plot"><h3>{title}</h3>{leg}{svg}</div>')
    return "".join(plots)


def reduction_bars(rows):
    w, h, ml, mr, mt, mb = 900, 230, 65, 15, 20, 45
    bound = max(5, max(abs(r["rms_reduction_Nm"] or 0) for r in rows)) * 1.18
    y = lambda v: mt + (bound - v) / (2 * bound) * (h - mt - mb)
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


def phase_table(rows):
    headers = ["phase", "τ_agent A RMS", "τ_agent B RMS", "RMS 감소량", "τ² 변화", "τ_assist B 평균", "τ_assist B RMS"]
    body = []
    for r in rows:
        body.append('<tr><th>' + f'{r["phase_start_pct"]:.0f}–{r["phase_end_pct"]:.0f}%' + '</th>' +
                    ''.join(f'<td>{number(r[k], k in ("rms_reduction_Nm", "tau2_change_pct", "assist_B"))}{"%" if k == "tau2_change_pct" else ""}</td>'
                            for k in ("agent_A_rms", "agent_B_rms", "rms_reduction_Nm", "tau2_change_pct", "assist_B", "assist_B_rms")) + '</tr>')
    return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(body) + '</tbody></table></div>'


def time_rows(data):
    t, ph = data["t"], data["phase"][:, 0]
    wraps = np.flatnonzero(np.diff(ph) < -.5) + 1
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


def label(key):
    if key.startswith("reference"):
        return "L1 ×0.5 · 기존 기준선"
    version = re.search(r"(?:^|_)(L[0-9]+)(?:_|$)", key)
    epoch = re.search(r"_e([0-9]+)(?:_|$)", key)
    return f'{version.group(1) if version else key} ×1 · {epoch.group(1) if epoch else "?"} epoch'


def rank_candidates(records, top_k=3):
    """Use both motions, reject falls/global regressions, then rank worst phase harm.

    This selects a display shortlist, NOT validated winners. Keep all source runs.
    """
    groups, diagnostics = {}, {}
    for key, m in records.items():
        run = Path(m["run_dir"])
        meta = json.loads((run / "meta.json").read_text())
        with np.load(run / "traces.npz") as f:
            tr = {k: f[k] for k in ("t", "gait_phase", "tau_pd_prev", "tau_ff_prev", "tau_assist")}
        data = aligned_torques(tr, meta)
        if len(data["t"]) < 2:
            diagnostics[key] = {"invalid": "insufficient samples after warmup"}
            continue
        rows = phase_profile(data, 10)
        increases = [max(0., -(r["rms_reduction_Nm"] or 0.)) for r in rows]
        diagnostics[key] = {"worst_phase_rms_increase_Nm": max(increases),
                            "mean_phase_rms_increase_Nm": float(np.mean(increases)),
                            "phase": rows, "intervals": {}}
        for lo, hi in ((20, 40), (70, 90)):
            sel = (data["phase"] >= lo/100) & (data["phase"] < hi/100)
            if not sel.any():
                diagnostics[key]["intervals"][f"{lo}-{hi}"] = {"samples": 0}
                continue
            a,b,u = (data[k][sel] for k in ("agent_A", "agent_B", "assist_B"))
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


def build_report(manifest, output):
    manifest, output = Path(manifest), Path(output)
    all_records = json.loads(manifest.read_text())
    records, ranking = rank_candidates(all_records)
    from phase_screen import screen_records

    screen = screen_records(all_records, ranking["evaluations"])
    manifest.with_name("phase_screen.json").write_text(json.dumps(screen, indent=2, ensure_ascii=False, allow_nan=False))
    manifest.with_name("phase_diagnostics.json").write_text(json.dumps(ranking, indent=2, ensure_ascii=False, allow_nan=False))
    panels, summary, payload, options = [], [], {}, {}
    for key, m in records.items():
        run = Path(m["run_dir"])
        meta = json.loads((run / "meta.json").read_text())
        with np.load(run / "traces.npz") as f:
            tr = {k: f[k] for k in ("t", "gait_phase", "tau_pd_prev", "tau_ff_prev", "tau_assist")}
        data = aligned_torques(tr, meta)
        fine, coarse = phase_profile(data, 50), phase_profile(data, 10)
        times, marks = time_rows(data)
        motion = int(meta["motion_id"])
        name = label(key)
        model_id = key.rsplit("_m", 1)[0]
        options[model_id] = name
        duration = float(meta["seconds"])
        status = f'{duration:g}초 완주' if meta["survived_s"] is None else f'{meta["survived_s"]:.2f}초 낙상'
        title = f'{name} · {"걷기" if motion == 0 else "달리기"}'
        limit = abs(meta["assist_scale"]) * min(meta["torque_limit"], meta.get("policy_torque_limit", meta["torque_limit"]))
        summary.append(f'<tr><th><button class="pick" data-model="{model_id}" data-motion="{motion}">{html.escape(title)}</button></th><td>{m["τ²"]:+.1f}%</td><td>{m["body"]:+.1f}%</td><td>{m["assist_rms"]:.1f}</td><td>{status}</td></tr>')
        phaseplots = torque_plots([r["phase_pct"] for r in fine], fine, "보행 phase [%]", (0, 100), limit)
        timeplots = torque_plots([r["time_s"] for r in times], times, "실험 시작 후 시간 [s]", (times[0]["time_s"], times[-1]["time_s"]), limit, marks)
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
{reduction_bars(coarse)}{phase_table(coarse)}
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
    option_html = ''.join(f'<option value="{k}" {"selected" if k==default else ""}>{html.escape(v)}</option>' for k,v in options.items())
    page = TEMPLATE.replace("__OPTIONS__", option_html).replace("__SUMMARY__", ''.join(summary)).replace("__PANELS__", ''.join(panels)).replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False, allow_nan=False))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page)
    print(f"saved {output} ({len(records)} evaluations)")


TEMPLATE = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="results/llp_wide100_sequence/comparison.json")
    parser.add_argument("--out", default="docs/ai_sbc/l6_l7_torque_report.html")
    args = parser.parse_args()
    build_report(args.manifest, args.out)
