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
"""Render the three discussed pendulum candidates as a standalone HTML snapshot.

Run from the repository root, then run docs/ai_sbc/organize_reports.py.
Inputs are deliberately pinned: new evaluations do not silently change selection.
"""

from datetime import datetime, timezone
from html import escape
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
ADAPT = "results/pendulum_adaptation_20261015"
FOLLOWUP = "results/pendulum_research_followup_20261015"
OUTPUT = ROOT / "docs/ai_sbc/pendulum_candidates_report.html"
COLORS = {"PA_E": "#167b65", "L1": "#376bc4", "C0": "#b46c20", "ref": "#b5c0cd"}
INK = "#23364b"
METRICS = [
    ("tau_sq_Nm2", "사람 토크 제곱 평균", "N·m² / 관절", 1),
    ("tracking_rmse_deg", "목표 각도 추종 오차", "RMSE · °", 3),
    ("abs_work_W", "절대 기계 일률", "W / 관절", 2),
]
INPUTS = {
    "L1": f"{ADAPT}/eval_L1_H1_legacy/summary.json",
    "C0": f"{FOLLOWUP}/eval_C0_e5000_H1_legacy/summary.json",
    **{
        f"PA_E_{epoch}": f"{ADAPT}/eval_PA_E_e{epoch}_H1_legacy/summary.json"
        for epoch in (100, 300, 600, 1000)
    },
}


def load_evidence():
    records, sources = {}, []
    for name, relative in INPUTS.items():
        path = ROOT / relative
        raw = path.read_bytes()
        records[name] = json.loads(raw)
        sources.append(
            {
                "label": name,
                "path": relative,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "modified_utc": datetime.fromtimestamp(
                    path.stat().st_mtime, timezone.utc
                ).isoformat(),
                "summary": records[name],
            }
        )
    base = records["L1"]
    for record in records.values():
        for key in (
            "base_checkpoint", "condition", "independent_limits", "combined_human_limit",
            "num_envs", "steps", "cohorts", "plant_seed", "trajectory_seed", "warmup",
            "hlp_checkpoint", "hlp_seed",
        ):
            assert record["protocol"][key] == base["protocol"][key], key
        # Separate GPU runs can differ at floating-point precision. Compare the
        # common exposure and reference metrics, retaining hashes as provenance.
        assert record["metrics"]["all"]["joint_samples"] == base["metrics"]["all"]["joint_samples"]
        assert record["failures"] == base["failures"] == {"reference": 5, "assisted": 0}
        assert record["total_episodes"] == 256
        assert record["injection_error_max_Nm"] == 0
        assert record["injection_clipped_fraction"] == 0
        for metric, _, _, _ in METRICS:
            assert np.isclose(
                record["metrics"]["all"]["reference_" + metric],
                base["metrics"]["all"]["reference_" + metric],
                rtol=1e-6,
            )
    return records, sources


def setup_plotting():
    plt.rcParams.update(
        {
            "font.family": "Noto Sans CJK JP",
            "font.size": 11,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": "#63758a",
            "ytick.color": INK,
            "axes.edgecolor": "#dce3ea",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "svg.fonttype": "path",
            "svg.hashsalt": "pendulum-candidates-20261015",
        }
    )


def svg(fig, title):
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    result = buffer.getvalue()
    result = result[result.index("<svg") :]
    result = re.sub(r'\s(?:width|height)="[^"]+"', "", result, count=2)
    # Every inline SVG needs its own namespace for glyphs and clipping paths.
    prefix = "chart-" + hashlib.sha256(title.encode()).hexdigest()[:10] + "-"
    for ident in set(re.findall(r'\bid="([^"]+)"', result)):
        result = result.replace(f'id="{ident}"', f'id="{prefix}{ident}"')
        result = result.replace(f'href="#{ident}"', f'href="#{prefix}{ident}"')
        result = result.replace(f'url(#{ident})', f'url(#{prefix}{ident})')
    result = result.replace(
        "<svg ", f'<svg role="img" aria-label="{escape(title, quote=True)}" ', 1
    )
    return result


def metric_value(record, metric, region="all", reference=False):
    prefix = "reference_" if reference else "assisted_"
    return record["metrics"][region][prefix + metric]


def metric_bars(records, spec):
    metric, title, unit, decimals = spec
    keys = ["PA_E_100", "L1", "C0"]
    values = [metric_value(records["L1"], metric, reference=True)]
    values += [metric_value(records[key], metric) for key in keys]
    labels = ["무보조", "PA_E @100", "L1 ×0.5", "C0 @5000"]
    colors = [COLORS[key] for key in ("ref", "PA_E", "L1", "C0")]
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    ax.barh(range(4), values, color=colors, height=0.48, zorder=3)
    ax.set_yticks(range(4), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.28)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.grid(axis="x", color="#e9eef3", zorder=0)
    ax.set_xlabel(unit + "  ·  작을수록 좋음", fontsize=10, labelpad=12)
    for index, value in enumerate(values):
        ax.text(value + max(values) * 0.025, index, f"{value:,.{decimals}f}", va="center", fontsize=11)
    return svg(fig, title + ": " + ", ".join(f"{a} {b:.{decimals}f}" for a, b in zip(labels, values)))


def tradeoff(records):
    fig, ax = plt.subplots(figsize=(10.7, 4.7))
    base = records["L1"]
    bx = metric_value(base, "tau_sq_Nm2")
    by = metric_value(base, "tracking_rmse_deg")
    ax.set_xlim(1790, 2040)
    ax.set_ylim(2.73, 3.25)
    ax.fill_between([1790, bx], 2.73, by, color="#e3f2eb", zorder=0)
    ax.axvline(bx, color=COLORS["L1"], lw=1, ls=(0, (4, 4)))
    ax.axhline(by, color=COLORS["L1"], lw=1, ls=(0, (4, 4)))
    offsets = {
        "PA_E_100": (-125, -6), "PA_E_300": (14, 14),
        "PA_E_600": (14, 16), "PA_E_1000": (-7, 25),
        "L1": (12, 17), "C0": (-135, -20),
    }
    for key in ("PA_E_1000", "PA_E_600", "PA_E_300", "PA_E_100", "L1", "C0"):
        rec = records[key]
        x, y = metric_value(rec, "tau_sq_Nm2"), metric_value(rec, "tracking_rmse_deg")
        color = COLORS["PA_E" if key.startswith("PA_E") else key]
        selected = key in ("PA_E_100", "L1", "C0")
        label = key.replace("PA_E_", "PA_E @")
        label = {"L1": "L1 ×0.5", "C0": "C0 @5000"}.get(key, label)
        ax.scatter(x, y, s=100 if selected else 55, color=color, marker="o" if selected else "D", zorder=4)
        ax.annotate(label, (x, y), xytext=offsets[key], textcoords="offset points", fontsize=11,
                    arrowprops={"arrowstyle": "-", "color": color, "lw": 0.8})
    ax.text(2035, 2.742, "← 부담 감소  /  ↓ 추종 오차 감소", ha="right", fontsize=10, color="#63758a")
    ax.set_xlabel("사람 토크 제곱 평균 [N·m² / 관절]", labelpad=12)
    ax.set_ylabel("추종 RMSE [°]", labelpad=14)
    ax.grid(color="#e9eef3", zorder=0)
    return svg(fig, "부담과 추종 오차: PA_E 100 epoch만 두 지표가 함께 L1보다 낮다. 축은 차이를 보기 위해 확대했다.")


def epoch_curves(records):
    fig, axes = plt.subplots(3, 1, figsize=(10.8, 7.5), sharex=True)
    epochs = [100, 300, 600, 1000]
    for ax, (metric, title, unit, decimals) in zip(axes, METRICS):
        values = [metric_value(records[f"PA_E_{epoch}"], metric) for epoch in epochs]
        baseline = metric_value(records["L1"], metric)
        ax.axhline(baseline, color=COLORS["L1"], ls=(0, (5, 4)), lw=1.5)
        ax.plot(epochs, values, "o-", color=COLORS["PA_E"], lw=2, ms=6)
        ax.scatter([100], [values[0]], s=150, facecolor="none", edgecolor=COLORS["PA_E"], lw=1.5)
        span = max(max(values), baseline) - min(min(values), baseline)
        ax.set_ylim(min(min(values), baseline) - span * 0.6, max(max(values), baseline) + span * 0.65)
        ax.text(1055, baseline, f"L1 {baseline:,.{decimals}f}", va="center", fontsize=10, color=COLORS["L1"], bbox={"facecolor": "white", "edgecolor": "none", "pad": 3})
        for epoch, value in zip(epochs, values):
            ax.annotate(f"{value:,.{decimals}f}", (epoch, value), xytext=(0, -19 if epoch == 100 else 13), textcoords="offset points", ha="center", fontsize=10)
        ax.set_title(f"{title}  [{unit}]", loc="left", fontsize=12, pad=10)
        ax.grid(axis="y", color="#e9eef3")
    axes[-1].set_xticks(epochs)
    axes[-1].set_xlim(50, 1210)
    axes[-1].set_xlabel("L1에서 시작한 PA_E 추가학습 epoch", labelpad=12)
    fig.subplots_adjust(hspace=0.65)
    return svg(fig, "PA_E 학습량 변화: 부담과 일률은 감소하지만 300 epoch부터 추종 오차가 L1보다 커진다.")


def subgroup_burden(records):
    regions = ["moving", "near_stationary", "heavy_load", "outside_load"]
    labels = ["움직이는 구간", "거의 정지한 구간", "무거운 하중 구간", "하중 구간 밖"]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 5.8))
    for ax, region, label in zip(axes.flat, regions, labels):
        values = [metric_value(records[key], "tau_sq_Nm2", region) for key in ("PA_E_100", "L1", "C0")]
        ax.barh(range(3), values, color=[COLORS[k] for k in ("PA_E", "L1", "C0")], height=0.48)
        ax.set_yticks(range(3), ["PA_E @100", "L1", "C0 @5000"])
        ax.tick_params(axis="y", length=0)
        ax.invert_yaxis()
        ax.set_xlim(0, max(values) * 1.3)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#e9eef3")
        ax.set_title(label, loc="left", fontsize=12, pad=12)
        for index, value in enumerate(values):
            ax.text(value + max(values) * 0.03, index, f"{value:,.1f}", va="center", fontsize=10)
    fig.subplots_adjust(wspace=0.55, hspace=0.55)
    return svg(fig, "운동·하중 구간별 사람 토크 제곱 평균. 패널마다 축 범위가 다르며 구간은 서로 겹친다.")


CSS = """
.report-shell{font-variant-numeric:tabular-nums}.eyebrow{font-size:12px;font-weight:750;letter-spacing:.13em;color:#167b65}
.report-shell h1{font-size:38px!important;line-height:1.28!important;max-width:850px}.lede{max-width:880px;font-size:17px;color:#536579}
.pills{display:flex;gap:8px;flex-wrap:wrap;margin:22px 0}.pill{background:#e8edf4;border:1px solid #d8e1ec;border-radius:30px;padding:5px 12px;font-size:12px}
.hero-note{border-left:4px solid #167b65;background:#e9f3ef;padding:18px 22px;border-radius:0 8px 8px 0}.hero-note strong{font-size:20px;color:#145d4d;display:block;margin-bottom:6px}
.section-head{display:flex;align-items:baseline;gap:14px;margin-top:42px}.section-head .number{font-size:12px;font-weight:750;color:#167b65;letter-spacing:.08em}.section-head h2{margin:0}
.chart-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin-top:22px}.chart-card,.wide-figure{background:white;border:1px solid #dce3eb;border-radius:12px;padding:20px;margin:0}
.chart-card h3{font-size:16px;margin:0 0 6px}.chart-card p{font-size:12px;color:#63758a;margin:0 0 10px}.chart-card svg,.wide-figure svg{display:block;width:100%;height:auto}
.wide-figure{margin:20px 0}.wide-figure figcaption{font-size:13px;color:#607287;padding-top:12px;border-top:1px solid #edf0f4;line-height:1.8}
.delta-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:15px;margin:20px 0}.delta{border-top:2px solid #cbded6;padding:12px 6px}.delta .value{display:block;font-size:29px;line-height:1.5;color:#167b65;font-weight:700}.delta small{color:#64748b;font-size:12px}.delta b{font-size:13px;font-weight:500;display:block}
.legend{display:flex;gap:20px;flex-wrap:wrap;font-size:13px;margin:20px 0 0}.legend span:before{content:'';display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px;background:var(--color)}
.readout{padding:15px 20px;background:#edf1f6;border-radius:8px;font-size:14px}.note{font-size:13px;color:#63758a}.model-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:15px;margin-top:20px}.model{background:white;border:1px solid #dce3eb;border-top:3px solid var(--color);border-radius:8px;padding:20px}.model h3{margin:4px 0 10px}.model p{font-size:14px}.model .rank{color:#63758a;font-size:12px}.model code{font-size:11px;word-break:break-all;line-height:1.7}.sources{font-size:13px}.sources li{margin:10px 0}.sources code{font-size:11px;overflow-wrap:anywhere}.method-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.method-grid p{margin:0;padding:14px;background:#edf1f6;border-radius:8px;font-size:14px}.method-grid b{display:block;margin-bottom:6px}button{background:#23364b;color:white;border:0;border-radius:6px;padding:10px 16px;cursor:pointer;font:inherit;font-size:13px}button:hover{background:#376bc4}.report-shell details{margin-top:20px}.report-shell summary{cursor:pointer;font-weight:650}
@media(max-width:800px){.chart-grid,.model-grid{grid-template-columns:1fr}.chart-card{max-width:none}.chart-card svg{max-width:480px;margin:auto}.delta-grid{gap:8px}.delta .value{font-size:24px}.wide-figure{overflow-x:auto;padding:14px}.wide-figure svg{min-width:660px}.method-grid{grid-template-columns:1fr}.report-shell h1{font-size:28px!important}.section-head{align-items:flex-start}.lede{font-size:15px}}
@media print{.chart-grid{grid-template-columns:repeat(3,1fr)}.wide-figure,.chart-card,.model{break-inside:avoid}.wide-figure svg{min-width:0}button{display:none}.section-head{break-after:avoid}.model-grid{grid-template-columns:repeat(3,1fr)}}
"""


def main():
    records, sources = load_evidence()
    setup_plotting()
    spec = importlib.util.spec_from_file_location("report_shell", ROOT / "docs/ai_sbc/organize_reports.py")
    organizer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(organizer)
    now = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M KST")
    selected, base = records["PA_E_100"], records["L1"]
    tau_delta, error_delta, work_delta = [
        metric_value(base, key) - metric_value(selected, key) for key, _, _, _ in METRICS
    ]
    bars = "".join(
        f'<article class="chart-card"><h3>{title}</h3><p>같은 평가 조건 · 0에서 시작하는 축</p>{metric_bars(records, metric)}</article>'
        for metric in METRICS
        for _, title, _, _ in [metric]
    )
    model_cards = ""
    for index, (key, name, text, family) in enumerate([
        ("PA_E_100", "PA_E @100", "L1 actor에서 합성 예측으로 진자 추가학습. 이번 고정 평가에서는 부담·추종·일률이 모두 소폭 개선됐다. 추가 seed 재현과 W1 전이 검증은 이 비교에 포함하지 않았다.", "PA_E"),
        ("L1", "L1 ×0.5", "진자에서 학습한 기존 비교 기준. 출력 0.5배로 적용한다. PA_E보다 부담 수치는 높지만, 현재까지 축적된 W1 비교의 기준 모델이다.", "L1"),
        ("C0", "C0 @5000", "수정된 진자 토크 처리에서 처음부터 학습한 대조군. C0의 실제 H1 평가 중 부담이 가장 낮은 체크포인트지만, L1보다 부담·추종·일률 모두 높다.", "C0"),
    ], 1):
        checkpoint = records[key]["protocol"]["checkpoint"]
        try:
            checkpoint = str(Path(checkpoint).relative_to(ROOT))
        except ValueError:
            pass
        model_cards += f'<article class="model" style="--color:{COLORS[family]}"><span class="rank">비교 후보 {index:02d}</span><h3>{name}</h3><p>{text}</p><code>{escape(checkpoint)}</code></article>'
    source_items = "".join(
        f'<li><a href="../{s["path"]}">{s["label"]} · 원본 summary.json</a><br><code>SHA-256: {s["sha256"]}</code></li>'
        for s in sources
    )
    evidence = {"generated_kst": now, "scope": "Pinned PA_E/L1/C0 comparison; not an exhaustive current ranking", "sources": sources}
    body = f"""
<header><p class="eyebrow">AI-SBC / PENDULUM LLP / MODEL COMPARISON</p>
<h1>진자에서 배운 세 후보,<br>부담과 추종을 함께 비교하다.</h1>
<p class="lede">PA_E @100 · L1 ×0.5 · C0 @5000. 앞서 선정한 세 계열의 저장 모델을 같은 진자에 연결한 비교다. 실제 H1 예측을 사용하며, 모든 보조 출력 한계는 ±20 N·m다.</p>
<div class="pills"><span class="pill">{now} 생성</span><span class="pill">실제 H1 · 기존 외란</span><span class="pill">256 에피소드 / 모델</span><span class="pill">세 후보 모두 보조 실패 0회</span></div>
<div class="hero-note"><strong>PA_E @100은 L1보다 부담과 추종 오차를 함께 소폭 줄였다.</strong>다만 한 평가 묶음의 결과다. 여러 학습 seed에서의 우위나 전신 보행 전이를 확정한 결과는 아니다.</div>
</header>
<nav class="report-toc" aria-label="본문 탐색"><a href="#compare">세 후보 비교</a><a href="#tradeoff">후보 선택 이유</a><a href="#epochs">학습량 변화</a><a href="#regions">구간별 부담</a><a href="#evidence">조건과 근거</a></nav>
<section id="compare"><div class="section-head"><span class="number">01 / COMPARE</span><h2>세 지표가 같은 방향으로 좋아지는가</h2></div>
<p>무보조를 함께 표시했다. 각 그래프는 절대값이며, 낮을수록 부담·오차·일률이 작다.</p>
<div class="legend"><span style="--color:{COLORS['PA_E']}">PA_E @100</span><span style="--color:{COLORS['L1']}">L1 ×0.5</span><span style="--color:{COLORS['C0']}">C0 @5000</span><span style="--color:{COLORS['ref']}">무보조</span></div>
<div class="chart-grid">{bars}</div>
<div class="delta-grid"><div class="delta"><b>PA_E @100의 L1 대비 부담 감소</b><span class="value">{tau_delta:.1f}</span><small>N·m² / 관절</small></div><div class="delta"><b>PA_E @100의 L1 대비 오차 감소</b><span class="value">{error_delta:.3f}°</span><small>목표 각도 RMSE</small></div><div class="delta"><b>PA_E @100의 L1 대비 일률 감소</b><span class="value">{work_delta:.2f}</span><small>W / 관절</small></div></div>
<p class="note">τ²는 시뮬레이션 사람 제어기의 부담 지표다. 실제 근활성·대사량을 측정한 값이 아니다. 일률은 |τ·ω|로, 순수 에너지나 대사 비용과 다르다. 통계적 유의성·학습 seed 재현성은 이 그림에서 주장하지 않는다.</p>
</section>
<section id="tradeoff"><div class="section-head"><span class="number">02 / SELECTION</span><h2>가장 낮은 부담이 곧 최선의 후보는 아니다</h2></div>
<p>왼쪽 아래로 갈수록 부담과 추종 오차가 함께 작다. 파란 점선은 L1이고, 연녹색 영역은 두 지표가 모두 L1 이하인 영역이다.</p>
<figure class="wide-figure">{tradeoff(records)}<figcaption>차이를 보기 위해 두 축을 확대했다. PA_E @1000의 부담은 1818.2로 가장 낮지만, 추종 오차는 2.827°로 L1의 2.777°보다 크다. PA_E @100은 부담 1849.5·오차 2.762°·일률 33.03 W로 세 지표 모두 L1보다 낮다.</figcaption></figure>
<div class="readout"><b>선택 기준:</b> 실제 H1 평가에서 부담·추종·일률을 함께 확인하고 보조 실패가 없어야 한다. PA_E 계열에서는 이 조건에 맞는 @100을 대표로 골랐다. C0는 비교 대조군이며 L1을 넘은 통과 후보가 아니다.</div>
<div class="model-grid">{model_cards}</div></section>
<section id="epochs"><div class="section-head"><span class="number">03 / LEARNING</span><h2>PA_E를 더 학습하면 무엇이 달라지는가</h2></div>
<p>점은 각각 저장된 체크포인트의 평가다. 파란 점선은 같은 L1 기준값이며, 원으로 강조한 첫 점이 선택한 @100이다.</p>
<figure class="wide-figure">{epoch_curves(records)}<figcaption>300 epoch부터 추종 오차가 L1보다 커진다. 부담·일률만 보면 더 긴 학습이 유리해 보이지만, 현재 선택 기준은 추종 유지도 요구한다. 각 패널의 세로축은 변화를 보기 위해 확대했으며, 연결선은 평가 사이를 보간한 안내선이다.</figcaption></figure></section>
<section id="regions"><div class="section-head"><span class="number">04 / REGIONS</span><h2>전체 평균 뒤의 운동·하중 구간</h2></div>
<p>같은 실제 H1 평가에서 구간별 사람 토크 제곱 평균을 분리했다. 각 패널 안에서 세 모델을 비교한다.</p>
<figure class="wide-figure">{subgroup_burden(records)}<figcaption>단위: N·m² / 관절. 모든 막대축은 0에서 시작하지만 패널마다 최대값이 다르다. 운동 구분과 하중 구분은 서로 겹치므로 네 패널을 합산하면 안 된다. 구간별 부담만으로 해당 구간의 종합 개선을 판정하지 않는다.</figcaption></figure></section>
<section id="evidence"><div class="section-head"><span class="number">05 / EVIDENCE</span><h2>같게 맞춘 조건과 해석 범위</h2></div>
<div class="method-grid"><p><b>공통 평가</b>128환경 × 2코호트 = 256회. 1,000 step, 초기 0.5초 제외. 기존 외란(legacy), 실제 동결 H1, plant seed 34011 / trajectory seed 41011 / H1 seed 777.</p><p><b>토크와 실패</b>사람 PD+FF 합계 ±150 N·m, 보조 ±20 N·m. 세 후보 보조 실패 0/256, 무보조 실패 5/256. 토크 주입 오차와 주입 clipping은 0이다.</p><p><b>평균의 분모</b>무보조와 보조가 함께 생존한 구간의 조건부 평균이다. 공통 생존 비율 약 {base['common_survival_fraction'] * 100:.2f}%. 생존 이후만 잘라 얻은 수치를 전체 실패 없는 성능으로 해석하지 않는다.</p><p><b>훈련 조건은 다르다</b>L1은 기존 토크 처리에서 학습했고 PA_E는 그 actor에서 추가학습했다. C0는 수정 환경에서 처음부터 학습했다. 20/40 N·m 학습 한계 하나의 인과 효과를 분리한 실험이 아니다.</p></div>
<p class="note">범위: 앞서 요청한 PA_E·L1·C0의 고정 비교다. 전신 추가학습 L13/L14, 진행 중인 PA_H1, 다른 조건으로 평가한 L2–L11을 섞어 전체 모델의 확정 순위를 만들지 않았다. W1 속도·낙상·다른 관절 부담은 이 진자 평가에 포함되지 않는다. 새 평가가 생겨도 이 리포트의 후보·epoch는 자동 교체되지 않는다.</p>
<details class="report-fold"><summary>원본 수치와 재생성 방법</summary><p>아래 원본 JSON에서 그림을 생성했다. 파일 하나로 열 수 있도록 모든 그래프와 사용한 원본 수치를 이 HTML에 포함했다.</p><button id="download-evidence" type="button">사용한 원본 수치 JSON 받기</button><ul class="sources">{source_items}</ul><p>재생성:</p><pre>/home/bak/miniconda3/envs/sbc/bin/python examples/experiments/assist_pendulum/report_candidates.py
/home/bak/miniconda3/envs/sbc/bin/python docs/ai_sbc/organize_reports.py</pre></details>
</section><footer class="report-footer">진자 LLP 후보 비교 · {now} · <a href="pendulum_research_report.html">전체 진자 실험 리포트</a> · <a href="reports.html">보고서 목록</a></footer>
<script type="application/json" id="report-evidence">{json.dumps(evidence, ensure_ascii=False).replace('<', chr(92) + 'u003c')}</script>
<script>document.getElementById('download-evidence').addEventListener('click',()=>{{const blob=new Blob([document.getElementById('report-evidence').textContent],{{type:'application/json'}});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='pendulum_candidates_evidence.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}});</script>
"""
    html = organizer.shell("진자 LLP 후보 3종 · 그래프 비교", body)
    html = html.replace("</head>", f"<style>{CSS}</style></head>")
    OUTPUT.write_text(html)
    print(f"Wrote {OUTPUT.relative_to(ROOT)} ({len(html.encode()):,} bytes, 6 SVG charts)")


if __name__ == "__main__":
    main()
