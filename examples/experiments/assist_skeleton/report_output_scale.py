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
"""Analyze matched torque scales with treatment-independent deceleration masks."""

import argparse
import hashlib
from html import escape
import io
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MODELS = ("L1", "G0_s0", "G4_s0", "G4_s1")
LABELS = {"L1": "L1 · 기존 기준", "G0_s0": "G0 · 공동 정책 · 첫 학습",
          "G4_s0": "G4 · 비용 강화 · 첫 학습", "G4_s1": "G4 · 비용 강화 · 독립 재학습"}
COLORS = {"L1": "#374151", "G0_s0": "#179275", "G4_s0": "#2767ad", "G4_s1": "#b65e38"}


def smooth(values, width=5):
    """Centered offline moving average; analysis masks exclude boundary padding."""
    return np.stack([np.convolve(values[:, j], np.ones(width) / width, mode="same")
                     for j in range(values.shape[1])], axis=1)


def align(traces):
    """Associate the torque at i-1 with wearer response measured at i."""
    index = np.arange(2, len(traces["t"]))
    return dict(t=traces["t"][index], index=index,
                A=(traces["tau_pd_prev"] + traces["tau_ff_prev"])[index, 0],
                B=(traces["tau_pd_prev"] + traces["tau_ff_prev"])[index, 1],
                assist=traces["tau_assist"][index - 1, 1],
                velocity=.5 * (traces["hip_vel"][index - 1] + traces["hip_vel"][index]))


def masks_from_reference(traces):
    aligned = align(traces)
    t = aligned["t"]
    if not np.allclose(np.diff(t), .01, atol=2e-5):
        raise ValueError("Expected uninterrupted 10 ms samples")
    velocity = smooth(aligned["velocity"][:, 0])
    acceleration = np.gradient(velocity, .01, axis=0)
    torque_rms = np.sqrt(smooth(np.square(aligned["A"])))
    valid = (t >= 1.5) & (t <= t[-1] - .05)
    deceleration = (velocity * acceleration < 0) & (np.abs(velocity) > .3) & (np.abs(acceleration) > 1.)
    masks = {"deceleration": valid[:, None] & deceleration}
    for threshold in (5, 10):
        masks[f"low_decel_{threshold}"] = masks["deceleration"] & (torque_rms <= threshold)
    masks["high_decel_20"] = masks["deceleration"] & (torque_rms > 20.)
    return dict(t=t, masks=masks, torque_rms=torque_rms, velocity=velocity,
                acceleration=acceleration, A=aligned["A"])


def selected_metrics(aligned, mask):
    count = int(mask.sum())
    result = dict(joint_samples=count, joint_seconds=count * .01,
                  samples_right=int(mask[:, 0].sum()), samples_left=int(mask[:, 1].sum()))
    if not count:
        return result
    a, b, assist = (aligned[key][mask].astype(np.float64) for key in ("A", "B", "assist"))
    velocity = aligned["velocity"][:, 1][mask]
    sq_a, sq_b = float(np.mean(a*a)), float(np.mean(b*b))
    result.update(A_tau_sq=sq_a, B_tau_sq=sq_b, tau_sq_increase=sq_b-sq_a,
                  A_rms=float(np.sqrt(sq_a)), B_rms=float(np.sqrt(sq_b)),
                  rms_increase=float(np.sqrt(sq_b)-np.sqrt(sq_a)),
                  assist_rms=float(np.sqrt(np.mean(assist*assist))),
                  assist_mean=float(np.mean(assist)),
                  braking_sample_pct=float(100*np.mean(assist*velocity < 0)),
                  absorbed_power_per_joint_W=float(np.maximum(-assist*velocity, 0).mean()),
                  absolute_assist_power_per_joint_W=float(np.abs(assist*velocity).mean()),
                  absorbed_work_selected_J=float(np.maximum(-assist*velocity, 0).sum()*.01),
                  absolute_assist_work_selected_J=float(np.abs(assist*velocity).sum()*.01))
    return result


def analyze_traces(traces, reference):
    aligned = align(traces)
    n = len(aligned["t"])
    if n > len(reference["t"]) or not np.allclose(aligned["t"], reference["t"][:n], atol=1e-6):
        raise ValueError("Cannot apply reference masks to different control times")
    overall_mask = np.broadcast_to((aligned["t"] >= 1.5)[:, None], aligned["A"].shape)
    result = {"overall": selected_metrics(aligned, overall_mask)}
    for name, mask in reference["masks"].items():
        selected = mask[:n] & overall_mask
        result[name] = selected_metrics(aligned, selected)
        result[name]["by_side"] = {}
        for side, label in enumerate(("right", "left")):
            side_mask = selected.copy()
            side_mask[:, 1-side] = False
            result[name]["by_side"][label] = selected_metrics(aligned, side_mask)
        if selected.any():
            result[name]["overall_tau_sq_contribution"] = float(
                ((aligned["B"]**2-aligned["A"]**2)*selected).sum()/overall_mask.sum())
    result["reference_A_difference_rms_Nm"] = float(np.sqrt(
        np.mean((aligned["A"][overall_mask]-reference["A"][:n][overall_mask])**2)))
    local_masks = masks_from_reference(traces)["masks"]
    for name in ("low_decel_5", "low_decel_10"):
        fixed, local = reference["masks"][name][:n], local_masks[name]
        union = int((fixed | local).sum())
        result[name]["fixed_vs_local_mask_jaccard"] = float((fixed & local).sum()/union) if union else None
    return result


def read_traces(record):
    with np.load(ROOT / record["path"] / "traces.npz") as file:
        return dict(file)


def summarize(study):
    sys.path.insert(0, str(ROOT / "examples/experiments/assist_pendulum"))
    from report_general_phase import digest_run

    plan = json.loads((study / "manifest.json").read_text())
    records = json.loads((study / "runs.json").read_text())
    references = {}
    for seed in (42, 7):
        key = f"L1_20Nm_seed{seed}"
        if key in records:
            references[seed] = masks_from_reference(read_traces(records[key]))
    results = {}
    for identity, record in records.items():
        if record["seed"] not in references:
            continue
        result = digest_run(record)
        # The older helper's "work" is time-mean bilateral absolute power [W].
        for condition in ("A", "B"):
            summary = result["summary"][condition]
            summary["absolute_power_W"] = summary.pop("work")
        traces = read_traces(record)
        result["segments"] = analyze_traces(traces, references[record["seed"]])
        if identity == f'L1_20Nm_seed{record["seed"]}':
            idx = align(traces)["index"]
            before, after = traces["gait_phase"][idx-1], traces["gait_phase"][idx]
            phase = (before + .5*((after-before+.5) % 1-.5)) % 1
            for threshold in (5, 10):
                name = f"low_decel_{threshold}"
                mask = references[record["seed"]]["masks"][name]
                result["segments"][name]["reference_phase_counts_4pct"] = {
                    label: np.histogram(phase[:, side][mask[:, side]], bins=np.linspace(0, 1, 26))[0].tolist()
                    for side, label in enumerate(("right", "left"))}
        result["worst_phase"] = [max(rows, key=lambda r: -r.get("reduction", 0)) for rows in result["rows"]]
        results[identity] = result
    groups = {}
    for model in MODELS:
        for limit in (10, 15, 20):
            runs = [r for r in results.values() if r["record"]["model"] == model and r["record"]["limit_Nm"] == limit]
            if not runs:
                continue
            values = {}
            extractors = {
                "overall_tau_sq_reduction": lambda r: -r["segments"]["overall"]["tau_sq_increase"],
                "B_hip_rms": lambda r: r["summary"]["B"]["tau_rms"],
                "assist_rms": lambda r: r["summary"]["assist_rms"],
                "saturation_pct": lambda r: r["summary"]["saturation_pct"],
                "tracking_mae": lambda r: r["summary"]["B"]["tracking_mae"],
                "speed": lambda r: r["summary"]["B"]["speed"],
                "low10_rms_increase": lambda r: r["segments"]["low_decel_10"].get("rms_increase"),
                "low5_rms_increase": lambda r: r["segments"]["low_decel_5"].get("rms_increase"),
                "worst_phase_rms_increase": lambda r: max(-row["reduction"] for row in r["worst_phase"]),
            }
            for name, extract in extractors.items():
                data = [extract(r) for r in runs]
                data = [v for v in data if v is not None]
                if data:
                    values[name] = dict(mean=float(np.mean(data)), min=float(min(data)), max=float(max(data)))
            groups[f"{model}_{limit}"] = dict(model=model, limit_Nm=limit, seeds=[r["record"]["seed"] for r in runs],
                                             complete=len(runs) == 2, falls=sum(r["meta"]["survived_s"] is not None for r in runs), metrics=values)
    analysis_sources = [Path(__file__), ROOT / "examples/experiments/assist_pendulum/report_general_phase.py",
                        ROOT / "examples/experiments/assist_pendulum/report_general_docs.py"]
    failures = study / "runtime_failures.json"
    evidence = dict(plan=plan, completed=len(results), total=len(plan["jobs"]), groups=groups, runs=results,
                    runtime_failures=json.loads(failures.read_text()) if failures.exists() else [],
                    analysis_source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in analysis_sources},
                    note="Group values are arithmetic means and observed ranges of two evaluation runs, not confidence intervals. No new trained model.")
    (study / "comparison.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False))
    return evidence


def render(evidence, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    font = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    plt.rcParams.update({"font.family": font.get_name(), "font.size": 10,
                         "axes.unicode_minus": False, "svg.fonttype": "path",
                         "axes.spines.top": False, "axes.spines.right": False})
    destination.parent.mkdir(parents=True, exist_ok=True)
    figures = destination.parent / "figures"
    figures.mkdir(exist_ok=True)
    groups = evidence["groups"]
    runs = evidence["runs"]

    def plot_html(figure, name):
        plt.rcParams["svg.hashsalt"] = "output-scale-" + name
        buffer = io.StringIO()
        figure.savefig(buffer, format="svg", bbox_inches="tight")
        figure.savefig(figures / (name + ".png"), dpi=160, bbox_inches="tight")
        plt.close(figure)
        value = buffer.getvalue()
        return '<div class="plot">' + value[value.index("<svg"):] + "</div>"

    def fmt(value, signed=False):
        return "—" if value is None else format(value, "+.2f" if signed else ".2f")

    def cell(group, metric, signed=False):
        values = group["metrics"].get(metric)
        if values is None:
            return "—"
        return (f'<b>{fmt(values["mean"], signed)}</b><br><small>'
                f'{fmt(values["min"], signed)} ~ {fmt(values["max"], signed)}</small>')

    css = """body{margin:0;background:#f3f6f8;color:#1c2d3d;font:16px/1.8 system-ui,sans-serif}
    main{max-width:1200px;margin:auto;padding:28px 26px 70px}h1{font-size:30px;line-height:1.4}
    h2{font-size:23px;margin-top:36px}h3{font-size:18px}a{color:#245aa0}table{border-collapse:collapse;width:100%;background:white;font-size:14px}
    td,th{border:1px solid #d8e1ea;padding:10px 12px;text-align:left;vertical-align:top}th{background:#e5edf5}
    small,.note{font-size:13px;color:#607080}.scroll{overflow-x:auto}.plot svg{width:100%;height:auto}
    .lead{background:#e5eef9;border-left:5px solid #2767ad;padding:18px 22px;border-radius:6px}
    details{background:white;border:1px solid #d8e1ea;padding:14px 20px;border-radius:7px;margin:18px 0}
    summary{cursor:pointer;font-weight:650}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}
    nav{display:flex;gap:18px;flex-wrap:wrap}select{font:inherit;max-width:100%;padding:6px 10px}
    @media(max-width:750px){main{padding:20px 14px}h1{font-size:25px}td,th{min-width:90px}}"""
    body = ['<nav><a href="../reports.html">보고서 목록</a><a href="../experiment_summary.html">전체 요약</a>'
            '<a href="../261015_1815/walking_phase_report.html">이전 23 N·m 비교</a></nav>',
            '<h1>출력을 맞추면 방해가 줄어드는가</h1>',
            '<p>진자에서 학습한 L1·G0·G4를 같은 10·15·20 N·m 출력 범위로 맞춘 동결 W1/H1 걷기 평가다. '
            '새 학습 없이 출력 크기만 바꾸고, 무보조에서 사람 토크가 작았던 감속 구간을 따로 비교한다.</p>']
    complete = evidence["completed"] == evidence["total"]
    falls = sum(run["meta"]["survived_s"] is not None for run in runs.values())
    body.append(f'<div class="lead"><b>{evidence["completed"]}/{evidence["total"]}건 완료 · '
                f'{"전체 평가 완료" if complete else "진행 중인 중간 결과"}</b><p>낙상·조기 종료 {falls}건. '
                '각 수치는 같은 실행의 무보조 A와 보조 B를 비교한 값이다. 두 평가 난수의 평균과 실제 범위를 표시한다.</p></div>')
    if evidence["runtime_failures"]:
        body.append(f'<p class="note">운영 오류 {len(evidence["runtime_failures"])}회: '
                    'L1·15 N·m·평가 난수 7의 첫 실행이 SIGSEGV로 중단되어 동일 명령으로 처음부터 재시도했다. '
                    '원인은 미해결이며 실패 로그를 보존했다. 표와 낙상 수는 정상적으로 기록을 저장한 실행을 대상으로 한다. '
                    '원본의 runtime_failures에 별도 기록했다.</p>')
    body.append('<details><summary>모델·단위·평가 난수 설명</summary><ul>'
                '<li>L1: 기존 진자 학습 모델. 원래 40 N·m 출력에 0.25/0.375/0.5배를 적용했다.</li>'
                '<li>G0: 양측 공동 정책의 첫 학습. G4: 보조 절약·명령 변화 비용을 강화한 관절별 공유 정책.</li>'
                '<li>G4 첫 학습/독립 재학습은 학습 seed 0/1로 각각 처음부터 만든 별개 모델이다. '
                'G0/G4 원래 23 N·m 출력에 10/23, 15/23, 20/23배를 적용했다.</li>'
                '<li>평가 seed 42/7은 모든 모델·출력 크기에 공통 적용한 두 평가 실행이다. 두 사람이나 두 보행 클립이 아니다.</li>'
                '<li>τ²: 토크 제곱 평균 [(N·m)²]. RMS: 그 제곱근 [N·m]. 실제 근활성이나 대사량 실측이 아니다.</li>'
                '<li>평균과 범위는 두 평가 실행의 값이며 신뢰구간이 아니다. 보행 위상은 각 다리의 참조 최대 굴곡이 0%다.</li>'
                '</ul></details>')

    if complete:
        eligible = [group for group in groups.values() if group["complete"] and not group["falls"]]
        best = max(eligible, key=lambda g: g["metrics"]["overall_tau_sq_reduction"]["mean"]) if eligible else None
        if best:
            body.append('<h2>이번 비교의 판단</h2>')
            body.append(f'<p>시험한 출력 범위에서 전체 고관절 토크² 감소가 가장 컸던 조건은 '
                        f'<b>{LABELS[best["model"]]} · {best["limit_Nm"]} N·m</b>이며 평균 감소량은 '
                        f'{fmt(best["metrics"]["overall_tau_sq_reduction"]["mean"])} (N·m)²다. '
                        '이는 이 걷기에서의 진단이며 새 일반 모델 선정이나 배포 성공을 뜻하지 않는다.</p>')
            body.append('<ul>')
            for model in MODELS:
                lower, higher = groups[f"{model}_10"], groups[f"{model}_20"]
                lo, hi = lower["metrics"], higher["metrics"]
                peak = max((groups[f"{model}_{limit}"] for limit in (10, 15, 20)),
                           key=lambda g: g["metrics"]["overall_tau_sq_reduction"]["mean"])
                body.append(f'<li>{LABELS[model]}: 20 → 10 N·m로 낮출 때 전체 토크² 감소량 '
                            f'{fmt(hi["overall_tau_sq_reduction"]["mean"])} → '
                            f'{fmt(lo["overall_tau_sq_reduction"]["mean"])} (N·m)², '
                            f'저부담 감속 RMS 증가 {fmt(hi["low10_rms_increase"]["mean"])} → '
                            f'{fmt(lo["low10_rms_increase"]["mean"])} N·m. '
                            f'이 모델에서 평균 전체 이득이 가장 컸던 시험 출력은 ±{peak["limit_Nm"]} N·m다.</li>')
            body.append('</ul><p class="note">근접한 평균값의 순위는 두 실행의 범위와 함께 읽어야 한다. '
                        '시험한 세 출력 중 평균 최대인 조건을 찾은 것이며 최적 출력을 확정한 것은 아니다.</p>')
            l1 = groups["L1_20"]["metrics"]
            g4 = groups["G4_s0_20"]["metrics"]
            body.append(f'<p>같은 ±20 N·m에서 L1과 G4 첫 학습본의 실제 보조 RMS는 '
                        f'{fmt(l1["assist_rms"]["mean"])} / {fmt(g4["assist_rms"]["mean"])} N·m, '
                        f'상한 95% 초과 빈도는 {fmt(l1["saturation_pct"]["mean"])} / '
                        f'{fmt(g4["saturation_pct"]["mean"])}%였다. 저부담 감속 RMS 증가는 '
                        f'{fmt(l1["low10_rms_increase"]["mean"])} / '
                        f'{fmt(g4["low10_rms_increase"]["mean"])} N·m였다. '
                        '출력 상한만 맞춰도 정책별 개입 방식의 차이는 남는다. '
                        '이 비교만으로 보상 계수·학습 환경·관측 구조 중 하나를 원인으로 확정할 수는 없다.</p>')
            body.append('<p><b>후속 학습 전에 확인할 문제:</b> 진자에서 비슷한 각도·속도·감속을 보이더라도 '
                        '외부 작용이 감속을 담당하는 경우와 사람이 감속 토크를 내야 하는 경우를 현재 관측과 이력으로 '
                        '구분할 수 있는지 확인한다. 저부담 조건에서 보조를 줄이는 능력과 고부담 조건의 유효한 보조를 '
                        '별도로 평가해야 한다. 이번 실험에서는 새 보상·관측·모델 학습을 수행하지 않았다.</p>')

    body.append('<h2>1. 전체 이득과 저부담 감속 손해</h2><p>왼쪽은 높을수록 고관절 전체 부담이 감소한다. '
                '오른쪽은 낮을수록 저부담 감속 구간에서 추가 부담이 작다. '
                '오차 막대는 두 평가 난수의 최소–최대이며, 출력 크기는 파형 전체에 곱한 값이다.</p>')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    for ax, key, ylabel in zip(axes, ("overall_tau_sq_reduction", "low10_rms_increase"),
                              ("전체 착용자 토크² 감소 A − B [(N·m)²]", "저부담 감속 착용자 RMS 증가 B − A [N·m]")):
        for model in MODELS:
            available = [groups[f"{model}_{limit}"] for limit in (10, 15, 20) if f"{model}_{limit}" in groups]
            if not available:
                continue
            values = [g["metrics"][key] for g in available]
            means = np.array([v["mean"] for v in values])
            errors = np.array([[v["mean"]-v["min"] for v in values], [v["max"]-v["mean"] for v in values]])
            ax.errorbar([g["limit_Nm"] for g in available], means, yerr=errors, marker="o", capsize=4,
                        color=COLORS[model], label=LABELS[model], ls="--" if model.endswith("s1") else "-")
        ax.axhline(0, color="#888", lw=.8)
        ax.set_xticks([10, 15, 20]); ax.set_xlabel("실제 보조 출력 범위 ±N·m")
        ax.set_ylabel(ylabel); ax.grid(alpha=.15); ax.legend(fontsize=8)
    body.append(plot_html(fig, "scale_tradeoff"))

    body.append('<div class="scroll"><table><tr><th>모델</th><th>출력 ±N·m</th><th>전체 토크² 감소<br>(N·m)² ↑</th>'
                '<th>보조 시 hip RMS<br>N·m ↓</th><th>저부담 감속 RMS 증가<br>N·m ↓</th>'
                '<th>최악 4% 위상 RMS 증가<br>N·m ↓</th><th>실제 보조 RMS<br>N·m</th><th>추종 MAE<br>° ↓</th></tr>')
    for model in MODELS:
        for limit in (10, 15, 20):
            g = groups.get(f"{model}_{limit}")
            if g:
                body.append(f'<tr><td>{LABELS[model]}</td><td>{limit}</td>' + ''.join(
                    '<td>'+cell(g, metric)+'</td>' for metric in (
                        "overall_tau_sq_reduction", "B_hip_rms", "low10_rms_increase",
                        "worst_phase_rms_increase", "assist_rms", "tracking_mae")) + '</tr>')
    body.append('</table></div><p class="note">굵은 값은 평가 실행 평균, 아래는 실행 범위다. '
                '같은 출력 상한이어도 실제 RMS와 파형은 다르다. 최악 위상 값은 각 실행의 좌우 50개 구간 중 '
                '최대 증가를 구한 뒤 두 실행을 평균한 값이다.</p>')

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for model in MODELS:
        available = [groups[f"{model}_{limit}"] for limit in (10, 15, 20) if f"{model}_{limit}" in groups]
        if not available:
            continue
        xx = [g["metrics"]["low10_rms_increase"]["mean"] for g in available]
        yy = [g["metrics"]["overall_tau_sq_reduction"]["mean"] for g in available]
        ax.plot(xx, yy, marker="o", color=COLORS[model], label=LABELS[model],
                ls="--" if model.endswith("s1") else "-")
        for x_value, y_value, g in zip(xx, yy, available):
            ax.annotate(str(g["limit_Nm"]), (x_value, y_value), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("저부담 감속 구간 RMS 증가 [N·m] · 왼쪽이 유리")
    ax.set_ylabel("전체 토크² 감소 [(N·m)²] · 위쪽이 유리")
    ax.grid(alpha=.15); ax.legend(fontsize=8)
    body.append('<p>아래 그림은 전체 이득과 국소 손해를 함께 보여 준다. 왼쪽 위가 유리하며, '
                '점 옆 숫자는 출력 범위다. 선은 측정한 세 조건을 연결한 것으로 중간 출력의 성능을 검증한 것은 아니다.</p>')
    body.append(plot_html(fig, "benefit_harm_tradeoff"))

    body.append('<h2>2. 힘이 거의 필요 없던 감속 구간에서도 개입하는가</h2>'
                '<p>각 평가 난수의 L1·20 N·m 실행에서 무보조 A의 기록만 사용해 구간을 정했다. '
                '그 시간 마스크를 모든 모델·출력 크기에 공통 적용한다. '
                '50 ms 토크 RMS ≤10 N·m, |각속도| &gt;0.3 rad/s, |각가속도| &gt;1 rad/s², '
                '각속도×각가속도 &lt;0인 구간이다. 5 N·m 이하는 엄격한 민감도 분석이다.</p>'
                '<p>평균 토크가 0에 가까워도 순간 토크가 클 수 있으므로 부호 있는 평균 대신 RMS를 사용한다. '
                '신전 방향 보조나 음의 일률 자체를 실패로 판정하지 않고, 실제 착용자 부담 증가를 판정한다.</p>')
    body.append('<div class="scroll"><table><tr><th>평가 난수</th><th>≤10 N·m 표본 / 관절·초</th>'
                '<th>≤5 N·m 표본 / 관절·초</th><th>기준 A RMS: ≤10 / ≤5 N·m</th></tr>')
    for seed in (42, 7):
        key = f"L1_20Nm_seed{seed}"
        if key in runs:
            segments = runs[key]["segments"]
            low, strict = segments["low_decel_10"], segments["low_decel_5"]
            body.append(f'<tr><td>{seed}</td><td>{low["joint_samples"]} / {low["joint_seconds"]:.2f}</td>'
                        f'<td>{strict["joint_samples"]} / {strict["joint_seconds"]:.2f}</td>'
                        f'<td>{low["A_rms"]:.2f} / {strict["A_rms"]:.2f} N·m</td></tr>')
    body.append('</table></div><p class="note">표본은 10 ms 창×관절 수다. 관절·초는 표본×0.01초이며 '
                '양쪽 다리를 합한 노출량이다. 독립 표본 수로 해석하지 않는다. 엄격한 ≤5 N·m 구간은 짧다.</p>')
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.2), layout="constrained", sharey=True)
    for side, label in enumerate(("right", "left")):
        for seed, color in ((42, "#374151"), (7, "#179275")):
            key = f"L1_20Nm_seed{seed}"
            if key in runs:
                counts = runs[key]["segments"]["low_decel_10"]["reference_phase_counts_4pct"][label]
                axes[side].plot((np.arange(25)+.5)*4, counts, label=f"평가 난수 {seed}", color=color)
        axes[side].set_title(("오른쪽", "왼쪽")[side])
        axes[side].set_xlabel("참조 최대 굴곡 = 0%, 위상 [%]")
        axes[side].set_ylabel("저부담 감속에 해당한 10 ms 창 수")
        axes[side].set_xlim(0, 100); axes[side].grid(alpha=.15); axes[side].legend(fontsize=8)
    body.append('<p>저부담 감속의 위상 분포는 좌우가 다르며 유각 후반에만 한정되지 않는다. '
                '아래는 기준 무보조 A에서 선택한 창의 분포다. 좌우를 합친 RMS는 선택된 창 수로 가중된다. '
                '각 다리의 RMS 변화도 실행별 원본의 by_side에 함께 저장한다.</p>')
    body.append(plot_html(fig, "reference_mask_distribution"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, threshold in zip(axes, (10, 5)):
        for offset, model in enumerate(MODELS):
            available = [groups[f"{model}_{limit}"] for limit in (10, 15, 20) if f"{model}_{limit}" in groups]
            if not available:
                continue
            x = np.array([g["limit_Nm"] for g in available], dtype=float)
            y = [g["metrics"][f"low{threshold}_rms_increase"]["mean"] for g in available]
            ax.bar(x+(offset-1.5)*.8, y, width=.75, color=COLORS[model], label=LABELS[model])
        ax.axhline(0, color="#777", lw=.8); ax.set_xticks([10, 15, 20])
        ax.set_title(f"무보조 50 ms 토크 RMS ≤{threshold} N·m")
        ax.set_xlabel("실제 출력 범위 ±N·m"); ax.set_ylabel("착용자 RMS 증가 [N·m]")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=.15)
    body.append(plot_html(fig, "low_burden_deceleration"))

    body.append('<h2>3. 좌우 보행 위상에서 어디가 바뀌는가</h2>'
                '<p>위상별 착용자 RMS 감소량 A−B다. 0 아래는 부담 증가다. '
                '같은 출력 범위에서 네 모델을 비교하며 선은 완료된 평가 실행 평균이다.</p>')
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained", sharex=True, sharey=True)
    x = (np.arange(25)+.5)*4
    for column, limit in enumerate((10, 15, 20)):
        for side in (0, 1):
            ax = axes[side, column]
            for model in MODELS:
                selected = [r for r in runs.values() if r["record"]["model"] == model and r["record"]["limit_Nm"] == limit]
                if selected:
                    y = np.mean([[row["reduction"] for row in r["rows"][side]] for r in selected], axis=0)
                    ax.plot(x, y, color=COLORS[model], label=LABELS[model], ls="--" if model.endswith("s1") else "-")
            ax.axhline(0, color="#777", lw=.8); ax.axhspan(-25, 0, color="#bb4444", alpha=.05)
            ax.set_title(f'{("오른쪽", "왼쪽")[side]} · ±{limit} N·m')
            ax.set_xlim(0, 100); ax.set_ylim(-25, 35); ax.grid(alpha=.15)
            if side == 1:
                ax.set_xlabel("참조 최대 굴곡 = 0%, 위상 [%]")
            if column == 0:
                ax.set_ylabel("착용자 RMS 감소 [N·m]")
    for ax in axes.flat:
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
            break
    body.append(plot_html(fig, "phase_comparison"))

    body.append('<h2>4. 보행 유지와 다른 관절</h2><p>몸통·무릎·발목 값은 해당 회전관절당 토크² 평균 변화 B−A다. '
                '양수면 증가다. 직선 관절의 힘은 포함하지 않는다.</p>'
                '<div class="scroll"><table><tr><th>모델·출력</th><th>속도 A→B<br>m/s</th><th>몸체 오차 A→B<br>mm</th>'
                '<th>무릎 변화<br>(N·m)²</th><th>발목 변화<br>(N·m)²</th><th>몸통 변화<br>(N·m)²</th><th>낙상</th></tr>')
    for model in MODELS:
        for limit in (10, 15, 20):
            selected = [r for r in runs.values() if r["record"]["model"] == model and r["record"]["limit_Nm"] == limit]
            if not selected:
                continue
            def mean_metric(condition, name):
                return float(np.mean([r["summary"][condition][name] for r in selected]))
            shifts = [np.mean([next(g["B"]-g["A"] for g in r["other_joints"] if g["group"] == group) for r in selected])
                      for group in ("무릎", "발목", "몸통")]
            body.append(f'<tr><td>{LABELS[model]} · {limit}</td><td>{mean_metric("A", "speed"):.4f} → '
                        f'{mean_metric("B", "speed"):.4f}</td><td>{mean_metric("A", "body_err")*1000:.2f} → '
                        f'{mean_metric("B", "body_err")*1000:.2f}</td>' +
                        ''.join(f'<td>{v:+.2f}</td>' for v in shifts) +
                        f'<td>{sum(r["meta"]["survived_s"] is not None for r in selected)} / {len(selected)}</td></tr>')
    body.append('</table></div>')

    body.append('<details><summary>실행별 수치·경로와 분석 범위</summary>'
                '<p>40초 기존 걷기, 첫 1.5초 제외. 사람 토크는 직전 10 ms 창 평균이므로 '
                '같은 창에 인가한 보조를 한 칸 앞에서 맞췄다. 저부담 마스크는 마지막 50 ms도 제외한다. '
                '시뮬레이터의 수치 산포 때문에 각 실행의 A는 기준 A와 완전히 같지 않을 수 있다.</p>'
                '<p>고정 구간과 각 실행 A로 다시 고른 구간의 겹침 비율(Jaccard), A 토크 차이, '
                '구간별 표본 수와 음의 보조 일률은 아래 원본에 포함한다. 선택 구간의 값은 전체 부담 기여와 다르다. '
                '일률·일은 10 ms 평균 각속도와 같은 창의 보조 토크로 근사하며, 일[J]은 선택된 양쪽 관절 창을 합산한다. '
                'absolute_power_W는 착용자 양측 |토크×각속도| 합의 시간 평균이다.</p>')
    for identity, run in runs.items():
        body.append(f'<details><summary>{escape(identity)}</summary><p><code>{escape(run["record"]["path"])}</code></p>'
                    '<pre>' + escape(json.dumps(dict(summary=run["summary"], segments=run["segments"],
                                                     worst_phase=run["worst_phase"], record=run["record"]),
                                               ensure_ascii=False, indent=2)) + '</pre></details>')
    body.append('</details><details><summary>고정 실험 계획·해시</summary><pre>' +
                escape(json.dumps(dict(plan=evidence["plan"], analysis_source_sha256=evidence["analysis_source_sha256"]),
                                  ensure_ascii=False, indent=2)) + '</pre></details>')
    body.append('<details><summary>원본과 재현 방법</summary>'
                '<p><a href="evidence.json">이 보고서의 전체 분석 수치</a> · '
                '<a href="../../results/llp_output_scale_20261016/runs.json">24개 실행 명령과 원본 경로</a> · '
                '<a href="../../results/llp_output_scale_20261016/runtime_environment.json">실행 환경</a></p>'
                '<p>저장소 루트에서 아래 명령으로 실행한다. 같은 study 경로는 이미 완료된 조건을 건너뛰며 '
                '평가 소스·계획 해시가 달라지면 중단한다. 다시 계산하려면 별도 --study 경로를 지정한다. '
                '새 학습은 실행하지 않는다.</p><pre>'
                '/home/bak/miniconda3/envs/sbc/bin/python examples/experiments/assist_skeleton/evaluate_output_scale.py\n'
                '/home/bak/miniconda3/envs/sbc/bin/python examples/experiments/assist_skeleton/report_output_scale.py\n'
                '/home/bak/miniconda3/envs/sbc/bin/python docs/ai_sbc/organize_reports.py'
                '</pre></details>')
    body.append('<p class="note">동결 W1과 기존 걷기 한 클립의 결과다. 같은 출력 상한에서의 비교는 '
                '서로 다른 학습 환경·학습량의 인과 효과를 분리하지 않는다. 실제 사람·다른 보행·실기기 성능은 미검증이다.</p>')
    embedded = json.dumps(evidence, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    document = '<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    document += '<title>공통 출력 크기와 저부담 감속 · AI-SBC</title><style>'+css+'</style></head><body><main>'
    document += ''.join(body) + '<script type="application/json" id="evidence">'+embedded+'</script></main></body></html>'
    destination.write_text(document)
    (destination.parent / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="results/llp_output_scale_20261016")
    parser.add_argument("--out", default="docs/ai_sbc/261016_output_scale/report.html")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    evidence = summarize(ROOT / args.study)
    if not args.analyze_only:
        render(evidence, ROOT / args.out)
    print(json.dumps(dict(completed=evidence["completed"], total=evidence["total"])))


if __name__ == "__main__":
    main()
