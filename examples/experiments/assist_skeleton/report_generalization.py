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
"""Graph-first report: does R0's existing-walk improvement over L1 hold on new conditions?

Reads stored W1 records only. Criteria were fixed in analysis_plan.json before any R0
run on a new condition. The existing walk row recomputes stored 2026-10-16 L1 and
2026-10-20 R0 records with the same code.
"""
from datetime import datetime, timezone
from html import escape
import io
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'examples/experiments/assist_skeleton'))
sys.path.insert(0, str(ROOT / 'examples/experiments/assist_pendulum'))
from analyze_need_bins import walking as need_bins  # noqa: E402
from evaluate_generalization import CONDITIONS, screen  # noqa: E402
from report_general_phase import digest_run  # noqa: E402
from report_output_scale import align, analyze_traces, masks_from_reference, read_traces  # noqa: E402

STUDY = ROOT / 'results/llp_r0_generalization_20261021'
OLD = ROOT / 'results/llp_output_scale_20261016'
RELATIVE = ROOT / 'results/llp_relative_20261020/walking'
OUT = ROOT / 'docs/ai_sbc/261021_r0_generalization/report.html'
PREVIEW = Path(os.environ['GEN_REPORT_PNG']) if os.environ.get('GEN_REPORT_PNG') else None
SEEDS = (42, 7)
LIMITS = {'L1': (10, 15, 20), 'R0': (15, 20)}
SHOWN = [('L1', 20), ('L1', 10), ('R0', 20)]
SHORT = {'walk': '기존 걷기', 'walk_seg400': '같은 걷기 다른 구간', 'run': '달리기',
         'stepinplace': '제자리 걷기', 'highknee': '무릎 높이 들기'}
LONG = {'walk': '기존 걷기 0–40 s (1.28 m/s, 저장 기록 재사용)', **{k: v['label'] for k, v in CONDITIONS.items()}}
COLORS = {('L1', 20): '#2f3640', ('L1', 15): '#7b8794', ('L1', 10): '#a7b1bc', ('R0', 20): '#245aa0', ('R0', 15): '#6f9bd1'}
NAMES = {('L1', 20): 'L1 ±20', ('L1', 15): 'L1 ±15', ('L1', 10): 'L1 ±10', ('R0', 20): 'R0 ±20', ('R0', 15): 'R0 ±15'}
NEED = ['0–5', '5–10', '10–20', '20–40', '40–80', '80+']
PHASE_BINS = 25
# Unseen random pendulum trajectories from the training generator, real H1, ±20 N·m (2026-10-20).
PENDULUM = ROOT / 'results/llp_relative_20261020/pendulum/R0_s0_H1_legacy/rollout.npz'
OMEGA = [0, 1, 2, 3, 4, 5, 6, 8]
FONT = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family': FONT.get_name(), 'font.size': 10, 'axes.unicode_minus': False, 'svg.fonttype': 'path',
                     'axes.spines.top': False, 'axes.spines.right': False})
CSS = ("html{scroll-behavior:smooth}body{margin:0;background:#f4f6f8;color:#1c2d3d;font:16px/1.8 system-ui,sans-serif}"
       "main{max-width:1180px;margin:auto;padding:28px 26px 70px}h1{font-size:30px;line-height:1.4}h2{font-size:23px;margin:38px 0 12px}"
       "a{color:#245aa0}table{width:100%;border-collapse:collapse;background:white;font-size:14px;line-height:1.6}th{background:#e5edf5;text-align:left}"
       "td,th{padding:8px 11px;border:1px solid #dce3eb;vertical-align:top}strong{color:#185b4a}code{font-size:.88em;overflow-wrap:anywhere}"
       "nav{display:flex;gap:18px;flex-wrap:wrap;border-bottom:1px solid #d8e1ea;padding-bottom:15px}"
       "details{background:white;border:1px solid #d8e1ea;border-radius:8px;padding:15px 20px;margin:20px 0}summary{cursor:pointer;font-weight:650}"
       ".note{color:#62748a;font-size:14px}.plot{background:white;border:1px solid #dce3eb;border-radius:10px;padding:10px;margin:12px 0}"
       ".plot svg{width:100%;height:auto;display:block}.status{padding:16px 18px;background:#e5eef9;border-radius:8px;border-left:5px solid #245aa0}"
       ".ok{color:#185b4a;font-weight:650}.part{color:#9a6700;font-weight:650}.no{color:#b42318;font-weight:650}small{color:#62748a}"
       ".scroll{overflow-x:auto}@media(max-width:750px){main{padding:20px 16px}h1{font-size:25px}table{display:block;overflow-x:auto}}")


def read(path):
    return json.loads(Path(path).read_text())


def svg(fig, identity):
    if PREVIEW:
        PREVIEW.mkdir(parents=True, exist_ok=True)
        fig.savefig(PREVIEW / f'{identity}.png', dpi=110, bbox_inches='tight')
    plt.rcParams['svg.hashsalt'] = identity
    buf = io.StringIO()
    fig.savefig(buf, format='svg', bbox_inches='tight')
    plt.close(fig)
    text = buf.getvalue()
    return '<div class="plot">' + text[text.index('<svg'):] + '</div>'


def gather():
    """(condition, model, limit, seed) -> record with a repository-relative path."""
    out = {}
    old = read(OLD / 'runs.json')
    relative = read(RELATIVE / 'runs.json')
    for seed in SEEDS:
        for limit in LIMITS['L1']:
            out['walk', 'L1', limit, seed] = old[f'L1_{limit}Nm_seed{seed}']
        for limit in LIMITS['R0']:
            out['walk', 'R0', limit, seed] = relative[f'R0_s0_{limit}Nm_seed{seed}']
    if (STUDY / 'runs.json').exists():
        for r in read(STUDY / 'runs.json').values():
            out[r['condition'], r['model'], r['limit_Nm'], r['seed']] = r
    return out


def digest(record, reference):
    run = digest_run(record)
    traces = read_traces(record)
    seg = analyze_traces(traces, reference)
    a = align(traces)
    keep = a['t'] >= 1.5
    side_benefit = (np.square(a['A'][keep]) - np.square(a['B'][keep])).mean(axis=0)
    worst = []
    for side in (0, 1):
        rows = [r for r in run['rows'][side] if r.get('n')]
        w = max(rows, key=lambda r: -r['reduction'])
        worst.append(dict(value=-w['reduction'], lo=w['lo'], hi=w['hi']))
    phase = traces['gait_phase']
    period = (traces['t'][-1] - traces['t'][0]) / max(int(np.sum(np.diff(phase[:, 0]) < -.5)), 1)
    s = run['summary']
    other = {o['group']: o['A'] - o['B'] for o in run['other_joints']}
    return dict(
        benefit=-seg['overall']['tau_sq_increase'], A_tau_sq=seg['overall']['A_tau_sq'],
        assist_rms=s['assist_rms'], saturation=s['saturation_pct'],
        low10=seg['low_decel_10'].get('rms_increase'), low10_s=seg['low_decel_10']['joint_seconds'],
        low5=seg['low_decel_5'].get('rms_increase'), low5_s=seg['low_decel_5']['joint_seconds'],
        low10_side=[seg['low_decel_10']['by_side'][k].get('rms_increase') for k in ('right', 'left')],
        benefit_side=side_benefit.tolist(), worst_side=worst, worst=max(w['value'] for w in worst),
        mask_A_diff=seg['reference_A_difference_rms_Nm'],
        body_err_mm=[1000 * s['A']['body_err'], 1000 * s['B']['body_err']], speed=[s['A']['speed'], s['B']['speed']],
        tracking=[s['A']['tracking_mae'], s['B']['tracking_mae']], fall=run['meta']['survived_s'],
        other=other, rows=run['rows'], period_s=float(period),
        need=need_bins(ROOT / record['path'] / 'traces.npz', s['limit'])['bins'], path=record['path'],
        omega=omega_bins(np.abs(a['velocity'][keep, 0]), np.abs(a['velocity'][keep, 1]), a['assist'][keep]))


def omega_bins(speed_a, speed_b, assist):
    """Share of unassisted time and applied assist RMS per hip angular speed bin [rad/s]."""
    rows = []
    for lo, hi in zip(OMEGA[:-1], OMEGA[1:]):
        share = (speed_a >= lo) & (speed_a < hi)
        sel = (speed_b >= lo) & (speed_b < hi)
        rows.append(dict(lo=lo, hi=hi, share=float(share.mean()),
                         assist_rms=float(np.sqrt(np.mean(assist[sel] ** 2))) if sel.sum() > 50 else None))
    return rows


def pendulum_omega():
    z = np.load(PENDULUM)
    keep = np.repeat((z['valid'] & (z['time_s'] > .5))[:, :, None], 2, 2)
    speed_a = np.abs(z['reference_velocity'][keep])
    return omega_bins(speed_a, np.abs(z['assisted_velocity'][keep]), z['assist_tau'][keep])


def curve(xs, ys, x):
    """Piecewise-linear L1 curve through the origin; None outside the measured range."""
    xs, ys = np.array([0.] + list(xs)), np.array([0.] + list(ys))
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    if x > xs[-1] + 1e-9:
        return None
    return float(np.interp(max(x, 0.), xs, ys))


def matched(runs, condition, seed, limit=20):
    l1 = [runs[condition, 'L1', lim, seed] for lim in LIMITS['L1']]
    r0 = runs[condition, 'R0', limit, seed]
    rms_value = curve([r['assist_rms'] for r in l1], [r['benefit'] for r in l1], r0['assist_rms'])
    harm_value = curve([r['low10'] for r in l1], [r['benefit'] for r in l1], r0['low10'])
    result = dict(seed=seed, R0_benefit=r0['benefit'], R0_assist_rms=r0['assist_rms'], R0_low10=r0['low10'],
                  L1_at_rms=rms_value, L1_at_harm=harm_value,
                  rms_diff=None if rms_value is None else r0['benefit'] - rms_value,
                  harm_diff=None if harm_value is None else r0['benefit'] - harm_value)
    result['rms_pct'] = None if rms_value in (None, 0) else 100 * result['rms_diff'] / rms_value
    result['harm_pct'] = None if harm_value in (None, 0) else 100 * result['harm_diff'] / harm_value
    return result


def verdict(rows, falls_l1, falls_r0):
    rms = [r['rms_diff'] for r in rows]
    harm = [r['harm_diff'] for r in rows if r['harm_diff'] is not None]
    if None in rms:
        return 'no', '판정 불가 (R0 보조 RMS가 L1 범위 밖)'
    if falls_r0 > falls_l1:
        return 'no', '유지 안 됨 (R0만 낙상)'
    m_rms, m_harm = np.mean(rms), (np.mean(harm) if len(harm) == len(rows) else None)
    if m_rms > 0 and all(v > 0 for v in rms) and m_harm is not None and m_harm > 0 and falls_l1 + falls_r0 == 0:
        return 'ok', '유지'
    if m_rms <= 0 and (m_harm is None or m_harm <= 0):
        return 'no', '유지 안 됨'
    return 'part', '일부 유지'


def collect():
    runs_raw = gather()
    order = ['walk', 'walk_seg400'] + [c for c in CONDITIONS if c != 'walk_seg400']
    conditions = [c for c in order if all((c, m, lim, s) in runs_raw for m in LIMITS for lim in LIMITS[m] for s in SEEDS)]
    runs = {}
    for c in conditions:
        for seed in SEEDS:
            reference = masks_from_reference(read_traces(runs_raw[c, 'L1', 20, seed]))
            for m in LIMITS:
                for lim in LIMITS[m]:
                    runs[c, m, lim, seed] = digest(runs_raw[c, m, lim, seed], reference)
    summary = {}
    for c in conditions:
        rows = [matched(runs, c, s) for s in SEEDS]
        rows15 = [matched(runs, c, s, 15) for s in SEEDS]
        falls = {m: sum(runs[c, m, lim, s]['fall'] is not None for lim in LIMITS[m] for s in SEEDS) for m in LIMITS}
        cls, text = verdict(rows, falls['L1'], falls['R0'])
        summary[c] = dict(matched=rows, matched15=rows15, falls=falls, verdict=text, verdict_class=cls)
    return conditions, runs, summary


def need_rms(runs, c, k, model='R0'):
    return float(np.mean([runs[c, model, 20, s]['need'][k]['assist_rms_Nm'] for s in SEEDS]))


def mean(runs, c, m, lim, key, idx=None):
    vals = [runs[c, m, lim, s][key] if idx is None else runs[c, m, lim, s][key][idx] for s in SEEDS]
    vals = [v for v in vals if v is not None]
    return (float(np.mean(vals)), float(min(vals)), float(max(vals))) if vals else (None, None, None)


def panels(n):
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    return rows, cols


def fig_curves(conditions, runs, summary, ykey, xkey, ylabel, xlabel, identity, note_key):
    rows, cols = panels(len(conditions))
    fig, axes = plt.subplots(rows, cols, figsize=(4.1 * cols, 3.3 * rows), squeeze=False)
    for ax, c in zip(axes.flat, conditions):
        for m in LIMITS:
            pts = [(mean(runs, c, m, lim, xkey), mean(runs, c, m, lim, ykey)) for lim in LIMITS[m]]
            x = [0.] + [p[0][0] for p in pts] if m == 'L1' else [p[0][0] for p in pts]
            y = [0.] + [p[1][0] for p in pts] if m == 'L1' else [p[1][0] for p in pts]
            ax.plot(x, y, '--' if m == 'L1' else '-', color=COLORS[m, 20], lw=1.3, zorder=1)
            for lim, (px, py) in zip(LIMITS[m], pts):
                ax.errorbar(px[0], py[0], xerr=[[px[0] - px[1]], [px[2] - px[0]]], yerr=[[py[0] - py[1]], [py[2] - py[0]]],
                            fmt='o' if m == 'L1' else 's', color=COLORS[m, lim], ms=6, capsize=2, lw=1, zorder=3)
                ax.annotate(f'{m} ±{lim}', (px[0], py[0]), xytext=(4, -11 if m == 'L1' else 5), textcoords='offset points',
                            fontsize=8, color=COLORS[m, 20])
        d = [r[note_key] for r in summary[c]['matched']]
        pct = [r[note_key.replace('diff', 'pct')] for r in summary[c]['matched']]
        if all(v is not None for v in d + pct):
            ax.set_title(f'{SHORT[c]} · R0 ±20 {np.mean(d):+.1f} (N·m)² ({np.mean(pct):+.0f} %)', fontsize=10)
        else:
            ax.set_title(f'{SHORT[c]} · 비교 범위 밖', fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.25)
        ax.axhline(0, color='#888', lw=.6)
    for ax in list(axes.flat)[len(conditions):]:
        ax.axis('off')
    fig.tight_layout()
    return svg(fig, identity)


def fig_summary(conditions, summary):
    fig, axes = plt.subplots(1, 2, figsize=(11, .75 * len(conditions) + 1.4), sharey=True)
    y = np.arange(len(conditions))[::-1]
    for ax, key, title in ((axes[0], 'rms_pct', '같은 보조 RMS에서 R0 ±20 이득 − L1 곡선 [%]'),
                           (axes[1], 'harm_pct', '같은 저부담 손해에서 R0 ±20 이득 − L1 곡선 [%]')):
        for yi, c in zip(y, conditions):
            vals = [r[key] for r in summary[c]['matched']]
            ok = [v for v in vals if v is not None]
            if ok:
                ax.barh(yi, np.mean(ok), color='#245aa0' if np.mean(ok) > 0 else '#c2410c', alpha=.75, height=.55)
                ax.scatter(ok, [yi] * len(ok), color='#1c2d3d', s=16, zorder=3)
                ax.text(max(ok + [np.mean(ok), 0.]), yi, f'  {np.mean(ok):+.0f} %', ha='left', va='center', fontsize=9)
            else:
                ax.text(0, yi, '범위 밖', ha='center', va='center', fontsize=9, color='#62748a')
        ax.axvline(0, color='#444', lw=.8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis='x', alpha=.25)
    axes[0].set_yticks(y, [SHORT[c] for c in conditions])
    fig.tight_layout()
    return svg(fig, 'summary_bars')


def fig_need(conditions, runs):
    fig, axes = plt.subplots(2, len(conditions), figsize=(2.6 * len(conditions) + 1, 5.4), squeeze=False, sharey='row')
    for j, c in enumerate(conditions):
        for key in SHOWN:
            per_seed = [runs[(c,) + key + (s,)]['need'] for s in SEEDS]
            labels = [NEED[[0, 5, 10, 20, 40, 80].index(b['need_lo_Nm'])] for b in per_seed[0]]
            rms = np.mean([[b['assist_rms_Nm'] for b in ns] for ns in per_seed], axis=0)
            sign = np.mean([[100 * b['sign_agreement'] for b in ns] for ns in per_seed], axis=0)
            x = np.arange(len(labels))
            axes[0, j].plot(x, rms, 'o-', color=COLORS[key], lw=1.4, ms=4, label=NAMES[key])
            axes[1, j].plot(x, sign, 'o-', color=COLORS[key], lw=1.4, ms=4)
        for i in (0, 1):
            axes[i, j].set_xticks(x, labels, rotation=45, fontsize=8)
            axes[i, j].grid(alpha=.25)
        axes[1, j].axhline(50, color='#c2410c', lw=.8, ls=':')
        axes[0, j].set_title(SHORT[c], fontsize=10)
        axes[1, j].set_xlabel('무보조 착용자 |토크| [N·m]', fontsize=8)
    axes[0, 0].set_ylabel('보조 RMS [N·m]')
    axes[1, 0].set_ylabel('보조 방향 일치 [%]')
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return svg(fig, 'need_bins')


def fig_omega(conditions, runs, pend):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    x = np.arange(len(OMEGA) - 1)
    labels = [f'{lo}–{hi}' for lo, hi in zip(OMEGA[:-1], OMEGA[1:])]
    styles = {'pendulum': ('#8b70b2', '진자 평가 (학습과 같은 궤적 생성기)')}
    palette = {'walk': '#1a8b67', 'walk_seg400': '#7cc3a6', 'run': '#c2410c'}
    axes[0].plot(x, [100 * r['share'] for r in pend], 'o-', color=styles['pendulum'][0], label=styles['pendulum'][1])
    axes[1].plot(x, [r['assist_rms'] for r in pend], 'o-', color=styles['pendulum'][0], label='R0 ±20 · 진자')
    for c in conditions:
        share = np.mean([[r['share'] for r in runs[c, 'L1', 20, s]['omega']] for s in SEEDS], axis=0)
        axes[0].plot(x, 100 * share, 'o-', color=palette.get(c, '#555'), label=SHORT[c])
        for m, ls in (('R0', '-'), ('L1', ':')):
            vals = [np.nan if v is None else v for v in
                    np.mean([[np.nan if r['assist_rms'] is None else r['assist_rms'] for r in runs[c, m, 20, s]['omega']] for s in SEEDS], axis=0)]
            if m == 'R0' or c == 'run':
                axes[1].plot(x, vals, 'o' + ls, color=palette.get(c, '#555'), label=f'{m} ±20 · {SHORT[c]}', ms=4)
    axes[0].set_ylabel('무보조 시간 비율 [%]')
    axes[1].set_ylabel('보조 RMS [N·m]')
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.set_xlabel('고관절 각속도 |ω| [rad/s]')
        ax.grid(alpha=.25)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return svg(fig, 'omega')


def fig_phase(conditions, runs):
    fig, axes = plt.subplots(len(conditions), 2, figsize=(11, 2.1 * len(conditions) + .6), squeeze=False, sharex=True)
    x = (np.arange(PHASE_BINS) + .5) * 100 / PHASE_BINS
    for i, c in enumerate(conditions):
        for side in (0, 1):
            ax = axes[i, side]
            for key in SHOWN:
                vals = np.mean([[-(r.get('reduction', np.nan)) for r in runs[(c,) + key + (s,)]['rows'][side]] for s in SEEDS], axis=0)
                ax.plot(x, vals, color=COLORS[key], lw=1.5, label=NAMES[key])
            ax.axhline(0, color='#888', lw=.6)
            ax.grid(alpha=.25)
            ax.set_title(f'{SHORT[c]} · {("오른쪽", "왼쪽")[side]}', fontsize=9)
            if side == 0:
                ax.set_ylabel('RMS 증가 [N·m]', fontsize=8)
    axes[-1, 0].set_xlabel('위상 [%] (0 = 그 다리 참조 최대 굴곡)')
    axes[-1, 1].set_xlabel('위상 [%] (0 = 그 다리 참조 최대 굴곡)')
    axes[0, 0].legend(fontsize=8, frameon=False, ncol=3)
    fig.tight_layout()
    return svg(fig, 'phase')


def fmt(v, spec='.1f'):
    return '—' if v is None else format(v, spec)


def cell(t, spec='.1f'):
    m, lo, hi = t
    return '—' if m is None else f'<b>{format(m, spec)}</b><br><small>{format(lo, spec)} ~ {format(hi, spec)}</small>'


def main():
    plan = read(STUDY / 'manifest.json')
    aplan = read(STUDY / 'analysis_plan.json')
    screening = read(STUDY / 'screening.json')
    records = read(STUDY / 'runs.json')
    failures = read(STUDY / 'runtime_failures.json') if (STUDY / 'runtime_failures.json').exists() else []
    conditions, runs, summary = collect()
    new = [c for c in conditions if c != 'walk']
    walk_screen = screen(read(OLD / 'runs.json')['L1_20Nm_seed42'])
    body = ['<nav><a href="../reports.html">보고서 목록</a><a href="../experiment_summary.html">전체 요약</a>'
            '<a href="../261020_relative/phase_report.html">R0 기존 걷기 그래프</a><a href="../261020_relative/report.html">R0/R1 상세</a></nav>',
            '<h1>R0 일반화 · 새 보행 조건에서도 L1보다 나은가</h1>',
            '<p>R0는 L1과 같은 진자 학습에서 토크 경로만 바로잡은 모델이다. 기존 걷기 앞 40초에서는 같은 보조 크기로 L1보다 고관절 부담을 더 줄였다. '
            '이번에는 R0가 평가된 적 없는 조건에서 L1(출력 ±10/15/20 N·m)과 R0(±15/20 N·m)를 같은 W1·H1으로 평가했다. 새 학습은 없고, 각 조건은 평가 난수 42·7 두 번이다.</p>']
    def avg(c, key):
        return float(np.mean([r[key] for r in summary[c]['matched']]))

    kept = [c for c in conditions if summary[c]['verdict_class'] == 'ok']
    lost = [c for c in conditions if summary[c]['verdict_class'] == 'no']
    total_runs = sum(1 for c in conditions for m in LIMITS for _ in LIMITS[m] for _ in SEEDS)
    falls = sum(summary[c]['falls']['L1'] + summary[c]['falls']['R0'] for c in conditions)
    lines = []
    if kept:
        lines.append('<b>같은 걷기 기록의 다른 구간에서는 유지된다.</b> 같은 보조 RMS에서 R0 ±20의 고관절 τ² 감소는 L1 곡선보다 '
                     + ', '.join(f'{SHORT[c]} {avg(c, "rms_pct"):+.0f} %' for c in kept) + ' 크고, 같은 저부담 손해에서도 '
                     + ', '.join(f'{avg(c, "harm_pct"):+.0f} %' for c in kept) + ' 크다.')
    if lost:
        lines.append('<b>달리기에서는 유지되지 않는다.</b> ' + ' '.join(
            f'{SHORT[c]}에서 R0 ±20은 같은 보조 RMS의 L1 곡선보다 {avg(c, "rms_pct"):+.0f} %, 같은 손해 기준 {avg(c, "harm_pct"):+.0f} %이고, '
            f'최악 위상 RMS 증가는 {mean(runs, c, "R0", 20, "worst")[0]:.2f} N·m로 L1 ±20({mean(runs, c, "L1", 20, "worst")[0]:.2f})과 같다. '
            f'같은 ±20 상한에서 이득은 L1 {mean(runs, c, "L1", 20, "benefit")[0]:.0f} 대 R0 {mean(runs, c, "R0", 20, "benefit")[0]:.0f} (N·m)²다.' for c in lost))
    body_change = [r['body_err_mm'][1] - r['body_err_mm'][0] for r in runs.values()]
    speed_change = [r['speed'][1] - r['speed'][0] for r in runs.values()]
    lines.append(f'{len(conditions)}조건 × 5출력 × 평가 난수 2 = {total_runs}건(기존 걷기 10건은 저장 기록, 나머지는 이번 신규 평가)에서 낙상 {falls}건. 보조에 따른 몸체 추종 오차 변화는 '
                 f'{min(body_change):+.1f} ~ {max(body_change):+.1f} mm, 전진 속도 변화는 {max(abs(v) for v in speed_change):.4f} m/s 이하다.')
    verdicts = ''.join(f'<li>{escape(SHORT[c])}: <span class="{summary[c]["verdict_class"]}">{summary[c]["verdict"]}</span></li>' for c in conditions)
    body.append('<div class="status">' + '<br>'.join(lines) + f'<ul>{verdicts}</ul>'
                '<small>판정 기준은 새 조건의 R0 결과가 나오기 전에 정했다(<code>analysis_plan.json</code>): 같은 보조 RMS의 L1 곡선 대비 이득 차이가 두 평가 난수 모두 0보다 크고, '
                '같은 저부담 손해의 L1 곡선 대비 이득 차이 평균이 0보다 크며, 낙상이 없으면 "유지"다.</small></div>')
    krows = []
    for c in conditions:
        mt = summary[c]['matched']
        krows.append(f'<tr><td>{escape(SHORT[c])}</td>'
                     f'<td>{mean(runs, c, "L1", 20, "benefit")[0]:.0f} / {mean(runs, c, "L1", 20, "assist_rms")[0]:.1f}</td>'
                     f'<td>{mean(runs, c, "L1", 10, "benefit")[0]:.0f} / {mean(runs, c, "L1", 10, "assist_rms")[0]:.1f}</td>'
                     f'<td>{mean(runs, c, "R0", 20, "benefit")[0]:.0f} / {mean(runs, c, "R0", 20, "assist_rms")[0]:.1f}</td>'
                     f'<td><b>{avg(c, "rms_diff"):+.1f} ({avg(c, "rms_pct"):+.0f} %)</b><br><small>{min(r["rms_pct"] for r in mt):+.0f} ~ {max(r["rms_pct"] for r in mt):+.0f} %</small></td>'
                     f'<td><b>{avg(c, "harm_diff"):+.1f} ({avg(c, "harm_pct"):+.0f} %)</b></td>'
                     f'<td>{mean(runs, c, "L1", 20, "worst")[0]:.2f} → {mean(runs, c, "R0", 20, "worst")[0]:.2f}</td>'
                     f'<td class="{summary[c]["verdict_class"]}">{summary[c]["verdict"]}</td></tr>')
    body.append('<div class="scroll"><table><tr><th>조건</th><th>L1 ±20 이득 / 보조 RMS</th><th>L1 ±10 이득 / 보조 RMS</th><th>R0 ±20 이득 / 보조 RMS</th>'
                '<th>같은 보조 RMS의 L1 곡선 대비</th><th>같은 저부담 손해의 L1 곡선 대비</th><th>최악 위상 RMS 증가 L1 ±20 → R0 ±20</th><th>판정</th></tr>'
                + ''.join(krows) + '</table></div>'
                '<p class="note">이득 = 고관절 τ² 감소 [(N·m)²], 보조 RMS [N·m], 최악 위상 [N·m]. 평가 난수 42·7 평균이며 작은 글씨는 두 실행의 범위다. '
                '"L1 곡선"은 원점과 L1 ±10/15/20 세 점을 직선으로 이은 것으로, L1의 출력만 줄여서 얻을 수 있는 이득이다. 기존 걷기 행은 저장 기록을 같은 코드로 다시 계산한 값이다.</p>')

    body.append('<h2>1. 평가 조건</h2><p>W1이 무보조로 40초를 완주하고, 무보조 몸체 추종 오차가 0.10 m 이하이며, 참조 위상이 10주기 이상 잡히는 조건만 본 평가에 넣었다. '
                '선별은 L1 ±20·평가 난수 42 한 번으로 했고, 그 기록은 본 평가의 저부담 구간 기준으로 그대로 쓴다.</p>')
    srows = [f'<tr><td>{escape(LONG["walk"])}</td><td>{walk_screen["unassisted_body_error_m"]*1000:.0f}</td><td>완주</td><td>{walk_screen["phase_cycles_right"]}</td><td>기존 기준 (재평가 없음)</td></tr>']
    for c in CONDITIONS:
        s = screening.get(c)
        if s is None:
            srows.append(f'<tr><td>{escape(LONG[c])}</td><td colspan="4">미평가</td></tr>')
            continue
        fall = '완주' if s['no_fall'] else f'낙상 {records[f"{c}_L1_20Nm_seed42"]["survived_s"]:.1f} s'
        used = '본 평가' if c in conditions else ('선별 통과, 본 평가 미완료' if s['passed'] else '제외')
        srows.append(f'<tr><td>{escape(LONG[c])}</td><td>{s["unassisted_body_error_m"]*1000:.0f}</td><td>{fall}</td><td>{s["phase_cycles_right"]}</td><td>{used}</td></tr>')
    srows.append('<tr><td>기존 걷기 0.85·1.15배 재생</td><td colspan="3">평가하지 않음</td><td>제외: 같은 자세를 시간 축만 바꿔 재생한 합성 동작이다. '
                 '보폭은 그대로 두고 걸음 주기만 바꾸므로 실제로 느리거나 빠르게 걷는 사람의 전신 동역학이 아니다.</td></tr>')
    body.append('<div class="scroll"><table><tr><th>조건</th><th>무보조 몸체 오차 [mm]</th><th>40초</th><th>위상 주기 수</th><th>사용</th></tr>'
                + ''.join(srows) + '</table></div>')
    body.append('<p class="note">무보조 몸체 오차는 W1 몸체 위치와 참조 동작의 평균 거리다. 실제 사람의 새 보행 기록(2026-10-13 변환본)은 W1이 16조건 중 15조건에서 넘어져 이번에도 쓰지 않았다. '
                '따라서 이 평가는 W1이 학습한 걷기·달리기 기록과, 그 밖의 같은 형식 동작 일부에 한정된다.</p>')

    body.append('<h2>2. 같은 보조 크기에서 R0가 더 줄이는가</h2>'
                '<p>점선은 L1의 출력 상한만 ±10/15/20 N·m로 바꾼 결과를 원점과 이은 곡선이다. 같은 보조 RMS에서 R0 점이 점선보다 위에 있으면, '
                'L1 출력을 줄여서는 얻을 수 없는 이득이다. 오차 막대는 평가 난수 두 번의 범위다.</p>')
    body.append(fig_summary(conditions, summary))
    body.append(fig_curves(conditions, runs, summary, 'benefit', 'assist_rms', '고관절 τ² 감소 [(N·m)²]', '보조 RMS [N·m]', 'benefit_vs_rms', 'rms_diff'))
    body.append('<h2>3. 같은 손해에서 이득이 큰가</h2><p>가로축은 저부담 감속 구간의 착용자 RMS 증가다. '
                '이 구간은 무보조 착용자 토크의 50 ms RMS가 10 N·m 이하이고 고관절이 감속 중인 시점으로, 조건·평가 난수마다 L1 ±20 실행의 무보조 기록에서 한 번 정해 모든 모델에 같은 시각을 쓴다. '
                '왼쪽 위에 있을수록 같은 방해로 더 많이 돕는다.</p>')
    exposure = ', '.join(f'{SHORT[c]} {mean(runs, c, "L1", 20, "low10_s")[0]:.2f}' for c in conditions)
    body.append(f'<p class="note">이 구간의 길이(좌우 관절을 합친 초)는 {exposure}이다. 달리기는 무보조 착용자 토크가 10 N·m 이하로 내려가는 감속 시점이 거의 없어 '
                '이 지표의 표본이 1초도 안 된다. 달리기의 손해는 최악 위상 RMS 증가로 함께 판단한다.</p>')
    body.append(fig_curves(conditions, runs, summary, 'benefit', 'low10', '고관절 τ² 감소 [(N·m)²]', '저부담 감속 RMS 증가 [N·m]', 'benefit_vs_harm', 'harm_diff'))

    trows = []
    for c in conditions:
        for i, key in enumerate(SHOWN + [('R0', 15)]):
            m, lim = key
            extra = ''
            if m == 'R0':
                mt = summary[c]['matched' if lim == 20 else 'matched15']
                a = [r['rms_diff'] for r in mt]
                h = [r['harm_diff'] for r in mt]
                extra = (f'<td>{"—" if None in a else f"<b>{np.mean(a):+.1f}</b><br><small>{min(a):+.1f} ~ {max(a):+.1f}</small>"}</td>'
                         f'<td>{"—" if None in h else f"<b>{np.mean(h):+.1f}</b><br><small>{min(h):+.1f} ~ {max(h):+.1f}</small>"}</td>')
            else:
                extra = '<td></td><td></td>'
            label = f'<td rowspan="4">{escape(SHORT[c])}<br><small>무보조 τ² {mean(runs, c, "L1", 20, "A_tau_sq")[0]:.0f}</small></td>' if i == 0 else ''
            trows.append(f'<tr>{label}<td>{NAMES[key]}</td><td>{cell(mean(runs, c, m, lim, "assist_rms"), ".2f")}</td>'
                         f'<td>{cell(mean(runs, c, m, lim, "saturation"))}</td><td>{cell(mean(runs, c, m, lim, "benefit"))}</td>'
                         f'<td>{cell(mean(runs, c, m, lim, "low10"), ".2f")}</td><td>{cell(mean(runs, c, m, lim, "worst"), ".2f")}</td>{extra}</tr>')
    body.append('<h2>4. 수치 표</h2><p>굵은 값은 평가 난수 두 번의 평균, 아래는 실제 범위다. 마지막 두 열은 R0 이득에서 같은 보조 RMS·같은 저부담 손해의 L1 곡선 값을 뺀 차이다.</p>'
                '<div class="scroll"><table><tr><th>조건</th><th>모델·상한</th><th>보조 RMS [N·m]</th><th>포화 [%]</th><th>고관절 τ² 감소 [(N·m)²]</th>'
                '<th>저부담 감속 RMS 증가 [N·m]</th><th>최악 위상 RMS 증가 [N·m]</th><th>같은 보조 RMS 대비 [(N·m)²]</th><th>같은 손해 대비 [(N·m)²]</th></tr>'
                + ''.join(trows) + '</table></div>')
    body.append('<p class="note">고관절 τ² 감소 = 같은 실행의 무보조 A와 보조 B에서 착용자 고관절 토크 제곱 평균의 차이(좌우 평균, 1.5초 이후). '
                '최악 위상 RMS 증가 = 4 % 위상 구간 중 착용자 RMS가 가장 많이 늘어난 구간의 증가량(좌우 중 큰 쪽). 포화 = 보조가 상한의 95 % 이상인 시간 비율. '
                '토크²는 시뮬레이션 부담 지표이며 근활성·대사량 측정값이 아니다.</p>')

    body.append('<h2>5. 필요한 만큼만 돕는가</h2><p>무보조 기록에서 착용자가 낸 토크 크기로 시간을 나눠, 각 구간의 보조 RMS(위)와 보조 방향이 무보조 착용자 토크 방향과 같은 비율(아래)을 봤다. '
                + ' '.join(f'{SHORT[c]}: R0 ±20 보조 RMS가 필요 토크 0–5 N·m에서 {need_rms(runs, c, 0):.1f}, 80 N·m 이상에서 {need_rms(runs, c, -1):.1f} N·m'
                           f'(L1 ±20 {need_rms(runs, c, 0, "L1"):.1f} → {need_rms(runs, c, -1, "L1"):.1f}).' for c in conditions)
                + ' 걷기에서 R0는 필요 토크가 클수록 더 크게 돕지만, 달리기에서는 필요 토크가 큰 구간에서 오히려 작아진다. '
                '보조 방향의 일치율은 모든 조건에서 L1과 R0가 거의 같고, 필요 토크 5 N·m 미만에서는 둘 다 50 % 근처(무작위)다. 아래 점선이 50 %다.</p>')
    body.append(fig_need(conditions, runs))

    pend = pendulum_omega()
    fast = {c: 100 * float(np.mean([sum(r['share'] for r in runs[c, 'L1', 20, sd]['omega'] if r['lo'] >= 4) for sd in SEEDS])) for c in conditions}
    pend_fast = 100 * sum(r['share'] for r in pend if r['lo'] >= 4)
    pend_rise = (pend[0]['assist_rms'], max(r['assist_rms'] for r in pend if r['assist_rms'] is not None))
    body.append('<h3>달리기는 진자에서 드물던 빠른 움직임이 많다 (관찰)</h3>'
                f'<p>고관절 각속도가 4 rad/s를 넘는 무보조 시간은 진자 평가 궤적에서 {pend_fast:.1f} %, '
                + ', '.join(f'{SHORT[c]}에서 {fast[c]:.1f} %' for c in conditions) + '다. '
                '진자 평가 궤적은 학습과 같은 무작위 궤적 생성기로 만든 학습 밖 표본이므로, 달리기에는 R0가 학습에서 거의 겪지 못한 상태가 많다. '
                f'진자에서는 각속도가 클수록 R0 보조가 커졌지만({pend_rise[0]:.1f} → {pend_rise[1]:.1f} N·m), W1 걷기·달리기에서는 각속도와 거의 무관했다. '
                '이 그림은 분포 차이를 보여 줄 뿐이며, 달리기에서 R0의 출력 배분이 바뀐 원인을 따로 검증하지는 않았다.</p>')
    body.append(fig_omega(conditions, runs, pend))

    body.append('<h2>6. 좌우 차이와 위상별 방해</h2>')
    lrows = []
    for c in conditions:
        for key in SHOWN:
            m, lim = key
            b = [mean(runs, c, m, lim, 'benefit_side', k)[0] for k in (0, 1)]
            h = [mean(runs, c, m, lim, 'low10_side', k)[0] for k in (0, 1)]
            w = [float(np.mean([runs[c, m, lim, s]['worst_side'][k]['value'] for s in SEEDS])) for k in (0, 1)]
            lrows.append(f'<tr><td>{escape(SHORT[c])}</td><td>{NAMES[key]}</td><td>{fmt(b[0])} / {fmt(b[1])}</td><td>{fmt(b[0]-b[1], "+.1f")}</td>'
                         f'<td>{fmt(h[0], ".2f")} / {fmt(h[1], ".2f")}</td><td>{fmt(w[0], ".2f")} / {fmt(w[1], ".2f")}</td></tr>')
    def side_gap(c, m):
        return mean(runs, c, m, 20, 'benefit_side', 0)[0] - mean(runs, c, m, 20, 'benefit_side', 1)[0]

    body.append('<p>고관절 τ² 감소를 좌우로 나눴다. R0 ±20은 ' + ', '.join(f'{SHORT[c]} {side_gap(c, "R0"):+.0f}' for c in conditions)
                + ' (N·m)²로 모든 조건에서 오른쪽 이득이 더 크다. L1 ±20은 ' + ', '.join(f'{SHORT[c]} {side_gap(c, "L1"):+.0f}' for c in conditions)
                + '이다. 걷기의 저부담 손해는 L1·R0 모두 오른쪽에 몰려 있고, 달리기에서는 좌우가 비슷하다.</p>')
    body.append('<div class="scroll"><table><tr><th>조건</th><th>모델·상한</th><th>τ² 감소 오른쪽 / 왼쪽</th><th>오른쪽 − 왼쪽</th>'
                '<th>저부담 감속 RMS 증가 오른쪽 / 왼쪽</th><th>최악 위상 RMS 증가 오른쪽 / 왼쪽</th></tr>' + ''.join(lrows) + '</table></div>')
    body.append('<p>위상별 착용자 RMS 변화다. 0보다 위면 그 위상에서 착용자 토크가 늘었다(방해).</p>')
    body.append(fig_phase(conditions, runs))

    srows = []
    for c in conditions:
        for key in SHOWN:
            m, lim = key
            r = [runs[(c,) + key + (s,)] for s in SEEDS]
            falls = sum(x['fall'] is not None for x in r)
            body_change = np.mean([x['body_err_mm'][1] - x['body_err_mm'][0] for x in r])
            speed = np.mean([x['speed'][1] for x in r])
            speed_change = np.mean([x['speed'][1] - x['speed'][0] for x in r])
            other = {g: np.mean([x['other'].get(g, np.nan) for x in r]) for g in ('무릎', '발목', '몸통')}
            srows.append(f'<tr><td>{escape(SHORT[c])}</td><td>{NAMES[key]}</td><td>{falls}/2</td>'
                         f'<td>{np.mean([x["body_err_mm"][0] for x in r]):.1f} → {np.mean([x["body_err_mm"][1] for x in r]):.1f} ({body_change:+.1f})</td>'
                         f'<td>{speed:.3f} ({speed_change:+.3f})</td>'
                         + ''.join(f'<td>{-other[g]:+.0f}</td>' for g in ('무릎', '발목', '몸통')) + '</tr>')
    body.append('<h2>7. 보행 안정성과 다른 관절</h2><p>다른 관절 열은 보조 B − 무보조 A의 토크 제곱 평균 변화 [(N·m)²]다. 음수면 부담이 줄었다.</p>'
                '<div class="scroll"><table><tr><th>조건</th><th>모델·상한</th><th>낙상</th><th>몸체 오차 A → B [mm]</th><th>속도 B [m/s] (B − A)</th>'
                '<th>무릎</th><th>발목</th><th>몸통</th></tr>' + ''.join(srows) + '</table></div>')

    reruns = [f for f in failures]
    body.append('<details><summary>해석 범위</summary><ul>'
                '<li>가상 착용자 W1 하나, 원본 걷기·달리기 기록 하나씩이다. 다른 사람·실제 속도 변화·경사·계단·실기기 성능은 보증하지 않는다.</li>'
                '<li>R0의 학습 seed는 0 하나다. 평가 난수 42·7은 같은 모델의 두 평가 실행이며 사람이나 학습 반복이 아니다.</li>'
                '<li>같은 걷기 다른 구간은 조건 변화가 아니라 구간 차이에 따른 산포를 보는 기준이다.</li>'
                '<li>W1이 학습하지 않은 동작(제자리 걷기·무릎 높이 들기)은 무보조에서도 2초 남짓에 넘어져 보조 모델을 비교하지 못했다. '
                '진짜 속도 변화(실제로 느리거나 빠르게 걸은 기록)도 W1이 따라갈 수 있는 자료가 없다.</li>'
                '<li>R0는 좌우 고관절을 한 번에 입력·출력하는 공동 정책이라 좌우 대칭이 보장되지 않는다. 좌우 차이는 정책과 원본 동작의 비대칭이 섞인 결과다.</li>'
                '<li>L1 곡선 대비 차이는 원점과 L1 세 점을 직선으로 이은 보간이다. 곡선 범위를 벗어나면 계산하지 않는다.</li></ul></details>')
    body.append('<details><summary>재현 경로·실행 기록</summary><ul>'
                '<li>평가 실행: <code>python examples/experiments/assist_skeleton/evaluate_generalization.py</code> '
                '(systemd 단위 <code>aisbc-r0-generalization-20261021</code>)</li>'
                '<li>이 페이지: <code>python examples/experiments/assist_skeleton/report_generalization.py</code></li>'
                f'<li>원본: <code>{STUDY.relative_to(ROOT)}/</code>(manifest, analysis_plan, screening, runs, comparison.json, runtime_failures). '
                f'신규 W1 평가 {len(records)}건, 운영 오류 후 재실행 {len(reruns)}회.</li>'
                '<li>기존 걷기 기록: L1 <code>results/llp_output_scale_20261016/</code>, R0 <code>results/llp_relative_20261020/walking/</code> (평가 소스 해시 동일 확인)</li>'
                '<li>모델: L1 <code>checkpoints/L1_llp_pendulum/last.ckpt</code>, R0 <code>results/llp_relative_R0_s0_20261020/epoch_6000.ckpt</code>. '
                'W1 <code>checkpoints/W1_wearer_human/last.ckpt</code>, H1은 manifest의 경로. 모두 동결.</li></ul></details>')
    body.append('<p class="note">생성 UTC ' + datetime.now(timezone.utc).isoformat(timespec='seconds') + '</p>')
    evidence = dict(analysis_plan=aplan, screening=screening, summary=summary,
                    runs={'/'.join(map(str, k)): {kk: vv for kk, vv in v.items() if kk not in ('rows',)} for k, v in runs.items()},
                    manifest_sha_inputs=dict(sources=plan['sources']))
    (STUDY / 'comparison.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False))
    body.append('<script type="application/json" id="evidence">' + json.dumps(dict(summary=summary), ensure_ascii=False, allow_nan=False).replace('</', '<\\/') + '</script>')
    html = ('<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>R0 일반화 평가</title><style>' + CSS + '</style></head><body><main>' + ''.join(body) + '</main></body></html>')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(f'wrote {OUT.relative_to(ROOT)} ({len(html) / 1e6:.1f} MB); conditions={conditions}')
    for c in conditions:
        print(c, summary[c]['verdict'], [(round(r['rms_diff'] or 0, 1), round(r['harm_diff'] or 0, 1)) for r in summary[c]['matched']])


if __name__ == '__main__':
    main()
