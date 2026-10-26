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
"""Graph-first W1 phase report: the top five candidates by the pre-registered score and L1.

Reads stored records only (no simulation). Selection rule, fixed before the R0/R1
training in results/llp_relative_20261020/manifest.json: at the matched ±20 N·m range,
score = hip torque² benefit / L1 benefit − low-burden deceleration RMS increase / L1 increase.
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
from matplotlib.colors import TwoSlopeNorm
from matplotlib.font_manager import FontProperties
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_need_bins import walking as need_bins  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'results/llp_relative_20261020'
OUT = ROOT / 'docs/ai_sbc/261020_relative/phase_report.html'
CAP = 20
BINS = 25
# Optional PNG copies of every figure for visual checks (not part of the page).
PREVIEW = Path(os.environ['PHASE_REPORT_PNG']) if os.environ.get('PHASE_REPORT_PNG') else None
X = (np.arange(BINS) + .5) * 100 / BINS
SIDES = ('오른쪽', '왼쪽')
NAMES = {'L1': 'L1 · 기준 모델 (과거 토크 경로)', 'R0_s0': 'R0 · 토크 경로만 수정', 'F1_s0': 'F1 · 토크 경로 + C2 궤적 연결',
         'R1_s0': 'R1 · R0 + 무보조 대비 토크 증가 비용', 'F2_s0': 'F2 · F1 + 흡수 파워 비용', 'G0_s0': 'G0 · ±23 N·m 학습 공동 actor',
         'G4_s0': 'G4 · 관절별 actor·비용 4배 (seed 0)', 'G4_s1': 'G4 · 관절별 actor·비용 4배 (seed 1)'}
SHORT = {'L1': 'L1', 'R0_s0': 'R0', 'F1_s0': 'F1', 'R1_s0': 'R1', 'F2_s0': 'F2', 'G0_s0': 'G0', 'R0n_s0': 'R0 배율 1'}
COLORS = {'L1': '#2f3640', 'R0_s0': '#245aa0', 'F1_s0': '#c48a2a', 'R1_s0': '#c2410c', 'F2_s0': '#8b70b2', 'G0_s0': '#1a8b67', 'R0n_s0': '#245aa0'}
NEED = ['0–5', '5–10', '10–20', '20–40', '40–80', '80+']
DEFINE_R1 = ('학습 중 같은 목표 궤적·관성·외란을 받는 무보조 진자를 함께 계산하고, 보조 진자의 착용자 토크²가 무보조보다 커진 만큼을 '
             '(무보조 착용자 토크²의 최근 50 ms 평균 + 10²)로 나눠 벌점(가중치 −0.5)으로 준다. 무보조보다 줄어든 경우는 벌점 0이다.')
FONT = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family': FONT.get_name(), 'font.size': 10, 'axes.unicode_minus': False, 'svg.fonttype': 'path',
                     'axes.spines.top': False, 'axes.spines.right': False})
CSS = ("html{scroll-behavior:smooth}body{margin:0;background:#f4f6f8;color:#1c2d3d;font:16px/1.8 system-ui,sans-serif}"
       "main{max-width:1180px;margin:auto;padding:28px 26px 70px}h1{font-size:32px;line-height:1.4}h2{font-size:23px;margin:38px 0 12px}"
       "a{color:#245aa0}table{width:100%;border-collapse:collapse;background:white;font-size:14px;line-height:1.6}th{background:#e5edf5;text-align:left}"
       "td,th{padding:9px 12px;border:1px solid #dce3eb;vertical-align:top}strong{color:#185b4a}code{font-size:.88em;overflow-wrap:anywhere}"
       "nav{display:flex;gap:18px;flex-wrap:wrap;border-bottom:1px solid #d8e1ea;padding-bottom:15px}"
       "details{background:white;border:1px solid #d8e1ea;border-radius:8px;padding:15px 20px;margin:20px 0}summary{cursor:pointer;font-weight:650}"
       ".note{color:#62748a;font-size:14px}.plot{background:white;border:1px solid #dce3eb;border-radius:10px;padding:10px;margin:12px 0}"
       ".plot svg{width:100%;height:auto;display:block}.status{padding:16px 18px;background:#e5eef9;border-radius:8px;border-left:5px solid #245aa0}"
       ".chips{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}.chip{background:white;border:1px solid #dce3eb;border-radius:999px;padding:3px 12px;font-size:14px}"
       ".chip i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}select{max-width:100%;font:inherit;padding:5px 8px}"
       ".scroll{overflow-x:auto}@media(max-width:750px){main{padding:20px 16px}h1{font-size:25px}table{display:block;overflow-x:auto}}")


def read(path):
    return json.loads(Path(path).read_text())


def svg(fig, identity):
    if PREVIEW:
        fig.savefig(PREVIEW / f'{identity}.png', dpi=110, bbox_inches='tight')
    plt.rcParams['svg.hashsalt'] = identity
    buf = io.StringIO()
    fig.savefig(buf, format='svg', bbox_inches='tight')
    plt.close(fig)
    text = buf.getvalue()
    return '<div class="plot">' + text[text.index('<svg'):] + '</div>'


def seed_mean_rows(runs):
    """Average 4% phase bins over the two evaluation seeds of one model."""
    out = []
    for side in (0, 1):
        rows = []
        for k in range(BINS):
            cells = [r['rows'][side][k] for r in runs if r['rows'][side][k].get('n')]
            row = dict(lo=k * 100 / BINS, hi=(k + 1) * 100 / BINS, n=sum(c['n'] for c in cells))
            for key in ('A_mean', 'B_mean', 'assist_mean', 'A_rms', 'B_rms', 'reduction', 'assist_rms', 'saturation_pct'):
                row[key] = float(np.mean([c[key] for c in cells]))
            row['assist_min'] = float(min(c['assist_min'] for c in cells))
            row['assist_max'] = float(max(c['assist_max'] for c in cells))
            rows.append(row)
        out.append(rows)
    return out


def arr(rows, side, key):
    return np.array([r[key] for r in rows[side]])


def collect():
    plan = read(STUDY / 'manifest.json')
    walking = read(STUDY / 'walking/comparison.json')
    native = read(STUDY / 'walking_native/comparison.json')
    pendulum = read(ROOT / 'docs/ai_sbc/261020_relative/evidence.json')['pendulum']
    groups = walking['groups']
    base = groups[f'L1_{CAP}']['metrics']

    def score(g):
        m = g['metrics']
        return (m['overall_tau_sq_reduction']['mean'] / base['overall_tau_sq_reduction']['mean']
                - m['low10_rms_increase']['mean'] / base['low10_rms_increase']['mean'])
    ranking = sorted(((score(g), g['model'], g['metrics']['overall_tau_sq_reduction']['mean'], g['metrics']['low10_rms_increase']['mean'])
                      for g in groups.values() if g['limit_Nm'] == CAP and g['model'] != 'L1'), reverse=True)
    top = [r[1] for r in ranking[:5]]
    models = ['L1'] + top
    data = {}
    for model in models + ['R0n_s0']:
        source = native if model.endswith('n_s0') else walking
        runs = [r for r in source['runs'].values() if r['record']['model'] == model and r['record']['limit_Nm'] == CAP]
        assert len(runs) == 2, (model, len(runs))
        g = source['groups'][f'{model}_{CAP}']
        bins = [need_bins(ROOT / r['record']['path'] / 'traces.npz', r['record']['limit_Nm'])['bins'] for r in runs]
        need = [{key: float(np.mean([b[i][key] for b in bins])) for key in bins[0][i] if isinstance(bins[0][i][key], (int, float))} for i in range(len(NEED))]
        for row in need:
            row['rms_change'] = float(np.sqrt(row['B_tau_sq_Nm2']) - np.sqrt(row['A_tau_sq_Nm2']))
        other = {name: float(np.mean([next(j['B'] - j['A'] for j in r['other_joints'] if j['group'] == name) for r in runs]))
                 for name in ('무릎', '발목', '몸통', '전체 비보조 회전관절')}
        data[model] = dict(metrics=g['metrics'], falls=g['falls'], rows=seed_mean_rows(runs), need=need, other=other,
                           seeds=[r['record']['seed'] for r in runs], paths=[r['record']['path'] for r in runs],
                           body_err_mm=float(np.mean([1000 * r['summary']['B']['body_err'] for r in runs])),
                           speed=float(np.mean([r['summary']['B']['speed'] for r in runs])))
    scaling = {cap: groups[f'L1_{cap}']['metrics'] for cap in (10, 15, 20)}
    pend = {(p['model'], p['condition']): p for p in pendulum if p['model'] in models}
    return dict(plan=plan, ranking=ranking, models=models, data=data, scaling=scaling, pendulum=pend,
                excluded_without_matched_20=['G1_s0', 'G1_s1', 'G2_s0', 'G3_s0'])


def fig_core(d):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), layout='constrained', sharey=True)
    values = np.concatenate([-arr(d[m]['rows'], side, 'reduction') for m in ('L1', 'R0_s0', 'R0n_s0') for side in (0, 1)])
    low, high = np.floor(values.min()) - 2, np.ceil(values.max()) + 2
    for side, ax in enumerate(axes):
        ax.axhspan(0, high, color='#c64448', alpha=.07)
        ax.axhline(0, color='#64748b', lw=.8)
        for model, style, width, label in (('L1', '-', 2.2, 'L1 · 0.5배'), ('R0_s0', '-', 2.2, 'R0 · 0.5배 (사전 등록 비교)'),
                                           ('R0n_s0', '--', 1.6, 'R0 · 배율 1·±20 클립 (추가 평가)')):
            ax.plot(X, -arr(d[model]['rows'], side, 'reduction'), color=COLORS[model], ls=style, lw=width, label=label)
        ax.set_title(SIDES[side] + ' 다리')
        ax.set_xlabel('참조 보행 위상 [%] · 최대 굴곡 = 0%')
        ax.set_xlim(0, 100); ax.set_xticks(np.arange(0, 101, 20)); ax.set_ylim(low, high); ax.grid(alpha=.15)
    axes[0].set_ylabel('착용자 토크 RMS 증가 B − A [N·m]\n위(빨강) = 방해 · 아래 = 도움')
    axes[1].legend(fontsize=8.5, loc='lower left')
    return svg(fig, 'core')


def fig_heatmap(d, models):
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.6), layout='constrained')
    data = [np.array([arr(d[m]['rows'], side, 'reduction') for m in models]) for side in (0, 1)]
    bound = max(np.nanmax(np.abs(x)) for x in data)
    for side, ax in enumerate(axes):
        im = ax.imshow(data[side], cmap='RdYlGn', norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound),
                       aspect='auto', extent=[0, 100, len(models) - .5, -.5])
        ax.set_yticks(range(len(models)), [NAMES[m] for m in models], fontsize=9)
        ax.set_title(SIDES[side] + ' · 착용자 토크 RMS 감소량 A − B (평가 seed 42/7 평균)')
        ax.set_xticks(np.arange(0, 101, 10))
    axes[1].set_xlabel('참조 보행 위상 [%] · 최대 굴곡 = 0%')
    fig.colorbar(im, ax=axes, label='[N·m]  음수(빨강): 방해 ← 0 → 양수(초록): 도움', shrink=.85)
    return svg(fig, 'heatmap')


def fig_metrics(d, models):
    panels = [('overall_tau_sq_reduction', '고관절 τ² 감소 [(N·m)²]', '클수록 좋음'),
              ('low10_rms_increase', '저부담 감속 RMS 증가 [N·m]', '작을수록 좋음'),
              ('worst_phase_rms_increase', '최악 위상 RMS 증가 [N·m]', '작을수록 좋음'),
              ('assist_rms', '보조 RMS [N·m]', '보조 크기'),
              ('saturation_pct', '출력 포화율 [%]', '±19 N·m 초과 비율')]
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.6), layout='constrained', sharey=True)
    y = np.arange(len(models))[::-1]
    for ax, (key, title, hint) in zip(axes, panels):
        for yy, m in zip(y, models):
            v = d[m]['metrics'][key]
            ax.barh(yy, v['mean'], color=COLORS[m], alpha=.9 if m != 'L1' else .55, height=.62)
            ax.errorbar(v['mean'], yy, xerr=[[v['mean'] - v['min']], [v['max'] - v['mean']]], color='#1c2d3d', capsize=2, lw=.8)
            ax.text(v['max'], yy, f'  {v["mean"]:.1f}', va='center', fontsize=8.5)
        ax.axvline(d['L1']['metrics'][key]['mean'], color=COLORS['L1'], ls='--', lw=.9)
        ax.set_title(f'{title}\n{hint}', fontsize=9.5); ax.margins(x=.28); ax.grid(axis='x', alpha=.15)
    axes[0].set_yticks(y, [SHORT[m] for m in models])
    fig.suptitle('W1 걷기 40초 · ±20 N·m · 막대 = 평가 seed 42/7 평균, 오차막대 = 두 실행의 범위, 점선 = L1', fontsize=10)
    return svg(fig, 'metrics')


def fig_tradeoff(d, models, scaling):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), layout='constrained', sharey=True)
    xs = [('low10_rms_increase', '저부담 감속 RMS 증가 [N·m] — 작을수록 좋음'), ('assist_rms', '실제 보조 RMS [N·m]')]
    for ax, (key, label) in zip(axes, xs):
        curve = [(scaling[c][key]['mean'], scaling[c]['overall_tau_sq_reduction']['mean'], c) for c in (10, 15, 20)]
        ax.plot([p[0] for p in curve], [p[1] for p in curve], ls='--', marker='s', mfc='white', color=COLORS['L1'], lw=1.1, label='L1 출력 축소 ±10/15/20')
        for x, yv, c in curve[:2]:
            ax.annotate(f'L1 ±{c}', (x, yv), xytext=(4, -12), textcoords='offset points', fontsize=8, color=COLORS['L1'])
        for m in models + ['R0n_s0']:
            met = d[m]['metrics']
            star = m.endswith('n_s0')
            ax.scatter(met[key]['mean'], met['overall_tau_sq_reduction']['mean'], s=150 if star else 70, marker='*' if star else 'o',
                       color='white' if star else COLORS[m], edgecolor=COLORS[m], linewidth=1.6, zorder=3)
            ax.annotate(SHORT[m] + ('' if star else ' ±20'), (met[key]['mean'], met['overall_tau_sq_reduction']['mean']),
                        xytext=(6, 4), textcoords='offset points', fontsize=8.5, color=COLORS[m])
        ax.set_xlabel(label); ax.grid(alpha=.15)
    axes[0].set_ylabel('전체 고관절 τ² 감소 [(N·m)²] — 클수록 좋음')
    axes[0].legend(fontsize=8, loc='lower right')
    axes[0].set_title('이득과 저부담 방해 · 점선 위쪽이 같은 방해에서 더 큰 이득')
    axes[1].set_title('이득과 보조 크기 · ☆ = R0를 배율 1·±20 클립으로 적용(추가 평가)')
    return svg(fig, 'tradeoff')


def fig_need(d, models):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), layout='constrained', sharex=True)
    x = np.arange(len(NEED))
    for m in models:
        need = d[m]['need']
        kw = dict(color=COLORS[m], marker='o', ms=4, lw=2.2 if m in ('L1', 'R0_s0') else 1.2, label=SHORT[m])
        axes[0].plot(x, [r['assist_rms_Nm'] for r in need], **kw)
        axes[1].plot(x, [100 * r['sign_agreement'] for r in need], **kw)
        axes[2].plot(x, [r['rms_change'] for r in need], **kw)
    axes[1].axhline(50, color='#64748b', ls=':', lw=1); axes[1].text(5, 51, '무작위 50 %', ha='right', fontsize=8, color='#64748b')
    axes[2].axhspan(0, 20, color='#c64448', alpha=.07); axes[2].axhline(0, color='#64748b', lw=.8)
    titles = ['보조 RMS [N·m]', '보조와 무보조 착용자 토크의 부호 일치 [%]', '착용자 토크 RMS 변화 B − A [N·m] · 위 = 방해']
    share = d['L1']['need']
    labels = [f'{name}\n({100*r["sample_fraction"]:.0f}%)' for name, r in zip(NEED, share)]
    for ax, t in zip(axes, titles):
        ax.set_title(t, fontsize=10); ax.set_xticks(x, labels, fontsize=8.5); ax.grid(alpha=.15)
        ax.set_xlabel('무보조 착용자 토크 |τ_A| 구간 [N·m] (괄호: 시간 비율)')
    axes[2].legend(fontsize=8, ncol=3, loc='upper right')
    return svg(fig, 'need')


def fig_other(d, models):
    groups = ['무릎', '발목', '몸통', '전체 비보조 회전관절']
    fig, ax = plt.subplots(figsize=(12, 3.6), layout='constrained')
    width = .8 / len(models)
    for i, m in enumerate(models):
        vals = [d[m]['other'][g] for g in groups]
        ax.bar(np.arange(len(groups)) + (i - (len(models) - 1) / 2) * width, vals, width=width * .92, color=COLORS[m], label=SHORT[m])
    ax.axhline(0, color='#64748b', lw=.8)
    ax.set_xticks(np.arange(len(groups)), ['무릎', '발목', '몸통(허리)', '비보조 회전관절 평균'])
    ax.set_ylabel('착용자 토크² 변화 B − A [(N·m)²]\n음수 = 감소, 양수 = 증가'); ax.grid(axis='y', alpha=.15); ax.legend(fontsize=8, ncol=6)
    ax.set_title('고관절 보조가 다른 관절 부담을 어떻게 바꿨나 (±20 N·m, 평가 2회 평균)')
    return svg(fig, 'other')


def fig_pendulum(pend, models):
    shown = [m for m in models if all((m, c) in pend and pend[(m, c)]['limit_Nm'] == 20 for c in ('legacy', 'phase', 'hold50'))]
    conditions = [('legacy', '학습 분포의\n미사용 궤적'), ('phase', '감속·반전\n외란 (학습 밖)'), ('hold50', '사람 명령 50 ms\n유지 (학습 밖)')]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), layout='constrained')
    width = .8 / len(shown)
    x = np.arange(len(conditions))
    for i, m in enumerate(shown):
        rec = [pend[(m, c)] for c, _ in conditions]
        off = x + (i - (len(shown) - 1) / 2) * width
        axes[0].bar(off, [r['tracking_rmse_deg'] for r in rec], width=width * .92, color=COLORS[m], label=SHORT[m])
        axes[1].bar(off, [100 * (1 - r['wearer_tau_sq_Nm2'] / r['reference_tau_sq_Nm2']) for r in rec], width=width * .92, color=COLORS[m])
        axes[2].bar(off, [r['failures']['assisted'] for r in rec], width=width * .92, color=COLORS[m])
    titles = ['목표 각도 추종 RMSE [°] · 작을수록 좋음', '사람 토크² 감소율 vs 무보조 [%] · 클수록 좋음', '보조 진자 실패 수 (/256) · 작을수록 좋음']
    for ax, t in zip(axes, titles):
        ax.set_title(t, fontsize=10); ax.set_xticks(x, [c[1] for c in conditions], fontsize=8.5); ax.grid(axis='y', alpha=.15)
    axes[0].set_ylim(2, None); axes[2].set_yticks(range(0, 5)); axes[0].legend(fontsize=8, ncol=3)
    return svg(fig, 'pendulum'), shown


def fig_model(m, d):
    rows, base = d[m]['rows'], d['L1']['rows']
    fig, axes = plt.subplots(3, 2, figsize=(12, 9.5), layout='constrained', sharex=True)
    fig.suptitle(NAMES[m] + ' · ±20 N·m · 평가 seed 42/7 평균', fontsize=12)
    for side in (0, 1):
        red = arr(rows, side, 'reduction')
        ax = axes[0, side]
        ax.axhline(0, color='#64748b', lw=.7)
        ax.bar(X, red, width=3.6, color=np.where(red >= 0, '#237b5b', '#c64448'), label=SHORT[m] + ' A − B')
        if m != 'L1':
            ax.plot(X, arr(base, side, 'reduction'), '--', color='#555', label='L1 A − B')
        ax.set_title(SIDES[side]); ax.set_ylabel('착용자 RMS 감소량 [N·m]\n양수 = 도움'); ax.legend(fontsize=8)
        ax = axes[1, side]
        ax.plot(X, arr(rows, side, 'A_mean'), color='#333', ls='--', label='무보조 A')
        ax.plot(X, arr(rows, side, 'B_mean'), color=COLORS[m], lw=1.8, label=SHORT[m] + ' 보조 B')
        if m != 'L1':
            ax.plot(X, arr(base, side, 'B_mean'), color='#b17c21', ls=':', label='L1 보조 B')
        ax.set_ylabel('착용자 평균 토크 [N·m]\n+굴곡 / −신전'); ax.legend(fontsize=8)
        ax = axes[2, side]
        ax.plot(X, arr(rows, side, 'assist_mean'), color=COLORS[m], lw=1.8, label=SHORT[m] + ' 보조 평균')
        ax.fill_between(X, arr(rows, side, 'assist_min'), arr(rows, side, 'assist_max'), color=COLORS[m], alpha=.13, label='관측 최소–최대')
        if m != 'L1':
            ax.plot(X, arr(base, side, 'assist_mean'), color='#b17c21', ls=':', label='L1 보조 평균')
        for sign in (-1, 1):
            ax.axhline(sign * CAP, color='#c64448', ls='--', lw=.8)
        ax.set_ylim(-23, 23); ax.set_ylabel('장치 보조 토크 [N·m]'); ax.legend(fontsize=8)
        ax.set_xlabel('참조 보행 위상 [%] · 최대 굴곡 = 0%')
        for a in axes[:, side]:
            a.set_xlim(0, 100); a.set_xticks(np.arange(0, 101, 10)); a.grid(alpha=.15)
    return svg(fig, 'model_' + m)


def worst(rows):
    """Largest RMS increase and its bin over both legs."""
    best = max(((-r['reduction'], side, r) for side in (0, 1) for r in rows[side]), key=lambda x: x[0])
    return best[0], SIDES[best[1]], best[2]


def main():
    ev = collect()
    d, models = ev['data'], ev['models']
    M = {m: d[m]['metrics'] for m in d}
    v = lambda m, k: M[m][k]['mean']  # noqa: E731
    w_l1, w_l1_side, w_l1_bin = worst(d['L1']['rows'])
    w_r0, w_r0_side, w_r0_bin = worst(d['R0_s0']['rows'])
    nl, nr = d['L1']['need'][0], d['R0_s0']['need'][0]
    body = ['<nav><a href="../experiment_summary.html">실험 요약</a><a href="../HANDOFF.md">현재 상태</a>'
            '<a href="report.html">R0/R1 전체 보고서(표·명령)</a><a href="../261015_1815/walking_phase_report.html">앞선 23 N·m 위상 보고서</a></nav>',
            '<h1>상위 5개 LLP와 기준 L1 · 걷기에서 어디서 돕고, 어디서 방해하는가</h1>',
            '<p>진자에서만 학습한 보조 정책(LLP)을 동결된 가상 착용자 W1과 각도 예측기 H1에 연결해 기존 걷기 40초를 평가했다. '
            '같은 초기 조건의 무보조 A와 보조 B를 짝지어 착용자 고관절 토크를 비교한다. 모든 모델은 실제 출력 범위 ±20 N·m로 맞췄고, 평가 seed 42와 7의 평균이다. 저장 기록만 사용했으며 새 시뮬레이션은 없다.</p>']
    body.append('<div class="status"><strong>R0만 L1보다 높은 점수를 받았다.</strong> '
                f'R0는 저부담 감속 방해({v("R0_s0","low10_rms_increase"):.2f} vs L1 {v("L1","low10_rms_increase"):.2f} N·m)와 최악 위상 방해({v("R0_s0","worst_phase_rms_increase"):.2f} vs {v("L1","worst_phase_rms_increase"):.2f} N·m)를 절반 이하로 줄이면서 고관절 이득의 {100*v("R0_s0","overall_tau_sq_reduction")/v("L1","overall_tau_sq_reduction"):.0f} %를 유지했다. '
                f'다만 보조 RMS도 {v("R0_s0","assist_rms"):.1f} N·m로 L1({v("L1","assist_rms"):.1f})의 절반이다. 같은 R0를 배율 1로 쓰면 이득 {v("R0n_s0","overall_tau_sq_reduction"):.1f} (N·m)²에 방해 {v("R0n_s0","low10_rms_increase"):.2f} N·m로 L1과 같아진다. '
                f'무보조 토크가 거의 0인 구간에서 보조 방향은 모든 모델이 무작위 수준(R0 {100*nr["sign_agreement"]:.0f} %, L1 {100*nl["sign_agreement"]:.0f} %)이라 저부담 방해의 근본 원인은 아직 남아 있다.</div>')
    body.append('<div class="chips">' + ''.join(f'<span class="chip"><i style="background:{COLORS[m]}"></i>{escape(NAMES[m])}</span>' for m in models) + '</div>')
    body.append('<ul class="note">'
                '<li><b>L1</b>: 기존 기준. 학습 출력 40 tanh(a), 평가 때 0.5배. 학습 진자에서 사람 PD와 속도 피드포워드를 따로 ±150 N·m로 자르고, 보조·피드포워드·외란의 합을 다시 잘랐다(과거 토크 경로).</li>'
                '<li><b>R0</b>: L1과 같은 학습에서 토크 경로만 바꿨다. 사람 PD+피드포워드를 한 번에 ±150 N·m로 자르고, 보조와 외란은 따로 더한다.</li>'
                '<li><b>F1</b>: R0 + 목표 궤적이 바뀌는 구간을 2차 미분까지 연속인 곡선으로 잇고 그 구간의 관성 토크를 40 N·m 이하로 제한.</li>'
                '<li><b>R1</b>: R0 + 무보조 대비 토크 증가 비용. ' + escape(DEFINE_R1) + '</li>'
                '<li><b>F2</b>: F1 + 보조 모터가 흡수하는 파워(보조 토크와 각속도의 부호가 반대일 때)에 비용.</li>'
                '<li><b>G0</b>: ±23 N·m 출력, 3,000 epoch, 출력 크기 벌점(bounds loss)을 쓴 양측 공동 actor. 평가 때 20/23배.</li></ul>')
    rank_rows = ''.join(f'<tr><td>{i+1}</td><td>{escape(NAMES.get(m, m))}</td><td>{sc:+.3f}</td><td>{b:.1f}</td><td>{h:.2f}</td></tr>'
                        for i, (sc, m, b, h) in enumerate(ev['ranking']))
    rank_rows += f'<tr><td>기준</td><td>{escape(NAMES["L1"])}</td><td>+0.000</td><td>{v("L1","overall_tau_sq_reduction"):.1f}</td><td>{v("L1","low10_rms_increase"):.2f}</td></tr>'
    body.append('<details><summary>상위 5개를 고른 방법</summary><p>R0/R1 학습 전에 manifest에 고정한 점수를 그대로 썼다. ±20 N·m 동일 출력에서 '
                '<code>점수 = 고관절 τ² 감소 / L1 값 − 저부담 감속 RMS 증가 / L1 값</code>. L1은 0이다. 이득과 방해를 같은 무게로 보는 규칙이라, 보조가 작은 모델(F1, R1)도 상위에 들어온다. '
                'G1·G2·G3은 ±23 N·m에서 seed 42만 평가해 같은 출력 비교가 없으므로 순위에서 뺐다. 모든 신규 모델의 학습 seed는 0 하나다.</p>'
                '<div class="scroll"><table><tr><th>순위</th><th>모델</th><th>점수</th><th>고관절 τ² 감소 (N·m)²</th><th>저부담 감속 RMS 증가 N·m</th></tr>' + rank_rows + '</table></div></details>')

    body.append('<h2>핵심 결과 · R0는 방해 구간을 줄였지만, 보조를 키우면 다시 나타난다</h2>')
    body.append(f'<p>선이 0보다 위면 그 위상에서 착용자 토크 RMS가 늘었다(방해). L1의 가장 큰 방해는 {w_l1_side} {w_l1_bin["lo"]:.0f}–{w_l1_bin["hi"]:.0f}%의 <strong>{w_l1:.2f} N·m</strong>, '
                f'R0는 {w_r0_side} {w_r0_bin["lo"]:.0f}–{w_r0_bin["hi"]:.0f}%의 <strong>{w_r0:.2f} N·m</strong>다. 점선은 같은 R0를 배율 1·±20 클립으로 적용한 추가 평가로, 후반 방해가 L1 수준으로 돌아온다.</p>')
    body.append(fig_core(d))
    body.append('<p class="note">위상 0 %는 각 다리의 참조 최대 굴곡이다. 발 접촉 기준 위상이 아니므로 입각기·유각기 경계를 이 그림으로 단정하지 않는다. 4 % 구간마다 평가 seed 두 번의 값을 평균했다.</p>')

    body.append('<h2>1. 위상별 도움과 방해 · 6개 모델</h2><p>초록은 부담 감소, 빨강은 증가다. 각 행은 그 모델의 짝지은 무보조 A를 기준으로 계산했다. L1·G0·F1은 70–90 %에 진한 빨간 띠가 있고, R0·R1은 같은 구간이 옅다. F2는 그 띠가 거의 사라진 대신 왼쪽 24–36 %에 새 방해가 생겼고, G0는 오른쪽 20–28 %에도 방해가 있다. 큰 도움(진한 초록)은 L1·G0의 40–48 %에 몰려 있다.</p>')
    body.append(fig_heatmap(d, models))

    body.append('<h2>2. 핵심 지표 한눈에</h2><p>왼쪽 두 지표는 같은 방향으로 좋아지기 어렵다. 보조 RMS와 포화율을 함께 보면, 방해가 작은 모델은 대체로 보조 자체가 작다. L1과 G0는 출력의 절반 이상이 상한에 붙어 있다.</p>')
    body.append(fig_metrics(d, models))

    body.append('<h2>3. 이득과 방해의 교환 · 보조 크기를 맞춰 보기</h2>'
                '<p>점선은 L1의 출력만 ±10/15/20 N·m로 줄인 저장 결과다. 같은 방해에서 점선보다 위에 있으면 L1을 단순히 줄인 것보다 낫다. '
                f'R0 ±20은 점선보다 약간 위에 있고, 보조 RMS가 거의 같은 L1 ±10({ev["scaling"][10]["assist_rms"]["mean"]:.1f} N·m)보다 이득이 {v("R0_s0","overall_tau_sq_reduction")-ev["scaling"][10]["overall_tau_sq_reduction"]["mean"]:+.1f} (N·m)² 크다. '
                'F2와 G0는 점선의 연장선 근처에서 이득과 방해가 함께 크고, F1·R1은 점선 아래다.</p>')
    body.append(fig_tradeoff(d, models, ev['scaling']))

    body.append('<h2>4. 필요한 만큼만 돕는가 · 무보조 착용자 토크 구간별</h2>'
                '<p>무보조 A에서 착용자가 낸 토크 크기로 시간을 나눴다. L1은 필요 토크와 상관없이 거의 같은 크기로 밀고, R0는 필요 토크가 클수록 더 민다. '
                f'하지만 필요 토크가 5 N·m 미만인 구간(표본의 {100*nl["sample_fraction"]:.0f} %)에서 보조 방향은 모든 모델이 50 % 근처이고, 그 구간에서 착용자 RMS가 늘어난다. 이것이 저부담 방해의 직접 원인이다.</p>')
    body.append(fig_need(d, models))

    body_err = [d[m]['body_err_mm'] for m in models]
    body.append('<h2>5. 다른 관절로 부담이 옮겨갔나</h2><p>보조가 큰 L1·G0는 무릎·발목 부담도 많이 줄이지만 몸통(허리) 부담을 가장 많이 늘린다. R0의 몸통 증가는 L1의 절반 이하이고, F2·R1은 더 작다. R1은 무릎 부담도 거의 바꾸지 않는다. '
                f'몸체 추종 오차는 {min(body_err):.1f}–{max(body_err):.1f} mm, 전진 속도는 모두 {d["L1"]["speed"]:.3f} m/s로 같았고 낙상은 없었다.</p>')
    body.append(fig_other(d, models))

    pend_fig, shown = fig_pendulum(ev['pendulum'], models)
    body.append('<h2>6. 진자에서의 일반화 · 학습에 쓰지 않은 조건</h2>'
                '<p>실제 H1 예측을 연결한 진자 128환경 × 2묶음 × 10초, ±20 N·m. 진자에서는 출력이 큰 L1이 사람 토크를 가장 많이 줄이고 추종도 가장 좋다. '
                'R0는 진자 점수만으로는 뽑히지 않았을 모델이며, 진자 성능과 W1 전이가 같은 순서가 아님을 보여 준다. '
                'G0는 ±23 N·m·다른 평가 seed로만 진자 평가가 있어 이 그림에서 뺐다.</p>')
    body.append(pend_fig)

    body.append('<h2>7. 모델별 자세히</h2><label for="model">모델 선택 </label><select id="model" onchange="document.querySelectorAll(\'.model\').forEach(e=>e.hidden=e.dataset.model!==this.value)">'
                + ''.join(f'<option value="{m}">{escape(NAMES[m])}</option>' for m in models[1:] + ['L1']) + '</select>')
    for i, m in enumerate(models[1:] + ['L1']):
        met = M[m]
        wv, ws, wb = worst(d[m]['rows'])
        body.append(f'<section class="model" data-model="{m}"' + (' hidden' if i else '') + '>')
        body.append(f'<p><strong>{escape(NAMES[m])}</strong> · 고관절 τ² 감소 {met["overall_tau_sq_reduction"]["mean"]:.1f} (N·m)² · 저부담 감속 RMS 증가 {met["low10_rms_increase"]["mean"]:.2f} N·m · '
                    f'가장 큰 방해 {ws} {wb["lo"]:.0f}–{wb["hi"]:.0f}% {wv:.2f} N·m · 보조 RMS {met["assist_rms"]["mean"]:.2f} N·m · 포화 {met["saturation_pct"]["mean"]:.1f} % · '
                    f'몸체 오차 {d[m]["body_err_mm"]:.1f} mm · 속도 {d[m]["speed"]:.3f} m/s · 낙상 {d[m]["falls"]}/2.</p>')
        body.append(fig_model(m, d))
        body.append('</section>')
    body.append('<p class="note">첫 줄: 착용자 RMS 감소량(초록 = 도움, 빨강 = 방해). 둘째 줄: 부호 있는 착용자 평균 토크. 셋째 줄: 장치 보조 평균과 관측 최소–최대(음영, 신뢰구간 아님); 빨간 점선은 ±20 N·m 한계. 비교선은 L1이다.</p>')

    body.append('<details><summary>해석 범위</summary><ul>'
                '<li>기존 걷기 한 클립, 가상 착용자 W1 하나, 평가 seed 두 번이다. 다른 사람·보행·실기기 성능을 보증하지 않는다.</li>'
                '<li>신규 모델(R0, R1, F1, F2)의 학습 seed는 0 하나다. 같은 설계의 반복 학습 산포는 아직 모른다.</li>'
                '<li>토크²는 시뮬레이션 부담 지표이며 근활성·대사량 측정값이 아니다.</li>'
                '<li>"배율 1·±20 클립"은 사전 등록하지 않은 추가 평가다. 같은 정책의 출력 크기만 바꿨다.</li>'
                '<li>저부담 감속 창은 과거 L1 ±20의 같은 평가 seed 무보조 기록에서 고정했다(50 ms RMS ≤ 10 N·m, 감속 중).</li></ul></details>')
    body.append('<details><summary>재현 경로</summary><ul>'
                '<li>이 페이지: <code>python examples/experiments/assist_pendulum/report_relative_phase.py</code></li>'
                '<li>W1 기록: <code>results/llp_relative_20261020/walking/comparison.json</code>, 추가 평가 <code>results/llp_relative_20261020/walking_native/comparison.json</code></li>'
                '<li>진자 기록: <code>docs/ai_sbc/261020_relative/evidence.json</code>의 pendulum 항목</li>'
                '<li>모델: R0 <code>results/llp_relative_R0_s0_20261020/epoch_6000.ckpt</code>, R1 <code>results/llp_relative_R1_s0_20261020/epoch_6000.ckpt</code>, '
                'F1/F2 <code>results/llp_feasible_F{1,2}_s0_20261020/epoch_6000.ckpt</code>, G0 <code>results/llp_general23_G0_s0_20261015_all/epoch_3000.ckpt</code>, L1 <code>checkpoints/L1_llp_pendulum/last.ckpt</code></li></ul></details>')
    body.append('<p class="note">생성 UTC ' + datetime.now(timezone.utc).isoformat(timespec='seconds') + '</p>')
    evidence = dict(selection_rule='score = benefit/benefit_L1 - low10/low10_L1 at ±20 N·m (fixed in results/llp_relative_20261020/manifest.json before R0/R1 training)',
                    ranking=ev['ranking'], excluded_without_matched_20=ev['excluded_without_matched_20'],
                    models={m: dict(name=NAMES.get(m, m), metrics=d[m]['metrics'], need_bins=d[m]['need'], other_joints=d[m]['other'],
                                    phase_rows=d[m]['rows'], paths=d[m]['paths']) for m in d},
                    L1_scaling={c: ev['scaling'][c] for c in ev['scaling']},
                    pendulum={f'{m}/{c}': {k: p[k] for k in ('tracking_rmse_deg', 'assist_rms_Nm', 'wearer_tau_sq_Nm2', 'reference_tau_sq_Nm2', 'failures', 'limit_Nm')}
                              for (m, c), p in ev['pendulum'].items()})
    body.append('<script type="application/json" id="evidence">' + json.dumps(evidence, ensure_ascii=False, allow_nan=False).replace('</', '<\\/') + '</script>')
    html = ('<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>상위 5개 LLP 위상 비교</title><style>' + CSS + '</style></head><body><main>' + ''.join(body) + '</main></body></html>')
    OUT.write_text(html)
    print(f'wrote {OUT.relative_to(ROOT)} ({len(html)/1e6:.1f} MB); top5 = {[r[1] for r in ev["ranking"][:5]]}')


if __name__ == '__main__':
    main()
