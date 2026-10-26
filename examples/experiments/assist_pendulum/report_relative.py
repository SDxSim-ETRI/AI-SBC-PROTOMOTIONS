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
"""Evidence collection and Korean HTML report for the R0/R1 pendulum pretraining study."""
import argparse
from datetime import datetime, timezone
from html import escape
import io
import json
from pathlib import Path
import shlex
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_feasible import all_joint_metrics, pendulum_record, read, save, stats  # noqa: E402
from analyze_need_bins import walking as need_bins_walking  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / 'results/pendulum_general23_20261015_all'
FEASIBLE = ROOT / 'results/llp_feasible_20261020'
COLORS = {'L1': '#374151', 'G0_s0': '#1a8b67', 'G4_s0': '#8b70b2', 'G4_s1': '#b39bcf',
          'F1_s0': '#c48a2a', 'F2_s0': '#e0b05a', 'R0_s0': '#2767ad', 'R1_s0': '#c2410c'}
LABELS = {'L1': 'L1 · 기존 기준(과거 토크 경로)', 'G0_s0': 'G0 · 23 N·m 공동', 'G4_s0': 'G4 · 비용 강화 s0', 'G4_s1': 'G4 · 비용 강화 s1',
          'F1_s0': 'F1 · 토크 경로+C2', 'F2_s0': 'F2 · F1+흡수 파워', 'R0_s0': 'R0 · 토크 경로만(대조)', 'R1_s0': 'R1 · R0+무보조 대비 토크 증가 비용'}
TAGS = ['info/episode_length', 'rewards/unnormalized_task_rewards', 'env/raw_r/tracking_mean',
        'env/raw_r/assist_effort_mean', 'env/raw_r/human_torque_sq_mean', 'env/raw_r/human_power_mean',
        'env/raw_r/paired_excess_mean', 'env/scaled_r/paired_excess_mean', 'env/raw_r/action_rate_mean',
        'env/scaled_r/tracking_mean', 'env/scaled_r/human_power_mean', 'env/scaled_r/human_torque_sq_mean',
        'env/scaled_r/assist_effort_mean', 'env/scaled_r/action_rate_mean',
        'env/paired/reference_done_fraction', 'env/paired/reference_only_truncation_fraction', 'env/paired/assisted_failure_fraction',
        'env/assist/injection_error_max_Nm', 'env/assist/injection_clipped_fraction', 'env/assist/limit_fraction',
        'env/termination/tracking_failure_mean', 'env/termination/dof_vel_failure_mean',
        'losses/actor_loss', 'losses/critic_loss', 'actor/bad_grads_count', 'critic/bad_grads_count',
        'times/last_epoch_seconds', 'actor/clip_frac', 'actor/update_skipped']


def training_record(directory):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    series = {}
    for version in sorted((directory / 'lightning_logs').glob('version_*')):
        a = EventAccumulator(str(version), size_guidance={'scalars': 0}); a.Reload()
        for tag in TAGS:
            if tag in a.Tags()['scalars']:
                series.setdefault(tag, []).extend([[int(e.step), float(e.value)] for e in a.Scalars(tag)])
    summary = {}
    for tag, values in series.items():
        values = sorted(dict(values).items()); series[tag] = values
        a = np.array([v for _, v in values]); finite = np.isfinite(a)
        summary[tag] = dict(count=len(a), nonfinite=int((~finite).sum()),
            first100_mean=float(a[:100].mean()) if finite.all() else None,
            last500_mean=float(a[-500:].mean()) if finite.all() else None,
            max=float(a.max()) if finite.all() else None, last=float(a[-1]) if finite.all() else None)
        # Keep the full statistics above; store every 5th epoch of the curve to bound the evidence size.
        series[tag] = values[::5] if len(values) > 1000 else values
    return dict(directory=str(directory.relative_to(ROOT)), summary=summary, series=series)


def collect(study):
    plan = read(study / 'manifest.json'); status = read(study / 'status.json')
    wpath = study / 'walking/comparison.json'
    walking = read(wpath) if wpath.exists() else read(FEASIBLE / 'walking/comparison.json')
    if wpath.exists():
        walking['note'] = 'Stored L1/G0/G4 (2026-10-16) and F1/F2 (2026-10-20) runs plus newly pendulum-trained R0/R1. Two evaluation seeds give means and observed ranges, not confidence intervals. W1 and H1 frozen.'
        walking['plan']['purpose'] = 'Evaluate freshly pendulum-trained R0/R1 at matched actual torque limits against stored L1/G0/G4/F1/F2 runs.'
        save(wpath, walking)
    training = {}
    for directory in sorted((ROOT / 'results').glob('llp_relative_R*_s*_20261020')):
        label = directory.name.removeprefix('llp_relative_').removesuffix('_20261020')
        training[label] = training_record(directory)
    pendulum = []
    for model, prefix in [('L1', 'L1_test'), ('G0_s0', 'G0_s0_dev'), ('G4_s0', 'G4_s0_test'), ('G4_s1', 'G4_s1_test')]:
        for condition in ('legacy', 'phase', 'hold50'):
            pendulum.append(pendulum_record(OLD / f'{prefix}_H1_{condition}', model, condition, True))
    for path in sorted((FEASIBLE / 'pendulum').glob('*_H1_*')):
        if (path / 'summary.json').exists():
            model, condition = path.name.split('_H1_')
            pendulum.append(pendulum_record(path, model, condition, True))
    for path in sorted((study / 'pendulum').glob('*_H1_*')):
        if (path / 'summary.json').exists():
            model, condition = path.name.split('_H1_')
            pendulum.append(pendulum_record(path, model, condition, False))
    for r in walking['runs'].values():
        r['all_rotational_joints'] = all_joint_metrics(r)
        r['need_bins'] = need_bins_walking(ROOT / r['record']['path'] / 'traces.npz', r['record']['limit_Nm'])
    for group in walking['groups'].values():
        runs = [r for r in walking['runs'].values() if r['record']['model'] == group['model'] and r['record']['limit_Nm'] == group['limit_Nm']]
        group['metrics']['body_error_mm'] = stats([1000 * r['summary']['B']['body_err'] for r in runs])
        group['metrics']['all_rotational_sum_tau_sq_reduction'] = stats([r['all_rotational_joints']['reduction_Nm2'] for r in runs])
        group['other_joints'] = {name: stats([next(j['B'] - j['A'] for j in r['other_joints'] if j['group'] == name) for r in runs])
                                 for name in ('전체 비보조 회전관절', '무릎', '발목', '몸통')}
        bins = {}
        for lo in (0, 5, 10, 20, 40, 80):
            rows = [b for r in runs for b in r['need_bins']['bins'] if b['need_lo_Nm'] == lo]
            if rows:
                bins[str(lo)] = {k: float(np.mean([b[k] for b in rows])) for k in rows[0] if isinstance(rows[0][k], float)}
        group['need_bins'] = bins
    comparison = []
    for key, g in walking['groups'].items():
        if not g['model'].startswith('R') or g['limit_Nm'] not in (15, 20): continue
        b = walking['groups'][f'L1_{g["limit_Nm"]}']; m = g['metrics']; bm = b['metrics']
        delta = {name: m[name]['mean'] - bm[name]['mean'] for name in m if name in bm}
        joint_primary = (delta['overall_tau_sq_reduction'] >= 0 and delta['low10_rms_increase'] <= 0
                         and delta['worst_phase_rms_increase'] <= 0 and g['falls'] == 0)
        guards = (delta['body_error_mm'] <= 1 and abs(delta['speed']) <= .02 and delta['all_rotational_sum_tau_sq_reduction'] >= 0)
        ps = [p for p in pendulum if p['model'] == g['model']]
        complete = len(ps) == 3
        base = {p['condition']: p for p in pendulum if p['model'] == 'L1'}
        stable = complete and all(p['failures']['assisted'] <= base[p['condition']]['failures']['assisted'] for p in ps)
        no_worse = complete and all(p['tracking_rmse_deg'] <= base[p['condition']]['tracking_rmse_deg'] for p in ps)
        entry = dict(model=g['model'], limit_Nm=g['limit_Nm'], delta_vs_L1=delta,
            walking_primary_jointly_better=bool(joint_primary), walking_other_guards=bool(guards),
            pendulum_complete=complete, pendulum_stable=bool(stable), pendulum_tracking_no_worse=bool(no_worse),
            all_criteria_met=bool(joint_primary and guards and stable and no_worse))
        if g['model'] == 'R1_s0' and f'R0_s0_{g["limit_Nm"]}' in walking['groups']:
            r0 = walking['groups'][f'R0_s0_{g["limit_Nm"]}']['metrics']
            entry['delta_vs_R0'] = {name: m[name]['mean'] - r0[name]['mean'] for name in m if name in r0}
        comparison.append(entry)
    native = None
    if (study / 'walking_native/comparison.json').exists():
        n = read(study / 'walking_native/comparison.json')
        native = dict(groups=n['groups'], runtime=read(study / 'walking_native/execution.json') if (study / 'walking_native/execution.json').exists() else {},
                      note='Supplementary, outside the pre-registered protocol: scale 1 with the controller clip at ±20 N·m (same actual range, different mapping). Masks reuse the stored L1 ±20 references.')
        for key, g in native['groups'].items():
            runs = [r for r in n['runs'].values() if r['record']['model'] == g['model'] and r['record']['limit_Nm'] == g['limit_Nm']]
            g['metrics']['body_error_mm'] = stats([1000 * r['summary']['B']['body_err'] for r in runs])
            g['other_joints'] = {name: stats([next(j['B'] - j['A'] for j in r['other_joints'] if j['group'] == name) for r in runs])
                                 for name in ('전체 비보조 회전관절', '무릎', '발목', '몸통')}
            g['low10_absorbed_power_W'] = float(np.mean([r['segments']['low_decel_10']['absorbed_power_per_joint_W'] for r in runs]))
    ev = dict(generated_utc=datetime.now(timezone.utc).isoformat(), plan=plan, status=status, walking_native=native,
        need_bins_L1=read(study / 'l1_need_bins.json') if (study / 'l1_need_bins.json').exists() else None,
        config_audit={k: v for k, v in read(study / 'config_audit.json').items() if k != 'configs'} if (study / 'config_audit.json').exists() else None,
        training=training, pendulum=pendulum, walking=walking, comparison=comparison,
        execution=read(study / 'execution.json') if (study / 'execution.json').exists() else {},
        runtime_failures=read(study / 'runtime_failures.json') if (study / 'runtime_failures.json').exists() else [],
        repeat_selection=read(study / 'repeat_selection.json') if (study / 'repeat_selection.json').exists() else None,
        limitations=['기존 걷기 한 클립·착용자 W1 하나. 평가 seed 42/7은 독립 학습 seed나 다른 사람이 아니다.',
            'L1/G0/G4/F1/F2 결과는 저장 기록 재사용이며 같은 seed의 Newton 재실행은 bit-동일하지 않다.',
            'L1 진자 기준은 ±20 N·m만 있다. G0/G4 진자 기록은 ±23 N·m이며 G0는 코호트 seed도 다르다.',
            'R0는 L1과 토크 경로만 다르다. R1은 R0와 보상항 하나(무보조 대비 토크 증가 비용)와 무보조 진자가 실패할 때 에피소드를 끊는 처리만 다르다. F1 = R0 + C2 전환.',
            '모든 신규 모델은 처음부터 6,000 epoch(786,432,000 표본). G0/G4는 3,000 epoch.',
            '반복 규칙은 학습 전에 고정했다. 보고서의 공동 기준 판정은 서술적 Pareto·가드 비교이며 통계 검정이 아니다. 진자 실패 수는 조건별로 L1을 넘지 않아야 한다.',
            '평가 2회의 평균·범위이며 신뢰구간이 아니다. 토크 제곱은 시뮬레이션 부담 지표이며 대사량이 아니다.'])
    save(study / 'evidence.json', ev)
    return ev


def render(ev, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family': font.get_name(), 'font.size': 10, 'axes.unicode_minus': False, 'svg.fonttype': 'path',
                         'axes.spines.top': False, 'axes.spines.right': False})
    destination.parent.mkdir(parents=True, exist_ok=True)
    figures = destination.parent / 'figures'; figures.mkdir(exist_ok=True)

    def fig_html(fig, name):
        fig.savefig(figures / f'{name}.png', dpi=155, bbox_inches='tight')
        buf = io.StringIO(); fig.savefig(buf, format='svg', bbox_inches='tight'); plt.close(fig)
        return '<div class="plot">' + buf.getvalue()[buf.getvalue().index('<svg'):] + '</div>'

    def table(headers, rows):
        return ('<div class="scroll"><table><thead><tr>' + ''.join('<th>' + escape(str(h)) + '</th>' for h in headers)
                + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + str(c) + '</td>' for c in row) + '</tr>' for row in rows) + '</tbody></table></div>')

    groups = [g for g in ev['walking']['groups'].values() if g['limit_Nm'] in (15, 20)]
    order = list(COLORS)
    groups.sort(key=lambda g: (g['limit_Nm'], order.index(g['model']) if g['model'] in order else 99))
    G = ev['walking']['groups']
    def val(g, k): return g['metrics'][k]['mean']
    def gm(model, cap, k): return val(G[f'{model}_{cap}'], k)
    complete = ev['status']['stage'] == 'all planned training and evaluations complete'
    have = {m for m in ('R0_s0', 'R1_s0') if f'{m}_20' in G and f'{m}_15' in G}
    body = ['<nav><a href="../experiment_summary.html">실험 요약</a> · <a href="../HANDOFF.md">현재 상태</a> · <a href="../AI_SBC_VERSIONS.md">모델 경로</a> · <a href="../261020_feasible/report.html">앞선 F1/F2</a> · <a href="phase_report.html">상위 5개 위상 그래프</a></nav>',
            '<h1>LLP 진자 사전학습 R0/R1 · 올바른 토크 경로 대조와 무보조 대비 토크 증가 비용</h1>',
            '<p class="muted">2026-10-20 · 학습 seed s0와 평가 seed 42/7을 구분한다. L1/G0/G4/F1/F2는 저장 기록을 재사용했고 재실행하지 않았다.</p>']
    fresh_walks = sum(r['record']['model'].startswith('R') for r in ev['walking']['runs'].values())
    fresh_pend = sum(not p['reused_baseline'] for p in ev['pendulum'])
    body.append(f'<p>신규 PPO 학습 {len(ev["training"])}개 · 신규 W1 평가 {fresh_walks}건 · 신규 진자 평가 {fresh_pend}건.</p>')
    accepted = [x for x in ev['comparison'] if x['all_criteria_met']]
    if not complete:
        headline = '계산 진행 중 — 완료된 평가만 표시하며 성능 결론은 보류한다.'
    elif accepted:
        headline = '같은 출력에서 공동 기준(이득·저부담·최악 위상·추종·낙상·진자 안정성)을 모두 통과한 설정: ' + ', '.join(f'{x["model"]} ±{x["limit_Nm"]} N·m' for x in accepted) + '. 학습 seed 하나의 결과이므로 반복 학습으로 확인해야 한다.'
    else:
        headline = 'PPO 학습과 전이 평가를 완료했다. 같은 출력에서 평균 이득·저부담 감속·최악 위상·진자 추종·안정성을 모두 L1 이상으로 만든 설정은 없다. 지표별 이득과 손실은 아래 표에서 구분한다.'
    body.append('<div class="callout">' + headline + '</div>')
    if complete and have == {'R0_s0', 'R1_s0'}:
        lines = []
        for cap in (20, 15):
            lines.append(f'±{cap} N·m: 고관절 τ² 감소 L1 {gm("L1",cap,"overall_tau_sq_reduction"):.2f} / R0 {gm("R0_s0",cap,"overall_tau_sq_reduction"):.2f} / R1 {gm("R1_s0",cap,"overall_tau_sq_reduction"):.2f} (N·m)², '
                         f'저부담 감속 RMS 증가 L1 {gm("L1",cap,"low10_rms_increase"):.2f} / R0 {gm("R0_s0",cap,"low10_rms_increase"):.2f} / R1 {gm("R1_s0",cap,"low10_rms_increase"):.2f} N·m, '
                         f'최악 위상 RMS 증가 L1 {gm("L1",cap,"worst_phase_rms_increase"):.2f} / R0 {gm("R0_s0",cap,"worst_phase_rms_increase"):.2f} / R1 {gm("R1_s0",cap,"worst_phase_rms_increase"):.2f} N·m, '
                         f'보조 RMS L1 {gm("L1",cap,"assist_rms"):.2f} / R0 {gm("R0_s0",cap,"assist_rms"):.2f} / R1 {gm("R1_s0",cap,"assist_rms"):.2f} N·m.')
        body.append('<p><strong>핵심 수치.</strong> ' + ' '.join(lines) + '</p>')
    body.append('<h2>1. 기존 L1/G0/G4에서 확인한 문제</h2>')
    nb = ev.get('need_bins_L1')
    if nb:
        pend = nb['pendulum_L1_H1_legacy']; w42 = nb['walking']['L1_20Nm_seed42']
        body.append('<p>저장 기록을 무보조 착용자 토크 |τ_A| 구간으로 나누어 L1의 출력을 보았다 (진자: 실제 H1·±20 N·m·legacy 조건, W1: 기존 걷기 ±20 N·m seed 42). '
                    '착용자가 거의 토크를 내지 않는 구간(|τ_A|&lt;5 N·m)이 진자 표본의 '
                    f'{100*pend["bins"][0]["sample_fraction"]:.1f}%, W1의 {100*w42["bins"][0]["sample_fraction"]:.1f}%인데, 그 구간에서 L1의 보조 부호는 무보조 토크와 '
                    f'{100*pend["bins"][0]["sign_agreement"]:.0f}% / {100*w42["bins"][0]["sign_agreement"]:.0f}% 일치(무작위 수준)했고 보조 RMS는 '
                    f'{pend["bins"][0]["assist_rms_Nm"]:.1f} / {w42["bins"][0]["assist_rms_Nm"]:.1f} N·m였다. 그 결과 착용자 τ²는 '
                    f'{pend["bins"][0]["A_tau_sq_Nm2"]:.1f}→{pend["bins"][0]["B_tau_sq_Nm2"]:.1f} (진자), {w42["bins"][0]["A_tau_sq_Nm2"]:.1f}→{w42["bins"][0]["B_tau_sq_Nm2"]:.1f} (W1) (N·m)²로 늘었다. '
                    '즉 저부담 손해는 전신 전이에서만 생기는 것이 아니라 진자 학습 분포 안에서도 이미 존재하며, |τω|·τ² 절대 비용은 큰 하중 구간이 수익 분산을 지배해 저부담 구간의 정밀도를 거의 평가하지 않는다.</p>')
        rows = []
        for b_p, b_w in zip(pend['bins'], w42['bins']):
            rows.append([f'{b_p["need_lo_Nm"]}–{b_p["need_hi_Nm"]}', f'{100*b_p["sample_fraction"]:.1f}% / {100*b_w["sample_fraction"]:.1f}%',
                         f'{b_p["assist_rms_Nm"]:.1f} / {b_w["assist_rms_Nm"]:.1f}', f'{100*b_p["saturation_fraction"]:.0f}% / {100*b_w["saturation_fraction"]:.0f}%',
                         f'{100*b_p["sign_agreement"]:.0f}% / {100*b_w["sign_agreement"]:.0f}%',
                         f'{b_p["A_tau_sq_Nm2"]:.0f}→{b_p["B_tau_sq_Nm2"]:.0f} / {b_w["A_tau_sq_Nm2"]:.0f}→{b_w["B_tau_sq_Nm2"]:.0f}'])
        body.append(table(['|τ_A| 구간 N·m', '표본 비율 진자 / W1', 'L1 보조 RMS N·m', '포화율', '보조·무보조 토크 부호 일치', '착용자 τ² 무보조→보조 (N·m)²'], rows))
        c = pend['stored_L1_costs_per_step']; e = pend['excess_floor10_window5']
        body.append(f'<p>L1의 저장 진자 기록에서 기존 사람 비용의 크기는 |τω| {c["human_power"]:.3f} + τ² {c["human_torque_sq"]:.3f} = {c["total"]:.3f}/step였다. 같은 기록에서 새 토크 증가 비용항(하한 10 N·m, 50 ms 창)의 평균은 {e["mean"]:.3f}(중앙값 {e["p50"]:.3f}, 99% {e["p99"]:.2f})이므로, R1의 가중치 −0.5는 새 항의 크기를 기존 사람 비용 합과 같게 맞춘 값이다. 실행 전에 고정했고 결과를 보고 조정하지 않았다.</p>')
    body.append('<p>G0/G4(±23 N·m, 3,000 epoch, bounds loss, G4는 관절별 공유 actor·비용 4배)와 F1/F2(올바른 토크 경로+C2 전환, F2는 흡수 파워 비용)는 모두 같은 출력에서 L1의 저부담 손해를 평균 이득 손실 없이 줄이지 못했다. F1의 낮은 이득이 C2 전환 때문인지 토크 경로 때문인지는 분리되지 않았고(앞선 보고서 후속 1), 저부담 손해를 직접 비용화하는 실험도 없었다(후속 2). 이번 R0/R1이 그 두 가지다.</p>')
    body.append('<h2>2. 이번 변경: L1 → R0 → R1, 한 번에 한 가지만 바꿈</h2>')
    body.append('<p>R0는 L1 학습에서 토크 경로(사람 PD+피드포워드의 토크 제한과 보조·외란을 더하는 방식)만 바로잡았다. R1은 R0에 무보조 대비 토크 증가 비용을 더했다. '
                '학습 중 같은 목표 궤적·관성·외란을 받는 무보조 진자를 함께 계산하고, 보조 진자의 착용자 토크²가 무보조보다 커진 만큼을 (무보조 착용자 토크²의 최근 50 ms 평균 + 10²)로 나눠 벌점으로 준다. 무보조보다 줄어든 경우는 벌점 0이다.</p>')
    body.append(table(['항목', 'L1', 'R0 (대조)', 'R1 (후보)'], [
        ['actor·관측·action', '공동 MLP 512–256–128, 137차원 관측, 40 tanh(a)', 'L1과 동일', 'L1과 동일 (새 actor 입력 없음)'],
        ['진자·궤적·외란', 'v4 무작위화(관성 ×1–25, ω_n 12–24, 입각 하중·섭동, 정지·블렌드)', 'L1과 동일', 'L1과 동일'],
        ['토크 경로', '사람 PD와 FF를 따로 제한하고 보조+FF+외란 합에 착용자 한계를 재적용', '사람 PD+FF를 한 actuator에서 한 번 제한, 보조·외란 독립 주입 (G/F와 동일)', 'R0와 동일'],
        ['보상', '추종 exp(−40 e²); 사람 |τω| −0.0016, τ² −3e−5; 장치 τ² −2e−5; action 변화 −0.005', 'L1과 동일', 'L1 항 유지 + 무보조 대비 토크 증가 비용 −0.5·mean_j relu(τ_B²−τ_A²)/(mean_50ms τ_A²+10²), step당 −3 클립'],
        ['학습 환경', 'AssistEnv', 'AssistEnv', 'PairedAssistEnv: 같은 의도·관성·게인·외란을 받는 무보조 진자를 함께 계산(같은 난수 사용 확인), 보상에만 사용'],
        ['학습량·seed', '6,000 epoch·4096환경·seed 0, 주기적 평가 200 epoch', '6,000 epoch·786,432,000 표본·seed 0, 주기 평가 없음', 'R0와 동일'],
        ['W1 출력', '0.375배=±15 / 0.5배=±20', '동일', '동일']]))
    if ev.get('config_audit'):
        ca = ev['config_audit']
        body.append('<details><summary>저장 설정 감사 (resolved_configs.pt 비교)</summary><p>R0−L1: ' + escape(', '.join(d['path'] for d in ca['differences_from_L1']['R0'])) + '</p><p>R1−R0: ' + escape(', '.join(d['path'] for d in ca['R1_minus_R0'])) + '</p><p>R0−F1: ' + escape(', '.join(d['path'] for d in ca['R0_minus_F1'])) + '</p></details>')
    body.append('<h2>3. W1 전이: 같은 출력 범위에서 이득과 방해</h2><p>동결 W1/H1, 기존 걷기 40초, 10 ms 보조, 평가 seed 42/7 평균. 고관절 τ² 감소는 클수록, 저부담 감속·최악 위상 RMS 증가는 작을수록 좋다. 회전관절 합은 고관절과 비보조 관절을 모두 포함한다.</p>')
    rows = []
    for g in groups:
        rows.append([g['model'], g['limit_Nm'], f'{val(g,"overall_tau_sq_reduction"):.2f}', f'{val(g,"low10_rms_increase"):.2f}',
                     f'{val(g,"worst_phase_rms_increase"):.2f}', f'{val(g,"assist_rms"):.2f}', f'{val(g,"saturation_pct"):.1f}%',
                     f'{val(g,"body_error_mm"):.2f}', f'{val(g,"speed"):.3f}', f'{val(g,"all_rotational_sum_tau_sq_reduction"):.1f}', g['falls']])
    body.append(table(['모델', '±N·m', '고관절 τ² 감소', '저부담 RMS 증가', '최악 위상 RMS 증가', '보조 RMS', '포화율', '몸체 오차 mm', '속도 m/s', '회전관절 합 τ² 감소', '낙상'], rows))
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    extra = [G['L1_10']] if 'L1_10' in G else []
    native_groups = [g for g in (ev.get('walking_native') or {}).get('groups', {}).values() if 'n_s0' in g['model']]
    for g in groups + extra + native_groups:
        m = g['metrics']; x = m['low10_rms_increase']; y = m['overall_tau_sq_reduction']
        native = g['model'].endswith('n_s0'); color = COLORS.get(g['model'].replace('n_s0', '_s0'), 'gray')
        fmt = '*' if native else ('o' if g['limit_Nm'] == 20 else ('^' if g['limit_Nm'] == 15 else 's'))
        ax.errorbar(x['mean'], y['mean'], xerr=[[x['mean'] - x['min']], [x['max'] - x['mean']]], yerr=[[y['mean'] - y['min']], [y['max'] - y['mean']]],
                    fmt=fmt, color=color, capsize=3, ms=11 if native else 7, mfc='white' if (native or g['limit_Nm'] == 10) else color)
        label = f'{g["model"].replace("n_s0", " native")} {g["limit_Nm"]}'
        ax.annotate(label, (x['mean'], y['mean']), xytext=(5, 5), textcoords='offset points', fontsize=8)
    if 'L1_10' in G:
        pts = sorted((G[f'L1_{k}']['metrics']['low10_rms_increase']['mean'], G[f'L1_{k}']['metrics']['overall_tau_sq_reduction']['mean']) for k in (10, 15, 20))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ls='--', color=COLORS['L1'], lw=1, alpha=.6, label='L1 출력 축소 곡선 (±10/15/20)')
        ax.legend(fontsize=8, loc='lower right')
    ax.set(xlabel='저부담 감속 RMS 증가 [N·m] — 작을수록 좋음', ylabel='전체 고관절 τ² 감소 [(N·m)²] — 클수록 좋음',
           title='동일 출력 범위 비교 · ○ ±20, △ ±15, □ ±10 N·m, ☆ 배율 1·±20 클립(추가 평가) · 오차막대는 평가 2회 범위'); ax.grid(alpha=.2)
    body.append(fig_html(fig, 'walking_tradeoff'))
    pts = []
    for g in groups + extra + native_groups:
        m = g['metrics']; pts.append((m['assist_rms']['mean'], g['model'].replace('n_s0', ' native'), g['limit_Nm'], m['overall_tau_sq_reduction']['mean'], m['low10_rms_increase']['mean'], m['worst_phase_rms_increase']['mean'], m['saturation_pct']['mean']))
    body.append('<h3>보조 크기를 맞춘 비교</h3><p>같은 출력 상한이라도 실제 보조 RMS는 모델마다 다르다. 저부담 손해는 보조 크기와 함께 커지므로, 모든 모델·출력 조건을 보조 RMS 순으로 나열하고 L1의 출력 축소 곡선(±10/15/20 N·m 저장 기록)과 비교한다. 곡선 위에 있으면 같은 손해에서 이득이 더 큰 것이다.</p>')
    body.append(table(['모델', '±N·m', '보조 RMS', '고관절 τ² 감소', '저부담 RMS 증가', '최악 위상 RMS 증가', '포화율'],
                      [[p[1], p[2], f'{p[0]:.2f}', f'{p[3]:.2f}', f'{p[4]:.2f}', f'{p[5]:.2f}', f'{p[6]:.1f}%'] for p in sorted(pts)]))
    body.append('<p>저부담 마스크는 과거 L1 ±20 N·m의 같은 평가 seed 무보조 기록에서 고정했다 (50 ms 중심 이동 RMS≤10 N·m, ωα&lt;0, |ω|&gt;0.3 rad/s, |α|&gt;1 rad/s²). 보조 명령 i−1과 착용자 응답 i를 맞춘다. 모든 모델에 같은 창을 쓴다.</p>')
    body.append('<h3>저부담 상태에서 무엇이 달라졌나 (무보조 토크 구간별)</h3><p>각 모델의 짝지은 무보조 A 토크 |τ_A| 구간별 보조 RMS·부호 일치·착용자 τ² 변화. 1절의 L1 분석과 같은 계산이며 평가 seed 2회 평균이다.</p>')
    for cap in (20, 15):
        rows = []
        for model in ('L1', 'R0_s0', 'R1_s0', 'F2_s0'):
            g = G.get(f'{model}_{cap}')
            if not g or not g.get('need_bins'): continue
            for lo in ('0', '5', '10', '20', '40', '80'):
                b = g['need_bins'].get(lo)
                if not b: continue
                rows.append([model if lo == '0' else '', f'{lo}–', f'{100*b["sample_fraction"]:.1f}%', f'{b["assist_rms_Nm"]:.1f}', f'{100*b["saturation_fraction"]:.0f}%', f'{100*b["sign_agreement"]:.0f}%', f'{b["A_tau_sq_Nm2"]:.0f}→{b["B_tau_sq_Nm2"]:.0f}'])
        body.append(f'<p>±{cap} N·m</p>' + table(['모델', '|τ_A| N·m', '표본', '보조 RMS', '포화율', '부호 일치', '착용자 τ² A→B'], rows))
    if ev.get('walking_native'):
        ng = ev['walking_native']['groups']
        body.append('<h3>추가 평가 (사전 등록 외): 출력 변환을 배율 1·±20 N·m 자르기로 바꾼 경우</h3><p>R0/R1은 ±20 N·m에서 출력 포화가 0.5 % 이하라 0.5배 변환이 요청 토크를 그대로 절반으로 줄인다. 같은 실제 범위 ±20 N·m에서 요청 토크를 그대로 적용하고 20 N·m 초과만 자르는 방식(제어기 <code>--assist-scale 1 --torque-limit 20</code>)으로 같은 두 정책을 다시 평가했다. 사전 등록한 비교(0.375/0.5배)와 별도로 보고하며, L1 행은 같은 저장 기록(0.5배)이다.</p>')
        rows = []
        for key in ('L1_20', 'R0n_s0_20', 'R1n_s0_20'):
            g = ng.get(key)
            if not g: continue
            m = g['metrics']
            rows.append([key.replace('n_s0', ' (native)').replace('_20', ''), 20, f'{m["overall_tau_sq_reduction"]["mean"]:.2f}', f'{m["low10_rms_increase"]["mean"]:.2f}', f'{m["worst_phase_rms_increase"]["mean"]:.2f}',
                         f'{m["assist_rms"]["mean"]:.2f}', f'{m["saturation_pct"]["mean"]:.1f}%', f'{m["body_error_mm"]["mean"]:.2f}', f'{m["speed"]["mean"]:.3f}',
                         *[f'{g["other_joints"][k]["mean"]:.1f}' for k in ('무릎', '발목', '몸통')], g['falls']])
        body.append(table(['모델', '±N·m', '고관절 τ² 감소', '저부담 RMS 증가', '최악 위상 RMS 증가', '보조 RMS', '포화율', '몸체 오차 mm', '속도 m/s', '무릎 τ² 증가', '발목 τ² 증가', '몸통 τ² 증가', '낙상'], rows))
        body.append('<div id="native_note">__NATIVE_NOTE__</div>')
    body.append('<h3>다른 관절에 부담이 옮겨갔는가</h3>')
    body.append(table(['모델', '±N·m', '무릎 τ² 증가', '발목 τ² 증가', '몸통 τ² 증가', '비보조 회전관절 평균 τ² 증가'],
                      [[g['model'], g['limit_Nm'], *[f'{g["other_joints"][k]["mean"]:.2f}' for k in ('무릎', '발목', '몸통', '전체 비보조 회전관절')]] for g in groups]))
    body.append('<details><summary>≤5 N·m 마스크 민감도·흡수 파워·고부담 감속</summary>')
    rows = []
    for g in groups:
        runs = [r for r in ev['walking']['runs'].values() if r['record']['model'] == g['model'] and r['record']['limit_Nm'] == g['limit_Nm']]
        def seg(s, k): return float(np.mean([r['segments'][s][k] for r in runs]))
        rows.append([g['model'], g['limit_Nm'], f'{val(g,"low5_rms_increase"):.3f}', f'{seg("low_decel_10","braking_sample_pct"):.1f}%', f'{seg("low_decel_10","absorbed_power_per_joint_W"):.3f}', f'{seg("high_decel_20","rms_increase"):.3f}'])
    body.append(table(['모델', '±N·m', '≤5 N·m 마스크 RMS 증가', '저부담 음의 파워 비율', '저부담 흡수 파워 W/관절', '고부담 감속 RMS 증가 N·m'], rows) + '</details>')
    for cap in (20, 15):
        fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
        for side, ax in enumerate(axs):
            for model in ('L1', 'F2_s0', 'R0_s0', 'R1_s0'):
                runs = [r for r in ev['walking']['runs'].values() if r['record']['model'] == model and r['record']['limit_Nm'] == cap]
                if not runs: continue
                y = np.mean([[-row['reduction'] for row in r['rows'][side]] for r in runs], axis=0)
                ax.plot(np.arange(25) * 4 + 2, y, label=model, color=COLORS.get(model))
            ax.axhline(0, color='black', lw=.7); ax.set(title=['오른쪽', '왼쪽'][side], xlabel='참조 고관절 위상 [%]'); ax.grid(alpha=.2)
        axs[0].set_ylabel('착용자 RMS 증가 [N·m] (양수 = 방해)'); axs[1].legend(fontsize=8)
        fig.suptitle(f'±{cap} N·m · 4% 위상 구간 (위상 0은 최대 고관절 굴곡)'); fig.tight_layout()
        body.append(fig_html(fig, f'phase_{cap}Nm'))
    body.append('<h2>4. 진자: 학습에 쓰지 않은 궤적·동역학 조건</h2><p>저장 평가와 같은 plant/trajectory/H1 seed(63031/52021/991), 128환경×2코호트×10초, 실제 동결 H1, ±20 N·m(출력 0.5배). legacy는 학습 분포의 미사용 무작위 궤적, phase는 감속·반전 외란, hold50은 사람 명령 50 ms 유지로 뒤 둘은 학습 조건 밖이다. 실패가 있으면 공통 생존 구간 평균만 보고 성공으로 판단하지 않는다.</p>')
    ps = [p for p in ev['pendulum'] if p['model'] in ('L1', 'F1_s0', 'F2_s0', 'R0_s0', 'R1_s0')]
    ps.sort(key=lambda p: (order.index(p['model']) if p['model'] in order else 99, ('legacy', 'phase', 'hold50').index(p['condition'])))
    body.append(table(['모델', '조건', '각도 RMSE °', '보조 RMS N·m', '포화율', 'Δτ RMS N·m/10ms', '사람 τ² (N·m)²', '무보조 τ²', '실패 무보조/보조 (/256)'],
                      [[p['model'], p['condition'], f'{p["tracking_rmse_deg"]:.3f}', f'{p["assist_rms_Nm"]:.2f}', f'{p["saturation_pct"]:.1f}%', f'{p["delta_tau_rms_Nm_per_10ms"]:.3f}',
                        f'{p["wearer_tau_sq_Nm2"]:.1f}', f'{p["reference_tau_sq_Nm2"]:.1f}', f'{p["failures"]["reference"]}/{p["failures"]["assisted"]}'] for p in ps]))
    body.append('<details><summary>과거 G0/G4 진자 결과 (±23 N·m, 직접 순위 비교 불가)</summary>' + table(['모델', '조건', '각도 RMSE °', '보조 RMS', '포화율', 'Δτ RMS/10ms'],
                [[p['model'], p['condition'], f'{p["tracking_rmse_deg"]:.3f}', f'{p["assist_rms_Nm"]:.2f}', f'{p["saturation_pct"]:.1f}%', f'{p["delta_tau_rms_Nm_per_10ms"]:.3f}'] for p in ev['pendulum'] if p['model'].startswith('G')]) + '</details>')
    body.append('<h2>5. 학습 기록</h2>')
    rows = []
    for model, t in ev['training'].items():
        s = t['summary']; latest = max((x[-1][0] for x in t['series'].values() if x), default=0)
        pe = s.get('env/raw_r/paired_excess_mean'); trunc = s.get('env/paired/reference_only_truncation_fraction')
        rows.append([model, latest, f'{s["info/episode_length"]["last500_mean"]:.1f}', f'{s["env/raw_r/tracking_mean"]["last500_mean"]:.4f}',
                     f'{s["env/raw_r/human_torque_sq_mean"]["last500_mean"]:.0f}', f'{s["env/raw_r/assist_effort_mean"]["last500_mean"]:.0f}',
                     f'{pe["last500_mean"]:.3f}' if pe else '—', f'{100*trunc["last500_mean"]:.2f}%' if trunc else '—',
                     f'{s["env/assist/injection_error_max_Nm"]["max"]:.6f}', sum(v['nonfinite'] for v in s.values()),
                     f'{s["times/last_epoch_seconds"]["last500_mean"]:.2f}' if 'times/last_epoch_seconds' in s else '—'])
    body.append(table(['모델', '기록 epoch', '후반 episode 길이/1000', '후반 추종 reward', '후반 사람 τ² 합', '후반 보조 τ² 합', '후반 토크 증가 비용항(가중치 전)', '무보조 진자 실패로 끊긴 비율', '최대 주입오차 N·m', '비유한 값', 'epoch 초'], rows))
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    panels = [('env/raw_r/tracking_mean', '추종 reward'), ('env/raw_r/human_torque_sq_mean', '사람 양쪽 τ² 합'), ('env/raw_r/assist_effort_mean', '보조 양쪽 τ² 합'),
              ('env/raw_r/paired_excess_mean', '토크 증가 비용항 (R1만)'), ('env/raw_r/human_power_mean', '사람 |τω| 합'), ('info/episode_length', 'episode 길이')]
    for ax, (tag, title) in zip(axs.flat, panels):
        for model, t in ev['training'].items():
            data = np.asarray(t['series'].get(tag, []))
            if len(data) < 5: continue
            width = min(100, len(data)); y = np.convolve(data[:, 1], np.ones(width) / width, mode='valid')
            ax.plot(data[width - 1:, 0], y, label=model, color=COLORS.get(model))
        ax.set(title=title, xlabel='epoch'); ax.grid(alpha=.2)
    axs[0, 0].legend(); fig.suptitle('학습 중 기록 · 100 epoch 이동 평균 (평가 성능과 구분)'); fig.tight_layout()
    body.append(fig_html(fig, 'training_curves'))
    body.append('<p>실행 오류 기록 ' + str(len(ev['runtime_failures'])) + '건. ' + ('원인·재개·로그는 runtime_failures.json에 보존했다.' if ev['runtime_failures'] else '프로세스 실패 기록은 없다.') + '</p>')
    if ev['runtime_failures']:
        body.append(table(['작업', '반환 코드', '실패 시도', '보존 로그'], [[e['label'], e['returncode'], e['attempt'], escape(e['log'])] for e in ev['runtime_failures']]))
    body.append('<h2>6. 판정과 다음 실험</h2>')
    body.append('<div id="verdict">__VERDICT__</div>')
    if complete and have == {'R0_s0', 'R1_s0'}:
        rows = []
        for cap in (20, 15):
            rows.append([f'R0−L1 ±{cap}', *[f'{gm("R0_s0",cap,k)-gm("L1",cap,k):+.2f}' for k in ('overall_tau_sq_reduction', 'low10_rms_increase', 'worst_phase_rms_increase', 'assist_rms', 'all_rotational_sum_tau_sq_reduction')]])
            rows.append([f'R1−R0 ±{cap}', *[f'{gm("R1_s0",cap,k)-gm("R0_s0",cap,k):+.2f}' for k in ('overall_tau_sq_reduction', 'low10_rms_increase', 'worst_phase_rms_increase', 'assist_rms', 'all_rotational_sum_tau_sq_reduction')]])
            if 'F1_s0_%d' % cap in G:
                rows.append([f'F1−R0 ±{cap} (C2 효과)', *[f'{gm("F1_s0",cap,k)-gm("R0_s0",cap,k):+.2f}' for k in ('overall_tau_sq_reduction', 'low10_rms_increase', 'worst_phase_rms_increase', 'assist_rms', 'all_rotational_sum_tau_sq_reduction')]])
        body.append('<p>한 가지만 다른 모델끼리의 차이. 이득·회전관절 합은 양수가 좋고, 저부담·최악 위상 차이는 음수가 좋다. 각각 학습 seed 하나의 관찰이다.</p>' + table(['비교', 'Δ고관절 τ² 감소', 'Δ저부담 RMS 증가', 'Δ최악 위상 RMS 증가', 'Δ보조 RMS', 'Δ회전관절 합 τ² 감소'], rows))
    body.append('<ul>' + ''.join('<li>' + escape(x) + '</li>' for x in ev['limitations']) + '</ul>')
    body.append('<h2>7. 재현 경로와 실행 명령</h2><p>학습·평가 명령과 소요 시간은 execution.json, 설정·소스 해시는 manifest.json, 설정 대조는 config_audit.json에 있다. 기존 출력 경로로 실행하면 재개되므로 독립 재현 시 experiment-name과 평가 out을 바꾼다.</p>')
    body.append('<p><a href="../../results/llp_relative_20261020/manifest.json">고정 계획</a> · <a href="../../results/llp_relative_20261020/execution.json">명령·소요 시간</a> · <a href="../../results/llp_relative_20261020/config_audit.json">설정 대조</a> · <a href="../../results/llp_relative_20261020/l1_need_bins.json">L1 구간 분석</a> · <a href="evidence.json">전체 수치</a></p>')
    for label, run in ev['execution'].items():
        if label.endswith('_train'):
            body.append('<details><summary>' + escape(label) + '</summary><pre>' + escape(shlex.join(run['command'])) + '</pre></details>')
    body.append('<p>정책 파일: results/llp_relative_R0_s0_20261020/epoch_6000.ckpt, results/llp_relative_R1_s0_20261020/epoch_6000.ckpt. 실험 파일 examples/experiments/assist_pendulum/mlp_relative.py, 실행기 run_relative.py, 무보조 진자 동시 계산 환경 protomotions/envs/base_env/paired_assist_env.py, 보상 protomotions/envs/rewards/assist_energy.py.</p>')
    body.append('<script type="application/json" id="evidence">' + json.dumps(ev, ensure_ascii=False, allow_nan=False).replace('</', '<\\/') + '</script>')
    css = 'body{font:16px/1.7 system-ui,sans-serif;color:#213547;background:#f5f7fa;margin:0}main{max-width:1180px;margin:auto;padding:36px 24px}h1{font-size:29px}h2{margin-top:38px}table{border-collapse:collapse;width:100%;background:white;font-size:13px}th,td{padding:9px 10px;border-bottom:1px solid #dbe2ea;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#eaf0f6}.scroll{overflow:auto}.plot svg{width:100%;height:auto}.callout{padding:20px;background:#e7f0f8;border-left:5px solid #3276a5}.muted{color:#607080}details{padding:12px;border:1px solid #dbe2ea;background:white;margin:12px 0}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}a{color:#146eab}'
    html = '<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LLP R0/R1 사전학습 결과</title><style>' + css + '</style></head><body><main>' + ''.join(body) + '</main></body></html>'
    verdict = (destination.parent / 'verdict.html').read_text() if (destination.parent / 'verdict.html').exists() else '<p class="muted">판정 문단은 계산 완료 후 작성한다.</p>'
    native_note = (destination.parent / 'native_note.html').read_text() if (destination.parent / 'native_note.html').exists() else ''
    destination.write_text(html.replace('__VERDICT__', verdict).replace('__NATIVE_NOTE__', native_note))
    save(destination.parent / 'evidence.json', ev)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--study', default='results/llp_relative_20261020'); p.add_argument('--out', default='docs/ai_sbc/261020_relative/report.html')
    args = p.parse_args(); ev = collect(ROOT / args.study); render(ev, ROOT / args.out)
    print(json.dumps(dict(status=ev['status']['stage'], comparison=ev['comparison']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
