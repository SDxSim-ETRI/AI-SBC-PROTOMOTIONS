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
"""Embed paired W1 walking phase figures and their numerical evidence in HTML."""
from datetime import datetime, timezone
from html import escape
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.font_manager import FontProperties
import numpy as np

from report_general_docs import FOLDER, ROOT, STUDY, glossary, model_label, shell

BINS = 25
X = (np.arange(BINS) + .5) * 100 / BINS
SIDES = ('오른쪽', '왼쪽')
FONT = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family': FONT.get_name(), 'font.size': 10,
                     'axes.unicode_minus': False, 'svg.fonttype': 'path',
                     'axes.spines.top': False, 'axes.spines.right': False})


def scalar(x):
    return float(x) if np.isfinite(x) else None


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def digest_run(record):
    directory = ROOT / record['path']
    meta = json.loads((directory / 'meta.json').read_text())
    with np.load(directory / 'traces.npz') as f:
        tr = dict(f)
    idx = np.flatnonzero((tr['t'] >= meta['warmup'] + .5) & (np.arange(len(tr['t'])) >= 2))
    if len(idx) < 2:
        return dict(record=record, meta=meta, error='1.5초 이후 분석 가능한 창이 없음')
    tau = (tr['tau_pd_prev'] + tr['tau_ff_prev'])[idx]
    assist = tr['tau_assist'][idx - 1, 1]
    before, after = tr['gait_phase'][idx - 1], tr['gait_phase'][idx]
    phase = (before + .5 * ((after - before + .5) % 1 - .5)) % 1
    assert np.isfinite(phase).all()
    limit = abs(meta['assist_scale']) * min(meta['torque_limit'], meta['policy_torque_limit'])
    assert np.max(np.abs(assist)) <= limit + 1e-4
    rows = []
    for side in (0, 1):
        values = []
        for k in range(BINS):
            sel = (phase[:, side] >= k / BINS) & (phase[:, side] < (k + 1) / BINS)
            a, b, s = tau[sel, 0, side], tau[sel, 1, side], assist[sel, side]
            row = dict(lo=k*100/BINS, hi=(k+1)*100/BINS, n=int(sel.sum()))
            if sel.any():
                row.update(A_mean=float(a.mean()), B_mean=float(b.mean()), assist_mean=float(s.mean()),
                           A_rms=rms(a), B_rms=rms(b), assist_rms=rms(s), reduction=rms(a)-rms(b),
                           assist_min=float(s.min()), assist_max=float(s.max()),
                           saturation_pct=float(100*np.mean(np.abs(s) > .95*limit)))
            values.append(row)
        rows.append(values)
    omega = .5*(tr['hip_vel'][idx-1] + tr['hip_vel'][idx])
    err = np.degrees(tr['hip_pos'][idx] - tr['ref_hip'][idx])
    summary = dict(samples=len(idx), t_start=float(tr['t'][idx[0]]), t_end=float(tr['t'][idx[-1]]),
                   limit=limit, assist_rms=rms(assist), assist_max=float(np.abs(assist).max()),
                   saturation_pct=float(100*np.mean(np.abs(assist) > .95*limit)))
    for j, name in enumerate(('A', 'B')):
        summary[name] = dict(tau_rms=rms(tau[:, j]), tau_sq=float(np.square(tau[:, j]).mean()),
                            work=float(np.abs(tau[:, j]*omega[:, j]).sum(axis=1).mean()),
                            tracking_mae=float(np.abs(err[:, j]).mean()))
        if 'root_xy' in tr:
            summary[name]['speed'] = float(np.linalg.norm(tr['root_xy'][idx[-1], j] - tr['root_xy'][idx[0]-1, j]) /
                                          (tr['t'][idx[-1]] - tr['t'][idx[0]-1]))
        if 'body_err' in tr:
            summary[name]['body_err'] = float(tr['body_err'][tr['body_err_t'] >= meta['warmup']+.5, j].mean())
    other = []
    if (directory / 'fullbody.npz').exists():
        fm = json.loads((directory / 'fullbody_meta.json').read_text())
        with np.load(directory / 'fullbody.npz') as f:
            fulltau = (f['control_tau_pd_prev'] + f['control_tau_ff_prev'])[idx]
        names = fm['dof_names_common']
        hip_indices = [names.index('hip_flexion_' + side) for side in ('r', 'l')]
        assert np.allclose(fulltau[:, :, hip_indices], tau, atol=1e-4), 'Full-body torque alignment mismatch'
        # Suit slide DOFs carry linear forces [N], not joint torques [N m].
        for group, indices in [('전체 비보조 회전관절', [i for i,n in enumerate(names) if n not in ('hip_flexion_r','hip_flexion_l') and not n.startswith('slide')]),
                               ('무릎', [i for i,n in enumerate(names) if 'knee' in n]),
                               ('발목', [i for i,n in enumerate(names) if 'ankle' in n]),
                               ('몸통', [i for i,n in enumerate(names) if any(s in n for s in ('lumbar','torso','abdomen'))])]:
            if indices:
                other.append(dict(group=group, dofs=[names[i] for i in indices],
                                  A=float(np.square(fulltau[:,0,indices]).mean()),
                                  B=float(np.square(fulltau[:,1,indices]).mean())))
    return dict(record=record, meta=meta, rows=rows, summary=summary, other_joints=other)


def svg(fig, identity):
    plt.rcParams['svg.hashsalt'] = identity
    buf = io.StringIO()
    fig.savefig(buf, format='svg', bbox_inches='tight')
    plt.close(fig)
    return '<div class="plot">' + buf.getvalue()[buf.getvalue().index('<svg'):] + '</div>'


def array(run, side, key):
    return np.array([r.get(key, np.nan) for r in run['rows'][side]])


def heatmap(runs):
    names = list(runs)
    fig, axes = plt.subplots(2, 1, figsize=(11, max(4, len(names)*.75+1.7)), layout='constrained')
    data = [np.array([array(runs[n], side, 'reduction') for n in names]) for side in (0, 1)]
    bound = max(1., *(np.nanmax(np.abs(d)) for d in data))
    for side, ax in enumerate(axes):
        im = ax.imshow(data[side], cmap='RdYlGn', norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound),
                       aspect='auto', extent=[0,100,len(names)-.5,-.5])
        ax.set_yticks(range(len(names)), [model_label(n, multiline=True) + (' · 조기 종료' if runs[n]['meta']['survived_s'] else '') for n in names], fontsize=9)
        ax.set_title(SIDES[side] + ' · 착용자 토크 RMS 감소량 A − B')
        ax.set_xticks(np.arange(0,101,10)); ax.set_xlabel('참조 보행 phase [%] · 최대 굴곡 = 0%')
    fig.colorbar(im, ax=axes, label='[N·m]  음수: 부담 증가 ← 0 → 양수: 부담 감소', shrink=.85)
    return svg(fig, 'heatmap')


def phase_fig(name, run, baseline):
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), layout='constrained', sharex=True)
    fig.suptitle(model_label(name), fontsize=12)
    label = model_label(name, compact=True)
    for side in (0, 1):
        reduction = array(run, side, 'reduction')
        ax = axes[0, side]
        ax.axhline(0, color='#64748b', lw=.7)
        ax.bar(X, reduction, width=3.6, color=np.where(reduction >= 0, '#237b5b', '#c64448'), label=label+'의 A − B')
        if baseline and name != 'L1':
            ax.plot(X, array(baseline,side,'reduction'), '--', color='#777', label='L1의 A − B')
        ax.set_title(SIDES[side]); ax.set_ylabel('착용자 RMS 감소량 [N·m]'); ax.legend(fontsize=8)
        ax = axes[1, side]
        ax.plot(X, array(run,side,'A_mean'), color='#333',ls='--', label='무보조 A')
        ax.plot(X, array(run,side,'B_mean'), color='#245aa0', label=label+' 보조 B')
        if baseline and name != 'L1':
            ax.plot(X, array(baseline,side,'B_mean'), color='#b17c21', ls=':',label='L1 보조 B')
        ax.set_ylabel('착용자 평균 토크 [N·m]'); ax.legend(fontsize=8)
        ax = axes[2, side]
        ax.plot(X, array(run,side,'assist_mean'), color='#245aa0',label=label+' 보조 평균')
        ax.fill_between(X, array(run,side,'assist_min'),array(run,side,'assist_max'), color='#245aa0',alpha=.12,label='기록한 최소–최대')
        if baseline and name != 'L1':
            ax.plot(X, array(baseline,side,'assist_mean'), color='#b17c21',ls=':',label='L1 보조 평균')
        for sign in (-1,1): ax.axhline(sign*run['summary']['limit'],color='#c64448',ls='--',lw=.8)
        ax.set_ylim(-25,25); ax.set_ylabel('장치 보조 토크 [N·m]'); ax.legend(fontsize=8)
        ax.set_xlabel('참조 보행 phase [%] · 최대 굴곡 = 0%')
        for ax in axes[:,side]:
            ax.set_xlim(0,100); ax.set_xticks(np.arange(0,101,10)); ax.grid(alpha=.15)
    return svg(fig, name)


def finalist_figure(runs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), layout='constrained', sharey=True)
    for side, ax in enumerate(axes):
        ax.axhspan(-25, 0, color='#c64448', alpha=.07)
        ax.axhline(0, color='#64748b', lw=.8)
        for name, color, style in [('L1', '#333333', '-'), ('G4_s0', '#245aa0', '-'), ('G4_s1', '#b86636', '--')]:
            ax.plot(X, array(runs[name], side, 'reduction'), color=color, ls=style, label=model_label(name, compact=True), lw=2)
        ax.set_title(SIDES[side]+' · 기존 L1과 보조 비용 강화 G4')
        ax.set_xlabel('참조 보행 phase [%] · 최대 굴곡 = 0%')
        ax.set_ylabel('착용자 RMS 감소량 A − B [N·m]')
        ax.set_xlim(0, 100); ax.set_ylim(-25, 30)
        ax.set_xticks(np.arange(0, 101, 20)); ax.grid(alpha=.15); ax.legend(fontsize=8)
    return svg(fig, 'finalist_comparison')


def audit_fig(name, run):
    s = run['summary']; metrics = [('tau_sq','고관절 토크² 평균 [(N·m)²]'),('work','고관절 |일률| 합 [W]'),
                                  ('tracking_mae','고관절 추종 평균절대오차 [°]'),('speed','평균 전진 속도 [m/s]')]
    fig, axes = plt.subplots(1, 4, figsize=(12,2.9), layout='constrained')
    for ax,(key,title) in zip(axes,metrics):
        vals=[s[k].get(key,np.nan) for k in ('A','B')]
        ax.bar(['무보조 A','보조 B'], vals,color=['#999','#245aa0']);ax.set_title(title,fontsize=9)
        for x,v in enumerate(vals): ax.text(x,v,f'{v:.2f}',ha='center',va='bottom',fontsize=9)
        ax.margins(y=.2)
    return svg(fig,name+'_health')


def other_fig(name, run):
    groups=run['other_joints']
    if not groups: return '<p>비보조 관절 토크 기록 없음.</p>'
    fig, ax=plt.subplots(figsize=(10,2.4),layout='constrained')
    yy=np.arange(len(groups)); delta=[g['B']-g['A'] for g in groups]
    ax.barh(yy,delta,color=['#c64448' if d>0 else '#237b5b' for d in delta])
    ax.set_yticks(yy,[g['group'] for g in groups]);ax.axvline(0,color='#666',lw=.7)
    ax.set_xlabel('비보조 관절 토크² 평균 변화 B − A [(N·m)²] · 양수는 부담 증가')
    ax.margins(x=.3)
    for y,d in zip(yy,delta):ax.text(d,y,f' {d:+.2f} ',va='center',ha='left' if d>=0 else 'right',fontsize=9)
    return svg(fig,name+'_other')


def appendix(runs):
    content='<details><summary>부록 · phase별 원시 요약표, 표본 수, 경로와 재현 조건</summary>'
    content+='<p>n은 해당 4% 구간에 들어간 10 ms 창 수다. 독립 사람 수나 독립 실험 반복 수가 아니다. 아래 표의 토크 단위는 N·m다. 한 번의 40초 시행이며 신뢰구간을 주장하지 않는다.</p>'
    for name,run in runs.items():
        content+=f'<details><summary>{escape(model_label(name))} · 수치·경로</summary><p>원본 식별자: <code>{escape(name)}</code></p><p><code>{escape(run["record"]["path"])}</code></p>'
        content+='<pre>'+escape(json.dumps(run['summary'],ensure_ascii=False,indent=2))+'</pre>'
        for side in (0,1):
            content+=f'<h3>{SIDES[side]}</h3><table><tr><th>phase %</th><th>n</th><th>A RMS</th><th>B RMS</th><th>A−B RMS</th><th>A 평균</th><th>B 평균</th><th>보조 평균</th><th>보조 RMS</th><th>포화 %</th></tr>'
            for r in run['rows'][side]:
                values=[r.get(k) for k in ('A_rms','B_rms','reduction','A_mean','B_mean','assist_mean','assist_rms','saturation_pct')]
                content+=f'<tr><td>{r["lo"]:.0f}–{r["hi"]:.0f}</td><td>{r["n"]}</td>'+''.join('<td>—</td>' if v is None else f'<td>{v:.3f}</td>' for v in values)+'</tr>'
            content+='</table>'
        content+='<pre>'+escape(json.dumps(run['record'],ensure_ascii=False,indent=2))+'</pre></details>'
    return content+'</details>'


def main():
    path=STUDY/'walking/runs.json'
    records=json.loads(path.read_text()) if path.exists() else {}
    runs={name:digest_run(records[name]) for name in sorted(records,key=lambda x:(x!='L1',x))}
    valid={k:v for k,v in runs.items() if 'error' not in v}
    missing=[f'G{k}_s0' for k in range(5) if f'G{k}_s0' not in runs]
    body='<h1>보행할 때 어디서 돕고, 어디서 방해하는가</h1><p>미래 각도 예측을 받아 ±23 N·m 보조 토크를 내는 LLP를 진자에서 처음부터 학습하는 실험이다. 동결 W1이 같은 걷기를 수행할 때, 무보조 A와 보조 B를 짝지어 비교한다. 기존 L1은 ±20 N·m다.</p>'+glossary()
    if missing:
        body+='<p class="status">실험 진행 중 · 아직 보행 평가가 없는 후보: '+', '.join(model_label(n) for n in missing)+'. 완료된 기록만 표시한다. 최종 선정이나 일반화 성공을 뜻하지 않는다.</p>'
    complete=(STUDY/'confirmation.json').exists() and all(n in valid for n in ('L1','G4_s0','G4_s1','G1_s1')) and not missing
    if complete:
        body+='<p class="status"><strong>8개 걷기 평가 모두 40초 완주. L1을 대체할 일반 모델은 확보하지 못했다.</strong> 진자에서 선정한 G4도 보행 후반의 방해가 남았다. 별도 진자 평가에서 L1 기준에 미달해 조건부 달리기는 실행하지 않았다. <a href="FINAL_RESULTS.html">최종 판단 전체</a></p>'
    body+='<p>각 다리의 참조 최대 굴곡을 phase 0%로 두고 4%씩 묶었다. 발 접촉 기준 위상이 아니므로 입각기·유각기의 경계를 이 그림에서 단정하지 않는다. 토크는 +굴곡 / −신전. <strong>RMS 감소량이 양수면 도움, 음수면 방해</strong>다. 부호 있는 평균 토크와 RMS를 구별한다.</p>'
    body+='<p class="note">조건: 기존 걷기 클립(동작 번호 0), 고정 W1/H1, 평가 난수 초기값 42, 40초, 보조 토크 갱신 10 ms(0.01초), 물리 계산 400 Hz(초당 400번). 초기 1.5초 제외. 모든 그래프는 동일한 10 ms 창에 착용자·보조 토크를 맞췄다. 새 후보는 각 3,000 epoch 마지막 모델. 수동 위상 게이트 없음.</p>'
    if complete:
        body+='<h2>핵심 결과 · G4도 방해 구간을 해결하지 못했다</h2><p>아래 선이 0보다 낮으면 해당 구간에서 착용자 토크 RMS가 증가했다. G4의 오른쪽 76–80% 증가량은 첫 학습/독립 재학습 순서로 각각 <strong>19.41 / 22.75 N·m</strong>, 왼쪽 84–88%는 <strong>18.35 / 21.00 N·m</strong>다. L1은 같은 구간에서 각각 13.95 / 14.98 N·m 증가했다.</p>'+finalist_figure(valid)
        body+='<p>G0는 전체 고관절 RMS가 34.27 N·m로 L1의 34.83 N·m보다 낮았지만, 오른쪽 24–28% 부담이 16.36 N·m 늘었다. 진자 실패와 독립 재학습을 통한 재검증 미실시도 있어 일반 모델의 우위로 판단하지 않는다. G1은 처음부터 별도로 학습한 두 모델 모두 68–72%에서 가장 크게 방해했다. 전체 평균과 아래 phase 그래프를 함께 본다.</p>'
    if valid:
        body+='<h2>1. 어느 구간에서 부담이 줄었나</h2><p>녹색은 감소, 빨강은 증가다. 각 행은 해당 모델의 짝지은 무보조 A를 기준으로 계산했다. 조기 종료는 관측된 시간만의 진단이며 완주 모델과 같은 기준으로 순위화하지 않는다.</p>'+heatmap(valid)
        body+='<h2>2. 모델별 토크와 보행 유지</h2><label for="model">모델 선택 </label><select id="model" onchange="document.querySelectorAll(\'.model\').forEach(e=>e.hidden=e.dataset.model!==this.value)">'
        body+=''.join(f'<option value="{escape(n)}">{escape(model_label(n))}</option>' for n in valid)+'</select>'
        for i,(name,run) in enumerate(valid.items()):
            s=run['summary']; ended=run['meta']['survived_s']
            body+=f'<section class="model" data-model="{escape(name)}"'+(' hidden' if i else '')+'>'
            body+=f'<h3>{escape(model_label(name))} · '+(f'{ended:.2f}초에 낙상 조건으로 쌍 종료' if ended else '40초 완주')+'</h3>'
            body+=f'<p>분석 {s["t_start"]:.2f}–{s["t_end"]:.2f}초 · 보조 RMS {s["assist_rms"]:.2f} N·m · 최대 절댓값 {s["assist_max"]:.2f} N·m · 출력 한계 ±{s["limit"]:.0f} N·m · 한계 95% 초과 {s["saturation_pct"]:.1f}%.</p>'
            for side, rows in enumerate(run['rows']):
                observed=[r for r in rows if r['n']]
                best=max(observed,key=lambda r:r['reduction'])
                worst=min(observed,key=lambda r:r['reduction'])
                body+=f'<p><strong>{SIDES[side]}</strong> · 가장 큰 RMS 감소량 {best["lo"]:.0f}–{best["hi"]:.0f}%: {best["reduction"]:+.2f} N·m. 가장 작은 감소량 {worst["lo"]:.0f}–{worst["hi"]:.0f}%: {worst["reduction"]:+.2f} N·m (음수면 그 크기만큼 부담 증가).</p>'
            body+=phase_fig(name,run,valid.get('L1'))
            body+='<p class="note">첫 줄: RMS 감소량. 둘째 줄: 부호 있는 착용자 평균 토크. 셋째 줄: 장치 보조 평균과 관측 최소–최대(음영); 빨간 점선은 장치 한계다. L1 점선은 별도 쌍 평가다. 음영은 신뢰구간이 아니다.</p>'
            body+=audit_fig(name,run)
            body+='<p>토크만 낮아졌더라도 덜 걷거나 추종이 나빠지면 유용한 보조라고 판정하지 않는다. 아래는 다른 관절로 부담이 이동했는지 확인하는 지표다. 회전관절당 토크² 평균이며 직선 slide 관절의 힘[N]은 제외했다. 전체에는 무릎·발목·몸통도 포함되므로 막대끼리 합산하지 않는다.</p>'+other_fig(name,run)+'</section>'
        body+=appendix(valid)
    else:
        body+='<p class="status">완료된 보행 평가가 아직 없어 그래프를 만들지 않았다.</p>'
    body+='<details><summary>해석 범위와 모델 이름</summary><p>G0: 양측 공동 actor 대조군. G1: 관절별 공유 actor 기본. G2: G1에 합성 감속·반전 외란과 구조 하중 추가. G3: G2의 사람 토크²·일률 비용 2배. G4: G2의 보조 절약·명령 변화 비용 4배. s0는 첫 학습(난수 초기값 0), s1은 같은 설계를 처음부터 다시 학습한 모델(난수 초기값 1)이다. PA 계열 가중치는 사용하지 않았다.</p><p>토크²는 시뮬레이션 부담 지표이며 근활성·대사량의 측정값이 아니다. 이 자료는 기존 걷기 한 클립의 전이 평가이며 다양한 사람·실기기의 성공을 입증하지 않는다. 진자의 몸체 관성은 환경별 고정이며 실제 heel strike 접촉·순간 관성 변화를 구현한 것은 아니다.</p></details>'
    body+='<p class="note">생성 UTC '+datetime.now(timezone.utc).isoformat()+'</p>'
    evidence=dict(protocol='4% bins; reference max-flexion phase; torque window midpoint; t>=warmup+.5s',
                  model_labels={name:model_label(name) for name in runs},
                  training_identity='s0: first training, seed 0; s1: independent training from scratch, seed 1; walking evaluation seed: 42 for all models',
                  runs=runs)
    data=json.dumps(evidence,ensure_ascii=False,allow_nan=False).replace('</','<\\/')
    body+='<button onclick="const b=new Blob([document.getElementById(\'evidence\').textContent],{type:\'application/json\'});const a=document.createElement(\'a\');a.href=URL.createObjectURL(b);a.download=\'walking_phase_evidence.json\';a.click();URL.revokeObjectURL(a.href)">그림의 수치·재현 조건 JSON 저장</button>'
    body+='<script type="application/json" id="evidence">'+data+'</script>'
    (FOLDER/'walking_phase_report.html').write_text(shell('보행 phase별 도움과 방해 · 23 N·m LLP',body))
    (FOLDER/'walking_phase_evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2,allow_nan=False))
    print('Walking report:',len(valid),'completed runs;',len(missing),'pending')


if __name__=='__main__':
    main()
