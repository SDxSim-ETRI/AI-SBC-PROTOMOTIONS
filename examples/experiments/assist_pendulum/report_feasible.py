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
"""Read-only comparison of new PPO candidates and preserved baseline evidence."""
import argparse
from datetime import datetime, timezone
from html import escape
import io
import json
from pathlib import Path
import shlex
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OLD=ROOT/'results/pendulum_general23_20261015_all'
COLORS={'L1':'#374151','G0_s0':'#1a8b67','G4_s0':'#8b70b2','G4_s1':'#b39bcf',
        'F1_s0':'#1872bd','F2_s0':'#dc7130','F1_s1':'#50a6ce','F2_s1':'#b04b22'}


def read(path):return json.loads(Path(path).read_text())
def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))
def stats(x):return dict(mean=float(np.mean(x)),min=float(np.min(x)),max=float(np.max(x)))


def pendulum_record(path,model,condition,reused):
    s=read(path/'summary.json');m=s['metrics']['all']
    with np.load(path/'rollout.npz') as f:
        mask=f['valid'] & (f['time_s']>=s['protocol']['warmup'])
        tau=f['assist_tau'].astype(np.float64)
        pairs=mask[1:] & mask[:-1]
        change=np.diff(tau,axis=0)[pairs]
        tracking=(f['assisted_q']-f['intent'])[mask]
        result=dict(model=model,condition=condition,reused_baseline=reused,
            limit_Nm=s['protocol']['evaluation_limit'],seed_protocol={k:s['protocol'][k] for k in ('plant_seed','trajectory_seed','hlp_seed')},
            tracking_rmse_deg=float(np.sqrt(np.mean(tracking**2))*180/np.pi),
            assist_rms_Nm=float(np.sqrt(np.mean(tau[mask]**2))),
            saturation_pct=100*float(np.mean(np.abs(tau[mask])>.95*s['protocol']['evaluation_limit'])),
            delta_tau_rms_Nm_per_10ms=float(np.sqrt(np.mean(change**2))),
            delta_tau_abs_mean_Nm_per_10ms=float(np.mean(np.abs(change))),
            wearer_tau_sq_Nm2=m['assisted_tau_sq_Nm2'],reference_tau_sq_Nm2=m['reference_tau_sq_Nm2'],
            failures=s['failures'],episodes=s['total_episodes'],survival_fraction=s['common_survival_fraction'],
            injection_error_max_Nm=s['injection_error_max_Nm'],reward_audit=s['reward_audit'],
            summary_path=str((path/'summary.json').relative_to(ROOT)))
    return result


def training_record(directory):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    tags=['info/episode_length','rewards/unnormalized_task_rewards','env/raw_r/tracking_mean',
          'env/raw_r/assist_effort_mean','env/raw_r/human_torque_sq_mean','env/raw_r/human_power_mean',
          'env/raw_r/assist_absorbed_power_mean','env/scaled_r/assist_absorbed_power_mean',
          'env/scaled_r/tracking_mean','env/scaled_r/human_power_mean','env/scaled_r/human_torque_sq_mean',
          'env/scaled_r/assist_effort_mean','env/scaled_r/action_rate_mean',
          'env/assist/injection_error_max_Nm','env/assist/injection_clipped_fraction',
          'env/termination/tracking_failure_mean','env/termination/dof_vel_failure_mean',
          'losses/actor_loss','losses/critic_loss','actor/bad_grads_count','critic/bad_grads_count',
          'times/last_epoch_seconds','actor/clip_frac','actor/update_skipped']
    series={}
    for version in sorted((directory/'lightning_logs').glob('version_*')):
        a=EventAccumulator(str(version),size_guidance={'scalars':0});a.Reload()
        for tag in tags:
            if tag in a.Tags()['scalars']:
                series.setdefault(tag,[]).extend([[int(e.step),float(e.value)] for e in a.Scalars(tag)])
    summary={}
    for tag,values in series.items():
        # A restart may duplicate a step: retain the later saved observation.
        unique=dict(values);values=sorted(unique.items());series[tag]=values
        a=np.array([v for _,v in values]);finite=np.isfinite(a)
        summary[tag]=dict(count=len(a),nonfinite=int((~finite).sum()),
            first100_mean=float(a[:100].mean()) if finite.all() else None,
            last500_mean=float(a[-500:].mean()) if finite.all() else None,
            max=float(a.max()) if finite.all() else None,last=float(a[-1]) if finite.all() else None)
    return dict(directory=str(directory.relative_to(ROOT)),summary=summary,series=series)


def all_joint_metrics(run):
    directory=ROOT/run['record']['path']
    with np.load(directory/'fullbody.npz') as f,np.load(directory/'traces.npz') as tr:
        idx=np.flatnonzero((tr['t']>=1.5)&(np.arange(len(tr['t']))>=2))
        tau=(f['control_tau_pd_prev']+f['control_tau_ff_prev'])[idx].astype(np.float64)
    names=read(directory/'fullbody_meta.json')['dof_names_common']
    rot=[i for i,n in enumerate(names) if not n.startswith('slide')]
    a=float(np.mean(np.sum(tau[:,0][:,rot]**2,axis=1)))
    b=float(np.mean(np.sum(tau[:,1][:,rot]**2,axis=1)))
    nonhip=next(j for j in run['other_joints'] if j['group']=='전체 비보조 회전관절')
    reconstructed=2*(run['summary']['A']['tau_sq']-run['summary']['B']['tau_sq'])+(len(rot)-2)*(nonhip['A']-nonhip['B'])
    assert np.isclose(a-b,reconstructed,atol=.02,rtol=1e-5), 'Whole-body torque aggregation mismatch'
    return dict(A_sum_tau_sq_Nm2=a,B_sum_tau_sq_Nm2=b,reduction_Nm2=a-b,
        rotational_dof_count=len(rot),note='Mean over time of sum across rotational wearer joints; slides excluded')


def collect(study):
    plan=read(study/'manifest.json');status=read(study/'status.json')
    wpath=study/'walking/comparison.json'
    if not wpath.exists():
        walking=read(ROOT/'results/llp_output_scale_20261016/comparison.json')
    else:
        walking=read(wpath)
        # The reused amplitude-study helper describes its original no-training
        # study. Correct only this new study's metadata, never the old evidence.
        walking['note']='Stored L1/G0/G4 baseline runs plus newly pendulum-trained F candidates. Two evaluation seeds provide means and observed ranges, not confidence intervals. W1 and H1 remain frozen.'
        walking['plan']['baseline_study_purpose']=walking['plan'].get('baseline_study_purpose',walking['plan']['purpose'])
        walking['plan']['purpose']='Evaluate freshly pendulum-trained F1/F2 at matched actual torque limits against stored L1/G0/G4 runs.'
        save(wpath,walking)
    training={}
    for directory in sorted((ROOT/'results').glob('llp_feasible_F*_s*_20261020')):
        label=directory.name.removeprefix('llp_feasible_').removesuffix('_20261020')
        training[label]=training_record(directory)
    pendulum=[]
    for model,prefix in [('L1','L1_test'),('G0_s0','G0_s0_dev'),('G4_s0','G4_s0_test'),('G4_s1','G4_s1_test')]:
        for condition in ('legacy','phase','hold50'):
            path=OLD/f'{prefix}_H1_{condition}'
            pendulum.append(pendulum_record(path,model,condition,True))
    for path in sorted((study/'pendulum').glob('*_H1_*')):
        if (path/'summary.json').exists():
            model,condition=path.name.split('_H1_')
            pendulum.append(pendulum_record(path,model,condition,False))
    for r in walking['runs'].values():r['all_rotational_joints']=all_joint_metrics(r)
    for group in walking['groups'].values():
        runs=[r for r in walking['runs'].values() if r['record']['model']==group['model'] and r['record']['limit_Nm']==group['limit_Nm']]
        group['metrics']['body_error_mm']=stats([1000*r['summary']['B']['body_err'] for r in runs])
        group['metrics']['all_rotational_sum_tau_sq_reduction']=stats([r['all_rotational_joints']['reduction_Nm2'] for r in runs])
        group['other_joints']={name:stats([next(j['B']-j['A'] for j in r['other_joints'] if j['group']==name) for r in runs])
            for name in ('전체 비보조 회전관절','무릎','발목','몸통')}
    comparison=[]
    for key,g in walking['groups'].items():
        if not g['model'].startswith('F') or g['limit_Nm'] not in (15,20):continue
        b=walking['groups'][f'L1_{g["limit_Nm"]}'];m=g['metrics'];bm=b['metrics']
        delta={name:m[name]['mean']-bm[name]['mean'] for name in m if name in bm}
        joint_primary=(delta['overall_tau_sq_reduction']>=0 and delta['low10_rms_increase']<=0
            and delta['worst_phase_rms_increase']<=0 and g['falls']==0)
        guards=(delta['body_error_mm']<=1 and abs(delta['speed'])<=.02
            and delta['all_rotational_sum_tau_sq_reduction']>=0)
        # Pendulum guard needs all three completed held-out domains.
        ps=[p for p in pendulum if p['model']==g['model']]
        pendulum_complete=len(ps)==3
        baseline_pendulum={p['condition']:p for p in pendulum if p['model']=='L1'}
        pendulum_stable=pendulum_complete and all(
            p['failures']['assisted']<=baseline_pendulum[p['condition']]['failures']['assisted'] for p in ps)
        pendulum_tracking_no_worse=pendulum_complete and all(
            p['tracking_rmse_deg']<=baseline_pendulum[p['condition']]['tracking_rmse_deg'] for p in ps)
        comparison.append(dict(model=g['model'],limit_Nm=g['limit_Nm'],delta_vs_L1=delta,
            walking_primary_jointly_better=bool(joint_primary),walking_other_guards=bool(guards),
            pendulum_complete=pendulum_complete,pendulum_stable=pendulum_stable,
            pendulum_tracking_no_worse=bool(pendulum_tracking_no_worse),
            all_criteria_met=bool(joint_primary and guards and pendulum_stable and pendulum_tracking_no_worse)))
    ev=dict(generated_utc=datetime.now(timezone.utc).isoformat(),plan=plan,status=status,
        trajectory_audit=read(study/'trajectory_audit.json'),training=training,pendulum=pendulum,
        walking=walking,comparison=comparison,
        execution=read(study/'execution.json') if (study/'execution.json').exists() else {},
        runtime_failures=read(study/'runtime_failures.json') if (study/'runtime_failures.json').exists() else [],
        orchestration_interruptions=read(study/'orchestration_interruptions.json') if (study/'orchestration_interruptions.json').exists() else [],
        repeat_selection=read(study/'repeat_selection.json') if (study/'repeat_selection.json').exists() else None,
        limitations=['Existing walking clip / one wearer; evaluation seeds are not independent training seeds or participants.',
            'L1/G0/G4 outcomes are preserved old runs; same seed Newton re-execution is not bit-identical.',
            'L1 pendulum baseline is available at 20 Nm only. G0/G4 old pendulum data uses 23 Nm; G0 also uses different cohort seeds.',
            'F1 differs from L1 in both trajectory blending and torque-accounting path. F1/F2 differ only in absorbed-power reward.',
            'Both candidates trained from scratch for 6000 epochs; G0/G4 used 3000. No reward-only causal claim against G4.',
            'The optional repeat rule was fixed before training. The report joint-criteria flag is a descriptive Pareto/guard comparison, not a preregistered statistical significance test; pendulum failures must not exceed L1 in each domain.',
            'Mean/range over evaluation runs, not confidence intervals. Torque squared is a simulated burden measure, not metabolism.'])
    save(study/'evidence.json',ev)
    return ev


def render(ev,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family':font.get_name(),'font.size':10,'axes.unicode_minus':False,'svg.fonttype':'path',
                        'axes.spines.top':False,'axes.spines.right':False})
    destination.parent.mkdir(parents=True,exist_ok=True)
    figures=destination.parent/'figures';figures.mkdir(exist_ok=True)
    def fig_html(fig,name):
        fig.savefig(figures/f'{name}.png',dpi=155,bbox_inches='tight')
        buf=io.StringIO();fig.savefig(buf,format='svg',bbox_inches='tight');plt.close(fig)
        return '<div class="plot">'+buf.getvalue()[buf.getvalue().index('<svg'):]+'</div>'
    def table(headers,rows):
        return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(c)+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>'
    body=['<nav><a href="../experiment_summary.html">실험 요약</a> · <a href="../HANDOFF.md">현재 상태</a> · <a href="../AI_SBC_VERSIONS.md">모델 경로</a></nav><h1>LLP 진자 사전학습 F1/F2 · 실제 PPO와 W1 전이</h1>',
        '<p class="muted">2026-10-20 · 학습 seed s0/s1과 평가 seed 42/7을 구분한다. 기존 L1/G0/G4는 저장 기록을 재사용했다.</p>']
    complete=ev['status']['stage']=='all planned training and evaluations complete'
    fresh_walks=sum(r['record']['model'].startswith('F') for r in ev['walking']['runs'].values())
    fresh_pendulum=sum(not p['reused_baseline'] for p in ev['pendulum'])
    body.append(f'<p>신규 PPO 학습 실행 {len(ev["training"])}개 · 신규 W1 평가 완료 {fresh_walks}건 · 신규 진자 평가 완료 {fresh_pendulum}건. 과거 W1 기준선 24건은 파일에서 재사용했다.</p>')
    accepted=[x for x in ev['comparison'] if x['all_criteria_met']]
    if not complete: headline='계산 진행 중 — 아래 수치 중 완료된 평가만 표시하며 성능 결론은 보류한다.'
    elif accepted: headline='동일 출력에서 공동 기준을 통과한 후보: '+', '.join(f'{x["model"]} ±{x["limit_Nm"]} N·m' for x in accepted)+'. 반복 seed와 진자 추종 결과를 함께 확인해야 한다.'
    else:headline='PPO 학습과 전이 평가는 완료했다. W1의 전체 이득·저부담 감속·최악 위상 및 진자 추종·안정성을 모두 통과한 L1 대체 모델은 아직 확정하지 않았다. 지표별 이득과 손실은 아래 표에서 구분한다.'
    body.append('<div class="callout">'+headline+'</div>')
    if complete and 'F2_s0_20' in ev['walking']['groups']:
        fm=ev['walking']['groups']['F2_s0_20']['metrics'];lm=ev['walking']['groups']['L1_20']['metrics']
        body.append('<p><strong>후속 연구 후보는 F2지만 L1 교체는 보류한다.</strong> ±20 N·m에서 보조 RMS는 '
            f'{lm["assist_rms"]["mean"]:.2f}→{fm["assist_rms"]["mean"]:.2f} N·m, 고관절 τ² 감소는 '
            f'{lm["overall_tau_sq_reduction"]["mean"]:.2f}→{fm["overall_tau_sq_reduction"]["mean"]:.2f} (N·m)²였다. '
            f'그러나 저부담 RMS 증가는 {lm["low10_rms_increase"]["mean"]:.2f}→{fm["low10_rms_increase"]["mean"]:.2f} N·m로 커졌고, 전신 회전관절 합의 이득과 미사용 진자 궤적 추종도 L1에 미달했다. '
            '작은 보조 RMS로 비슷한 평균 고관절 이득을 낸 점과 방해·일반화 실패를 함께 보존한다.</p>')
        body.append('<p>고정한 반복 선정 규칙(같은 상한에서 이득·저부담·최악 위상 동시 충족)을 통과한 설정이 없어 학습 seed 1 반복은 수행하지 않았다. 시간 부족으로 중단한 것이 아니며, 각 설정의 독립 학습 결과는 seed 0 한 개다.</p>')
    body.append('<h2>기존 구조에서 확인한 문제와 이번 변경</h2>')
    body.append('<p>L1은 이미 고조파·정지·비주기 연결·관성 무작위화를 사용한다. 각 정상 세그먼트의 관성 토크는 75 N·m 이하로 제한하지만, 두 궤적을 연결하는 미분항은 이 제한에 포함되지 않는다. F1은 기존 주파수·진폭·외란 분포를 유지하고 연결부만 C2 연속 곡선으로 바꾸며, 추가 관성 토크의 상계를 40 N·m로 제한한다. 이는 중력·외란을 모두 포함한 완전한 역동역학 실현 가능성 보장은 아니다.</p>')
    body.append('<p>G4는 절약 비용뿐 아니라 관절별 공유 actor, 외란, 학습 출력(23 N·m), 학습량(3,000 epoch)이 L1과 다르다. 포화된 tanh 출력에서는 raw action 변화 비용과 물리 토크 변화가 같은 양이 아니다. 비용 계수를 키웠다는 사실만으로 W1 방해가 줄어야 한다고 볼 수 없다.</p>')
    body.append('<p>관측·정규화·출력 부호의 계약은 L1과 동일하게 유지했다. 다만 합성 예측은 의도 궤적의 미래이고 실제 H1은 측정된 운동의 미래를 예측한다는 의미 차이가 남는다. 현재 상태·이력만으로 사람이 해야 하는 감속과 외력이 맡는 감속을 항상 구분할 수 있다는 근거는 없다. 이번 실험은 새 센서 정보로 이 문제를 해결하는 실험이 아니다.</p>')
    body.append('<p>저장된 L1 ±20 N·m 기록의 저부담 감속 창에서는 약 89.4%가 보조의 음의 기계적 파워였다. F2는 이 관찰을 바탕으로 흡수 파워를 비용화하되, 제동을 금지하거나 보행 위상에 따라 토크를 차단하지 않는다. 추종·사람 부담의 이득이 크면 제동 보조를 사용할 수 있는 보상 설계다.</p>')
    body.append('<p>기존 경로는 사람 PD와 속도 피드포워드를 따로 제한하고, 보조·외란과 합친 주입량에도 한계를 적용했다. F1/F2는 사람 PD+FF 전체를 한 번 제한하고 보조와 외란을 독립 주입한다. 따라서 L1 대비 변화는 궤적 하나의 인과 효과로 해석하지 않는다. F1/F2 간 유일한 설계 차이는 흡수 파워 비용이다.</p>')
    body.append(table(['항목','L1','G0 / G4','F1 / F2'],[
        ['actor','공동 MLP 512–256–128','G0 공동 / G4 단일 관절 공유','L1 공동 MLP 유지'],
        ['관측','137차원: 상태·250 ms 이력·0/10/20 ms 예측·이전 action','같은 정보; G4는 관절별 분리','L1과 동일, 신규 센서·위상 입력 없음'],
        ['학습 action','40 tanh(a)','23 tanh(a)','40 tanh(a)'],
        ['학습량','6,000 epoch','3,000 epoch','각 6,000 epoch, 4096환경×32 step/epoch'],
        ['학습 seed','0 (보관 config.yaml)','s0=0 / s1=1','각 0; 조건부 독립 반복만 1'],
        ['기본 비용','추종 exp(−40 mean(e²)); 사람 |τω| −0.0016, τ² −3e−5; 장치 τ² −2e−5; raw action 변화 −0.005','G4 장치 비용·변화 비용 4배','L1 기본값; F2만 −0.0032 Σmax(−τassist ω,0) 추가'],
        ['정규화/PPO','관측 running norm ±5, reward norm; Adam actor1e−4/critic5e−4; γ .99, GAE .95, clip .2','G0/G4 bounds loss .001','L1 설정 유지; bounds loss 0'],
        ['주입·사람 한계','과거 합산 제한 경로','독립 주입 + 사람 PD/FF 전체 한계','G0/G4와 같은 올바른 토크 경로'],
        ['W1 출력','0.375배=±15 / 0.5배=±20','15/23·20/23배 (저장 평가)','0.375배=±15 / 0.5배=±20']]))
    a=ev['trajectory_audit']
    body.append('<p>정책을 실행하지 않고 동일 전환 파라미터 4,096개를 검사했다. 기존 관성 토크 최대 '+f'{a["old_cubic_inertial_torque_Nm"]["max"]:.2f} → {a["new_quintic_inertial_torque_Nm"]["max"]:.2f} N·m'+', 전환 시간 중앙값 '+f'{a["requested_duration_s"][1]:.2f} → {a["actual_duration_s"][1]:.2f} s'+'. 전환 중 '+f'{100*a["duration_extended_fraction"]:.1f}%'+ '만 늘어났다. 이는 학습 성능 지표가 아니다.</p>')
    body.append('<h2>W1: 같은 출력 범위에서 이득과 방해</h2><p>동결 W1/H1, 기존 걷기 40초, 10 ms 보조. 표는 평가 seed 42/7의 평균이며, 모든 구간의 착용자 고관절 토크² 감소는 클수록 좋다. 저부담 감속·최악 위상 RMS 증가는 작을수록 좋다. 전체 회전관절 합에는 고관절과 비보조 관절을 함께 포함하며 직선 slide DOF는 제외했다.</p>')
    groups=[g for g in ev['walking']['groups'].values() if g['limit_Nm'] in (15,20)]
    groups.sort(key=lambda g:(g['limit_Nm'],list(COLORS).index(g['model']) if g['model'] in COLORS else 99))
    def val(g,k):return g['metrics'][k]['mean']
    rows=[]
    for g in groups:
        rows.append([g['model'],g['limit_Nm'],f'{val(g,"overall_tau_sq_reduction"):.2f}',f'{val(g,"low10_rms_increase"):.2f}',
            f'{val(g,"worst_phase_rms_increase"):.2f}',f'{val(g,"assist_rms"):.2f}',f'{val(g,"saturation_pct"):.1f}%',
            f'{val(g,"body_error_mm"):.2f}',f'{val(g,"speed"):.3f}',f'{val(g,"all_rotational_sum_tau_sq_reduction"):.1f}',g['falls']])
    body.append(table(['모델','±N·m','고관절 τ² 감소','저부담 RMS 증가','최악 위상 RMS 증가','보조 RMS','포화율','몸체 오차 mm','속도 m/s','회전관절 합 τ² 감소','낙상'],rows))
    if 'F1_s0_15' in ev['walking']['groups']:
        f=ev['walking']['groups']['F1_s0_15'];l=ev['walking']['groups']['L1_10']
        body.append('<p>추가 학습 없이 출력을 줄인 저장 기준선도 함께 보면, F1 ±15 N·m의 고관절 τ² 감소 / 저부담 RMS 증가는 '
            f'{val(f,"overall_tau_sq_reduction"):.2f} (N·m)² / {val(f,"low10_rms_increase"):.2f} N·m, '
            f'L1 ±10 N·m는 {val(l,"overall_tau_sq_reduction"):.2f} (N·m)² / {val(l,"low10_rms_increase"):.2f} N·m다. '
            '이는 출력이 다른 보조 비교이며, 주 비교는 위의 같은 ±15/20 N·m 표다. 포화율 감소만으로 새 학습의 유용성을 판단하지 않는다.</p>')
    fig,ax=plt.subplots(figsize=(9,5.5))
    for g in groups:
        m=g['metrics'];x=m['low10_rms_increase'];y=m['overall_tau_sq_reduction']
        ax.errorbar(x['mean'],y['mean'],xerr=[[x['mean']-x['min']],[x['max']-x['mean']]],
            yerr=[[y['mean']-y['min']],[y['max']-y['mean']]],fmt='o' if g['limit_Nm']==20 else '^',
            color=COLORS.get(g['model'],'gray'),capsize=3,ms=7)
        offset=(-6,-16) if g['model']=='L1' and g['limit_Nm']==20 else (5,5)
        ax.annotate(f'{g["model"]} {g["limit_Nm"]}',(x['mean'],y['mean']),xytext=offset,textcoords='offset points',fontsize=8,
            ha='right' if offset[0]<0 else 'left')
    ax.set(xlabel='저부담 감속 RMS 증가 [N·m] — 작을수록 좋음',ylabel='전체 고관절 τ² 감소 [(N·m)²] — 클수록 좋음',
        title='동일 출력 범위 비교 · 오차막대는 평가 2회의 관측 범위');ax.grid(alpha=.2)
    body.append(fig_html(fig,'walking_tradeoff'))
    body.append('<p>저부담 마스크는 과거 L1 ±20 N·m의 같은 평가 seed 무보조 기록에서 고정했다. 50 ms 중심 이동 RMS≤10 N·m, ωα&lt;0, |ω|&gt;0.3 rad/s, |α|&gt;1 rad/s²이며 ≤5 N·m도 민감도 분석으로 보존한다. 보조 명령 i−1과 착용자 응답 i를 맞춘다. 좌우 위상 분포가 달라 이 마스크를 유각 말기와 동일시하지 않는다.</p>')
    body.append('<details><summary>더 낮은 부담 기준(≤5 N·m)의 민감도와 무보조 기준의 변동</summary>')
    sensitivity=[]
    for g in groups:
        runs=[r for r in ev['walking']['runs'].values() if r['record']['model']==g['model'] and r['record']['limit_Nm']==g['limit_Nm']]
        drift=np.mean([r['segments']['reference_A_difference_rms_Nm'] for r in runs])
        sensitivity.append([g['model'],g['limit_Nm'],f'{val(g,"low5_rms_increase"):.3f}',f'{drift:.3f}'])
    body.append(table(['모델','±N·m','≤5 N·m 마스크 RMS 증가','과거 고정 기준과 새 무보조 A의 토크 RMS 차이'],sensitivity))
    body.append('<p>≤5 N·m 창은 평가 seed당 양쪽 합계 0.95초로 적고 시간적으로 상관된다. 표본 수를 독립 관측 수로 해석하지 않는다. 무보조 A의 변동은 새 모델 효과가 아니라 실행 간 수치·난수 차이다.</p></details>')
    body.append('<h3>다른 관절에 부담이 옮겨갔는가</h3>')
    body.append(table(['모델','±N·m','무릎 τ² 증가','발목 τ² 증가','몸통 τ² 증가','비보조 회전관절 평균 τ² 증가'],[
        [g['model'],g['limit_Nm'],*[f'{g["other_joints"][k]["mean"]:.2f}' for k in ('무릎','발목','몸통','전체 비보조 회전관절')]] for g in groups]))
    body.append('<details><summary>흡수 파워와 고부담 감속 구간의 변화</summary>')
    mechanical=[]
    for g in groups:
        runs=[r for r in ev['walking']['runs'].values() if r['record']['model']==g['model'] and r['record']['limit_Nm']==g['limit_Nm']]
        def segment_mean(segment,key):return float(np.mean([r['segments'][segment][key] for r in runs]))
        mechanical.append([g['model'],g['limit_Nm'],f'{segment_mean("low_decel_10","braking_sample_pct"):.1f}%',
            f'{segment_mean("low_decel_10","absorbed_power_per_joint_W"):.3f}',
            f'{segment_mean("high_decel_20","rms_increase"):.3f}',
            f'{segment_mean("high_decel_20","absorbed_power_per_joint_W"):.3f}'])
    body.append(table(['모델','±N·m','저부담 음의 파워 비율','저부담 흡수 파워 W/관절','고부담 감속 RMS 증가 N·m','고부담 흡수 파워 W/관절'],mechanical))
    body.append('<p>고부담은 같은 고정 무보조 기준에서 RMS&gt;20 N·m인 감속 창이다. 음의 파워 자체가 방해의 정의는 아니다. 필요한 제동을 도울 때에도 음의 파워가 생기므로, F2의 비용은 저부담 손해와 유용한 제동을 함께 바꿀 수 있다.</p></details>')
    for cap in (15,20):
        fig,axs=plt.subplots(1,2,figsize=(11,3.8),sharey=True)
        for side,ax in enumerate(axs):
            for model in ['L1']+[k for k in ev['training'] if any(g['model']==k and g['limit_Nm']==cap for g in groups)]:
                runs=[r for r in ev['walking']['runs'].values() if r['record']['model']==model and r['record']['limit_Nm']==cap]
                if not runs:continue
                y=np.mean([[-row['reduction'] for row in r['rows'][side]] for r in runs],axis=0)
                ax.plot(np.arange(25)*4+2,y,label=model,color=COLORS.get(model))
            ax.axhline(0,color='black',lw=.7);ax.set(title=['오른쪽','왼쪽'][side],xlabel='참조 고관절 위상 [%]');ax.grid(alpha=.2)
        axs[0].set_ylabel('착용자 RMS 증가 [N·m]');axs[1].legend(fontsize=8)
        fig.suptitle(f'±{cap} N·m · 4% 위상 구간 (위상 0은 최대 고관절 굴곡)');fig.tight_layout()
        body.append(fig_html(fig,f'phase_{cap}Nm'))
    body.append('<h2>진자: 보지 않은 궤적과 동역학 조건</h2><p>기존 저장 평가와 동일한 plant/trajectory/H1 seed를 사용하는 128환경×2코호트×10초다. 학습에 쓰지 않은 무작위 궤적을 기존 연결 방식으로 평가한다. phase는 감속·반전 외란, hold50은 사람 명령의 50 ms 유지로 둘 다 새 정책의 학습 조건 밖이다. 아래 L1·F1/F2는 모두 ±20 N·m다. 진자 ±15 N·m의 기존 L1 기록은 없어 재실행하지 않았다. 실패가 있으면 생존 공통 구간의 평균만 보고 성공으로 판단하지 않는다.</p>')
    ps=[p for p in ev['pendulum'] if p['model']=='L1' or p['model'].startswith('F')]
    body.append(table(['모델','조건','각도 RMSE °','보조 RMS N·m','포화율','Δτ RMS N·m/10ms','사람 τ²','실패 무보조/보조'],[
        [p['model'],p['condition'],f'{p["tracking_rmse_deg"]:.3f}',f'{p["assist_rms_Nm"]:.2f}',f'{p["saturation_pct"]:.1f}%',
         f'{p["delta_tau_rms_Nm_per_10ms"]:.3f}',f'{p["wearer_tau_sq_Nm2"]:.1f}',f'{p["failures"]["reference"]}/{p["failures"]["assisted"]} / {p["episodes"]}'] for p in ps]))
    body.append('<details><summary>과거 G0/G4 진자 결과 (출력·seed가 달라 직접 순위 비교 불가)</summary>')
    body.append(table(['모델','조건','출력 ±N·m','각도 RMSE °','보조 RMS','포화율','Δτ RMS/10ms'],[
        [p['model'],p['condition'],p['limit_Nm'],f'{p["tracking_rmse_deg"]:.3f}',f'{p["assist_rms_Nm"]:.2f}',f'{p["saturation_pct"]:.1f}%',f'{p["delta_tau_rms_Nm_per_10ms"]:.3f}'] for p in ev['pendulum'] if p['model'].startswith('G')]))
    body.append('<p>G0는 plant/trajectory/H1=34011/41011/777, L1·G4 test와 신규 모델은 63031/52021/991. 모두 원본 그대로 보존한다.</p></details>')
    body.append('<h2>학습 결과와 실패 기록</h2>')
    tr=[]
    for model,t in ev['training'].items():
        s=t['summary'];latest=max((x[-1][0] for x in t['series'].values() if x),default=0)
        tr.append([model,latest,f'{s["info/episode_length"]["last500_mean"]:.1f}',
            f'{s["env/raw_r/tracking_mean"]["last500_mean"]:.4f}',
            f'{s["env/assist/injection_error_max_Nm"]["max"]:.6f}',
            sum(v['nonfinite'] for v in s.values())])
    body.append(table(['모델','기록 epoch','후반 평균 episode 길이/1000','후반 추종 reward','최대 주입오차 N·m','비유한 로그값'],tr))
    fig,axs=plt.subplots(2,2,figsize=(11,7))
    for ax,(tag,title) in zip(axs.flat,[('env/raw_r/tracking_mean','추종 reward'),('env/raw_r/human_torque_sq_mean','사람 양쪽 τ² 합'),('env/raw_r/assist_effort_mean','보조 양쪽 τ² 합'),('info/episode_length','episode 길이')]):
        for model,t in ev['training'].items():
            data=np.asarray(t['series'].get(tag,[]))
            if len(data)<5:continue
            width=min(100,len(data));y=np.convolve(data[:,1],np.ones(width)/width,mode='valid')
            ax.plot(data[width-1:,0],y,label=model,color=COLORS.get(model))
        ax.set(title=title,xlabel='epoch');ax.grid(alpha=.2)
    axs[0,0].legend();fig.suptitle('학습 중 기록 · 100 epoch 이동 평균, 평가 성능과 구분');fig.tight_layout()
    body.append(fig_html(fig,'training_curves'))
    body.append('<p>실행 오류 기록 '+str(len(ev['runtime_failures']))+'건. '+('원인·재개 명령·실패 로그는 evidence.json 및 runtime_failures.json에 보존했다.' if ev['runtime_failures'] else '현재 저장된 프로세스 실패 기록은 없다.')+'</p>')
    if ev['runtime_failures']:
        body.append(table(['작업','프로세스 반환 코드','실패 시도','보존 로그'],[
            [e['label'],e['returncode'],e['attempt'],escape(e['log'])] for e in ev['runtime_failures']]))
        body.append('<p>F1 phase 평가의 SIGILL(−4)은 시뮬레이터 상태 변환 중 발생했다. 같은 명령·seed·checkpoint로 처음부터 재실행해 완료했으며, 원인은 아직 확정하지 못했다. 물리 실패 횟수와 프로세스 오류를 구분한다. 복구 기록은 <a href="../../results/llp_feasible_20261020/recovery_F1_phase.json">recovery_F1_phase.json</a>에 있다.</p>')
        if 'F2_s0_15Nm_seed42' in ev['execution'] and ev['execution']['F2_s0_15Nm_seed42']['attempt']>1:
            body.append('<p>F2 ±15 N·m·평가 seed 42에서도 접촉 센서 계산의 Warp launch 중 SIGSEGV(−11)가 발생했다. 자동 재실행 2차 시도에서 같은 설정으로 40초를 완료했다. 원인은 미확정이며, 진단 위치와 당시 GPU 상태는 <a href="../../results/llp_feasible_20261020/runtime_diagnosis.json">runtime_diagnosis.json</a>에 보존했다.</p>')
    if ev['orchestration_interruptions']:
        body.append('<p>별도로 마지막 F2 hold50 평가를 시작하던 실행 세션이 종료 코드 143으로 끝났다. 자식 평가의 종료 코드는 확인할 수 없었고, 명령만 기록된 상태였다. 완료된 작업을 건너뛰고 같은 마지막 평가만 다시 실행했다. 시뮬레이터 오류 두 건과 구분한 기록은 <a href="../../results/llp_feasible_20261020/orchestration_interruptions.json">orchestration_interruptions.json</a>에 있다.</p>')
    body.append('<h2>판정 범위와 다음 실험</h2>')
    if all(f'F2_s0_{cap}' in ev['walking']['groups'] for cap in (15,20)):
        contrast=[]
        for cap in (15,20):
            f1=ev['walking']['groups'][f'F1_s0_{cap}'];f2=ev['walking']['groups'][f'F2_s0_{cap}']
            contrast.append([cap,*[f'{val(f2,k)-val(f1,k):+.3f}' for k in ('overall_tau_sq_reduction','low10_rms_increase','worst_phase_rms_increase','assist_rms')]])
        body.append('<p>설정 차이가 보상항 하나인 F1→F2 비교다. 이득 차이는 양수가 좋고, 저부담·최악 위상 차이는 음수가 좋다. 학습 seed 하나의 관찰이므로 반복 학습의 평균 효과로 일반화하지 않는다.</p>')
        body.append(table(['실제 출력 ±N·m','고관절 τ² 감소량의 차이','저부담 RMS 증가량의 차이','최악 위상 RMS 증가량의 차이','보조 RMS 차이'],contrast))
        means={}
        for model in ('F1_s0','F2_s0'):
            rr=[r for r in ev['walking']['runs'].values() if r['record']['model']==model and r['record']['limit_Nm']==15]
            means[model]={k:float(np.mean([r['segments']['low_decel_10'][k] for r in rr])) for k in ('absorbed_power_per_joint_W','rms_increase')}
        a,b=means['F1_s0'],means['F2_s0']
        body.append('<p>±15 N·m의 저부담 감속 창에서 흡수 파워는 '
            f'{a["absorbed_power_per_joint_W"]:.2f}→{b["absorbed_power_per_joint_W"]:.2f} W/관절, 착용자 RMS 증가는 '
            f'{a["rms_increase"]:.2f}→{b["rms_increase"]:.2f} N·m였다. '
            '흡수 파워 감소와 부담 감소가 서로 다른 목표임을 보여 준다. 이 비용 하나로 불필요한 보조를 구분할 수 있다고 판단하지 않는다.</p>')
        body.append('<p>F2 ±20 N·m는 후반의 방해를 줄였지만 왼쪽 참조 위상 28–32%에 새 최악 구간이 생겼다. 이 구간의 착용자 RMS 증가는 '
            f'{val(ev["walking"]["groups"]["F2_s0_20"],"worst_phase_rms_increase"):.2f} N·m로, '
            'L1의 왼쪽 84–88% 방해를 해결한 범용 정책이 아니라 방해 위치와 보조 분배가 바뀐 정책으로 해석한다. 위상은 참조 고관절 각도 기준이며 발 접촉 기준이 아니다.</p>')
    body.append('<p>최종 채택 여부는 고관절 평균 이득과 저부담 감속·최악 위상, 추종·낙상, 다른 관절로 옮겨간 부담을 함께 본다. 진자 개선만으로 W1 전이 개선이라 부르지 않는다. 흡수 파워 비용은 필요한 제동도 줄일 수 있어 급정지·반전 추종의 손해를 반드시 함께 확인한다. 현재 운동 관측만으로 외력이 맡는 감속과 사람이 맡는 감속을 완전히 구분한다는 검증은 아니다.</p>')
    body.append('<p>단순 진자 식에서 Iq̈ + g(q) + bq̇ − τassist = τhuman + τexternal이다. 관성과 중력까지 안다고 해도 같은 운동·보조 이력에 같은 우변 합을 만드는 두 토크 분해가 가능하다면, 운동 관측만으로 그 분해를 유일하게 정할 수 없다. 이는 현재 정책이 항상 실패한다는 증명이 아니라, 부하의 시간 패턴 같은 추가 가정이나 측정 없이 비용 계수만으로 모든 상황의 구분을 보장하기 어렵다는 구조적 한계다.</p>')
    body.append('<p>후속 우선순위 1: 같은 ±40 N·m·6,000 epoch에서 올바른 토크 경로만 적용한 대조 모델을 두어, 이번 C2 궤적 연결의 효과를 분리한다. F1의 낮은 출력이 궤적 수정 때문인지 토크 처리 수정 때문인지는 이번 두 설정만으로 확정할 수 없다.</p>')
    body.append('<p>후속 우선순위 2: 같은 목표·관성·외란을 받는 무보조/보조 진자 쌍에서, 무보조 대비 사람 토크²가 증가한 양을 직접 비용화하는 실험을 검토한다. 저부담에서 분모가 0에 가까워지지 않도록 RMS 창과 하한을 두고 추종 보상을 유지한다. 외력이 감속을 맡는 조건과 사람이 감속해야 하는 조건을 별도로 평가해야 한다. 단순히 음의 파워나 비용 계수를 더 키우는 실험과 구분하며, 이 후속 학습은 이번에 실행하지 않았다.</p>')
    body.append('<ul>'+''.join('<li>'+escape(x)+'</li>' for x in ev['limitations'])+'</ul>')
    body.append('<h2>재현 경로와 실행 명령</h2><p>모든 학습·평가 명령은 execution.json, 설정과 소스 해시는 manifest.json에 있다. 기존 출력 경로로 실행하면 해당 실행을 재개하므로 독립 재현 시 experiment-name과 평가 out을 새 경로로 바꾼다.</p>')
    body.append('<p><a href="../../results/llp_feasible_20261020/manifest.json">고정 계획</a> · <a href="../../results/llp_feasible_20261020/execution.json">정확한 명령·소요 시간</a> · <a href="../../results/llp_feasible_20261020/config_audit.json">설정 차이</a> · <a href="evidence.json">전체 수치</a></p>')
    for label,run in ev['execution'].items():
        if label.endswith('_train'):
            body.append('<details><summary>'+escape(label)+'</summary><pre>'+escape(shlex.join(run['command']))+'</pre></details>')
    body.append('<p>정책 파일: results/llp_feasible_F1_s0_20261020/epoch_6000.ckpt, results/llp_feasible_F2_s0_20261020/epoch_6000.ckpt. 독립 seed 1은 repeat_selection.json에서 실행 여부를 확인한다.</p>')
    body.append('<script type="application/json" id="evidence">'+json.dumps(ev,ensure_ascii=False,allow_nan=False).replace('</','<\/')+'</script>')
    css='body{font:16px/1.7 system-ui,sans-serif;color:#213547;background:#f5f7fa;margin:0}main{max-width:1180px;margin:auto;padding:36px 24px}h1{font-size:29px}h2{margin-top:38px}table{border-collapse:collapse;width:100%;background:white;font-size:13px}th,td{padding:9px 10px;border-bottom:1px solid #dbe2ea;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#eaf0f6}.scroll{overflow:auto}.plot svg{width:100%;height:auto}.callout{padding:20px;background:#e7f0f8;border-left:5px solid #3276a5}.muted{color:#607080}details{padding:12px;border:1px solid #dbe2ea;background:white;margin:12px 0}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}a{color:#146eab}'
    destination.write_text('<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LLP F1/F2 사전학습 결과</title><style>'+css+'</style></head><body><main>'+''.join(body)+'</main></body></html>')
    save(destination.parent/'evidence.json',ev)


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',default='results/llp_feasible_20261020');p.add_argument('--out',default='docs/ai_sbc/261020_feasible/report.html')
    args=p.parse_args();ev=collect(ROOT/args.study);render(ev,ROOT/args.out)
    print(json.dumps(dict(status=ev['status']['stage'],comparison=ev['comparison']),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
