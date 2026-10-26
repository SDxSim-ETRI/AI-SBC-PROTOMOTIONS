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
"""Plot completed general-study pendulum evaluations and weighted reward terms."""
from html import escape
import json
import numpy as np
import matplotlib.pyplot as plt

from report_general_phase import svg
from report_general_docs import FOLDER, STUDY, glossary, model_label, shell

DOMAINS = [('legacy','기존 외란'),('phase','혼합 구간 외란'),('hold50','목표 50 ms 유지')]
REWARDS = [('tracking','추종'),('human_torque_sq','사람 토크²'),('human_power','사람 |일률|'),('assist_effort','보조 토크²'),('action_rate','명령 변화')]


def evaluation_label(name):
    split = 'dev' if '_dev_' in name else 'test'
    model, condition = name.split('_'+split+'_')
    predictor, domain = condition.split('_', 1)
    return ' · '.join((model_label(model), '선정용 평가' if split=='dev' else '선정에 쓰지 않은 조건 평가',
                      '실제 각도 예측기 H1' if predictor=='H1' else '합성 잡음 예측', dict(DOMAINS)[domain]))


def comparison_figure(records, split):
    selected={k:v for k,v in records.items() if '_'+split+'_H1_' in k}
    names=sorted({k.split('_'+split+'_')[0] for k in selected},key=lambda n:(n!='L1',n))
    if not names: return ''
    fig,axes=plt.subplots(3,3,figsize=(12,max(7,len(names)*.65+3)),layout='constrained')
    metrics=[('tau_sq_Nm2','사람 토크² [(N·m)²]'),('tracking_rmse_deg','추종 RMS 오차 [°]'),('abs_work_W','사람 |일률| [W]')]
    for col,(domain,title) in enumerate(DOMAINS):
        for row,(metric,ylabel) in enumerate(metrics):
            ax=axes[row,col]
            for i,name in enumerate(names):
                r=selected.get(name+'_'+split+'_H1_'+domain)
                if not r:continue
                m=r['metrics']['all']; a=m['reference_'+metric]; b=m['assisted_'+metric]
                ax.plot([a,b],[i,i],color='#c9d1dc',lw=3)
                ax.plot(a,i,'o',color='#888',ms=6)
                ax.plot(b,i,'o',color='#c64448' if b>a else '#237b5b',ms=6)
                if row==0:
                    ax.annotate(f" 실패 {r['failures']['assisted']}/{r['total_episodes']}",(max(a,b),i),xytext=(4,0),textcoords='offset points',fontsize=7,va='center')
            labels=[model_label(n, multiline=True) for n in names] if col==0 else ['']*len(names)
            ax.set_yticks(range(len(names)),labels,fontsize=8);ax.invert_yaxis();ax.set_xlabel(ylabel);ax.grid(axis='x',alpha=.2);ax.margins(x=.4)
            if row==0:ax.set_title(title)
    return svg(fig,'pendulum_'+split)


def rewards_figure(records):
    names=sorted(k for k in records if k.endswith('_dev_H1_phase'))
    if not names:return ''
    fig,axes=plt.subplots(1,2,figsize=(12,max(3,len(names)*.65+1)),layout='constrained')
    colors=['#245aa0','#c64448','#b86636','#8655a1','#6c737f']
    y=np.arange(len(names))
    left=np.zeros(len(names))
    positive=np.array([records[n]['reward_audit']['tracking']['effective_mean'] for n in names])
    axes[0].barh(y,positive,color=colors[0],label='추종 보상')
    for (key,label),color in zip(REWARDS[1:],colors[1:]):
        vals=np.array([records[n]['reward_audit'][key]['effective_mean'] for n in names])
        axes[1].barh(y,vals,left=left,color=color,label=label);left+=vals
    for i,ax in enumerate(axes):
        labels=[model_label(n.split('_dev_')[0], multiline=True) for n in names] if i==0 else ['']*len(names)
        ax.set_yticks(y,labels,fontsize=9);ax.invert_yaxis();ax.set_xlabel('가중치·clip 적용 후 평균 reward 기여');ax.legend(fontsize=8);ax.grid(axis='x',alpha=.15)
    axes[0].set_title('양수 보상');axes[1].set_title('음수 비용의 구성')
    return svg(fig,'reward_audit')


def main():
    path=STUDY/'comparison.json'
    records=json.loads(path.read_text()) if path.exists() else {}
    body='<h1>진자 일반화 평가와 보상 크기</h1>'+glossary()
    body+='<p>이 문서는 합성 진자에서의 후보 선정 근거다. 실제 궁금한 보행 구간별 효과는 <a href="walking_phase_report.html">보행 phase 보고서</a>에서 본다. 새 모델은 ±23 N·m, L1은 ±20 N·m다. PA 추가학습 가중치는 사용하지 않는다.</p>'
    body+=f'<p class="status">완료된 평가 {len(records)}개. 미완료 조건은 빈칸으로 남긴다. 아래 수치는 보조·무보조가 함께 생존한 구간이며, 실패 횟수와 함께 판단한다.</p>'
    confirmation=STUDY/'confirmation.json'
    if confirmation.exists() and not json.loads(confirmation.read_text()).get('passes_L1'):
        body+='<p><strong>최종 판정: G4는 이번 설계 중 선정 점수가 가장 좋았지만 L1을 대체하지 못했다.</strong> 별도 실제 H1·혼합 외란에서 사람 토크²는 L1 1256.40, G4의 첫 학습/독립 재학습 순서로 1371.80 / 1352.69 (N·m)², 추종 RMS 오차는 L1 2.642°, G4 3.194° / 3.237°다. 목표 50 ms 유지에서는 L1과 G1/G4 모두 보조 실패 1/256회였다. <a href="FINAL_RESULTS.html">최종 판단·한계</a></p>'
    body+='<h2>1. 실제 H1 입력에서의 부담·추종·일률</h2><p>회색 점은 무보조, 녹색 점은 감소한 보조 결과, 빨간 점은 증가한 보조 결과다. 세 지표는 낮을수록 좋다. 토크²·일률은 관절당 평균이다. 실패 수는 보조 환경의 실패한 시행 수다. 같은 행의 모델을 세 외란 조건에서 비교한다.</p>'
    for split,title in [('dev','선정용 조건'),('test','선정에 쓰지 않은 평가 난수 조건')]:
        figure=comparison_figure(records,split)
        if figure:body+='<h3>'+title+'</h3>'+figure
    if not records:body+='<p>아직 완료된 본 평가가 없다. 사전 검사 결과를 학습 완료 모델의 성능으로 섞지 않는다.</p>'
    body+='<h2>2. 음수 reward가 실제로 얼마나 기여했나</h2><p>선정용 실제 H1·혼합 구간 외란 평가에서 학습 설정의 가중치와 clip을 적용한 평균이다. 음수 계수 숫자만의 비교가 아니다. 양수·음수 그림의 축 범위가 다르므로 축 숫자를 함께 읽는다. raw action bounds loss는 optimizer 손실로, 환경 reward 그림에는 포함하지 않는다.</p>'+rewards_figure(records)
    if all(n+'_s0_dev_H1_phase' in records for n in ('G2','G3','G4')):
        body+='<p><strong>비용을 키우는 것만으로는 해결되지 않았다.</strong> G2의 사람 토크²·일률 비용 기여는 −0.089 / −0.103으로 이미 측정 가능한 크기다. 이를 2배로 한 G3은 출력 포화가 48.97% → 59.59%로 늘었다. 보조 절약·명령 변화 비용을 4배로 한 G4는 포화를 31.81%로 줄였으나 L1과 보행 phase 기준을 넘지 못했다. 포화는 한계 95% 초과 비율이다. 이 평가의 항별 clip 비율은 G3 최대 0.0316%, 나머지는 0%로, 비용 대부분이 clip된 상황은 아니다.</p>'
    recovery=STUDY/'recoveries.json'
    if recovery.exists():
        body+='<details><summary>학습 중단·재개 기록</summary><p>프로세스 오류 후 같은 실험의 저장된 모델·optimizer에서 이어서 학습했다. PA 가중치를 가져온 추가학습이 아니다. 시뮬레이터와 난수는 재초기화되며 재개 직후 정책 갱신 한 번을 생략하므로, 중단 없는 실행과 비트 단위로 같은 궤적은 아니다. 고정 마지막 epoch에서 평가한다.</p><pre>'+escape(recovery.read_text())+'</pre></details>'
    for filename,title in [('selection.json','진자 후보 선정'),('confirmation.json','독립 재학습·L1 비교 판정')]:
        p=STUDY/filename
        if p.exists():body+='<details><summary>부록 · '+title+'</summary><pre>'+escape(p.read_text())+'</pre></details>'
    body+='<details><summary>부록 · 전체 평가 수치, 보상 가중치·clip·포화·구간 표본 수</summary>'
    body+='<p>dev: 후보 선정용 평가 난수. test: 환경 물성·목표 궤적·H1 예측의 난수를 바꾼 별도 평가. emulator: 실제 H1 대신 합성 잡음 예측을 주는 진단. legacy: 기존 외란. phase: 혼합 구간 외란. hold50: 사람 목표를 50 ms마다 갱신. injection error: 요청된 보조 토크와 실제 주입값의 차이. saturation: 해당 모델 한계의 95% 초과 비율.</p>'
    for name,r in records.items():
        m=r['metrics']['all']
        body+=f'<details><summary>{escape(evaluation_label(name))} · 보조 실패 {r["failures"]["assisted"]}/{r["total_episodes"]}</summary><p>원본 식별자: <code>{escape(name)}</code></p><pre>'+escape(json.dumps(r,ensure_ascii=False,indent=2))+'</pre></details>'
    body+='</details>'
    body+='<p>실제 heel strike 접촉·순간 관성 변화는 이 진자에 구현하지 않았다. 합성 감속·반전 감쇠 사건을 실제 보행 phase와 동일시하지 않는다. 진자 통과 여부와 무관하게 완료된 후보의 W1 걷기 진단은 별도로 남긴다.</p>'
    (FOLDER/'pendulum_results.html').write_text(shell('진자 평가·보상 기여 · 23 N·m LLP',body))
    print('Pendulum report:',len(records),'evaluations')


if __name__=='__main__':
    main()
