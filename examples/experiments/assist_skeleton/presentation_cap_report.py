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
"""Audit the fixed wearer-cap experiment and prepare truthful media evidence."""

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from presentation_off_media import comparison
from presentation_off_report import first_fall

STUDY = Path('results/presentation_cap_20261023')
OUT = STUDY / 'deliverables'


def analyze(runs):
    rows = {}
    traces = {}
    for key, rec in runs.items():
        with np.load(Path(rec['path']) / 'traces.npz') as f:
            tr = dict(f)
        traces[key] = tr
        assert len(tr['t']) == 2000 and len(tr['physics_human_tau']) == 8000
        for field in ['physics_human_tau', 'physics_assist_tau', 'presentation_body_pos', 'presentation_body_rot']:
            assert np.isfinite(tr[field]).all(), (key, field)
        assert np.allclose(np.linalg.norm(tr['presentation_body_rot'], axis=-1), 1, atol=2e-4)
        hips = [tr['presentation_dof_names'].tolist().index(n) for n in ['hip_flexion_r', 'hip_flexion_l']]
        human = tr['physics_human_tau'][:, 1]
        assist = tr['physics_assist_tau'][:, 1]
        cap = tr['hip_effort_caps']
        assert (np.abs(human[2000:, hips])-cap).max() < 1e-3
        assert np.max(np.abs(assist)) <= 20.0001
        assert np.allclose(human, tr['physics_raw_pd_tau'][:,1], atol=1e-4)
        if key.endswith('off'):
            assert np.max(np.abs(assist[4000:])) == 0
        sat = (np.abs(human[:, hips]) >= cap-1e-3)
        other = [i for i,n in enumerate(tr['presentation_dof_names']) if i not in hips and not n.startswith('slide')]
        fall = first_fall(tr['t'], tr['root_z'][:,1])
        row = {'caps_right_left_Nm': cap.tolist(), 'fall_video_s': None if fall is None else fall-5,
               'PD_and_other_limits_preserved': json.loads(str(tr['hip_effort_cap_audit'])),
               'peak_after_cap_Nm': np.abs(human[2000:,hips]).max(0).tolist()}
        for name, a,b in [('before_OFF',5,10),('after_OFF',10,20)]:
            lo,hi = a*400,b*400
            mask = (tr['body_err_t']>a)&(tr['body_err_t']<=b)
            rootmask = (tr['t']>=a)&(tr['t']<b)
            xy = tr['root_xy'][rootmask,1]
            # This is horizontal path speed, not reference-forward projection.
            speed = np.linalg.norm(np.diff(xy,axis=0),axis=1).mean()/0.01
            row[name] = {'hip_tau_squared_mean_Nm2': float(np.mean(human[lo:hi,hips]**2)),
                         'assist_RMS_Nm': float(np.sqrt(np.mean(assist[lo:hi,hips]**2))),
                         'hip_saturation_fraction_right_left': sat[lo:hi].mean(0).tolist(),
                         'body_tracking_error_mean_mm':float(tr['body_err'][mask,1].mean()*1000),
                         'other_21_rotational_dof_tau_squared_sum_mean_Nm2':float(np.mean(np.sum(human[lo:hi,other]**2,axis=1))),
                         'horizontal_path_speed_m_s':float(speed),
                         'minimum_pelvis_height_m':float(tr['root_z'][rootmask,1].min())}
        rows[key] = row
    for model in ['L1','R0']:
        on,off=traces[model+'_on'],traces[model+'_off']
        assert np.array_equal(on['hip_effort_caps'],off['hip_effort_caps'])
        mask=(on['t']>=5)&(on['t']<10)
        p=on['presentation_body_names'].tolist().index('pelvis')
        delta=(on['presentation_body_pos'][mask,1]-on['presentation_body_pos'][0,1,p])-(off['presentation_body_pos'][mask,1]-off['presentation_body_pos'][0,1,p])
        rows[model+'_off']['pre_OFF_pair_pose_RMS_after_initial_translation_alignment_mm']=float(np.sqrt(np.mean(delta**2))*1000)
    (OUT/'metrics_and_validation.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n')
    plt.rcParams.update({'font.family':FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name(),
                        'font.size':14,'axes.unicode_minus':False,'svg.fonttype':'path','axes.spines.top':False,'axes.spines.right':False})
    fig, axes=plt.subplots(2,2,figsize=(16,9),sharex=True)
    fig.subplots_adjust(left=.08,right=.97,top=.82,bottom=.13,hspace=.3,wspace=.22)
    for col,model in enumerate(['L1','R0']):
        for side in range(2):
            ax=axes[side,col]
            for cond,color,label in [('on','#16867b','ON 유지'),('off','#d46d42','5초 후 OFF')]:
                tr=traces[model+'_'+cond];hip=[0,5][side]
                x=np.arange(8000)/400-5;sel=x>=0
                ax.plot(x[sel],tr['physics_human_tau'][sel,1,hip],color=color,lw=.8,alpha=.85,label=label)
            cap=tr['hip_effort_caps'][side]
            ax.axhline(cap,color='#637386',ls='--',lw=1);ax.axhline(-cap,color='#637386',ls='--',lw=1)
            ax.axvline(5,color='#333333',ls=':',lw=1.4)
            ax.set(title=f'{model} · {"오른쪽" if side==0 else "왼쪽"} · 한계 ±{cap:.1f} N·m',xlim=(0,15),ylim=(-210,210))
            ax.grid(alpha=.15)
    for ax in axes[1]:ax.set_xlabel('영상 시간 [s]')
    for ax in axes[:,0]:ax.set_ylabel('착용자 고관절 토크 [N·m]')
    axes[0,0].legend(loc='upper right',ncol=2,frameon=False)
    fig.suptitle('자세 보정은 유지하고, 고관절 힘의 최대값만 제한',x=.08,ha='left',fontsize=25,y=.97,fontweight='bold')
    fig.text(.08,.89,'점선: 기존 보조 ON 기록의 절대 최대값 · 세로선: 오른쪽 영상의 보조 OFF · 400 Hz 실제 토크',fontsize=14)
    for ext in ['png','svg']:fig.savefig(OUT/('graph_01_hip_torque_caps.'+ext),dpi=180,facecolor='white')
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,7))
    fig.subplots_adjust(left=.07,right=.97,top=.77,bottom=.22,wspace=.36)
    specs=[('hip_tau_squared_mean_Nm2','착용자 고관절 부담','평균 τ² [(N·m)²]'),
           ('body_tracking_error_mean_mm','전신 동작 추종 오차','평균 오차 [mm]'),
           ('other_21_rotational_dof_tau_squared_sum_mean_Nm2','나머지 관절 부담','21개 회전축 τ² 합의 평균 [(N·m)²]')]
    for ax,(field,title,ylabel) in zip(axes,specs):
        for i,(cond,label,color) in enumerate([('on','ON 유지','#16867b'),('off','5초 후 OFF','#d46d42')]):
            v=[rows[m+'_'+cond]['after_OFF'][field] for m in ['L1','R0']]
            bars=ax.bar(np.arange(2)+(i-.5)*.34,v,width=.32,label=label,color=color)
            ax.bar_label(bars,fmt='%.1f',padding=4,fontsize=11)
        ax.set(xticks=[0,1],xticklabels=['L1','R0'],title=title,ylabel=ylabel);ax.margins(y=.16);ax.grid(axis='y',alpha=.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5,.88), ncol=2, frameon=False, fontsize=13)
    fig.suptitle('보조 OFF 이후 10초: 자세가 비슷해도 착용자 부담은 달라질 수 있음',x=.07,ha='left',fontsize=23,y=.96,fontweight='bold')
    fig.text(.07,.08,'동일 모델 안에서 같은 출력 한계로 비교 · L1/R0의 착용자 한계는 서로 다름 · 평가 seed 42 한 번\n토크²는 시뮬레이션 부담 지표이며, 근활성·대사량 측정이 아님',fontsize=13)
    for ext in ['png','svg']:fig.savefig(OUT/('graph_02_after_OFF.'+ext),dpi=180,facecolor='white')
    plt.close(fig)
    return rows


def render(runs):
    for key,record in runs.items():
        filename=key+'_clean.mp4'
        cmd=[sys.executable,'examples/experiments/assist_skeleton/presentation_render.py','--run',record['path'],'--out',str(OUT/filename),'--env','1','--camera-env','1','--camera-height-mode','body-center','--start','5','--duration','15']
        with (STUDY/'logs'/f'render_{key}.log').open('w') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        meta_path=(OUT/filename).with_suffix('.json')
        meta=json.loads(meta_path.read_text());meta.pop('policy_or_physics_modified',None)
        meta.update(rendering_changes_dynamics=False,policy_weights_changed=False,wearer_feedback_retained=True,
                    intervention='Fixed hip effort cap from simulation 5s; assist OFF at simulation 10s in off condition only')
        meta_path.write_text(json.dumps(meta,indent=2))
        print('RENDERED',key,flush=True)
    for model in ['L1','R0']:
        comparison(OUT,[model+'_on_clean.mp4',model+'_off_clean.mp4'],[model+' | Hip effort cap',model+' | Hip effort cap'],model+'_ON_vs_OFF.mp4',[False,True])
        print('COMPOSED',model,flush=True)


def main():
    runs=json.loads((STUDY/'runs.json').read_text())
    assert len(runs)==4
    rows=analyze(runs)
    print(json.dumps(rows,indent=2),flush=True)
    if '--analyze-only' not in sys.argv:render(runs)


if __name__=='__main__':main()
