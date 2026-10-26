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
"""Prepare a three-slide HTML briefing from verified walking results."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/presentation_three_slides_20261023'
A=OUT/'assets'
NAMES=['기본형','효율형'];COLORS=['#4274b8','#12866e'];KEYS=['L1_20','R0_s0_20']
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
plt.rcParams.update({'font.family':FontProperties(fname=FONT).get_name(),'font.size':15,'axes.unicode_minus':False,'svg.fonttype':'path','axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#233745','text.color':'#233745'})


def save(fig,name):
    for ext in ['svg','png']:fig.savefig(A/(name+'.'+ext),dpi=180,facecolor='white',bbox_inches='tight',pad_inches=.18)
    plt.close(fig)


def figures():
    archive=json.loads((ROOT/'results/llp_relative_20261020/walking/comparison.json').read_text());g=archive['groups']
    fig,ax=plt.subplots(figsize=(8.3,3.4));fig.subplots_adjust(left=.15,right=.95,bottom=.23,top=.93)
    for key,name,c in zip(KEYS,NAMES,COLORS):
        m=g[key]['metrics'];x=m['assist_rms']['mean'];y=m['overall_tau_sq_reduction']['mean']
        ax.scatter(x,y,s=135,c=c,zorder=3)
        ax.annotate(f'{name}  {y:.1f}\nRMS {x:.2f} N·m',(x,y),xytext=(-15,-54),textcoords='offset points',ha='right' if key=='L1_20' else 'center',fontsize=15,color=c,fontweight='bold')
    ax.set(xlabel='보조 강도 (RMS) [N·m]',ylabel='고관절 τ² 감소 [(N·m)²]',xlim=(0,20),ylim=(0,340));ax.grid(alpha=.18);save(fig,'01_same_limit')
    fig,axs=plt.subplots(1,2,figsize=(10,3.8),sharey=True);fig.subplots_adjust(left=.10,right=.98,bottom=.23,top=.80,wspace=.14)
    selected=[]
    for model in ['L1','R0_s0']:
        selected.append([r for r in archive['runs'].values() if r['record']['model']==model and r['record']['limit_Nm']==20])
    for side,ax in enumerate(axs):
        ax.axhspan(-20,0,color='#fbe9e5');ax.axhline(0,color='#72828e',lw=1)
        for rows,name,c in zip(selected,NAMES,COLORS):
            y=np.mean([[r['rows'][side][b]['reduction'] for b in range(25)] for r in rows],axis=0)
            ax.plot((np.arange(25)+.5)*4,y,color=c,lw=2.5,label=name)
        ax.set(title=['오른쪽','왼쪽'][side],xlabel='보행 주기 [%]',xlim=(0,100),ylim=(-20,35));ax.grid(axis='y',alpha=.15)
    axs[0].set_ylabel('착용자 토크 크기 감소 [N·m]');h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,ncol=2,loc='upper center',frameon=False,bbox_to_anchor=(.56,1.02));save(fig,'02_phase')
    fig,ax=plt.subplots(figsize=(6.5,3.8));fig.subplots_adjust(left=.16,right=.98,bottom=.18,top=.80)
    for i,(rows,name,c) in enumerate(zip(selected,NAMES,COLORS)):
        cats=['무릎','발목','몸통'];v=[np.mean([next(q['B']-q['A'] for q in r['other_joints'] if q['group']==group) for r in rows]) for group in cats]
        x=np.arange(3)+(i-.5)*.32;b=ax.bar(x,v,width=.30,color=c,label=name)
        ax.bar_label(b,fmt='%+.0f',padding=4,fontsize=13)
    ax.set(xticks=range(3),xticklabels=cats,ylabel='부위별 평균 τ² 변화 [(N·m)²]',ylim=(-260,190));ax.axhline(0,c='#72828e',lw=1);ax.grid(axis='y',alpha=.15);ax.legend(ncol=2,loc='upper center',bbox_to_anchor=(.5,1.28),frameon=False);save(fig,'03_other_joints')
    fig,axs=plt.subplots(1,2,figsize=(9,5));fig.subplots_adjust(left=.11,right=.98,bottom=.16,top=.86,wspace=.25)
    for ax,metric,title in zip(axs,['low10_rms_increase','worst_phase_rms_increase'],['低부담 감속 구간'.replace('低','저'),'가장 방해가 큰 구간']):
        v=[g[k]['metrics'][metric]['mean'] for k in KEYS];bars=ax.bar(range(2),v,color=COLORS,width=.55);ax.bar_label(bars,fmt='%.2f',padding=6,fontsize=20,fontweight='bold')
        ax.set(xticks=[0,1],xticklabels=NAMES,title=title,ylim=(0,18));ax.grid(axis='y',alpha=.15)
    axs[0].set_ylabel('착용자 토크 크기 증가 [N·m]');save(fig,'04_remaining')
    # Same measurements and layout as the ON/OFF/ON report, with audience names.
    runs=json.loads((ROOT/'results/presentation_toggle_20261023/runs.json').read_text())
    fig,axs=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey='row');fig.subplots_adjust(left=.11,right=.98,bottom=.12,top=.83,wspace=.18,hspace=.18)
    end=np.rint((5+np.arange(900)/60)*400).astype(int);t=np.arange(900)/60
    for col,(model,name) in enumerate(zip(['L1','R0'],NAMES)):
        cs=[]
        for i,cond in enumerate(['on','toggle']):
            z=np.load(ROOT/Path(runs[model+'_'+cond]['path'])/'traces.npz');p=np.mean(z['physics_human_tau'][:,1][:,[0,5]]**2,axis=1);c=np.r_[0,np.cumsum(p,dtype=np.float64)];cs.append(c)
            rms=np.sqrt((c[end]-c[end-400])/400);axs[0,col].plot(t,rms,c=['#12866e','#d46d42'][i],label=['계속 ON','ON → OFF → ON'][i],lw=2)
        excess=(cs[1][end]-cs[1][2000]-cs[0][end]+cs[0][2000])/400;axs[1,col].plot(t,excess,c='#d46d42',lw=2)
        axs[0,col].set(title=name,ylim=(0,70));axs[1,col].set(ylim=(-500,2500),xlabel='영상 시간 [s]')
        for ax in axs[:,col]:
            ax.axvspan(5,10,color='#faebe5');ax.axvline(5,c='#8b969d',ls=':',lw=1);ax.axvline(10,c='#8b969d',ls=':',lw=1);ax.grid(alpha=.15);ax.set_xlim(0,15)
    axs[0,0].set_ylabel('착용자 고관절 토크 크기\n직전 1초 RMS [N·m]');axs[1,0].set_ylabel('계속 ON 대비 추가 부담 누계\n[(N·m)²·s]')
    h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,ncol=2,loc='upper center',bbox_to_anchor=(.55,.96),frameon=False);save(fig,'05_toggle')


def videos():
    records=[]
    for model,name in zip(['L1','R0'],NAMES):
        for mode,folder,file in [('cap','presentation_cap_20261023',model+'_ON_vs_OFF_with_torque.mp4'),('toggle','presentation_toggle_20261023',model+'_ON_OFF_ON.mp4')]:
            source=ROOT/'results'/folder/'deliverables'/file;target=A/f'{model}_{mode}.mp4'
            filters=[]
            for panel in range(2):
                label=name+' · '+('고관절 힘 제한' if mode=='cap' else ('계속 ON' if panel==0 else 'ON → OFF → ON'))
                filters+=[f'drawbox=x={panel*960}:y=0:w=680:h=73:color=white:t=fill',f"drawtext=fontfile={FONT}:text='{label}':fontcolor=0x253545:fontsize=27:x={panel*960+25}:y=20"]
            subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(source),'-vf',','.join(filters),'-an','-c:v','libx264','-preset','fast','-crf','18','-threads','4','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],check=True)
            subprocess.run(['ffmpeg','-y','-loglevel','error','-ss','3','-i',str(target),'-frames:v','1',str(target.with_suffix('.jpg'))],check=True)
            records.append({'source':str(source),'output':str(target),'change':'Header labels only; body motion, status timing and torque meters unchanged'})
            print('VIDEO',target.name,flush=True)
    (OUT/'media_sources.json').write_text(json.dumps(records,indent=2,ensure_ascii=False))


if __name__=='__main__':
    figures()
    if '--figures-only' not in sys.argv:videos()
