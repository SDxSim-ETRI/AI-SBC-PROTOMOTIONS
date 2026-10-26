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
"""Validate and render a measured ON/OFF/ON walking demonstration."""
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from presentation_off_report import first_fall

S=Path('results/presentation_toggle_20261023');D=S/'deliverables'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
COLORS=['#16867b','#d46d42']


def load_analyze(runs):
    traces={};metrics={}
    for k,r in runs.items():
        with np.load(Path(r['path'])/'traces.npz') as f:tr=dict(f)
        traces[k]=tr
        assert len(tr['t'])==2000 and len(tr['physics_human_tau'])==8000
        assert np.isfinite(tr['presentation_body_pos']).all() and np.isfinite(tr['physics_human_tau']).all()
        assert np.allclose(np.linalg.norm(tr['presentation_body_rot'],axis=-1),1,atol=2e-4)
        assert np.max(np.abs(tr['physics_assist_tau']))<=20.0001
        assert np.allclose(tr['physics_human_tau'],tr['physics_raw_pd_tau'],atol=1e-4)
        if k.endswith('toggle'):
            assert np.max(np.abs(tr['physics_assist_tau'][4000:6000]))==0
            assert np.max(np.abs(tr['physics_assist_tau'][6000:,1]))>1
            assert tr['assist_motor_enabled'][999] and not tr['assist_motor_enabled'][1000]
            assert not tr['assist_motor_enabled'][1499] and tr['assist_motor_enabled'][1500]
        hips=[0,5];other=[i for i,n in enumerate(tr['presentation_dof_names']) if i not in hips and not n.startswith('slide')]
        row={'fall_simulation_s':first_fall(tr['t'],tr['root_z'][:,1]),'stages':{}}
        for stage,lo,hi in [('ON_before',5,10),('OFF',10,15),('ON_restored',15,20)]:
            h=tr['physics_human_tau'][lo*400:hi*400,1];a=tr['physics_assist_tau'][lo*400:hi*400,1]
            sel=(tr['body_err_t']>lo)&(tr['body_err_t']<=hi)
            pos=tr['root_xy'][lo*100:hi*100,1]
            row['stages'][stage]={'hip_tau2_Nm2':float(np.mean(h[:,hips]**2)),
                'assist_RMS_Nm':float(np.sqrt(np.mean(a[:,hips]**2))),
                'tracking_error_mm':float(np.mean(tr['body_err'][sel,1])*1000),
                'other_21_rotation_tau2_sum_Nm2':float(np.mean(np.sum(h[:,other]**2,axis=1))),
                'horizontal_path_speed_m_s':float(np.mean(np.linalg.norm(np.diff(pos,axis=0),axis=1))/.01)}
        metrics[k]=row
    for m in ['L1','R0']:
        on,tog=traces[m+'_on'],traces[m+'_toggle']
        p=on['presentation_body_names'].tolist().index('pelvis')
        delta=(on['presentation_body_pos'][500:1000,1]-on['presentation_body_pos'][0,1,p])-(tog['presentation_body_pos'][500:1000,1]-tog['presentation_body_pos'][0,1,p])
        metrics[m+'_toggle']['pre_OFF_pose_RMS_mm_after_initial_translation_alignment']=float(np.sqrt(np.mean(delta**2))*1000)
        for stage in metrics[m+'_toggle']['stages']:
            metrics[m+'_toggle']['stages'][stage]['hip_tau2_excess_vs_time_matched_ON_Nm2']=metrics[m+'_toggle']['stages'][stage]['hip_tau2_Nm2']-metrics[m+'_on']['stages'][stage]['hip_tau2_Nm2']
    (D/'metrics_and_validation.json').write_text(json.dumps(metrics,indent=2))
    return traces,metrics


def series(traces,model):
    out=[]
    end=np.rint((5+np.arange(900)/60)*400).astype(int)
    for cond in ['on','toggle']:
        p=np.mean(traces[model+'_'+cond]['physics_human_tau'][:,1][:,[0,5]]**2,axis=1)
        c=np.r_[0,np.cumsum(p,dtype=np.float64)]
        out.append((np.sqrt((c[end]-c[end-400])/400),c))
    excess=(out[1][1][end]-out[1][1][2000]-(out[0][1][end]-out[0][1][2000]))/400
    return np.array([x[0] for x in out]),excess


def plots(traces,metrics):
    plt.rcParams.update({'font.family':FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name(),'font.size':14,'axes.unicode_minus':False,'svg.fonttype':'path','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(16,9),sharex=True)
    fig.subplots_adjust(left=.08,right=.96,top=.81,bottom=.15,hspace=.28,wspace=.2)
    for col,m in enumerate(['L1','R0']):
        rms,excess=series(traces,m);t=np.arange(900)/60
        for i,label in enumerate(['계속 ON','ON → OFF → ON']):axes[0,col].plot(t,rms[i],color=COLORS[i],lw=2,label=label)
        axes[1,col].plot(t,excess,color=COLORS[1],lw=2)
        axes[0,col].set_title(m+' · 원래 W1 출력 한계·자세 보정 유지')
        axes[0,col].set_ylim(0,70)
        axes[1,col].axhline(0,color='#8897a1',lw=1)
        axes[1,col].set_xlabel('영상 시간 [s]')
        axes[1,col].set_ylim(-500,2500)
        for ax in axes[:,col]:
            ax.axvspan(5,10,color='#d46d42',alpha=.10);ax.axvline(5,color='#77828d',ls=':',lw=1);ax.axvline(10,color='#77828d',ls=':',lw=1);ax.grid(alpha=.15);ax.set_xlim(0,15)
    axes[0,0].set_ylabel('착용자 고관절 토크 RMS [N·m]\n직전 1초 · 좌우 통합')
    axes[1,0].set_ylabel('계속 ON 대비 추가 부담 누계\n∫Δτ² dt [(N·m)²·s]')
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.5,.9),ncol=2,frameon=False)
    fig.suptitle('같은 보행 중 보조를 끄고, 다시 켜면 착용자 부담은?',x=.08,ha='left',fontsize=25,y=.97,fontweight='bold')
    fig.text(.08,.04,'음영 5–10초: 보조 OFF · 10초: 다시 ON · 누계는 동일 시각의 계속 ON 실행과 비교\n토크 제곱의 시간 적분은 부담 지표이며 에너지·대사량이 아님 · 평가 seed 42 한 번',fontsize=13)
    for ext in ['png','svg']:fig.savefig(D/('graph_toggle_response.'+ext),dpi=180,facecolor='white')
    plt.close(fig)


    fig,axes=plt.subplots(1,2,figsize=(16,8),sharey=True)
    fig.subplots_adjust(left=.08,right=.97,top=.77,bottom=.18,wspace=.16)
    stages=['ON_before','OFF','ON_restored']
    for ax,m in zip(axes,['L1','R0']):
        for i,(cond,label) in enumerate([('on','계속 ON 대조군'),('toggle','ON → OFF → ON')]):
            vals=[metrics[m+'_'+cond]['stages'][stage]['hip_tau2_Nm2'] for stage in stages]
            bars=ax.bar(np.arange(3)+(i-.5)*.34,vals,width=.32,color=COLORS[i],label=label)
            ax.bar_label(bars,fmt='%.0f',padding=4,fontsize=12)
        ax.set(title=m,xticks=[0,1,2],xticklabels=['ON\n0–5초','OFF\n5–10초','다시 ON\n10–15초'],ylim=(0,2000))
        ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('착용자 고관절 평균 τ² [(N·m)²]')
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.5,.86),ncol=2,frameon=False)
    fig.suptitle('보조 OFF에서 부담 증가 → 다시 ON에서 대조군 수준으로 복귀',x=.08,ha='left',fontsize=24,y=.97,fontweight='bold')
    fig.text(.08,.05,'막대는 동일 시간 구간끼리 비교 · 원래 W1의 힘 한계와 자세 보정 유지 · 새 학습 없음\n평가 seed 42 한 번 · 실제 사람의 근활성·대사량이나 다른 보행에서 검증한 결과가 아님',fontsize=13)
    for ext in ['png','svg']:fig.savefig(D/('graph_stage_comparison.'+ext),dpi=180,facecolor='white')
    plt.close(fig)


def render(runs):
    for key,rec in runs.items():
        cmd=[sys.executable,'examples/experiments/assist_skeleton/presentation_render.py','--run',rec['path'],'--out',str(D/(key+'_clean.mp4')),'--env','1','--camera-env','1','--camera-height-mode','body-center','--start','5','--duration','15']
        with (S/'logs'/('render_'+key+'.log')).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
        p=D/(key+'_clean.json');meta=json.loads(p.read_text());meta.pop('policy_or_physics_modified',None);meta.update(rendering_changes_dynamics=False,policy_weights_changed=False,wearer_feedback_retained=True,wearer_effort_limits_unchanged=True,assist_toggle='OFF at simulation 10s, ON at 15s' if key.endswith('toggle') else 'Continuous ON after standard calibration')
        p.write_text(json.dumps(meta,indent=2));print('RENDERED',key,flush=True)


def compose(model):
    cmd=['ffmpeg','-y','-loglevel','error','-filter_complex_threads','1']
    for cond in ['on','toggle']:cmd+=['-i',str(D/(model+'_'+cond+'_clean.mp4'))]
    filters=[]
    for panel,label in enumerate([model+' | Continuous ON',model+' | ON / OFF / ON']):
        f=f'[{panel}:v]scale=1360:765,crop=960:765:(iw-960)/2:0,pad=960:1080:0:75:color=white,'
        f+=f"drawtext=fontfile={FONT}:text='{label}':fontcolor=0x253545:fontsize=25:x=25:y=22,"
        f+=f"drawtext=fontfile={FONT}:text='ASSIST ON':fontcolor=0x16867b:fontsize=24:x=w-tw-25:y=22"
        if panel:
            f+=f":enable='lt(t,5)+gte(t,10)',drawtext=fontfile={FONT}:text='ASSIST OFF':fontcolor=0xd46d42:fontsize=24:x=w-tw-25:y=22:enable='gte(t,5)*lt(t,10)'"
        filters.append(f+f'[v{panel}]')
    filters.append('[v0][v1]hstack=inputs=2[out]')
    cmd+=['-filter_complex',';'.join(filters),'-map','[out]','-an','-c:v','libx264','-preset','fast','-crf','18','-threads','4','-pix_fmt','yuv420p','-movflags','+faststart',str(S/(model+'_panels.mp4'))]
    subprocess.run(cmd,check=True)


def meters(traces,model):
    rms,excess=series(traces,model)
    assert rms.max()<70
    low=min(-500,np.floor(excess.min()/500)*500);high=max(2500,np.ceil(excess.max()/500)*500)
    font=ImageFont.truetype(FONT,23);small=ImageFont.truetype(FONT,16)
    cmd=['ffmpeg','-y','-loglevel','error','-filter_complex_threads','1','-i',str(S/(model+'_panels.mp4')),'-f','rawvideo','-pix_fmt','rgb24','-s','1920x240','-r','60','-i','pipe:0','-filter_complex','[0:v][1:v]overlay=0:840:shortest=1[out]','-map','[out]','-an','-c:v','libx264','-preset','fast','-crf','18','-threads','4','-pix_fmt','yuv420p','-movflags','+faststart',str(D/(model+'_ON_OFF_ON.mp4'))]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    for i in range(900):
        im=Image.new('RGB',(1920,240),'white');d=ImageDraw.Draw(im)
        d.text((30,5),f'WEARER HIP RMS (1 s)   ON {rms[0,i]:.1f} / switched {rms[1,i]:.1f} N.m',font=font,fill='#203243')
        d.text((990,5),f'EXCESS BURDEN vs ON: {excess[i]:+.0f} (N.m)^2 s',font=font,fill='#203243')
        for panel in range(2):
            left=panel*960+75;right=panel*960+915;top=51;bottom=188
            d.rectangle((left+(right-left)/3,top,left+2*(right-left)/3,bottom),fill='#fbede7')
            lo,hi=(0,70) if panel==0 else (low,high)
            ticks=[0,35,70] if panel==0 else [low,0,high]
            for value in ticks:
                y=bottom-(value-lo)/(hi-lo)*(bottom-top)
                d.line((left,y,right,y),fill='#dae2e7');d.text((left-65,y-9),f'{value:.0f}',font=small,fill='#667582')
            for sec in [0,5,10,15]:
                x=left+sec/15*(right-left);d.line((x,top,x,bottom),fill='#dae2e7');d.text((x-8,bottom+7),str(sec),font=small,fill='#667582')
            d.text((left+(right-left)/2-30,218),'Time [s]',font=small,fill='#667582')
            values=[rms[0],rms[1]] if panel==0 else [excess]
            colors=COLORS if panel==0 else [COLORS[1]]
            for vals,color in zip(values,colors):
                pts=[(left+j/900*(right-left),bottom-(float(vals[j])-lo)/(hi-lo)*(bottom-top)) for j in range(i+1)]
                if len(pts)>1:d.line(pts,fill=color,width=3)
            x=left+i/900*(right-left);d.line((x,top,x,bottom),fill='#344755',width=2)
        proc.stdin.write(im.tobytes())
    proc.stdin.close();assert proc.wait()==0
    (D/(model+'_ON_OFF_ON.json')).write_text(json.dumps({'physics_modified_by_rendering':False,'source_runs':{c:json.loads((S/'runs.json').read_text())[model+'_'+c]['path'] for c in ['on','toggle']},'meter_RMS_window':'preceding 1s of 400Hz actual wearer torque, mean over time and both flexion axes','excess_burden':'integral from video start of bilateral mean(tau_toggle^2 - tau_continuousON^2), units (Nm)^2 s; not energy','display_rms_range':[0,70],'display_excess_range':[float(low),float(high)],'video_offset_s':5,'OFF_video_s':5,'ON_again_video_s':10},indent=2))
    print('COMPOSED',model,flush=True)


def main():
    runs=json.loads((S/'runs.json').read_text());assert len(runs)==4
    traces,metrics=load_analyze(runs);plots(traces,metrics)
    print(json.dumps(metrics,indent=2),flush=True)
    if '--analyze-only' in sys.argv:return
    render(runs)
    for model in ['L1','R0']:compose(model);meters(traces,model)


if __name__=='__main__':main()
