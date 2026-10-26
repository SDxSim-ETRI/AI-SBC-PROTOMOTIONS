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
"""Audit recorded assist-removal forces and produce presentation diagnostics."""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from report_output_scale import align


def first_fall(t, z):
    below = np.convolve((z < .5).astype(int), np.ones(50, dtype=int), mode="valid")
    found = np.flatnonzero(below == 50)
    return float(t[found[0]]) if len(found) else None


def main():
    study = Path("results/presentation_off_20261023")
    folder = study/"deliverables"
    runs = json.loads((study/"runs.json").read_text())
    traces = {}
    for key, record in runs.items():
        with np.load(Path(record["path"])/"traces.npz") as file:
            traces[key] = dict(file)
    audit = {"protocol": {
        "video_start_simulation_s": 5, "video_duration_s": 15,
        "off_video_s": 5, "off_simulation_s": 10,
        "physics_hz": 400, "assist_hz": 100, "tracker_hz": 20,
        "replay_scope": "All 27 wearer actuator DOFs, prerecorded from continuous-assist reference, aligned by simulation time",
        "replay_on_control": "Same wearer replay and removal of PD feedback, assistance retained from continuous-ON reference",
        "falls": "Root z <0.5m continuously for 0.5s; reported time is beginning of this interval",
        "termination": "Simulation continues to 20s, including after a fall; no reset or kinematic pose enforcement",
    }, "runs": {}}
    for key, tr in traces.items():
        model, mode = key.split("_", 1)
        ref = traces[model+"_reference"]
        assert len(tr["t"]) == 2000
        assert len(tr["physics_human_tau"]) == 8000
        assert np.isfinite(tr["presentation_body_pos"]).all()
        assert np.isfinite(tr["physics_human_tau"]).all()
        assert np.allclose(np.linalg.norm(tr["presentation_body_rot"],axis=-1),1,atol=2e-4)
        assert np.abs(tr["physics_assist_tau"]).max() <= 20.0001
        if mode in ("feedback", "replay_off"):
            assert np.abs(tr["physics_assist_tau"][4000:, 1]).max() == 0
        fall = first_fall(tr["t"], tr["root_z"][:,1])
        pre = (tr["t"] >= 5) & (tr["t"] < 10)
        names = tr["presentation_body_names"].tolist()
        assert names == ref["presentation_body_names"].tolist()
        pelvis = names.index("pelvis")
        origin = tr["presentation_body_pos"][0,1,pelvis]
        reference_origin = ref["presentation_body_pos"][0,1,pelvis]
        # Separate runs sample different world spawn XY before their paired reset.
        # Align only the constant initial translation; retain subsequent drift.
        pose_difference = ((tr["presentation_body_pos"][pre,1] - origin)
                           - (ref["presentation_body_pos"][pre,1] - reference_origin))
        row = {
            "source_run": runs[key]["path"],
            "fall_start_simulation_s": fall,
            "fall_start_video_s": None if fall is None else fall-5,
            "minimum_root_height_m": float(tr["root_z"][:,1].min()),
            "pre_switch_pose_difference_rms_m_after_initial_translation_alignment": float(np.sqrt(np.mean(pose_difference**2))),
            "initial_world_translation_difference_m": (origin-reference_origin).tolist(),
            "assist_max_after_switch_Nm": float(np.abs(tr["physics_assist_tau"][4000:,1]).max()),
            "all_values_finite": True,
        }
        if mode.startswith("replay"):
            error = np.abs(tr["physics_human_tau"][4000:,1]-ref["physics_human_tau"][4000:,1]).max()
            pd = np.abs(tr["physics_raw_pd_tau"][4000:,1]).max()
            assert error < 2e-4 and pd < 1e-5, (key,error,pd)
            row.update(wearer_replay_max_error_Nm=float(error), original_PD_max_after_switch_Nm=float(pd))
            if mode == "replay_on":
                assert np.allclose(tr["physics_assist_tau"][4000:,1],ref["physics_assist_tau"][4000:,1],atol=1e-5)
        audit["runs"][key] = row
    audit["replay_causal_interpretation"] = {
        model: "Not isolated: continuous-assist torque-replay control also falls" if audit["runs"][model+"_replay_on"]["fall_start_video_s"] is not None
        else "Continuous-assist replay control completes; interpret paired removal with remaining replay limitations"
        for model in ("L1","R0")
    }
    (folder/"experiment_validation.json").write_text(json.dumps(audit,indent=2,ensure_ascii=False)+"\n")
    plt.rcParams.update({"font.family":FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc").get_name(),
                         "font.size":15,"axes.unicode_minus":False,"svg.fonttype":"path","axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(16,9),sharey=True)
    fig.subplots_adjust(left=.08,right=.97,top=.77,bottom=.2,wspace=.13)
    styles=[("reference","보조 계속 ON / W1 반응","#566779","-"),
            ("feedback","5초 OFF / W1 반응","#4274b8","-"),
            ("replay_on","5초부터 토크 재생 / 보조 ON","#8e6e99","--"),
            ("replay_off","5초 OFF / 토크 재생","#c65e50","-")]
    for model,ax in zip(("L1","R0"),axes):
        for mode,label,color,ls in styles:
            tr=traces[model+"_"+mode];sel=tr["t"]>=5
            ax.plot(tr["t"][sel]-5,tr["root_z"][sel,1],label=label,color=color,ls=ls,lw=2.2)
        ax.axvline(5,color="#81909d",lw=1);ax.axhline(.5,color="#a8b2bb",ls=":",lw=1)
        ax.set(title=model+" · ±20 N·m",xlabel="영상 시간 [s]",xlim=(0,15),ylim=(0,1.12))
        ax.grid(alpha=.2)
    axes[0].set_ylabel("골반 높이 [m]")
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc="upper center",bbox_to_anchor=(.53,.87),ncol=2,frameon=False,fontsize=13)
    fig.suptitle("보조 OFF와 착용자 반응 제거를 구분",x=.08,ha="left",y=.96,fontsize=27,fontweight="bold")
    fig.text(.08,.07,"세로선 5초: 보조 중단 또는 전신 토크 재생 시작 · 점선 0.5 m: 낙상 판정 높이\n"
             "토크 재생 ON 대조군도 넘어지면, 재생 OFF의 낙상을 보조 중단의 효과로만 해석할 수 없음",fontsize=14)
    for ext in ("png","svg"):fig.savefig(folder/("graph_OFF_and_replay_control."+ext),dpi=180,facecolor="white")
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(16,9),sharex=True,sharey="row")
    fig.subplots_adjust(left=.09,right=.97,top=.85,bottom=.16,wspace=.18,hspace=.18)
    for col,model in enumerate(("L1","R0")):
        for mode,color,label in [("reference","#7d8790","보조 계속 ON"),("feedback","#247ca4","5초에 보조 OFF")]:
            a=align(traces[model+"_"+mode]);x=a["t"]-5;sel=(x>=3)&(x<=9)
            burden=np.mean(a["B"]**2,axis=1)
            averaged=np.convolve(burden,np.ones(5)/5,mode="same")
            axes[0,col].plot(x[sel],averaged[sel],color=color,label=label,lw=1.8)
            assist=np.sqrt(np.mean(a["assist"]**2,axis=1))
            axes[1,col].plot(x[sel],assist[sel],color=color,lw=2)
        for row in range(2):
            axes[row,col].axvline(5,color="#b35349",ls="--",lw=1.3);axes[row,col].grid(alpha=.18)
        axes[0,col].set_title(model+" · W1 자세 보정 유지")
        axes[1,col].set(xlabel="영상 시간 [s]",xlim=(3,9),ylim=(-1,22))
    axes[0,0].set_ylabel("착용자 고관절 τ² [(N·m)²]\n좌우 평균 · 50 ms 이동평균")
    axes[1,0].set_ylabel("보조 토크 좌우 RMS [N·m]")
    axes[0,0].legend(frameon=False,fontsize=13)
    fig.suptitle("보조를 끄면 착용자 토크가 어떻게 변하는가",x=.08,ha="left",y=.97,fontsize=26,fontweight="bold")
    fig.text(.09,.04,"동일 모델의 계속 ON 실행과 OFF 실행 비교 · 토크²는 시뮬레이션 부담 지표 · W1/H1 동결\n"
             "작은 실행 간 차이는 5초 전에도 존재 · 위 그래프의 50 ms 이동평균은 표시용이며 제어에는 사용하지 않음",fontsize=12)
    for ext in ("png","svg"):fig.savefig(folder/("graph_feedback_OFF_torques."+ext),dpi=180,facecolor="white")
    plt.close(fig)
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
