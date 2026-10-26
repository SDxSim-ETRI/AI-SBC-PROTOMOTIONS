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
"""Package and audit ON/OFF/ON demos without modifying the simulated motion."""
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import zipfile
from PIL import Image,ImageDraw

S=Path('results/presentation_toggle_20261023');D=S/'deliverables'
runs=json.loads((S/'runs.json').read_text());metrics=json.loads((D/'metrics_and_validation.json').read_text())
summary=[];table=''
for m in ['L1','R0']:
    values=[]
    for stage,label in [('ON_before','ON 0–5초'),('OFF','OFF 5–10초'),('ON_restored','ON 복귀 10–15초')]:
        a=metrics[m+'_on']['stages'][stage];b=metrics[m+'_toggle']['stages'][stage];delta=b['hip_tau2_excess_vs_time_matched_ON_Nm2']
        table+=f"<tr><td>{m}</td><td>{label}</td><td>{a['hip_tau2_Nm2']:.1f}</td><td>{b['hip_tau2_Nm2']:.1f}</td><td>{delta:+.1f}</td></tr>"
        values.append(f"{label}: 계속 ON {a['hip_tau2_Nm2']:.2f}, 전환 {b['hip_tau2_Nm2']:.2f}, 차이 {delta:+.2f} (N·m)²")
    summary.append(m+'\n'+'\n'.join(values))
text='\n\n'.join(summary)
readme='''발표용 추가 제안: 보조 ON → OFF → ON 회복 시연

추천 파일: L1_ON_OFF_ON.mp4, R0_ON_OFF_ON.mp4
- 왼쪽: 보조 계속 ON. 오른쪽: 영상 5초에 OFF, 10초에 다시 ON.
- 원래 suit skeleton의 실제 물리 시뮬레이션 추론 결과. 카메라·그래프만 사후 렌더링.
- 하단 왼쪽: 실제 착용자 고관절 토크의 직전 1초 RMS. 초록=계속 ON, 주황=전환 실험.
- 하단 오른쪽: 같은 시각의 계속 ON 대비 추가 부담 누계 ∫(좌우 평균 τ²전환 − τ²ON)dt.
  이 값은 (N·m)²·s 단위의 부담 지표이며, 에너지·근활성·대사량 측정이 아님.
- *_clean.mp4는 표시 없는 개별 시연 4개.
- 동영상 1920×1080, 60 fps, 15초. OFF/ON 전환은 5/10초.

실험 조건
- 시뮬레이션 0–20초. 영상은 5–20초. OFF 10초, ON 복귀 15초.
- W1/H1/L1/R0 가중치 동결. 모터 토크는 ±20 N·m 운용.
- W1 원래 출력 한계·PD 강성·감쇠·관측 유지. 별도 근력 제한 없음. 전신 토크 재생 없음.
- OFF 동안에도 H1/LLP의 추론과 이력은 계속 진행. 모터 외력 주입만 0으로 만들고 ON 복귀 시 현재 추론값을 다시 주입. 전환 시 모델/상태 리셋 없음.
- seed 42, 기존 보행. 새 전환 실행 2개와 앞서 저장한 같은 조건의 계속 ON 실행 2개를 비교.
- 독립 실행의 미세한 차이는 존재. 전환 전 초기 위치 정렬 후 pose RMS와 각 구간 토크 차이를 함께 기록.
- 전체 구간을 포함한 5초 평균이며, 전환 직후 과도응답을 임의로 제외하지 않음.
- 낙상 기준은 골반 높이 0.5m 미만 0.5초 지속. 네 실행 모두 낙상 없음.

발표에서 전달할 내용
현재 보행에서는 보조를 꺼도 W1이 힘을 더 내며 걷는다. 다시 켜면 같은 시각의 ON 대조군 수준으로 고관절 부담이 돌아온다.
현재 W1과 이 보행에서 확인한 결과이며 실제 사람 또는 다양한 보행에서의 검증으로 확대하지 않는다.
이 실험의 5초 구간 수치로 L1/R0의 일반 성능 순위를 바꾸지 않는다.

구간별 결과: 400 Hz 실제 착용자 고관절 토크 제곱의 좌우·시간 평균
'''
(D/'README.txt').write_text(readme+text+'\n');(D/'RESULTS.txt').write_text(text+'\n')
shutil.copy2(S/'protocol.json',D/'protocol.json')
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>보조 ON → OFF → ON</title><style>body{font-family:system-ui,sans-serif;background:#f4f6f8;color:#203243;max-width:1200px;margin:auto;padding:36px 24px;line-height:1.65}h1{font-size:34px}h2{margin-top:28px}video,img{width:100%;border-radius:12px;background:white}section,.note{background:white;border:1px solid #dae2e8;padding:22px;border-radius:14px;margin:20px 0}.note{border-left:5px solid #16867b}table{width:100%;border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #dae2e8;text-align:left}a{color:#146b8f}summary{cursor:pointer}</style>
<h1>보조를 끄고, 다시 켜면?</h1><p>같은 보행 중 보조 전환과 착용자 토크의 변화를 함께 보여주는 추가 시연입니다.</p>
<div class="note"><b>왼쪽은 계속 ON · 오른쪽은 5초 OFF → 10초 다시 ON</b><br>착용자의 원래 힘 한계와 자세 보정을 유지합니다. 별도 근력 제한이나 토크 재생은 없습니다.<br>그래프의 초록은 계속 ON, 주황은 전환 실험입니다. 오른쪽 그래프는 에너지가 아닌 토크² 부담의 시간 누계입니다.</div>
'''
for m in ['L1','R0']:page+=f'<section><h2>{m} · ON / OFF / ON</h2><video controls preload="metadata" src="{m}_ON_OFF_ON.mp4"></video><a href="{m}_ON_OFF_ON.mp4" download>영상 저장</a></section>'
page+='<section><img src="graph_stage_comparison.png"><a href="graph_stage_comparison.svg">벡터 그래프</a></section><section><img src="graph_toggle_response.png"><a href="graph_toggle_response.svg">벡터 그래프</a></section>'
page+='<section><h2>같은 시각의 계속 ON 대조군과 비교</h2><p>착용자 고관절 평균 τ² [(N·m)²]</p><table><tr><th>모델</th><th>영상 구간</th><th>계속 ON</th><th>전환 실험</th><th>차이</th></tr>'+table+'</table><p>seed 42 한 번의 기존 보행 결과. 실제 사람의 근활성·대사량 측정이 아닙니다.</p><a href="README.txt">상세 조건</a> · <a href="metrics_and_validation.json">측정값·검증</a></section>'
page+='<details><summary>표시 없는 개별 영상</summary>'
for k in runs:page+=f'<h3>{k}</h3><video controls preload="none" src="{k}_clean.mp4"></video><a href="{k}_clean.mp4" download>저장</a>'
page+='</details><p><a href="../../presentation_cap_20261023/deliverables/index.html">앞선 고관절 최대 힘 제한 실험</a></p></html>'
(D/'index.html').write_text(page)
checks={};frames=[]
for movie in sorted(D.glob('*.mp4')):
    v=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(movie)]))['streams'][0]
    assert (v['width'],v['height'],v['r_frame_rate'],int(v['nb_frames']))==(1920,1080,'60/1',900)
    subprocess.run(['ffmpeg','-v','error','-i',str(movie),'-f','null','-'],check=True)
    checks[movie.name]={'full_decode':'passed','resolution':[1920,1080],'frames':900,'fps':60,'sha256':hashlib.sha256(movie.read_bytes()).hexdigest()}
    for t in [4.983333,5,9.983333,10,14.9]:
        target=S/'previews'/f'{movie.stem}_{t:.3f}.jpg'
        subprocess.run(['ffmpeg','-y','-loglevel','error','-ss',str(t),'-i',str(movie),'-frames:v','1','-vf','scale=480:270',str(target)],check=True)
        tile=Image.new('RGB',(480,300),'white');tile.paste(Image.open(target),(0,30));ImageDraw.Draw(tile).text((8,6),f'{movie.stem} / {t:.3f}s',fill='black');frames.append(tile)
for start in range(0,len(frames),10):
    sheet=Image.new('RGB',(2400,600),'#e8edf2')
    for i,f in enumerate(frames[start:start+10]):sheet.paste(f,((i%5)*480,(i//5)*300))
    sheet.save(S/'previews'/f'contact_sheet_{start//10}.jpg')
(D/'video_validation.json').write_text(json.dumps(checks,indent=2))
manifest={'runs':runs,'code':{},'inputs':{}}
for name in ['presentation_toggle_capture.py','presentation_toggle_media.py','presentation_toggle_package.py','presentation_off_capture.py','presentation_capture.py','presentation_render.py','run_system.py']:
    p=Path('examples/experiments/assist_skeleton')/name;shutil.copy2(p,S/'source_snapshot'/name);manifest['code'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
for r in runs.values():
    p=Path(r['path'])/'traces.npz';manifest['inputs'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    meta=json.loads((Path(r['path'])/'meta.json').read_text())
    for k,v in meta.items():
        if 'checkpoint' in k and isinstance(v,str) and Path(v).is_file():manifest['inputs'][v]=hashlib.sha256(Path(v).read_bytes()).hexdigest()
(D/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
cmds=[shlex.join(r['command']) for r in runs.values() if 'command' in r]
cmds+=[f'/home/bak/miniconda3/envs/sbc/bin/python examples/experiments/assist_skeleton/{name}.py' for name in ['presentation_toggle_media','presentation_toggle_package']]
(D/'reproduce.txt').write_text('Repository root: /home/bak/Projects/AI-SBC-PROTOMOTIONS\nReused reference paths in source_manifest.json\n\n'+'\n\n'.join(cmds)+'\n')
print(text,flush=True);print('Media decoded; contact sheets ready.',flush=True)
