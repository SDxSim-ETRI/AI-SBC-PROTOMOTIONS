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
"""Package the measured hip-cap demos and verify every encoded frame."""
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

from PIL import Image, ImageDraw

S=Path('results/presentation_cap_20261023')
D=S/'deliverables'
rows=json.loads((D/'metrics_and_validation.json').read_text())
runs=json.loads((S/'runs.json').read_text())
lines=[]
for m in ['L1','R0']:
 on,off=rows[m+'_on'],rows[m+'_off']
 a,b=on['after_OFF'],off['after_OFF']
 lines.append(f"{m}: 오른쪽/왼쪽 고관절 한계 ±{on['caps_right_left_Nm'][0]:.2f}/±{on['caps_right_left_Nm'][1]:.2f} N·m\n"
              f"  낙상 영상시각 ON={on['fall_video_s']}, OFF={off['fall_video_s']} (None=15초 영상 안에서 낙상 없음)\n"
              f"  OFF 이후 구간 고관절 평균 τ²: ON {a['hip_tau_squared_mean_Nm2']:.2f}, OFF {b['hip_tau_squared_mean_Nm2']:.2f} (N·m)²\n"
              f"  전신 추종 오차: ON {a['body_tracking_error_mean_mm']:.2f}, OFF {b['body_tracking_error_mean_mm']:.2f} mm\n"
              f"  한계 도달 시간 비율 [오른쪽, 왼쪽]: ON {[round(v*100,3) for v in a['hip_saturation_fraction_right_left']]}, OFF {[round(v*100,3) for v in b['hip_saturation_fraction_right_left']]} %")
summary='\n\n'.join(lines)
protocol='''고관절 출력 한계 내 보조 ON/OFF 시연

L1_ON_vs_OFF.mp4 / R0_ON_vs_OFF.mp4: 왼쪽 ON 유지, 오른쪽 영상 5초에 OFF.
모델명·ON/OFF 상태만 표시. *_clean.mp4는 표시 없는 개별 시연 영상.
*_with_torque.mp4: 착용자 좌우 고관절 토크의 직전 1초 RMS를 동기화한 표시 버전. 표시는 0–70 N·m 고정 축이며 제어에는 사용하지 않음.
모든 영상: 1920×1080, 60 fps, 15초, 원래 suit skeleton의 실제 추론 결과.

방법
- W1과 H1, L1/R0 가중치 동결. LLP는 두 모델 모두 ±20 N·m 운용.
- 기존 각 모델의 보조 ON 실행에서 시뮬레이션 5–20초, 400 Hz 실제 착용자 토크의 절대 최대값을 좌우 따로 산출.
- 좌우 hip_flexion 축만 그 절대 최대값의 ±대칭 한계로 설정. 모든 순간에 같은 한계.
- 제한은 시뮬레이션 5초(영상 시작)에 양 조건 동일하게 적용. 보조 OFF는 시뮬레이션 10초(영상 5초).
- W1 정책·PD 게인·감쇠 유지. 무릎, 발목, 고관절 내전/회전 등 다른 관절의 한계는 기존값 유지.
- 뉴턴의 native joint actuator effort limit 사용. 실제 PD 토크를 제한하며 보조 외력은 별도 가산.
- 저장 토크 재생 없음. ON도 H1/LLP를 현재 상태에서 계속 추론.
- 같은 시작 동작/평가 seed 42. 독립 실행이라 작은 수치 차이가 있으며 pre-OFF pose RMS를 기록.
- 모델별 한계가 달라 L1/R0 우열 비교에는 사용하지 않음.
- 낙상: 골반 높이 0.5 m 미만이 0.5초 연속. 첫 낮아진 시점을 보고. 이후에도 종료·리셋 없이 20초 계산.

해석
이 실험은 보조 ON에서 관측한 최대 힘을 넘지 못하는 가상 착용자의 진단이다.
사람의 근력 저하를 검증한 모델이나 실제 착용자 반응으로 해석하지 않는다.
OFF에서 쓰러지도록 한계를 추가로 낮추거나 선택하지 않았다.
실루엣이 비슷해도 토크 차이가 존재할 수 있다. 토크²는 시뮬레이션 부담 지표이며 근활성·대사량 측정이 아니다.
나머지 관절 지표는 hip_flexion 두 축과 병진 slide 네 축을 제외한 회전축 21개의 토크² 합이다.

결과 (OFF 이후 시뮬레이션 10–20초)
'''
(D/'README.txt').write_text(protocol+summary+'\n')
(D/'RESULTS.txt').write_text(summary+'\n')
shutil.copy2(S/'protocol.json',D/'protocol.json')
results_table=''
for m in ['L1','R0']:
 for cond,label in [('on','ON 유지'),('off','5초 후 OFF')]:
  r=rows[m+'_'+cond];a=r['after_OFF']
  fall='없음' if r['fall_video_s'] is None else f"{r['fall_video_s']:.2f}초"
  results_table+=f"<tr><td>{m}</td><td>{label}</td><td>{a['hip_tau_squared_mean_Nm2']:.1f}</td><td>{a['body_tracking_error_mean_mm']:.2f}</td><td>{fall}</td></tr>"
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>고관절 힘 한계 · 보조 ON/OFF</title><style>body{font-family:system-ui,sans-serif;background:#f4f6f8;color:#203243;max-width:1200px;margin:auto;padding:36px 24px;line-height:1.65}h1{font-size:34px;margin-bottom:8px}h2{margin-top:34px}video,img{width:100%;border-radius:12px;background:white}section,.note{background:white;border:1px solid #dae2e8;padding:22px;border-radius:14px;margin:20px 0}.note{border-left:5px solid #16867b}table{width:100%;border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #dae2e8;text-align:left}a{color:#146b8f}.muted{color:#5e6b77}summary{cursor:pointer}</style>
<h1>고관절 힘의 최대값을 제한한 보조 ON/OFF</h1>
<p>W1의 자세 보정은 유지합니다. 좌우 고관절 굴곡·신전 축의 최대 출력만 제한합니다.</p>
<div class="note"><b>왼쪽: ON 유지 · 오른쪽: 5초에 OFF</b><br>두 조건에 같은 착용자 힘 한계가 적용됩니다. 기록한 토크를 재생하지 않습니다.<br>하단 막대: 실제 착용자 고관절 토크의 직전 1초 RMS. 두 영상에 같은 0–70 N·m 축을 사용합니다.</div>
'''
for m in ['L1','R0']:
 cap=rows[m+'_on']['caps_right_left_Nm']
 page+=f'<section><h2>{m} · 오른쪽 ±{cap[0]:.1f} / 왼쪽 ±{cap[1]:.1f} N·m</h2><video controls preload="metadata" src="{m}_ON_vs_OFF_with_torque.mp4"></video><p><a href="{m}_ON_vs_OFF_with_torque.mp4" download>계측 표시 영상 저장</a> · <a href="{m}_ON_vs_OFF.mp4" download>계측 없는 비교 영상</a></p></section>'
page+='<section><h2>보조 OFF 이후 10초의 측정값</h2><table><tr><th>모델</th><th>조건</th><th>고관절 평균 τ² [(N·m)²]</th><th>전신 오차 [mm]</th><th>낙상</th></tr>'+results_table+'</table><p class="muted">평가 seed 42 한 번. 모델마다 착용자 힘 한계가 다르므로 L1/R0 순위를 매기는 비교가 아닙니다.</p></section>'
for g in ['graph_01_hip_torque_caps','graph_02_after_OFF']:
 page+=f'<section><img src="{g}.png"><a href="{g}.svg">벡터 그래프</a></section>'
page+='''<section><h2>이 실험에서 말할 수 있는 범위</h2><p>한계값은 기존 보조 ON 기록 5–20초의 절대 최대값입니다. OFF에서 넘어지도록 값을 낮추지 않았습니다. 고관절 이외의 관절과 W1의 피드백은 계속 작동합니다. 토크²는 시뮬레이션 부담 지표입니다.</p><a href="README.txt">상세 조건과 해석</a> · <a href="metrics_and_validation.json">원시 수치·검증</a></section><details><summary>표시 없는 개별 시연 영상 4개</summary>'''
for k in runs:
 page+=f'<h3>{k}</h3><video controls preload="none" src="{k}_clean.mp4"></video><a href="{k}_clean.mp4" download>저장</a>'
page+='</details></html>'
(D/'index.html').write_text(page)
# Decode all videos, retain proof and inspectable frame sheets.
previous_checks=json.loads((D/"video_validation.json").read_text()) if (D/"video_validation.json").exists() else {}
video_checks={}
frames=[]
for movie in sorted(D.glob('*.mp4')):
 raw=subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(movie)])
 info=json.loads(raw);v=next(s for s in info['streams'] if s['codec_type']=='video')
 assert (v['width'],v['height'],v['r_frame_rate'],int(v['nb_frames']))==(1920,1080,'60/1',900)
 sha=hashlib.sha256(movie.read_bytes()).hexdigest()
 if previous_checks.get(movie.name,{}).get('sha256')!=sha:
  subprocess.run(['ffmpeg','-v','error','-i',str(movie),'-f','null','-'],check=True)
 video_checks[movie.name]={'width':1920,'height':1080,'fps':60,'frames':900,'duration_s':float(info['format']['duration']),'full_decode':'passed','sha256':sha}
 for t in [4.983333,5.0,10.0,14.9]:
  target=S/'previews'/f'{movie.stem}_{t:.3f}.jpg';target.parent.mkdir(exist_ok=True)
  subprocess.run(['ffmpeg','-y','-loglevel','error','-ss',str(t),'-i',str(movie),'-frames:v','1','-vf','scale=480:270',str(target)],check=True)
  im=Image.open(target).convert('RGB');canvas=Image.new('RGB',(480,300),'white');canvas.paste(im,(0,30));ImageDraw.Draw(canvas).text((8,6),f'{movie.stem} / {t:.3f}s',fill='black');frames.append(canvas)
for start in range(0,len(frames),12):
 sheet=Image.new('RGB',(4*480,3*300),'#e8edf2')
 for n,frame in enumerate(frames[start:start+12]):sheet.paste(frame,((n%4)*480,(n//4)*300))
 sheet.save(S/'previews'/f'contact_sheet_{start//12}.jpg')
(D/'video_validation.json').write_text(json.dumps(video_checks,indent=2))
files=[Path('examples/experiments/assist_skeleton')/n for n in ['presentation_cap_capture.py','presentation_cap_report.py','presentation_cap_package.py','presentation_cap_meter.py','presentation_off_capture.py','presentation_capture.py','presentation_render.py','presentation_off_media.py','run_system.py']]
manifest={'runs':runs,'code':{},'inputs':{}}
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for p in files:
 shutil.copy2(p,S/'source_snapshot'/p.name);manifest['code'][str(p)]={'sha256':digest(p)}
for rec in runs.values():
 for root in [rec['path'],rec['reference']]:
  p=Path(root)/'traces.npz';manifest['inputs'][str(p)]={'sha256':digest(p)}
 meta=json.loads((Path(rec['path'])/'meta.json').read_text())
 for key,val in meta.items():
  if 'checkpoint' in key and isinstance(val,str) and Path(val).is_file():manifest['inputs'][val]={'sha256':digest(Path(val))}
(D/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
print(summary,flush=True)
print('Videos decoded; contact sheets prepared',flush=True)
