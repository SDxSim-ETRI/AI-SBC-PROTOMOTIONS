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
"""Add a synchronized measured hip-torque RMS meter to the comparison movies."""
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

S=Path('results/presentation_cap_20261023');D=S/'deliverables'
runs=json.loads((S/'runs.json').read_text())
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',24)
small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
for model in ['L1','R0']:
 series=[]
 for condition in ['on','off']:
  with np.load(Path(runs[model+'_'+condition]['path'])/'traces.npz') as z:
   power=np.mean(z['physics_human_tau'][:,1][:,[0,5]]**2,axis=1)
   cumulative=np.r_[0,np.cumsum(power,dtype=np.float64)]
   # Poses at time t precede the next physics step; use only [t-1,t).
   end=np.rint((5+np.arange(900)/60)*400).astype(int)
   rms=np.sqrt((cumulative[end]-cumulative[end-400])/400)
   assert rms.max()<70
   series.append(rms)
 out=D/(model+'_ON_vs_OFF_with_torque.mp4')
 cmd=['ffmpeg','-y','-loglevel','error','-filter_complex_threads','1','-i',str(D/(model+'_ON_vs_OFF.mp4')),
      '-f','rawvideo','-pix_fmt','rgb24','-s','1920x90','-r','60','-i','pipe:0',
      '-filter_complex','[0:v][1:v]overlay=0:990:shortest=1[out]','-map','[out]','-an','-c:v','libx264','-preset','fast','-crf','18','-threads','4','-pix_fmt','yuv420p','-movflags','+faststart',str(out)]
 proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
 for i in range(900):
  im=Image.new('RGB',(1920,90),'white');draw=ImageDraw.Draw(im)
  for panel in range(2):
   x=panel*960+25;v=float(series[panel][i]);color='#16867b' if panel==0 or i<300 else '#d46d42'
   draw.text((x,8),f'WEARER HIP RMS (1 s): {v:.1f} N.m',fill='#203243',font=font)
   draw.rounded_rectangle((x,47,x+785,65),radius=6,fill='#e5eaf0')
   draw.rounded_rectangle((x,47,x+785*v/70,65),radius=6,fill=color)
   draw.text((x+800,44),'70 N.m',fill='#566779',font=small)
  proc.stdin.write(im.tobytes())
 proc.stdin.close();assert proc.wait()==0
 meta={'source_comparison':str(D/(model+'_ON_vs_OFF.mp4')),'measure':'sqrt(mean(actual 400 Hz wearer hip torque squared across preceding 1 second and both flexion axes))','window':'causal [t-1s,t), including history before video start','fixed_display_max_Nm':70,'display_fps':60,'simulation_offset_s':5,'torque_replayed':False,'rendering_changes_dynamics':False,'policy_weights_changed':False}
 out.with_suffix('.json').write_text(json.dumps(meta,indent=2))
 print('METER_COMPLETE',model,flush=True)
