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
"""Audit assist output when the wearer's negative hip torque exceeds 100 Nm."""
import json
from pathlib import Path
import numpy as np
from report_output_scale import align

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/presentation_three_slides_20261023/diagnostics'

def main():
    archive=json.loads((ROOT/'results/llp_relative_20261020/walking/comparison.json').read_text())
    result={}
    for model in ['L1','R0_s0']:
        samples={'assisted':[],'unassisted':[]};records=[]
        for key,run in archive['runs'].items():
            rec=run['record']
            if rec['model']!=model or rec['limit_Nm']!=20:continue
            with np.load(ROOT/rec['path']/'traces.npz') as source:a=align(source)
            selected=a['t']>=1.5
            for basis,field in [('assisted','B'),('unassisted','A')]:
                mask=(a[field]<=-100)&selected[:,None];v=a['assist'][mask]
                samples[basis].extend(v.tolist());records.append({'run':key,'basis':basis,'count':len(v),'total_leg_samples':int(selected.sum()*2),'path':rec['path']})
        result[model]={'records':records}
        for basis,values in samples.items():
            v=np.asarray(values);assert len(v)>0 and np.isfinite(v).all() and abs(v).max()<=20.0001
            result[model][basis]={'count':len(v),'mean_Nm':float(v.mean()),'median_Nm':float(np.median(v)),'p10_p90_Nm':np.quantile(v,[.1,.9]).tolist(),'negative_fraction':float(np.mean(v<0)),'near_minus20_fraction_tau_le_minus19':float(np.mean(v<=-19)),'wrong_sign_fraction':float(np.mean(v>0)),'min_max_Nm':[float(v.min()),float(v.max())]}
    (OUT/'high_torque.json').write_text(json.dumps(result,indent=2))
    refs=json.loads((ROOT/'results/presentation_off_20261023/runs.json').read_text());check={}
    for model in ['L1','R0']:
        p=Path(refs[model+'_reference']['path'])
        with np.load(ROOT/p/'traces.npz') as z:
            human=z['physics_human_tau'][2000:,1][:,[0,5]];assist=z['physics_assist_tau'][2000:,1][:,[0,5]]
        v=assist[human<=-100]
        check[model]={'source':str(p),'interval_s':[5,20],'sample_hz':400,'count':len(v),'mean_assist_Nm':float(v.mean()),'negative_fraction':float(np.mean(v<0)),'near_minus20_fraction_tau_le_minus19':float(np.mean(v<=-19)),'min_max_Nm':[float(v.min()),float(v.max())]}
    (OUT/'high_torque_400Hz_check.json').write_text(json.dumps(check,indent=2))
    print(json.dumps({k:v['assisted'] for k,v in result.items()},indent=2))

if __name__=='__main__':main()
