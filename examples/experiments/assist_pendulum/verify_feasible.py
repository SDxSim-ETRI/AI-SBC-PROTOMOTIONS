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
"""Verify finished study provenance, raw traces, checkpoints and report evidence."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
STUDY=ROOT/'results/llp_feasible_20261020'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    plan=json.loads((STUDY/'manifest.json').read_text())
    status=json.loads((STUDY/'status.json').read_text())
    assert status['stage']=='all planned training and evaluations complete'
    checks={}
    checks['source_hashes']={name:sha(ROOT/name)==h for name,h in plan['sources'].items()}
    assert all(checks['source_hashes'].values())
    checks['baseline_asset_hashes']={name:sha(item['path'])==item['sha256'] for name,item in plan['assets'].items()}
    assert all(checks['baseline_asset_hashes'].values())
    checks['baseline_evidence_unchanged']=sha(ROOT/'results/llp_output_scale_20261016/comparison.json')==plan['baseline_evidence_sha256']
    assert checks['baseline_evidence_unchanged']
    execution=json.loads((STUDY/'execution.json').read_text())
    training={}
    for label,record in execution.items():
        if not label.endswith('_train'):continue
        model=label.removesuffix('_train')
        directory=ROOT/'results'/f'llp_feasible_{model}_20261020'
        path=directory/f'epoch_{plan["epochs"]}.ckpt'
        ckpt=torch.load(path,map_location='cpu',weights_only=False)
        assert ckpt['epoch']==plan['epochs']
        assert all(not torch.is_tensor(v) or torch.isfinite(v).all() for v in ckpt['model'].values())
        cfg=torch.load(directory/'resolved_configs.pt',map_location='cpu',weights_only=False)
        assert cfg['env'].action_config['torque_scale']==cfg['env'].assist_torque_limit==40
        assert cfg['env'].human_feedforward_inside_pd and not cfg['env'].clamp_combined_injection
        assert cfg['agent'].training_max_steps==786432000
        assert cfg['agent'].model.actor.mu_model._target_=='protomotions.agents.common.mlp.MLPWithConcat'
        assert '--checkpoint' not in record['command']
        training[model]=dict(checkpoint=str(path.relative_to(ROOT)),epoch=ckpt['epoch'],sha256=sha(path),all_parameters_finite=True)
    assert 'F1_s0' in training and 'F2_s0' in training
    checks['training']=training
    from audit_feasible import canonical,differences
    cfgs={name:canonical(torch.load(ROOT/'results'/f'llp_feasible_{name}_20261020/resolved_configs.pt',map_location='cpu',weights_only=False)) for name in ('F1_s0','F2_s0')}
    diff=differences(cfgs['F1_s0']['env'],cfgs['F2_s0']['env'])
    assert len(diff)==1 and diff[0]['path']=='.reward_components.assist_absorbed_power'
    assert cfgs['F1_s0']['agent']==cfgs['F2_s0']['agent']
    checks['F1_F2_environment_difference']=diff
    records=json.loads((STUDY/'walking/runs.json').read_text())
    new={k:v for k,v in records.items() if not v.get('reused_baseline',False)}
    assert len(new)==len(training)*4
    checks['walking']={}
    for key,record in new.items():
        path=ROOT/record['path'];meta=json.loads((path/'meta.json').read_text())
        with np.load(path/'traces.npz') as f:
            assert all(np.isfinite(f[k]).all() for k in f.files if np.issubdtype(f[k].dtype,np.number))
            assert np.allclose(np.diff(f['t']),.01,atol=2e-5)
            assert abs(f['tau_assist'][:,1]).max()<=record['limit_Nm']+1e-4
            n=len(f['t'])
            if meta['survived_s'] is None:assert n==4000
        checks['walking'][key]=dict(samples=n,completed_40s=meta['survived_s'] is None,
            trace_sha256=sha(path/'traces.npz'),meta_sha256=sha(path/'meta.json'))
    checks['pendulum']={}
    for path in sorted((STUDY/'pendulum').glob('*/summary.json')):
        summary=json.loads(path.read_text())
        assert summary['policy_epoch']==6000
        assert summary['injection_error_max_Nm']<1e-4
        assert summary['assist_max_abs_Nm']<=20.00001
        assert summary['protocol']['plant_seed']==63031 and summary['protocol']['trajectory_seed']==52021
        with np.load(path.parent/'rollout.npz') as f:
            assert all(np.isfinite(f[k]).all() for k in f.files if np.issubdtype(f[k].dtype,np.number))
        checks['pendulum'][path.parent.name]=dict(failures=summary['failures'],episodes=summary['total_episodes'],sha256=sha(path))
    assert len(checks['pendulum'])==3*len(training)
    report=ROOT/'docs/ai_sbc/261020_feasible/report.html'
    evidence=json.loads((report.parent/'evidence.json').read_text())
    embedded=json.loads(re.search(r'id="evidence">(.*?)</script>',report.read_text(),re.S).group(1))
    assert embedded==evidence
    checks['embedded_evidence_identical']=True
    for match in re.findall(r'href="([^"#]+)"',report.read_text()):
        if not match.startswith(('http:','https:')):
            assert (report.parent/match).exists(),match
    # Physical-control tests passed before training; frozen source hashes above
    # establish that their implementation did not change during the study.
    checks['preflight_tests']=json.loads((STUDY/'preflight_validation.json').read_text())
    assert checks['preflight_tests']['passed']
    tests=[sys.executable,'-m','pytest','protomotions/tests/test_output_scale_analysis.py','-q']
    result=subprocess.run(tests,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (STUDY/'final_tests.log').write_text(result.stdout)
    assert result.returncode==0,result.stdout
    checks['tests']=dict(command=tests,returncode=result.returncode,log=str((STUDY/'final_tests.log').relative_to(ROOT)))
    analysis_files=['examples/experiments/assist_pendulum/'+name for name in ('audit_feasible.py','report_feasible.py','verify_feasible.py')]
    checks['analysis_sources']={}
    for name in analysis_files:
        dest=STUDY/'analysis_sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
        checks['analysis_sources'][name]=sha(ROOT/name)
    checks['report_artifacts']={str(p.relative_to(ROOT)):sha(p) for p in report.parent.rglob('*') if p.is_file()}
    checks['finished_utc']=datetime.now(timezone.utc).isoformat()
    (STUDY/'verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
    print(json.dumps(dict(training_models=len(training),new_W1_evaluations=len(new),pendulum_evaluations=len(checks['pendulum']),verified=True)))


if __name__=='__main__':main()
