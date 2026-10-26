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
"""R0/R1 (2026-10-20): sequential fresh PPO -> frozen W1 -> held-out pendulum, with bounded recovery.

R0 = L1 recipe on the corrected torque path (control). R1 = R0 + paired relative burden cost.
Stored L1/G0/G4 and today's F1/F2 W1 records are reused as baselines; nothing is re-evaluated.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
PYTHON = sys.executable
RECIPE = "examples/experiments/assist_pendulum/mlp_relative.py"
EVALUATOR = "examples/experiments/assist_pendulum/evaluate_feasible.py"
BASE = "results/llp_pendulum20_C0_20261014_nativeff/epoch_2000.ckpt"
OLD = ROOT / "results/llp_output_scale_20261016"
FEASIBLE = ROOT / "results/llp_feasible_20261020"
VARIANTS = {"R0": ["--paired-excess-weight=0.0"],
            "R1": ["--paired-excess-weight=-0.5", "--excess-floor=10", "--excess-window-steps=5"]}
BASELINE_MODELS = ("L1", "G0_s0", "G4_s0", "G4_s1", "F1_s0", "F2_s0")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',default='results/llp_relative_20261020')
    p.add_argument('--epochs',type=int,default=6000)
    p.add_argument('--hours',type=float,default=8.)
    args=p.parse_args()
    study=(ROOT/args.study).resolve(); study.mkdir(parents=True,exist_ok=True)
    lock=(study/'worker.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=json.loads((OLD/'manifest.json').read_text())
    sources=list(previous['sources']) + [RECIPE,EVALUATOR,
        'examples/experiments/assist_pendulum/run_relative.py',
        'examples/experiments/assist_pendulum/mlp_relative.py',
        'examples/experiments/assist_pendulum/mlp_feasible.py',
        'protomotions/envs/context_views.py',
        'protomotions/tests/test_pendulum_relative.py',
        'examples/experiments/assist_pendulum/mlp.py',
        'examples/experiments/assist_pendulum/mlp_v3.py',
        'examples/experiments/assist_pendulum/mlp_v4.py',
        'examples/experiments/assist_pendulum/runtime_utils.py',
        'examples/experiments/assist_pendulum/evaluate_research.py',
        'protomotions/envs/control/pendulum_feasible_control.py',
        'protomotions/envs/control/assist_target_control.py',
        'protomotions/envs/control/assist_motion_control.py',
        'protomotions/envs/control/assist_disturbance_control.py',
        'protomotions/envs/control/pendulum_general_control.py',
        'protomotions/envs/control/pendulum_research_control.py',
        'protomotions/envs/base_env/assist_env.py',
        'protomotions/envs/base_env/paired_assist_env.py',
        'protomotions/envs/rewards/assist.py',
        'protomotions/envs/rewards/assist_energy.py',
        'protomotions/envs/obs/assist_obs.py',
        'protomotions/agents/ppo/agent.py',
        'protomotions/agents/base_agent/agent.py',
        'protomotions/agents/ppo/config.py',
        'protomotions/robot_configs/hip_pendulum.py',
        'protomotions/tests/test_pendulum_feasible.py',
        'examples/experiments/assist_skeleton/report_output_scale.py',
        'examples/experiments/assist_pendulum/report_general_phase.py',
        'examples/experiments/assist_pendulum/report_general_docs.py']
    sources={name:sha(ROOT/name) for name in dict.fromkeys(sources)}
    plan=dict(started_utc=datetime.now(timezone.utc).isoformat(),started_unix=time.time(),
        time_budget_hours=args.hours,epochs=args.epochs,num_envs=4096,batch_size=16384,
        transitions_per_model=args.epochs*4096*32,first_training_seed=0,
        variants={
            'R0':'L1 recipe (mlp_v4 defaults, bilateral actor, 40 Nm action) with only the corrected torque path: one human PD+FF actuator limit, independent assist/load injection; no C2 blending',
            'R1':'R0 + reward-only paired relative burden cost -0.5 * mean_j relu(tau_B^2 - tau_A^2) / (mean_50ms(tau_A^2) + 10^2), clipped at -3 per step, unassisted twin integrated with identical intent/plant/loads'},
        variant_arguments={k:list(v) for k,v in VARIANTS.items()},
        baseline='Stored L1/G0/G4 (2026-10-16) and F1/F2 (2026-10-20) results only; no old checkpoint re-evaluation',
        native_limit_Nm=40,output_scaling={'15':.375,'20':.5},
        initialization='Independent random initialization, no source checkpoint / W1 training',
        selection='Fixed final epoch, never select intermediate checkpoint by W1 outcomes',
        repeat_rule='If time allows, repeat one recipe on training seed 1 only if at a matched cap mean whole hip torque squared benefit >= L1, low10 burden <= L1, worst-phase burden <= L1, zero falls, and body tracking <= L1+1 mm. Choose largest normalized benefit-minus-harm score; otherwise no repeat.',
        common_differences_from_L1=['Corrected human PD+FF combined actuator and independent qfrc sources','Periodic training evaluation disabled; fixed epoch and same PPO sample count','R1 only: paired twin truncates an episode when the unassisted twin fails (bootstrap, not terminal)'],
        weight_calibration='R1 weight -0.5 mean-matches the new term to the sum of L1 stored wearer costs (|tau w| and tau^2, 0.22 per step) at the stored L1 pendulum rollout where mean excess (floor 10, 50 ms window) was 0.414',
        pendulum_protocol=dict(base_checkpoint=BASE,conditions=['legacy','phase','hold50'],
            plants=128,cohorts=2,seconds=10,plant_seed=63031,trajectory_seed=52021,hlp_seed=991,
            HLP='frozen H1',limit_Nm=20,warmup_s=.5,
            caveat='Unseen random trajectories/plants; phase force fields and 50 ms human command hold are outside training. No new natural-motion family claim.'),
        walking_protocol=previous['protocol'],masks=previous['masks'],
        evaluation_seeds=[42,7],assets=previous['assets'],sources=sources,
        baseline_evidence_sha256=sha(OLD/'comparison.json'))
    manifest=study/'manifest.json'
    if manifest.exists():
        old=json.loads(manifest.read_text())
        assert old['sources']==sources and old['epochs']==args.epochs,'Source/plan changed'
        plan=old
    else:
        save(manifest,plan)
        for name in sources:
            dest=study/'sources'/name; dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,dest)
        save(study/'initial_git_status.json',subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True).splitlines())
        (study/'initial_nvidia_smi.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    deadline=plan['started_unix']+args.hours*3600
    records_path=study/'execution.json'
    records=json.loads(records_path.read_text()) if records_path.exists() else {}
    errors_path=study/'runtime_failures.json'
    errors=json.loads(errors_path.read_text()) if errors_path.exists() else []
    environment=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONFAULTHANDLER='1')
    walking=study/'walking'; walking.mkdir(exist_ok=True)
    wrecords=json.loads((walking/'runs.json').read_text()) if (walking/'runs.json').exists() else json.loads((FEASIBLE/'walking/runs.json').read_text())
    for key,r in wrecords.items():
        if r['model'] in BASELINE_MODELS: r['reused_baseline']=True
    wplan=json.loads((walking/'manifest.json').read_text()) if (walking/'manifest.json').exists() else json.loads((FEASIBLE/'walking/manifest.json').read_text())
    save(walking/'runs.json',wrecords); save(walking/'manifest.json',wplan)
    active={}

    def status(stage,**extra):
        save(study/'status.json',dict(stage=stage,pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_s=time.time()-plan['started_unix'],completed_jobs=len(records),
            new_walking_completed=sum(not x.get('reused_baseline',False) for x in wrecords.values()),**extra))

    def execute(command,label,timeout):
        if label in records: return records[label]
        assert all(sha(ROOT/name)==digest for name,digest in sources.items()),'Frozen source changed'
        for attempt in range(1,4):
            logfile=study/'logs'/f'{label}_attempt{attempt}.log'; logfile.parent.mkdir(exist_ok=True)
            started=time.time()
            with logfile.open('a') as stream:
                stream.write(json.dumps(command)+'\n'); stream.flush()
                child=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
                active.update(child_pid=child.pid,current=label,command=command,log=str(logfile),attempt=attempt)
                print(f'START {label} attempt {attempt} pid {child.pid}',flush=True)
                while child.poll() is None:
                    with logfile.open('rb') as inp:
                        inp.seek(max(0,logfile.stat().st_size-5000))
                        epochs=re.findall(r'Epoch (\d+)',inp.read().decode(errors='replace'))
                    status('running',**active,job_elapsed_s=time.time()-started,observed_epoch=int(epochs[-1]) if epochs else None)
                    if time.time()-started>timeout or time.time()>deadline:
                        child.terminate()
                        try: child.wait(timeout=10)
                        except subprocess.TimeoutExpired: child.kill(); child.wait()
                        raise RuntimeError(f'Time budget exceeded: {label}')
                    time.sleep(10)
            outcome=dict(command=command,returncode=child.returncode,attempt=attempt,
                started_unix=started,elapsed_s=time.time()-started,log=str(logfile.relative_to(ROOT)))
            active.clear()
            if child.returncode==0:
                records[label]=outcome; save(records_path,records)
                print(f'DONE {label} {outcome["elapsed_s"]:.1f}s',flush=True)
                return outcome
            errors.append(dict(label=label,**outcome)); save(errors_path,errors)
            if child.returncode not in (-11,-6) or attempt==3:
                raise RuntimeError(f'{label} failed: {child.returncode}')
            # Training detects only its own last.ckpt and resumes. Evaluations
            # restart from time zero; unsuccessful raw files/logs are preserved.
            print(f'RETRY {label}: {child.returncode}',flush=True)
        raise AssertionError('Unreachable')

    def train(model,seed):
        name=f'llp_relative_{model}_s{seed}_20261020'
        checkpoint=ROOT/'results'/name/f'epoch_{args.epochs}.ckpt'
        command=[PYTHON,'-u','protomotions/train_agent.py','--robot-name','hip_pendulum',
            '--simulator','newton','--experiment-path',RECIPE,'--experiment-name',name,
            '--motion-file','none','--num-envs','4096','--batch-size','16384',
            '--seed',str(seed),'--training-max-steps',str(args.epochs*4096*32),*VARIANTS[model]]
        execute(command,f"{model}_s{seed}_train",timeout=12000)
        import torch
        ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert ckpt['epoch']==args.epochs
        assert all(not torch.is_tensor(v) or torch.isfinite(v).all() for v in ckpt['model'].values())
        resolved=torch.load(checkpoint.parent/'resolved_configs.pt',map_location='cpu',weights_only=False)['env']
        assert resolved.clamp_combined_injection is False and resolved.human_feedforward_inside_pd is True
        assert ('paired_excess' in resolved.reward_components)==(model=='R1')
        if model=='R1': assert resolved.paired_reference_window_steps==5 and resolved._target_.endswith('PairedAssistEnv')
        save(study/f'{model}_s{seed}_checkpoint.json',dict(path=str(checkpoint.relative_to(ROOT)),sha256=sha(checkpoint),epoch=ckpt['epoch']))
        return checkpoint

    def transfer(model,seed,checkpoint):
        label=f'{model}_s{seed}'
        for cap in (15,20):
            for evalseed in (42,7):
                identity=f'{label}_{cap}Nm_seed{evalseed}'
                if identity in wrecords: continue
                job=dict(id=identity,model=label,limit_Nm=cap,seed=evalseed,
                    checkpoint=str(checkpoint.relative_to(ROOT)),output_scale=cap/40.)
                if not any(j['id']==identity for j in wplan['jobs']):
                    wplan['jobs'].append(job);save(walking/'manifest.json',wplan)
                folder=walking/identity;folder.mkdir(exist_ok=True)
                command=[PYTHON,'-u','examples/experiments/assist_skeleton/run_system.py',
                    '--human','tracker','--tracker-checkpoint',plan['assets']['W1']['path'],
                    '--motion-file',plan['assets']['motion']['path'],
                    '--llp-checkpoint',str(checkpoint),'--hlp-checkpoint',plan['assets']['H1']['path'],
                    '--assist-scale',str(cap/40.),'--motion-id','0','--seconds','40','--seed',str(evalseed),
                    '--tag',identity,'--out',str(folder),'--fullbody-diagnostic']
                result=execute(command,identity,timeout=900)
                outputs=[x.parent for x in folder.glob('*/metrics.json') if (x.parent/'fullbody.npz').exists()]
                assert len(outputs)==1,outputs
                output=outputs[0];meta=json.loads((output/'meta.json').read_text())
                assert Path(meta['llp_checkpoint']).resolve()==checkpoint.resolve()
                assert meta['assist_scale']==cap/40.
                assert not meta['oracle'] and not meta['swing_gate'] and not meta['mask_window']
                assert abs(meta['assist_scale']*min(meta['policy_torque_limit'],meta['torque_limit'])-cap)<1e-8
                wrecords[identity]=dict(**job,path=str(output.relative_to(ROOT)),**result,survived_s=meta['survived_s'])
                save(walking/'runs.json',wrecords)
        sys.path.insert(0,str(ROOT/'examples/experiments/assist_skeleton'))
        import report_output_scale as analysis
        analysis.MODELS=tuple(dict.fromkeys(r['model'] for r in wrecords.values()))
        return analysis.summarize(walking)

    def pendulum(model,seed,checkpoint):
        for condition in plan['pendulum_protocol']['conditions']:
            label=f'{model}_s{seed}_H1_{condition}'
            dest=study/'pendulum'/label
            command=[PYTHON,'-u',EVALUATOR,'--base-checkpoint',BASE,'--checkpoint',str(checkpoint),
                '--independent-limits','--combined-human-limit','--condition',condition,
                '--evaluation-limit','20','--output-scale','.5','--out',str(dest),
                '--hlp-checkpoint',plan['assets']['H1']['path'],'--plant-seed','63031',
                '--trajectory-seed','52021','--hlp-seed','991']
            execute(command,label,timeout=1200)
            summary=json.loads((dest/'summary.json').read_text())
            assert summary['injection_error_max_Nm']<1e-4
            assert summary['assist_max_abs_Nm']<=20.00001

    try:
        for model in VARIANTS:
            checkpoint=train(model,0)
            # A finished model's W1 transfer is guaranteed before starting the
            # next expensive training job or optional diagnostic evaluation.
            transfer(model,0,checkpoint)
            pendulum(model,0,checkpoint)
        comparison=json.loads((walking/'comparison.json').read_text())
        eligible=[]
        for model in VARIANTS:
            for cap in (15,20):
                g=comparison['groups'][f'{model}_s0_{cap}'];b=comparison['groups'][f'L1_{cap}']
                m=g['metrics'];bm=b['metrics']
                runs=[r for r in comparison['runs'].values() if r['record']['model']==f'{model}_s0' and r['record']['limit_Nm']==cap]
                bases=[r for r in comparison['runs'].values() if r['record']['model']=='L1' and r['record']['limit_Nm']==cap]
                body=sum(r['summary']['B']['body_err'] for r in runs)/2
                basebody=sum(r['summary']['B']['body_err'] for r in bases)/2
                if (g['falls']==0 and m['overall_tau_sq_reduction']['mean']>=bm['overall_tau_sq_reduction']['mean']
                    and m['low10_rms_increase']['mean']<=bm['low10_rms_increase']['mean']
                    and m['worst_phase_rms_increase']['mean']<=bm['worst_phase_rms_increase']['mean']
                    and body<=basebody+.001):
                    score=m['overall_tau_sq_reduction']['mean']/bm['overall_tau_sq_reduction']['mean']-m['low10_rms_increase']['mean']/bm['low10_rms_increase']['mean']
                    eligible.append(dict(model=model,cap=cap,score=score))
        selection=dict(eligible=eligible,selected=None,remaining_hours=(deadline-time.time())/3600)
        if eligible and deadline-time.time()>2.5*3600:
            model=max(eligible,key=lambda x:x['score'])['model'];selection['selected']=model
            save(study/'repeat_selection.json',selection)
            checkpoint=train(model,1);transfer(model,1,checkpoint);pendulum(model,1,checkpoint)
        else:
            selection['reason']='No joint-endpoint qualifying recipe or insufficient reserved time'
            save(study/'repeat_selection.json',selection)
        status('all planned training and evaluations complete')
        print('COMPLETE',flush=True)
    except BaseException as error:
        status('stopped with error',error=repr(error),**active)
        raise


if __name__=='__main__':
    main()
