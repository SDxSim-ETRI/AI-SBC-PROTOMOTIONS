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
"""Read saved configurations and audit intention curvature without policy rollouts."""
import argparse
from dataclasses import fields, is_dataclass
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from protomotions.envs.control.pendulum_feasible_control import (
    PendulumFeasibleControl, PendulumFeasibleControlConfig,
)
from protomotions.robot_configs.hip_pendulum import HipPendulumRobotConfig


def canonical(value):
    if is_dataclass(value): return {f.name:canonical(getattr(value,f.name)) for f in fields(value)}
    if isinstance(value,dict): return {str(k):canonical(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [canonical(v) for v in value]
    if isinstance(value,torch.Tensor): return value.tolist()
    if callable(value): return value.__module__+'.'+value.__qualname__
    if isinstance(value,Path): return str(value)
    if hasattr(value,'__dict__'): return canonical(vars(value))
    if value is None or isinstance(value,(str,int,float,bool)): return value
    return str(value)


def differences(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        return sum([differences(a.get(k),b.get(k),path+'.'+k) for k in sorted(set(a)|set(b))],[])
    if a!=b: return [dict(path=path,old=a,new=b)]
    return []


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='results/llp_feasible_20261020')
    args=p.parse_args();out=Path(args.out);out.mkdir(exist_ok=True,parents=True)
    paths={'L1':'checkpoints/L1_llp_pendulum','G0':'results/llp_general23_G0_s0_20261015_all',
           'G4':'results/llp_general23_G4_s0_20261015_all','F1':'results/llp_feasible_F1_s0_20261020'}
    configs={name:canonical(torch.load(Path(path)/'resolved_configs.pt',map_location='cpu',weights_only=False)) for name,path in paths.items()}
    (out/'config_audit.json').write_text(json.dumps(dict(configs=configs,differences={
        name:differences(configs['L1'],cfg) for name,cfg in configs.items() if name!='L1'}),ensure_ascii=False,indent=2))
    n=512;torch.manual_seed(73021)
    env=SimpleNamespace(num_envs=n,dt=.01,device=torch.device('cpu'),robot_config=HipPendulumRobotConfig(),
        progress_buf=torch.zeros(n,dtype=torch.long),
        state_history=SimpleNamespace(dof_pos=torch.zeros(n,1,2),dof_vel=torch.zeros(n,1,2)))
    raw=torch.load(Path(paths['F1'])/'resolved_configs.pt',map_location='cpu',weights_only=False)['env'].control_components['assist']
    c=PendulumFeasibleControl(raw,env)
    inertia=.2863+.1316*(1+24*torch.rand(n,2))
    c.set_wearer_gains(torch.ones(n,2)*500,torch.ones(n,2)*30,torch.ones(n,2)*150,inertia)
    # Compare old cubic vs new quintic on IDENTICAL independently sampled
    # segments/start times/plant inertias; no checkpoint is evaluated here.
    old_tau=[];new_tau=[];durations=[];requested=[]
    sample_grid=np.linspace(.0001,.9999,101)[None,:,None]
    for batch in range(8):
        ids=torch.arange(n)
        c._sample_segment(0,ids,True);c._sample_segment(1,ids,True)
        c._seg_start[:]=torch.rand(n)*10
        c._blend_dur[:]=.25+2.25*torch.rand(n)
        c._limited_start[:]=float('nan');c._limit_new_transitions(ids)
        pp={k:v.numpy().astype(np.float64) for k,v in c._params.items()}
        for kind,target in [('cubic',old_tau),('quintic',new_tau)]:
            duration=(c._requested_blend if kind=='cubic' else c._blend_dur).numpy()[:,None,None]
            t=c._seg_start.numpy()[:,None,None]+duration*sample_grid
            q=[];v=[];a=[]
            for slot in (0,1):
                f=pp['freq'][slot,:,None,:];w=2*np.pi*f
                amp=pp['amp'][slot,:,None,:];amp2=pp['amp2'][slot,:,None,:]
                z=w*t+pp['phase'][slot,:,None,:];z2=2*w*t+pp['phase2'][slot,:,None,:]
                q.append(pp['offset'][slot,:,None,:]+amp*np.sin(z)+amp2*np.sin(z2))
                v.append(amp*w*np.cos(z)+2*amp2*w*np.cos(z2))
                a.append(-amp*w*w*np.sin(z)-4*amp2*w*w*np.sin(z2))
            x=sample_grid
            if kind=='cubic': s=x*x*(3-2*x);ds=6*x*(1-x);dds=6-12*x
            else: s=x**3*(10-15*x+6*x*x);ds=30*x*x*(1-x)**2;dds=60*x-180*x*x+120*x**3
            acceleration=(1-s)*a[0]+s*a[1]+2*ds/duration*(v[1]-v[0])+dds/duration**2*(q[1]-q[0])
            target.append(np.abs(acceleration)*inertia.numpy()[:,None,:])
        durations.append(c._blend_dur.numpy().copy());requested.append(c._requested_blend.numpy().copy())
    old_tau=np.concatenate(old_tau);new_tau=np.concatenate(new_tau)
    d=np.concatenate(durations);r=np.concatenate(requested)
    def stats(x): return dict(mean=float(x.mean()),p95=float(np.quantile(x,.95)),p99=float(np.quantile(x,.99)),max=float(x.max()),above_150_fraction=float((x>150).mean()))
    audit=dict(seed=73021,transitions=n*8,joint_time_samples=old_tau.size,
        old_cubic_inertial_torque_Nm=stats(old_tau),new_quintic_inertial_torque_Nm=stats(new_tau),
        requested_duration_s=np.quantile(r,[0,.5,.95,1]).tolist(),actual_duration_s=np.quantile(d,[0,.5,.95,1]).tolist(),
        duration_extended_fraction=float((d>r).mean()),
        analytical_inertial_torque_bound_Nm=75+40,
        caveat='Offline analytic intentions, identical parameters. No policy/W1 rollout. Gravity, external load and actuator transients are not included; not a complete feasibility certificate.')
    assert new_tau.max()<115.001
    (out/'trajectory_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
