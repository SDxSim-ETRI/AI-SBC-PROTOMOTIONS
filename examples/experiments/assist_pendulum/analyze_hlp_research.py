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
"""Compare frozen H1 and emulated forecasts on the same recorded pendulum states.

This is an offline prediction audit, not a closed-loop assistance evaluation.
Trunk pitch/roll are explicitly zero for the fixed-base pendulum; hip angles use
its flexion sign (-1). No walking motion or torque is used as a training target.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from protomotions.maniflow.angle_estimator import ManiFlowAngleEstimator
from protomotions.maniflow.chunk_source import ManiFlowChunkSource


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--out", required=True)
    args=parser.parse_args()
    device=torch.device("cuda")
    torch.manual_seed(args.seed)
    data=np.load(args.rollout)
    total=data["assisted_q"].shape[1]
    ids=np.linspace(0,total-1,min(total,args.num_envs),dtype=int)
    q=torch.as_tensor(data["assisted_q"][:,ids],device=device)
    valid=data["valid"][:,ids] & (data["time_s"][:,ids] >= .5)
    emulated=data["predicted_angles"][:,ids]
    estimator=ManiFlowAngleEstimator.from_checkpoint(args.checkpoint,
        num_envs=len(ids),device=device,denoise_steps=1)
    source=ManiFlowChunkSource(estimator,len(ids),.01,device,flexion_sign=-1.)
    assert abs(source.knot_dt-.01) < 1e-6 and source.num_knots >= 3
    env_ids=torch.arange(len(ids),device=device)
    source.reset(env_ids,q[0],trunk=None)
    forecasts=[]
    with torch.no_grad():
        for i in range(q.shape[0]-2):
            if i:
                source.observe(q[i],trunk=None)
            knots,_=source.predict_knots(env_ids,torch.full((len(ids),),(i+1)*.01,device=device))
            forecasts.append(knots[:,:3].cpu().numpy())
    real=np.stack(forecasts)
    rows=[]
    for k in range(3):
        target=data["assisted_q"][k:k+len(real),ids]
        # Saved emulation was observed BEFORE its physics step; q[i] was
        # recorded AFTER step i, hence emulated[i+1] is the matching forecast.
        old=emulated[1:1+len(real),:,k]
        mask=valid[:len(real)] & valid[k:k+len(real)]
        e_real=(real[:,:,k]-target)[mask]
        e_old=(old-target)[mask]
        rows.append(dict(lead_ms=10*k,joint_samples=int(e_real.size),
            H1_MAE_deg=float(np.abs(e_real).mean()*180/np.pi),
            emulated_MAE_deg=float(np.abs(e_old).mean()*180/np.pi),
            H1_RMSE_deg=float(np.sqrt(np.mean(e_real**2))*180/np.pi),
            emulated_RMSE_deg=float(np.sqrt(np.mean(e_old**2))*180/np.pi)))
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/"predictions.npz",actual_h1=real,env_ids=ids)
    report=dict(protocol=vars(args),trunk_pitch_roll_rad=[0.,0.],flexion_sign=-1.,
        dt_s=.01,denoise_steps=1,metrics=rows,
        caveat="Offline common-state prediction comparison only; no closed-loop assistance claim.")
    (out/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__ == "__main__":
    main()
