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
"""Frozen candidate, matched L9 controls, and independent walking conditions."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'examples/experiments/assist_skeleton'))
import run_walk_finetune_sequence as base


def evaluate(ckpt, tag, condition, fullbody=False):
    done=base.OUT/f'eval_{tag}.json'
    if done.exists():
        return json.loads(done.read_text())
    out=base.OUT/'evaluations'
    runs=sorted(p for p in out.glob(f'*_{tag}') if (p/'metrics.json').exists())
    if not runs:
        cmd=[sys.executable,'examples/experiments/assist_skeleton/run_system.py',
             '--human','tracker','--tracker-checkpoint','checkpoints/W1_wearer_human/last.ckpt',
             '--llp-checkpoint',ckpt,'--hlp-checkpoint',base.HLP,
             '--motion-file',condition['motion_file'],'--motion-id',condition['motion_id'],
             '--seconds',40,'--llp-dt',.01,'--calibration','wearer',
             '--assist-scale',1,'--torque-limit',100,'--seed',condition['seed'],
             '--start-time',condition['start'],'--out',out,'--tag',tag]
        if fullbody:
            cmd+=['--fullbody-diagnostic']
        base.run(cmd,f'eval_{tag}')
        runs=sorted(p for p in out.glob(f'*_{tag}') if (p/'metrics.json').exists())
    assert len(runs)==1,runs
    row=base.summarize(runs[0])
    row['condition']=condition
    base.save(done,row)
    return row


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--candidate',required=True)
    p.add_argument('--name',required=True)
    p.add_argument('--conditions',required=True)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    base.OUT=ROOT/args.out
    base.OUT.mkdir(exist_ok=True,parents=True)
    conditions=json.loads(Path(args.conditions).read_text())
    ckpt=Path(args.candidate)
    protocol=dict(candidate=str(ckpt),sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),
                  candidate_name=args.name,conditions=conditions,
                  fixed_before_evaluation=True,alter_training_from_holdout=False,
                  comparator='L9@2000, same seed/start/motion and each paired unassisted control',
                  duration_s=40,gain=1,limit_Nm=100)
    base.save(base.OUT/'protocol.json',protocol)
    rows={}
    for name,condition in conditions.items():
        for label,path in [('L9',ROOT/'results/llp_L9_opposition/epoch_2000.ckpt'),(args.name,ckpt)]:
            tag=f'{label}_{name}'
            base.status(f'walking validation: {tag}')
            rows[tag]=evaluate(path,tag,condition,fullbody=bool(condition.get('fullbody')) and label!= 'L9')
            base.save(base.OUT/'comparison.json',rows)
        gate=base.screen(rows[f'{args.name}_{name}'],rows[f'L9_{name}'])
        base.save(base.OUT/f'screen_{name}.json',gate)
        print(name,gate,flush=True)
    base.status('complete; frozen walking candidate independently evaluated')

if __name__=='__main__':
    main()
