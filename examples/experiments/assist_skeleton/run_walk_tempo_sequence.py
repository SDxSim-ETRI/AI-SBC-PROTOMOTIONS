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
"""L14: bounded walking tempo curriculum with predeclared unseen tempos."""
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'examples/experiments/assist_skeleton'))
import run_walk_finetune_sequence as base
# The published helper is installed after the current range evaluation finishes.
import validate_walk_finetune as validation
from compare_runs import metrics
from report_torque_html import rank_candidates
from phase_screen import screen_motion

OUT=ROOT/'results/llp_L14_walk_tempo_20261011'
RUN=ROOT/'results/llp_L14_walk_tempo'
PARENT=ROOT/'results/llp_L13R_walk_continue/epoch_400.ckpt'
base.OUT,base.RUN=OUT,RUN


def command(config_only=False):
    cmd=base.train_command(RUN.name,256,200)
    cmd[cmd.index('examples/experiments/assist_skeleton/llp_walk_finetune.py')]='examples/experiments/assist_skeleton/llp_walk_tempo.py'
    cmd[cmd.index('data/motion_for_trackers/skeleton_torque_suit_gait.pt')]='results/llp_walk_curriculum_20261011/walk_tempo_variants.pt'
    cmd+=['--initial-actor-checkpoint',PARENT,'--local-excess-weight',.14901784920034208]
    if config_only:cmd+=['--create-config-only']
    return cmd


def original_screen(rows):
    ref='reference_L1_x0.5_m0'
    p=ROOT/'results/oracle/2026-10-01_18-38-39_tracker_L1s05_verified'
    rec={k:dict(metrics(Path(v['run_dir'])),run_dir=v['run_dir']) for k,v in rows.items()}
    rec[ref]=dict(metrics(p),run_dir=str(p))
    _,diag=rank_candidates(rec);diag=diag['evaluations']
    screen={k:screen_motion(rec[k],rec[ref],diag[k],diag[ref],r['survived'],1,100) for k,r in rows.items() if k.startswith('L14')}
    (OUT/'original_L1_walk_screen.json').write_text(json.dumps(screen,indent=2,default=float))
    base.save(OUT/'L9_development_screen.json',{k:base.screen(r,rows['L9_reference']) for k,r in rows.items() if k.startswith('L14')})


def main():
    base.save(OUT/'protocol.json',dict(parent=str(PARENT),parent_sha256=hashlib.sha256(PARENT.read_bytes()).hexdigest(),
              changed='Training tempos0.90/1.00/1.10 plus body nondegradation margin5mm->1mm to align with the existing evaluation gate. Same actor inputs, plant, other rewards, learning rates and split.',
              source_training='actor copied; critic/optimizer/return stats fresh',
              epochs=200,development_evaluations=[100,200],heldout_policy='fixed final epoch200, no choosing from held-out results',
              train_ids=[1,3,5],heldout_ids=[0,2,4,6],train_start_s=[0,20],episode_s=8,heldout_start_s=60,
              limitation='Synthetic speed scaling of one pattern; not new people/terrains/stairs. L1 development failure is never relabelled as pass.',
              scope='Walking remediation/diversity trial. Running is conditional on walking review.'))
    base.status('configure walking tempo curriculum and verify data split')
    if not (RUN/'resolved_configs.pt').exists():base.run(command(True),'config')
    config=torch.load(RUN/'resolved_configs.pt',map_location='cpu',weights_only=False)
    assert config['env'].motion_manager.exclude_motion_ids==[0,2,4,6]
    verified=OUT/'preflight_32.json'
    if not verified.exists() or not json.loads(verified.read_text()).get('pass'):
        base.run([sys.executable,'examples/experiments/assist_skeleton/verify_walk_finetune.py','--run',RUN,'--num-envs',32,'--steps',250,'--expected-motion-ids','1,3,5','--out',verified],'preflight_32')
    assert json.loads(verified.read_text())['pass']
    base.status('L14 walking tempo learning:200epochs,256pairs')
    ckpt=RUN/'last.ckpt'
    if not ckpt.exists() or torch.load(ckpt,map_location='cpu',weights_only=False)['epoch']<200:base.run(command(),'train')
    assert torch.load(ckpt,map_location='cpu',weights_only=False)['epoch']==200
    old=json.loads((ROOT/'results/llp_L13R_continuation_20261011/comparison.json').read_text())
    rows={k:old[k] for k in ['L9_reference','L13R_e400']}
    for epoch in [100,200]:
        base.status(f'L14 independent nominal walking evaluation:{epoch}')
        condition=dict(motion_file='data/motion_for_trackers/skeleton_torque_suit_gait.pt',motion_id=0,seed=42,start=0)
        rows[f'L14_e{epoch}']=validation.evaluate(RUN/f'epoch_{epoch}.ckpt',f'L14_e{epoch}',condition,fullbody=epoch==200)
        base.save(OUT/'development.json',rows);original_screen(rows)
    tests=json.loads((ROOT/'results/llp_walk_curriculum_20261011/heldout_conditions.json').read_text())
    held={}
    for name,condition in tests.items():
        for label,checkpoint in [('L13R400',PARENT),('L14_200',RUN/'epoch_200.ckpt')]:
            tag=f'{label}_{name}';base.status(f'unseen walking tempo evaluation:{tag}')
            held[tag]=validation.evaluate(checkpoint,tag,condition)
            base.save(OUT/'heldout.json',held)
    base.status('complete:walking tempo curriculum trained and all planned evaluations obtained')

if __name__=='__main__':main()
