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
"""L13R: one justified objective refinement; same walking plant and actor inputs."""

import json
import sys
from pathlib import Path

import torch

import run_walk_finetune_sequence as base

ROOT = base.ROOT
OUT = ROOT / 'results/llp_L13R_fullbody_walk_20261011'
RUN = ROOT / 'results/llp_L13R_fullbody_walk'
base.OUT, base.RUN = OUT, RUN


def command(epochs=200, config_only=False):
    weight = json.loads((OUT / 'calibration.json').read_text())['mean_matched_weight']
    cmd = base.train_command(RUN.name, 256, epochs)
    cmd[cmd.index('examples/experiments/assist_skeleton/llp_walk_finetune.py')] = 'examples/experiments/assist_skeleton/llp_walk_finetune_relative.py'
    cmd += ['--local-excess-weight', weight]
    if config_only:
        cmd += ['--create-config-only']
    return cmd


def report(rows):
    base.report(rows)
    p = OUT / 'comparison.html'
    s = p.read_text().replace('초기정책은 L9의 좌표 부호만 등가 변환했다.',
        'L13R 초기정책은 L13@100이다. 무보조 대비 부담 증가 비용을 상대값으로 바꾸고, '
        '학습 구간에서 측정한 평균 비용이 이전과 같도록 계수를 맞췄다. phase 규칙은 없다.')
    p.write_text(s)


def main():
    calibration = json.loads((OUT / 'calibration.json').read_text())
    base.save(OUT / 'protocol.json', dict(parent='L13@100', changed='relative paired local harm only; actor warm start, fresh critic/optimizer',
               calibration=calibration, epochs=200, evaluation_epochs=[50,100,200],
               actor_inputs=137, training_start_s=[0,20], episode_s=8,
               development_gate='unchanged L1 per-motion screen plus L9-based walking screen',
               next='held-out walking conditions only after walking screen; running contingent on walking success'))
    base.status('configure L13R and verify unchanged starting actor')
    if not (RUN / 'resolved_configs.pt').exists():
        base.run(command(config_only=True), 'config')
    verified = OUT / 'preflight_32.json'
    if not verified.exists() or not json.loads(verified.read_text()).get('pass'):
        base.run([sys.executable, 'examples/experiments/assist_skeleton/verify_walk_finetune.py',
                  '--run', RUN, '--num-envs', 32, '--steps', 250, '--out', verified], 'preflight_32')
    assert json.loads(verified.read_text())['pass']
    base.status('L13R walking learning: 200 epochs')
    ckpt = RUN / 'last.ckpt'
    if not ckpt.exists() or torch.load(ckpt, map_location='cpu', weights_only=False)['epoch'] < 200:
        base.run(command(), 'train')
    assert torch.load(ckpt, map_location='cpu', weights_only=False)['epoch'] == 200
    previous = json.loads((ROOT/'results/llp_L13_fullbody_walk_20261011/comparison.json').read_text())
    rows = {key: base.summarize(Path(previous[key]['run_dir'])) for key in ('L9_reference','L13_e100','L13_e200')}
    for epoch in (50,100,200):
        base.status(f'L13R independent walking evaluation: epoch {epoch}')
        rows[f'L13R_e{epoch}'] = base.evaluate(RUN/f'epoch_{epoch}.ckpt', f'L13R_e{epoch}')
        report(rows)
    base.status('relative_pilot_complete: review before expansion')


if __name__ == '__main__':
    main()
