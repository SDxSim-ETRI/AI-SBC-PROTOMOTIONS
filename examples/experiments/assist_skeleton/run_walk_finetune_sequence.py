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
"""Validated, bounded L13 walking pilot, with independent deployment evaluations."""

import hashlib
import html
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from report_torque_html import aligned_torques, phase_profile, reduction_bars, phase_table

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/llp_L13_fullbody_walk_20261011"
RUN = ROOT / "results/llp_L13_fullbody_walk"
HLP = "/home/bak/Projects/ManiFlow_Policy/ManiFlow/data/outputs/hip_flexion_seed-maniflow_lowdim_policy_walking-flexion100-h15-run01_seed42/checkpoints/epoch=0006-val_loss=0.000762.ckpt"


def save(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def status(stage):
    save(OUT / 'status.json', dict(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat()))
    print(stage, flush=True)


def snapshot():
    paths = [str(p.relative_to(ROOT)) for folder in ('protomotions/assist', 'protomotions/envs',
             'protomotions/agents/ppo', 'protomotions/maniflow', 'examples/experiments/assist_skeleton')
             for p in (ROOT / folder).rglob('*.py')]
    paths.append('protomotions/train_agent.py')
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}
    path = OUT / 'source_sha256.json'
    if path.exists():
        assert json.loads(path.read_text()) == hashes, 'Source changed during sequence'
    else:
        for p in paths:
            dest = OUT / 'source_snapshot' / p
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / p, dest)
        save(path, hashes)


def run(command, name):
    snapshot()
    print('RUN', ' '.join(map(str, command)), flush=True)
    for attempt in range(3):
        with (OUT / f'{name}.log').open('a') as f:
            result = subprocess.run(list(map(str, command)), cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        if result.returncode == 0:
            return
        if result.returncode != -11 or attempt == 2:
            raise RuntimeError(f'{name} exited {result.returncode}; see preserved log')
        print(f'{name}: native segfault; resume attempt {attempt + 1}', flush=True)


def train_command(name, n, epochs):
    return [sys.executable, 'protomotions/train_agent.py', '--robot-name', 'skeleton_torque_suit_human',
            '--simulator', 'newton', '--experiment-path', 'examples/experiments/assist_skeleton/llp_walk_finetune.py',
            '--experiment-name', name, '--motion-file', 'data/motion_for_trackers/skeleton_torque_suit_gait.pt',
            '--num-envs', n, '--batch-size', n * 32, '--training-max-steps', epochs * n * 32, '--seed', 13,
            '--distributed-backend', 'gloo']


def summarize(path):
    meta = json.loads((path / 'meta.json').read_text())
    summary = json.loads((path / 'metrics.json').read_text())['summary']
    with np.load(path / 'traces.npz') as tr:
        data = aligned_torques(tr, meta)
        idx = np.flatnonzero((tr['t'] >= meta['warmup']) & (np.arange(len(tr['t'])) >= 1))
        vel = .5 * (tr['hip_vel'][idx - 1] + tr['hip_vel'][idx]) if 'hip_vel' in tr else None
    a, b = data['agent_A'], data['agent_B']
    regions = {}
    for lo, hi in ((20, 40), (70, 80), (80, 90), (70, 90)):
        mask = (data['phase'] >= lo / 100) & (data['phase'] < hi / 100)
        rms = lambda x: float(np.sqrt(np.mean(x[mask] ** 2)))
        regions[f'{lo}-{hi}'] = dict(A_rms=rms(a), B_rms=rms(b), increase=rms(b) - rms(a))
    return dict(run_dir=str(path), tau2_change_pct=float(100 * (np.mean(b*b) / np.mean(a*a) - 1)),
                regions=regions, phase=phase_profile(data, 10),
                body_increase_m=summary['body_err_m_B'] - summary['body_err_m_A'],
                speed_change_pct=summary['walk_speed_change_pct'],
                survived=meta['survived_s'] is None and meta['seconds'] >= 40,
                omega_change_pct=(None if vel is None else float(100 * (np.abs(vel[:, 1]).mean() / np.abs(vel[:, 0]).mean() - 1))),
                seed=meta.get('seed'), start_time=meta['start_time'])


def evaluate(ckpt, tag, seed=42, start=0.):
    done = OUT / f'eval_{tag}.json'
    if done.exists():
        return json.loads(done.read_text())
    root = OUT / 'evaluations'
    candidates = sorted(p for p in root.glob(f'*_{tag}') if (p / 'metrics.json').exists())
    if not candidates:
        run([sys.executable, 'examples/experiments/assist_skeleton/run_system.py', '--human', 'tracker',
             '--tracker-checkpoint', 'checkpoints/W1_wearer_human/last.ckpt', '--llp-checkpoint', ckpt,
             '--hlp-checkpoint', HLP, '--motion-file', 'data/motion_for_trackers/skeleton_torque_suit_gait.pt',
             '--motion-id', 0, '--seconds', 40, '--llp-dt', .01, '--calibration', 'wearer',
             '--assist-scale', 1, '--torque-limit', 100, '--seed', seed, '--start-time', start,
             '--out', root, '--tag', tag], f'eval_{tag}')
        candidates = sorted(p for p in root.glob(f'*_{tag}') if (p / 'metrics.json').exists())
    assert len(candidates) == 1, candidates
    row = summarize(candidates[0])
    save(done, row)
    return row


def screen(row, reference):
    checks = dict(survived=row['survived'], global_burden=row['tau2_change_pct'] <= reference['tau2_change_pct'],
                  late_swing=row['regions']['70-90']['increase'] <= reference['regions']['70-90']['increase'] - 2,
                  phase20_40=row['regions']['20-40']['increase'] <= reference['regions']['20-40']['increase'],
                  all_bins=all(a['rms_reduction_Nm'] >= b['rms_reduction_Nm'] - 1
                               for a,b in zip(row['phase'],reference['phase'])),
                  body=row['body_increase_m'] <= .001, speed=abs(row['speed_change_pct']) <= 2,
                  angular_speed=row['omega_change_pct'] is not None and abs(row['omega_change_pct']) <= 9.153601456140912)
    return {'pass': all(checks.values()), 'checks': checks}


def report(rows):
    save(OUT / 'comparison.json', rows)
    s = ['<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width">',
         '<title>L13 전신 걷기 추가학습</title><style>body{max-width:1150px;margin:32px auto;padding:20px;font:16px/1.7 system-ui;background:#faf9f6;color:#222}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}.scroll{overflow:auto}svg{width:100%;height:auto}.grid{stroke:#ddd}.tick{font:12px sans-serif}.axis{font:13px sans-serif}</style>',
         '<h1>L13 · L9에서 전신 걷기 추가학습</h1><p>W1과 실제 HLP는 동결, LLP만 학습. '
         '보조 gain 1, 최종 ±100 N·m. 초기정책은 L9의 좌표 부호만 등가 변환했다. '
         '학습 시작점 0–20초, 에피소드8초; 60초 이후는 LLP 추가학습 미사용 구간이다. '
         '같은 클립의 다른 구간은 새로운 보행 패턴 전체에 대한 일반화 입증이 아니다. 모든 그림은 내장.</p>',
         '<table><tr><th>정책</th><th>τ² 변화</th><th>70–90 RMS 증가</th><th>몸체 오차 증가</th><th>속도 변화</th><th>40초</th></tr>']
    for name, r in rows.items():
        s.append(f'<tr><th>{html.escape(name)}</th><td>{r["tau2_change_pct"]:+.2f}%</td><td>{r["regions"]["70-90"]["increase"]:+.2f} Nm</td><td>{1000*r["body_increase_m"]:+.2f} mm</td><td>{r["speed_change_pct"]:+.3f}%</td><td>{r["survived"]}</td></tr>')
    s.append('</table>')
    for name, r in rows.items():
        s += [f'<h2>{html.escape(name)}</h2>', reduction_bars(r['phase']), phase_table(r['phase'])]
    if 'L9_reference' in rows:
        checks = {k: screen(v, rows['L9_reference']) for k,v in rows.items() if k.startswith('L13') and '_e' in k}
        save(OUT / 'walk_screen.json', checks)
        s.append('<h2>L9 대비 사전 개발 판정</h2><pre>' + html.escape(json.dumps(checks, ensure_ascii=False, indent=2)) + '</pre>')
    s.append('</html>')
    (OUT / 'comparison.html').write_text(''.join(s))


def main():
    snapshot()
    status('preflight: 256 pairs, real HLP, partial reset and ending checks')
    verified = OUT / 'preflight_256.json'
    if not verified.exists() or not json.loads(verified.read_text()).get('pass'):
        run([sys.executable, 'examples/experiments/assist_skeleton/verify_walk_finetune.py',
             '--num-envs', 256, '--steps', 250, '--out', verified], 'preflight_256')
    assert json.loads(verified.read_text())['pass']
    status('PPO smoke: 8 epochs, 32 pairs')
    smoke = ROOT / 'results/llp_L13_ppo_smoke/last.ckpt'
    if not smoke.exists():
        run(train_command('llp_L13_ppo_smoke', 32, 8), 'ppo_smoke')
    assert torch.load(smoke, map_location='cpu', weights_only=False)['epoch'] == 8
    rows = {'L9_reference': summarize(ROOT / 'results/walk_fullbody_diagnostic_20261010/evaluations/2026-10-10_23-31-12_tracker_L9_fullbody40')}
    status('independent 40s evaluation of unchanged initial policy')
    rows['L13_e0'] = evaluate(RUN / 'initial.ckpt', 'L13_e0')
    report(rows)
    # Gross mismatch blocks optimization; fine repeatability remains reported.
    assert abs(rows['L13_e0']['tau2_change_pct'] - rows['L9_reference']['tau2_change_pct']) < 2
    status('walking PPO: 200 epochs, 256 pairs')
    last = RUN / 'last.ckpt'
    if not last.exists() or torch.load(last, map_location='cpu', weights_only=False)['epoch'] < 200:
        run(train_command('llp_L13_fullbody_walk', 256, 200), 'train')
    assert torch.load(last, map_location='cpu', weights_only=False)['epoch'] == 200
    for epoch in (50, 100, 200):
        status(f'independent walking evaluation: epoch {epoch}')
        rows[f'L13_e{epoch}'] = evaluate(RUN / f'epoch_{epoch}.ckpt', f'L13_e{epoch}')
        report(rows)
    status('pilot_complete: review walking screen before held-out expansion or running')


if __name__ == '__main__':
    main()
