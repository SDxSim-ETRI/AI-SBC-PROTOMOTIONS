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
"""Run frozen W1 walking diagnostics for every completed general-policy arm."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'results/pendulum_general23_20261015_all'
RUNNER = 'examples/experiments/assist_skeleton/run_system.py'


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    out = STUDY / 'walking'
    out.mkdir(exist_ok=True)
    lock = (out / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((STUDY / 'manifest.json').read_text())
    sources = [RUNNER, 'protomotions/assist/controller.py',
               'examples/experiments/assist_skeleton/fullbody_diagnostic.py',
               'examples/experiments/assist_skeleton/analysis.py',
               'protomotions/assist/llp_policy.py',
               'protomotions/agents/common/shared_joint_mlp.py']
    hashes = {s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in sources}
    amendment = dict(purpose='All-candidate walking phase diagnostics requested by user; no change to frozen training or pendulum selection',
                     protocol='Frozen W1/H1; motion 0; seed 42; 40 seconds; no manual gates; full-body readback',
                     candidate_limit_Nm=23, L1_limit_Nm=20, source_sha256=hashes)
    amendment_path = STUDY / 'reporting_amendment.json'
    if amendment_path.exists():
        assert json.loads(amendment_path.read_text()) == amendment
    else:
        save(amendment_path, amendment)
        for source in sources:
            dest = out / 'sources' / source
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((ROOT / source).read_bytes())
    records_path = out / 'runs.json'
    records = json.loads(records_path.read_text()) if records_path.exists() else {}
    environment = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')

    def status(stage, **extra):
        save(out / 'status.json', dict(stage=stage, pid=os.getpid(), updated_utc=datetime.now(timezone.utc).isoformat(), completed=list(records), **extra))
        print(stage, extra, flush=True)

    def render():
        with (STUDY / 'reporting.lock').open('a') as report_lock:
            fcntl.flock(report_lock, fcntl.LOCK_EX)
            for script in ('examples/experiments/assist_pendulum/report_general_phase.py',
                           'examples/experiments/assist_pendulum/report_general_docs.py',
                           'docs/ai_sbc/organize_reports.py'):
                if (ROOT / script).exists():
                    subprocess.run([sys.executable, script], cwd=ROOT, check=True, env=environment, timeout=60)

    try:
        while True:
            jobs = [('L1', ROOT / 'checkpoints/L1_llp_pendulum/last.ckpt', .5)]
            for seed in (0, 1):
                for label in plan['variants']:
                    ckpt = ROOT / 'results' / f'llp_general23_{label}_s{seed}_20261015_all' / f"epoch_{plan['epochs']}.ckpt"
                    # The final checkpoint must be completely written and the training process finished.
                    trained = any(k.startswith(f'{label}_s{seed}_') for k in json.loads((STUDY / 'comparison.json').read_text())) if (STUDY / 'comparison.json').exists() else False
                    if ckpt.exists() and trained:
                        jobs.append((f'{label}_s{seed}', ckpt, 1.0))
            for label, ckpt, scale in jobs:
                if label in records:
                    continue
                assert all(hashlib.sha256((ROOT / s).read_bytes()).hexdigest() == h for s, h in hashes.items()), 'Walking evaluation sources changed'
                tag = 'phase_' + label
                command = [sys.executable, '-u', RUNNER, '--llp-checkpoint', str(ckpt),
                           '--hlp-checkpoint', plan['hlp_checkpoint'], '--assist-scale', str(scale),
                           '--motion-id', '0', '--seconds', '40', '--seed', '42', '--tag', tag,
                           '--out', str(out), '--fullbody-diagnostic']
                existing = [p for p in out.glob(f'*_tracker_{tag}') if (p / 'metrics.json').exists() and (p / 'fullbody.npz').exists()]
                if not existing:
                    log = out / (label + '.log')
                    with log.open('a') as stream:
                        stream.write(json.dumps(command) + '\n'); stream.flush()
                        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
                        while child.poll() is None:
                            status('evaluating ' + label, child_pid=child.pid, log=str(log), command=command)
                            time.sleep(30)
                        if child.returncode:
                            raise RuntimeError(f'{label}: exit {child.returncode}; see {log}')
                    existing = [p for p in out.glob(f'*_tracker_{tag}') if (p / 'metrics.json').exists() and (p / 'fullbody.npz').exists()]
                assert len(existing) == 1, f'Ambiguous results for {label}: {existing}'
                result = existing[0]
                meta = json.loads((result / 'meta.json').read_text())
                assert Path(meta['llp_checkpoint']).resolve() == ckpt.resolve()
                assert meta['assist_scale'] == scale and not meta['oracle'] and not meta['swing_gate']
                records[label] = dict(path=str(result.relative_to(ROOT)), checkpoint=str(ckpt.relative_to(ROOT)),
                                      checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),
                                      command=command, survived_s=meta['survived_s'])
                save(records_path, records)
                status('completed ' + label)
                render()
            stage = json.loads((STUDY / 'status.json').read_text())['stage']
            if stage.startswith('complete') or args.once:
                missing = [f'{v}_s0' for v in plan['variants'] if f'{v}_s0' not in records]
                if stage.startswith('complete') and missing:
                    raise RuntimeError(f'Training ended but walking diagnostics missing: {missing}')
                status('complete' if not missing else 'available checkpoints evaluated', missing=missing)
                render()
                break
            if stage == 'stopped with error':
                status('waiting for main study error recovery')
            else:
                status('waiting for completed checkpoint')
            time.sleep(30)
    except BaseException as error:
        status('stopped with error', error=repr(error))
        raise


if __name__ == '__main__':
    main()
