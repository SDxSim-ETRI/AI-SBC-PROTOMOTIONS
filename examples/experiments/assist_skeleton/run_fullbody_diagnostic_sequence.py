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
"""Two walking diagnostics, with validation between cases; no training."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from analyze_fullbody_diagnostic import analyze, render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results/walk_fullbody_diagnostic_20261010')
    args = parser.parse_args()
    out = Path(args.out).resolve()
    cmd = json.loads((out / 'instrumented8_command.json').read_text())
    cmd[cmd.index('--seconds') + 1] = '40'
    source_files = [Path(__file__), Path(__file__).with_name('run_system.py'),
                    Path(__file__).with_name('fullbody_diagnostic.py'),
                    Path(__file__).with_name('analyze_fullbody_diagnostic.py'),
                    Path('protomotions/simulator/newton/substep_hook.py'),
                    Path('protomotions/assist/controller.py'), Path('protomotions/assist/llp_policy.py')]
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    snapshot = out / 'source_snapshot_40s'
    snapshot.mkdir(exist_ok=False)
    for p in source_files:
        target = snapshot / p.resolve().relative_to(Path.cwd())
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    (snapshot / 'hashes.json').write_text(json.dumps(hashes, indent=2))
    status = dict(state='running', completed={}, source_snapshot=str(snapshot), training=False)

    def save():
        (out / 'sequence_status.json').write_text(json.dumps(status, indent=2, ensure_ascii=False))

    try:
        for name, options in [('L9_fullbody40', []), ('L9_fullbody40_mask70_90', ['--mask-window', '70,90'])]:
            for file, expected in hashes.items():
                if hashlib.sha256(Path(file).read_bytes()).hexdigest() != expected:
                    raise RuntimeError(f'Diagnostic source changed during sequence: {file}')
            case = cmd.copy()
            case[case.index('--tag') + 1] = name
            case += options
            status['current'] = name
            save()
            print('Starting', name, flush=True)
            with (out / (name + '.log')).open('x') as log:
                subprocess.run(case, stdout=log, stderr=subprocess.STDOUT, check=True)
            candidates = list((out / 'evaluations').glob('*_tracker_' + name))
            if len(candidates) != 1:
                raise RuntimeError(f'Ambiguous result paths for {name}: {candidates}')
            run = candidates[0]
            meta = json.loads((run / 'meta.json').read_text())
            if meta['survived_s'] is not None:
                raise RuntimeError(f'Walking failed at {meta["survived_s"]}: {run}')
            diagnostic = analyze(run)
            render(diagnostic, run / 'fullbody_report')
            status['completed'][name] = dict(run_dir=str(run), validation=diagnostic['validation'])
            save()
            print('Validated', name, diagnostic['validation'], flush=True)
        status['state'] = 'complete'
        status['current'] = None
        save()
    except Exception as exc:
        status['state'] = 'failed'
        status['error'] = repr(exc)
        save()
        raise


if __name__ == '__main__':
    main()
