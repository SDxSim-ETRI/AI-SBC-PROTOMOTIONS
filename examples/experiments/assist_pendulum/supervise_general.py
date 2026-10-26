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
"""Supervise the fixed study and walking worker; retry until completed or stopped."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'results/pendulum_general23_20261015_all'
POLL_SECONDS = 30
STALLED_SECONDS = 15 * 60
# systemd restarts every other exit, but never this verified completion status.
COMPLETED_EXIT_CODE = 42


def notify_systemd(message):
    address = os.environ.get('NOTIFY_SOCKET')
    if not address:
        return
    if address.startswith('@'):
        address = '\0' + address[1:]
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
        sock.sendto(message.encode(), address)


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save(path, value):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def process_matches(pid, command):
    """Reject dead/zombie/reused PIDs; act only on the recorded exact command."""
    if not isinstance(pid, int) or not command:
        return False
    try:
        proc = Path('/proc') / str(pid)
        if proc.joinpath('stat').read_text().split(') ', 1)[1].split()[0] == 'Z':
            return False
        argv = proc.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')
        return argv == [str(x) for x in command] and proc.joinpath('cwd').resolve() == ROOT
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return False


def stalled(log, started, now):
    """Allow setup time and require an actual output stall, not an old status age."""
    if not log or now - started < STALLED_SECONDS:
        return False
    try:
        return now - Path(log).stat().st_mtime > STALLED_SECONDS
    except FileNotFoundError:
        return now - started > STALLED_SECONDS


def terminate_recorded(pid, command):
    if not process_matches(pid, command):
        return
    os.kill(pid, signal.SIGTERM)
    for _ in range(10):
        if not process_matches(pid, command):
            return
        time.sleep(.5)
    if process_matches(pid, command):
        os.kill(pid, signal.SIGKILL)


def valid_checkpoint(path):
    import torch
    value = torch.load(path, map_location='cpu', weights_only=False)
    if not all(torch.isfinite(t).all() for t in value['model'].values() if isinstance(t, torch.Tensor)):
        raise ValueError('non-finite model')
    if not all(k in value for k in ('epoch', 'actor_optimizer', 'critic_optimizer')):
        raise ValueError('missing epoch/optimizer')
    return value


def preserve_failure(state, attempt, lane):
    folder = STUDY / 'recovery' / f'attempt_{attempt:02d}'
    folder.mkdir(parents=True, exist_ok=True)
    save(folder / 'failure.json', state)
    record = dict(attempt=attempt, lane=lane, recorded_utc=datetime.now(timezone.utc).isoformat(),
                  reason=state.get('error', 'process exited or stopped progressing'),
                  resume='same experiment/checkpoint and optimizer; completed evaluations are reused',
                  caveat='Simulator/RNG restart; not bitwise equivalent to uninterrupted training. train_agent skips the first resumed policy update.')
    command = state.get('command', [])
    if '--experiment-name' in command:
        name = command[command.index('--experiment-name') + 1]
        source = ROOT / 'results' / name
        record['experiment'] = name
        for filename in ('last.ckpt', 'env_null.ckpt', 'config.yaml', 'resolved_configs.pt', 'resolved_configs_inference.pt'):
            p = source / filename
            if p.exists():
                shutil.copy2(p, folder / filename)
        p = source / 'last.ckpt'
        if p.exists() or any(source.glob('epoch_*.ckpt')):
            try:
                checkpoint = valid_checkpoint(p)
            except Exception as error:
                record['invalid_last_checkpoint'] = repr(error)
                # Preserve the invalid file above; roll back only to a valid saved
                # epoch from this exact run, never to an unrelated policy.
                alternatives = sorted(source.glob('epoch_*.ckpt'), key=lambda x: int(x.stem.split('_')[1]), reverse=True)
                for alternative in alternatives:
                    try:
                        checkpoint = valid_checkpoint(alternative)
                    except Exception:
                        continue
                    shutil.copy2(alternative, p)
                    record['restored_from'] = str(alternative)
                    break
                else:
                    record['checkpoint_repair'] = 'No valid saved epoch; retry remains enabled, manual diagnosis required'
                    checkpoint = None
            if checkpoint is not None:
                record.update(resume_epoch=checkpoint['epoch'], checkpoint_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    save(folder / 'resume.json', record)
    return record


def main():
    lock = (STUDY / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = read(STUDY / 'manifest.json')
    if not all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == digest for p, digest in manifest['source_sha256'].items()):
        save(STUDY / 'supervisor_status.json', dict(stage='frozen source verification failed; systemd retry remains enabled', updated_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid()))
        raise RuntimeError('Frozen source hashes changed; restore/diagnose before training. systemd will retry.')
    environment = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONFAULTHANDLER='1')
    commands = {
        'study': [sys.executable, '-u', 'examples/experiments/assist_pendulum/run_general.py', '--hlp-checkpoint', manifest['hlp_checkpoint']],
        'walking': [sys.executable, '-u', 'examples/experiments/assist_pendulum/run_general_transfer.py'],
    }
    states = {'study': STUDY / 'status.json', 'walking': STUDY / 'walking/status.json'}
    logs = {'study': STUDY / 'runner.log', 'walking': STUDY / 'walking/worker.log'}
    owned = {}
    known = {}
    last_start = {'study': 0., 'walking': 0.}
    recoveries = read(STUDY / 'recoveries.json', [])
    heartbeat = 0
    progress = {}
    # Adopt already running workers without interrupting their training/evaluation.
    now = time.time()
    for lane in commands:
        state = read(states[lane], {})
        pid = state.get('pid')
        if process_matches(pid, commands[lane]):
            known[lane] = pid
            last_start[lane] = now

    def render():
        with (STUDY / 'reporting.lock').open('a') as report_lock:
            try:
                fcntl.flock(report_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            for script in ('examples/experiments/assist_pendulum/report_general_docs.py', 'docs/ai_sbc/organize_reports.py'):
                try:
                    result = subprocess.run([sys.executable, script], cwd=ROOT, env=environment, capture_output=True, text=True, timeout=45)
                    if result.returncode:
                        (STUDY / 'report_refresh_error.log').write_text(result.stdout + result.stderr)
                except subprocess.TimeoutExpired:
                    (STUDY / 'report_refresh_error.log').write_text('Report refresh timed out; computation monitoring continues')

    def recovery(state, lane):
        record = preserve_failure(state, len(recoveries) + 1, lane)
        recoveries.append(record)
        save(STUDY / 'recoveries.json', recoveries)

    notify_systemd('READY=1\nWATCHDOG=1')
    while True:
        notify_systemd('WATCHDOG=1')
        state = {lane: read(path, {}) for lane, path in states.items()}
        done = {'study': state['study'].get('stage', '').startswith('complete'),
                'walking': state['walking'].get('stage') == 'complete'}
        now = time.time()
        monitor = {}
        for lane in commands:
            if done[lane]:
                monitor[lane] = dict(status='complete')
                continue
            parent = known.get(lane, state[lane].get('pid'))
            parent_alive = process_matches(parent, commands[lane])
            leaf = state[lane].get('child_pid')
            leaf_command = state[lane].get('command', [])
            leaf_alive = process_matches(leaf, leaf_command)
            # Walking status historically omitted the child command. Read its
            # exact argv only when parent is known to be our worker and out is ours.
            if lane == 'walking' and isinstance(leaf, int) and parent_alive and not leaf_command:
                try:
                    candidate = (Path('/proc') / str(leaf) / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
                    if 'examples/experiments/assist_skeleton/run_system.py' in candidate and '--out' in candidate and candidate[candidate.index('--out')+1] == str(STUDY/'walking'):
                        leaf_command = candidate
                        leaf_alive = process_matches(leaf, candidate)
                except (FileNotFoundError, ProcessLookupError):
                    pass
            if parent_alive or leaf_alive:
                log = state[lane].get('log') if leaf_alive else str(states[lane])
                epoch = state[lane].get('observed_epoch')
                marker = (leaf, epoch)
                epoch_stalled = False
                if leaf_alive and isinstance(epoch, int) and any(str(x).endswith('train_agent.py') for x in leaf_command):
                    if progress.get(lane, (None,))[0] != marker:
                        progress[lane] = (marker, now)
                    epoch_stalled = now - progress[lane][1] > STALLED_SECONDS
                if stalled(log, last_start[lane], now) or epoch_stalled:
                    failure = dict(state[lane], error=f'No log or training epoch progress for {STALLED_SECONDS}s; terminating only recorded processes')
                    recovery(failure, lane)
                    terminate_recorded(leaf, leaf_command)
                    terminate_recorded(parent, commands[lane])
                    known.pop(lane, None)
                    monitor[lane] = dict(status='stalled process stopped; restart next poll')
                else:
                    monitor[lane] = dict(status='running' if parent_alive else 'adopting surviving child', pid=parent if parent_alive else leaf,
                                         stage=state[lane].get('stage'), epoch=state[lane].get('observed_epoch'))
                continue
            if now - last_start[lane] < POLL_SECONDS:
                monitor[lane] = dict(status='retry backoff')
                continue
            if lane in owned:
                owned[lane].poll()  # Reap the exited process before replacing it.
            if state[lane]:
                recovery(state[lane], lane)
            with logs[lane].open('a') as stream:
                child = subprocess.Popen(commands[lane], cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            owned[lane] = child
            known[lane] = child.pid
            last_start[lane] = now
            monitor[lane] = dict(status='restarted', pid=child.pid)
        value = dict(stage='complete' if all(done.values()) else 'monitoring', updated_utc=datetime.now(timezone.utc).isoformat(),
                     pid=os.getpid(), poll_seconds=POLL_SECONDS, stalled_seconds=STALLED_SECONDS,
                     retry_limit=None, recovery_count=len(recoveries), lanes=monitor)
        save(STUDY / 'supervisor_status.json', value)
        print(json.dumps(value, ensure_ascii=False), flush=True)
        if all(done.values()):
            render()
            save(STUDY / 'supervision_complete.json', value)
            return COMPLETED_EXIT_CODE
        if heartbeat % 2 == 0:
            render()
        heartbeat += 1
        time.sleep(POLL_SECONDS)


if __name__ == '__main__':
    sys.exit(main())
