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
"""Process identity, stall detection, and checkpoint recovery invariants."""
import os
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from examples.experiments.assist_pendulum import supervise_general as supervisor


def test_recovery_never_terminates_a_different_recorded_command():
    command = [sys.executable, '-c', 'import time; print("ready", flush=True); time.sleep(60)']
    child = subprocess.Popen(command, cwd=supervisor.ROOT, stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'ready'
        assert supervisor.process_matches(child.pid, command)
        supervisor.terminate_recorded(child.pid, command + ['different-job'])
        assert child.poll() is None
        supervisor.terminate_recorded(child.pid, command)
        assert child.wait(timeout=5) != 0
        assert not supervisor.process_matches(child.pid, command)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_stall_uses_output_progress_and_startup_grace(tmp_path):
    log = tmp_path / 'job.log'
    log.write_text('progress')
    now = time.time()
    os.utime(log, (now-1000, now-1000))
    assert not supervisor.stalled(log, now-60, now)
    assert supervisor.stalled(log, now-1000, now)
    os.utime(log, (now, now))
    assert not supervisor.stalled(log, now-1000, now)


def test_invalid_last_checkpoint_is_preserved_and_uses_latest_valid_epoch(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, 'ROOT', tmp_path)
    monkeypatch.setattr(supervisor, 'STUDY', tmp_path/'study')
    directory = tmp_path/'results'/'G1_test'
    directory.mkdir(parents=True)
    (directory/'last.ckpt').write_bytes(b'corrupt interrupted write')
    (directory/'epoch_2000.ckpt').write_bytes(b'also corrupt')
    saved = {'epoch': 1500, 'model': {'weight': torch.ones(1)}, 'actor_optimizer': {}, 'critic_optimizer': {}}
    torch.save(saved, directory/'epoch_1500.ckpt')
    state = {'command': ['python', '--experiment-name', 'G1_test'], 'error': 'test interrupted write'}
    record = supervisor.preserve_failure(state, 1, 'study')
    assert record['resume_epoch'] == 1500
    assert supervisor.valid_checkpoint(directory/'last.ckpt')['epoch'] == 1500
    assert (tmp_path/'study/recovery/attempt_01/last.ckpt').read_bytes() == b'corrupt interrupted write'
    assert record['restored_from'].endswith('epoch_1500.ckpt')


def test_completed_study_exits_without_starting_workers(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, 'ROOT', tmp_path)
    monkeypatch.setattr(supervisor, 'STUDY', tmp_path)
    (tmp_path/'walking').mkdir()
    (tmp_path/'manifest.json').write_text(json.dumps({'source_sha256': {}, 'hlp_checkpoint': 'unused.ckpt'}))
    (tmp_path/'status.json').write_text(json.dumps({'stage': 'complete: held-out L1 criterion failed'}))
    (tmp_path/'walking/status.json').write_text(json.dumps({'stage': 'complete'}))
    monkeypatch.setattr(supervisor, 'notify_systemd', lambda message: None)
    monkeypatch.setattr(supervisor.subprocess, 'run', lambda *a, **kw: subprocess.CompletedProcess(a, 0))

    def no_workers(*args, **kwargs):
        raise AssertionError('Completed studies must never start another worker')

    monkeypatch.setattr(supervisor.subprocess, 'Popen', no_workers)
    assert supervisor.main() == 42
    result = json.loads((tmp_path/'supervision_complete.json').read_text())
    assert result['stage'] == 'complete'
    assert result['lanes'] == {'study': {'status': 'complete'}, 'walking': {'status': 'complete'}}
