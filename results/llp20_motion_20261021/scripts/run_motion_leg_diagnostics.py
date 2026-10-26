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
"""예측 신호 진단 실행기 (2026-10-22): 네 조건 학습이 끝나면 새 모델 4개와 L1·R0를
학습 분포 진자에서 에뮬레이션 예측 / 실제 H1 예측으로 각각 평가한다 (`diagnose_motion_leg_chunk.py`).

    python examples/experiments/assist_pendulum/run_motion_leg_diagnostics.py
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "results/llp20_motion_20261021"
SCRIPT = "examples/experiments/assist_pendulum/diagnose_motion_leg_chunk.py"
CONDITIONS = ("A0D0", "A1D1", "A1D0", "A0D1")


def main():
    records = STUDY / "execution_right.json"
    while True:
        done = json.loads(records.read_text()) if records.exists() else {}
        if all(f"{c}_right" in done for c in CONDITIONS):
            break
        time.sleep(120)
    hlp = json.loads((ROOT / "results/llp_output_scale_20261016/manifest.json").read_text())["assets"]["H1"]["path"]
    config = done["A0D0_right"]["checkpoint"]
    models = {c: done[f"{c}_right"]["checkpoint"] for c in CONDITIONS}
    models.update(L1="checkpoints/L1_llp_pendulum/last.ckpt", R0="results/llp_relative_R0_s0_20261020/epoch_6000.ckpt")
    environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONFAULTHANDLER="1")
    for name, checkpoint in models.items():
        for source in ("emulated", "h1"):
            out = STUDY / "diagnostics" / f"chunk_{name}_{source}.json"
            if out.exists():
                continue
            command = [sys.executable, "-u", SCRIPT, "--config-checkpoint", config, "--checkpoint", checkpoint,
                       "--out", str(out)] + (["--hlp-checkpoint", hlp] if source == "h1" else [])
            for attempt in range(1, 4):
                log = STUDY / "diagnostics" / f"chunk_{name}_{source}_attempt{attempt}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("w") as stream:
                    stream.write(json.dumps(command) + "\n")
                    stream.flush()
                    code = subprocess.call(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
                if code == 0:
                    break
                if code not in (-11, -6) or attempt == 3:
                    raise RuntimeError(f"{name} {source}: exit {code}; see {log}")
            print("DONE", name, source, flush=True)
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
