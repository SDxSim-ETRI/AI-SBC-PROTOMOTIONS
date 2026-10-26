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
"""Supplementary W1 evaluation of R0/R1 with the native output mapping: scale 1, clip ±20 N·m.

Not part of the pre-registered R0/R1 protocol (which scales 40 tanh(a) by 0.5). The
actual output range is the same ±20 N·m, but the mapping differs: outputs below 20 N·m
are applied unscaled and only larger requests are clipped. Masks and analysis reuse the
stored L1 ±20 N·m references of the same evaluation seed.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "results/llp_relative_20261020"
PYTHON = sys.executable


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def main():
    plan = json.loads((STUDY / "manifest.json").read_text())
    base = json.loads((STUDY / "walking/runs.json").read_text())
    folder = STUDY / "walking_native"; folder.mkdir(exist_ok=True)
    records_path = folder / "runs.json"
    records = json.loads(records_path.read_text()) if records_path.exists() else {}
    for key in ("L1_20Nm_seed42", "L1_20Nm_seed7"):
        records[key] = dict(base[key], reused_baseline=True)
    execution = {}
    for model in ("R0", "R1"):
        checkpoint = ROOT / f"results/llp_relative_{model}_s0_20261020/epoch_6000.ckpt"
        for seed in (42, 7):
            identity = f"{model}n_s0_20Nm_seed{seed}"
            if identity in records and not records[identity].get("reused_baseline"):
                continue
            out = folder / identity; out.mkdir(exist_ok=True)
            command = [PYTHON, "-u", "examples/experiments/assist_skeleton/run_system.py",
                       "--human", "tracker", "--tracker-checkpoint", plan["assets"]["W1"]["path"],
                       "--motion-file", plan["assets"]["motion"]["path"],
                       "--llp-checkpoint", str(checkpoint), "--hlp-checkpoint", plan["assets"]["H1"]["path"],
                       "--assist-scale", "1.0", "--torque-limit", "20", "--motion-id", "0", "--seconds", "40",
                       "--seed", str(seed), "--tag", identity, "--out", str(out), "--fullbody-diagnostic"]
            for attempt in range(1, 4):
                log = folder / f"{identity}_attempt{attempt}.log"
                started = time.time()
                with log.open("w") as stream:
                    stream.write(json.dumps(command) + "\n"); stream.flush()
                    code = subprocess.call(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                execution[f"{identity}_attempt{attempt}"] = dict(command=command, returncode=code, elapsed_s=time.time() - started, log=str(log.relative_to(ROOT)))
                save(folder / "execution.json", execution)
                if code == 0:
                    break
                if code not in (-11, -6, -4):
                    raise RuntimeError(f"{identity} failed with {code}")
            else:
                raise RuntimeError(f"{identity} failed three times")
            outputs = [x.parent for x in out.glob("*/metrics.json") if (x.parent / "fullbody.npz").exists()]
            assert len(outputs) == 1, outputs
            meta = json.loads((outputs[0] / "meta.json").read_text())
            assert meta["assist_scale"] == 1.0 and meta["torque_limit"] == 20.0 and not meta["oracle"]
            records[identity] = dict(id=identity, model=f"{model}n_s0", limit_Nm=20, seed=seed,
                                     checkpoint=str(checkpoint.relative_to(ROOT)), output_scale=1.0, torque_limit_Nm=20,
                                     mapping="native: 40 tanh(a) clipped to ±20", path=str(outputs[0].relative_to(ROOT)),
                                     survived_s=meta["survived_s"], returncode=code, attempt=attempt)
            save(records_path, records)
    save(folder / "manifest.json", dict(purpose="Supplementary native-mapping (scale 1, clip ±20 N·m) W1 evaluation of R0/R1; stored L1 ±20 references for masks",
                                        protocol=plan["walking_protocol"], masks=plan["masks"], assets=plan["assets"],
                                        jobs=[{k: r[k] for k in ("id", "model", "limit_Nm", "seed", "checkpoint", "output_scale")} for r in records.values()]))
    sys.path.insert(0, str(ROOT / "examples/experiments/assist_skeleton"))
    import report_output_scale as analysis
    analysis.MODELS = tuple(dict.fromkeys(r["model"] for r in records.values()))
    analysis.summarize(folder)
    comparison = json.loads((folder / "comparison.json").read_text())
    for key, g in comparison["groups"].items():
        m = g["metrics"]
        print(f"{key:12s} tau2red={m['overall_tau_sq_reduction']['mean']:7.2f} low10={m['low10_rms_increase']['mean']:5.2f} "
              f"worst={m['worst_phase_rms_increase']['mean']:6.2f} assistRMS={m['assist_rms']['mean']:5.2f} sat={m['saturation_pct']['mean']:5.1f}% falls={g['falls']}")


if __name__ == "__main__":
    main()
