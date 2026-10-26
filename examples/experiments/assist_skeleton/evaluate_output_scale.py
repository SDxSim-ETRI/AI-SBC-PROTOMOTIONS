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
"""Frozen walking output-scale comparison, with bounded retries and provenance."""

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
RUNNER = "examples/experiments/assist_skeleton/run_system.py"
MODELS = {
    "L1": ("checkpoints/L1_llp_pendulum/last.ckpt", 40.0),
    "G0_s0": ("results/llp_general23_G0_s0_20261015_all/epoch_3000.ckpt", 23.0),
    "G4_s0": ("results/llp_general23_G4_s0_20261015_all/epoch_3000.ckpt", 23.0),
    "G4_s1": ("results/llp_general23_G4_s1_20261015_all/epoch_3000.ckpt", 23.0),
}
SOURCES = [
    "examples/experiments/assist_skeleton/evaluate_output_scale.py",
    RUNNER,
    "examples/experiments/assist_skeleton/fullbody_diagnostic.py",
    "examples/experiments/assist_skeleton/analysis.py",
    "protomotions/assist/controller.py",
    "protomotions/assist/llp_policy.py",
    "protomotions/assist/skeleton_features.py",
    "protomotions/agents/common/shared_joint_mlp.py",
    "protomotions/maniflow/angle_estimator.py",
    "protomotions/maniflow/chunk_source.py",
    "protomotions/maniflow/hybrid_control.py",
    "protomotions/simulator/newton/substep_hook.py",
]


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="results/llp_output_scale_20261016")
    args = parser.parse_args()
    study = (ROOT / args.study).resolve()
    study.mkdir(parents=True, exist_ok=True)
    lock = (study / "worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    old_plan = json.loads((ROOT / "results/pendulum_general23_20261015_all/manifest.json").read_text())
    hlp = old_plan["hlp_checkpoint"]
    jobs = []
    # A single freshly evaluated reference supplies fixed analysis masks per seed.
    for seed in (42, 7):
        for limit in (20, 10, 15):
            for model, (checkpoint, native_limit) in MODELS.items():
                jobs.append(dict(id=f"{model}_{limit}Nm_seed{seed}", model=model,
                                 limit_Nm=limit, seed=seed, checkpoint=checkpoint,
                                 output_scale=limit / native_limit))
    assets = {model: str(ROOT / checkpoint) for model, (checkpoint, _) in MODELS.items()}
    assets.update(W1=str(ROOT / "checkpoints/W1_wearer_human/last.ckpt"), H1=hlp,
                  motion=str(ROOT / "data/motion_for_trackers/skeleton_torque_suit_gait.pt"))
    plan = dict(
        purpose="Separate output amplitude effects from persistent low-burden deceleration interference; no training",
        jobs=jobs, seconds=40, analysis_start_s=1.5,
        protocol="Frozen W1/H1, existing walk motion 0, wearer calibration, 400 Hz physics, 100 Hz assistance, no gates/filters",
        scale_contract="Multiply the existing bounded policy torque by limit/native_limit; do not replace scaling with clipping",
        masks=dict(source="Unassisted A of fresh L1 20 Nm run for the SAME evaluation seed; reuse at identical times across all policies/scales",
                   smoothing="Centered 5-sample (50 ms) moving average, offline only; torque uses sqrt(mean(tau_A squared))",
                   deceleration="smoothed omega * derivative(smoothed omega) < 0, abs(omega)>0.3 rad/s, abs(alpha)>1 rad/s^2",
                   primary_low_burden_Nm=10, strict_sensitivity_Nm=5,
                   edge_exclusion_s=.05,
                   rationale="Existing reference inspection found few strict <=5 Nm samples; report both fixed thresholds and exposure, without retuning on new results"),
        primary_outcomes=["paired overall hip torque squared/RMS", "bilateral 4%-phase worst RMS increase",
                          "fixed-mask low-burden deceleration paired RMS/torque-squared increase"],
        secondary_outcomes=["tracking", "speed", "falls", "knee/ankle/lumbar torque squared",
                            "assist RMS and saturation", "braking work and torque direction"],
        limitations="One existing walk and one frozen wearer; two evaluation seeds are not participants or independent training seeds. Low-burden samples are temporally correlated.",
        sources={source: sha(ROOT / source) for source in SOURCES},
        assets={name: dict(path=path, sha256=sha(path)) for name, path in assets.items()},
    )
    manifest = study / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != plan:
            raise RuntimeError("Frozen manifest differs; use a new study directory")
    else:
        save(manifest, plan)
        for source in SOURCES:
            target = study / "sources" / source
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / source).read_bytes())
        for model, (checkpoint, _) in MODELS.items():
            for filename in ("resolved_configs.yaml", "resolved_configs_inference.yaml", "config.yaml"):
                source = (ROOT / checkpoint).parent / filename
                if source.exists():
                    target = study / "configs" / model / filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(source.read_bytes())
    records_path = study / "runs.json"
    records = json.loads(records_path.read_text()) if records_path.exists() else {}
    environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")

    def status(stage, **extra):
        save(study / "status.json", dict(stage=stage, pid=os.getpid(), completed=len(records),
                                        total=len(jobs), updated_utc=datetime.now(timezone.utc).isoformat(), **extra))

    try:
        for job in jobs:
            identity = job["id"]
            if identity in records:
                continue
            if any(sha(ROOT / source) != digest for source, digest in plan["sources"].items()):
                raise RuntimeError("Frozen evaluation source changed")
            checkpoint = str(ROOT / job["checkpoint"])
            if sha(checkpoint) != plan["assets"][job["model"]]["sha256"]:
                raise RuntimeError("Checkpoint changed")
            folder = study / identity
            folder.mkdir(exist_ok=True)
            command = [sys.executable, "-u", RUNNER, "--human", "tracker",
                       "--tracker-checkpoint", assets["W1"], "--motion-file", assets["motion"],
                       "--llp-checkpoint", checkpoint, "--hlp-checkpoint", hlp,
                       "--assist-scale", str(job["output_scale"]), "--motion-id", "0",
                       "--seconds", "40", "--seed", str(job["seed"]), "--tag", identity,
                       "--out", str(folder), "--fullbody-diagnostic"]
            for attempt in range(1, 4):
                logfile = folder / f"attempt_{attempt}.log"
                started = time.perf_counter()
                print(f"START {len(records)+1}/{len(jobs)} {identity}, attempt {attempt}", flush=True)
                with logfile.open("a") as stream:
                    stream.write(json.dumps(command) + "\n")
                    stream.flush()
                    child = subprocess.Popen(command, cwd=ROOT, env=environment,
                                             stdout=stream, stderr=subprocess.STDOUT)
                    while child.poll() is None:
                        elapsed = time.perf_counter() - started
                        status("evaluating", current=identity, child_pid=child.pid,
                               attempt=attempt, elapsed_s=elapsed, log=str(logfile))
                        if elapsed > 900:
                            child.terminate()
                            try:
                                child.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                child.kill()
                                child.wait()
                            raise RuntimeError(f"Evaluation timeout: {identity}")
                        time.sleep(5)
                elapsed = time.perf_counter() - started
                outputs = [p.parent for p in folder.glob("*/metrics.json")
                           if (p.parent / "fullbody.npz").exists()]
                if child.returncode == 0 and len(outputs) == 1:
                    result = outputs[0]
                    meta = json.loads((result / "meta.json").read_text())
                    assert Path(meta["llp_checkpoint"]).resolve() == Path(checkpoint).resolve()
                    assert meta["assist_scale"] == job["output_scale"]
                    assert not meta["oracle"] and not meta["swing_gate"] and not meta["mask_window"]
                    assert abs(meta["assist_scale"] * min(meta["policy_torque_limit"], meta["torque_limit"]) - job["limit_Nm"]) < 1e-8
                    records[identity] = dict(**job, path=str(result.relative_to(ROOT)), command=command,
                                             elapsed_s=elapsed, attempt=attempt, survived_s=meta["survived_s"])
                    save(records_path, records)
                    status("between evaluations", last_completed=identity)
                    print(f"DONE {identity}: {elapsed:.1f} s, fall={meta['survived_s']}", flush=True)
                    break
                if attempt == 3 or child.returncode not in (-11, -6):
                    raise RuntimeError(f"{identity}: exit {child.returncode}, outputs={outputs}; see {logfile}")
                print(f"RETRY {identity}: process exit {child.returncode}", flush=True)
        status("evaluations complete")
        print("All evaluations complete", flush=True)
    except BaseException as error:
        status("stopped with error", error=repr(error))
        raise


if __name__ == "__main__":
    main()
