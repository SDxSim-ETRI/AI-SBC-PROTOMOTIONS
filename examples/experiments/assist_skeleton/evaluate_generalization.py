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
"""L1 vs R0 on walking conditions never used for R0 (2026-10-21), frozen W1/H1, no training.

Stage 1 screens every candidate with the L1 20 N m seed-42 run (that run is also the
mask reference of the main matrix). Stage 2 evaluates conditions that pass the fixed
rule with L1 10/15/20 and R0 15/20 N m on evaluation seeds 42 and 7. Stored
existing-walk records (2026-10-16 L1, 2026-10-20 R0) are reused, never re-run.
"""

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

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUNNER = "examples/experiments/assist_skeleton/run_system.py"
OLD = ROOT / "results/llp_output_scale_20261016"
RELATIVE = ROOT / "results/llp_relative_20261020"
MODELS = {
    "L1": "checkpoints/L1_llp_pendulum/last.ckpt",
    "R0": "results/llp_relative_R0_s0_20261020/epoch_6000.ckpt",
}
NATIVE_LIMIT = 40.0
MATRIX = [("L1", 20), ("L1", 15), ("L1", 10), ("R0", 20), ("R0", 15)]
GAIT = "data/motion_for_trackers/skeleton_torque_suit_gait.pt"
MOTIONS14 = "data/motion_for_trackers/skeleton_torque_suit_motions14.pt"
# Priority order: W1 training motion families first, then movements W1 never trained on.
CONDITIONS = {
    "run": dict(motion_file=GAIT, motion_id=1, start=0.0,
                label="달리기 2.59 m/s (W1 학습 동작, R0 미평가)"),
    "walk_seg400": dict(motion_file=GAIT, motion_id=0, start=400.0,
                        label="기존 걷기 기록의 400–440 s 구간 (조건 변화 없음, 구간 차이 기준)"),
    "stepinplace": dict(motion_file=MOTIONS14, motion_id=7, start=0.0,
                        label="제자리 걷기 stepinplace1 (W1 미학습 동작)"),
    "highknee": dict(motion_file=MOTIONS14, motion_id=0, start=0.0,
                     label="무릎 높이 들기 highknee (W1 미학습 동작)"),
}
# Retimed walks (walk_tempo_variants.pt) keep poses and stride length and only rescale time,
# so gravity/inertia no longer match a real slower or faster walk; excluded on 2026-10-21.
EXCLUDED = {"walk_tempo085/115": "Synthetic time rescaling of the same poses; whole-body dynamics are not a real speed change"}
SCREEN_RULE = dict(no_fall_s=40, max_unassisted_body_error_m=0.10, min_phase_cycles=10,
                   note="Applied to the L1 20 N m seed-42 screening run before any R0 evaluation; "
                        "existing walk unassisted body error is 0.060 m")
SOURCES = [
    "examples/experiments/assist_skeleton/evaluate_generalization.py",
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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def job(condition, model, limit, seed):
    return dict(id=f"{condition}_{model}_{limit}Nm_seed{seed}", condition=condition, model=model,
                limit_Nm=limit, seed=seed, checkpoint=MODELS[model], output_scale=limit / NATIVE_LIMIT)


def screen(record):
    """Fixed W1 feasibility rule on the screening run."""
    directory = ROOT / record["path"]
    meta = json.loads((directory / "meta.json").read_text())
    with np.load(directory / "traces.npz") as file:
        traces = dict(file)
    keep = traces["body_err_t"] >= meta["warmup"] + .5
    body_error = traces["body_err"][keep].mean(axis=0)
    phase = traces["gait_phase"]
    cycles = int(np.sum(np.diff(phase[:, 0]) < -.5)) if np.isfinite(phase).all() else 0
    checks = dict(no_fall=meta["survived_s"] is None,
                  unassisted_body_error_m=float(body_error[0]), assisted_body_error_m=float(body_error[1]),
                  body_error_ok=bool(body_error[0] <= SCREEN_RULE["max_unassisted_body_error_m"]),
                  phase_cycles_right=cycles, phase_ok=cycles >= SCREEN_RULE["min_phase_cycles"])
    checks["passed"] = checks["no_fall"] and checks["body_error_ok"] and checks["phase_ok"]
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="results/llp_r0_generalization_20261021")
    args = parser.parse_args()
    study = (ROOT / args.study).resolve()
    study.mkdir(parents=True, exist_ok=True)
    lock = (study / "worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = json.loads((OLD / "manifest.json").read_text())
    hlp = previous["assets"]["H1"]["path"]
    assets = {model: str(ROOT / path) for model, path in MODELS.items()}
    assets.update(W1=str(ROOT / "checkpoints/W1_wearer_human/last.ckpt"), H1=hlp,
                  **{Path(f).stem: str(ROOT / f) for f in (GAIT, MOTIONS14)})
    plan = dict(
        purpose="Does R0's existing-walk improvement over L1 hold on walking/movement conditions never used for R0?",
        conditions=CONDITIONS, excluded_conditions=EXCLUDED, screen_rule=SCREEN_RULE,
        screening_job="L1 20 N m, evaluation seed 42 (kept as the main-matrix mask reference)",
        matrix=[dict(model=m, limit_Nm=lim) for m, lim in MATRIX], evaluation_seeds=[42, 7],
        seconds=40, analysis_start_s=1.5,
        protocol="Frozen W1/H1, wearer calibration, 400 Hz physics, 100 Hz assistance, no gates/filters; "
                 "identical to results/llp_output_scale_20261016 except motion file/id/start",
        scale_contract="Multiply the bounded 40 N m policy torque by limit/40; L1 and R0 both use 40 N m actions",
        masks={**previous["masks"], "source": "Unassisted A of the L1 20 N m run of the SAME condition and evaluation seed"},
        reused=dict(existing_walk_L1=str(OLD.relative_to(ROOT)), existing_walk_R0=str(RELATIVE.relative_to(ROOT) / "walking"),
                    reason="Evaluation sources are byte-identical to the 2026-10-16 manifest"),
        not_reused="2026-10-13/14 L1 running records lack full-body diagnostics and source fingerprints",
        limitations="One frozen wearer (W1) and one source walk/run recording; no physically valid walking-speed "
                    "change is available for W1; two evaluation seeds are not participants or training seeds. "
                    "R0 has one training seed.",
        sources={source: sha(ROOT / source) for source in SOURCES},
        assets={name: dict(path=path, sha256=sha(path)) for name, path in assets.items()},
    )
    shared = set(plan["sources"]) & set(previous["sources"])
    assert len(shared) == len(SOURCES) - 1 and all(plan["sources"][k] == previous["sources"][k] for k in shared), \
        "Shared evaluation sources differ from the 2026-10-16 study"
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
        save(study / "initial_git_status.json",
             subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines())
        (study / "initial_nvidia_smi.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    records_path = study / "runs.json"
    records = json.loads(records_path.read_text()) if records_path.exists() else {}
    failures_path = study / "runtime_failures.json"
    failures = json.loads(failures_path.read_text()) if failures_path.exists() else []
    screening_path = study / "screening.json"
    screening = json.loads(screening_path.read_text()) if screening_path.exists() else {}
    environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONFAULTHANDLER="1")

    def status(stage, **extra):
        save(study / "status.json", dict(stage=stage, pid=os.getpid(), completed=len(records),
                                        updated_utc=datetime.now(timezone.utc).isoformat(), **extra))

    def execute(spec):
        identity = spec["id"]
        if identity in records:
            return records[identity]
        if any(sha(ROOT / source) != digest for source, digest in plan["sources"].items()):
            raise RuntimeError("Frozen evaluation source changed")
        if sha(assets[spec["model"]]) != plan["assets"][spec["model"]]["sha256"]:
            raise RuntimeError("Checkpoint changed")
        condition = CONDITIONS[spec["condition"]]
        folder = study / "runs" / identity
        folder.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-u", RUNNER, "--human", "tracker",
                   "--tracker-checkpoint", assets["W1"], "--motion-file", str(ROOT / condition["motion_file"]),
                   "--llp-checkpoint", assets[spec["model"]], "--hlp-checkpoint", hlp,
                   "--assist-scale", str(spec["output_scale"]), "--motion-id", str(condition["motion_id"]),
                   "--start-time", str(condition["start"]), "--seconds", "40", "--seed", str(spec["seed"]),
                   "--tag", identity, "--out", str(folder), "--fullbody-diagnostic"]
        for attempt in range(1, 4):
            logfile = folder / f"attempt_{attempt}.log"
            started = time.perf_counter()
            print(f"START {identity}, attempt {attempt}", flush=True)
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
            outputs = [p.parent for p in folder.glob("*/metrics.json") if (p.parent / "fullbody.npz").exists()]
            if child.returncode == 0 and len(outputs) == 1:
                result = outputs[0]
                meta = json.loads((result / "meta.json").read_text())
                assert Path(meta["llp_checkpoint"]).resolve() == Path(assets[spec["model"]]).resolve()
                assert meta["motion_id"] == condition["motion_id"] and meta["start_time"] == condition["start"]
                assert Path(meta["motion_file"]).resolve() == (ROOT / condition["motion_file"]).resolve()
                assert meta["assist_scale"] == spec["output_scale"]
                assert not meta["oracle"] and not meta["swing_gate"] and not meta["mask_window"]
                assert abs(meta["assist_scale"] * min(meta["policy_torque_limit"], meta["torque_limit"]) - spec["limit_Nm"]) < 1e-8
                records[identity] = dict(**spec, path=str(result.relative_to(ROOT)), command=command,
                                         elapsed_s=elapsed, attempt=attempt, survived_s=meta["survived_s"])
                save(records_path, records)
                print(f"DONE {identity}: {elapsed:.1f} s, fall={meta['survived_s']}", flush=True)
                return records[identity]
            failures.append(dict(id=identity, attempt=attempt, returncode=child.returncode,
                                 elapsed_s=elapsed, log=str(logfile.relative_to(ROOT))))
            save(failures_path, failures)
            if attempt == 3 or child.returncode not in (-11, -6):
                raise RuntimeError(f"{identity}: exit {child.returncode}, outputs={outputs}; see {logfile}")
            print(f"RETRY {identity}: process exit {child.returncode}", flush=True)
        raise AssertionError("Unreachable")

    try:
        for condition in CONDITIONS:
            if condition not in screening:
                screening[condition] = screen(execute(job(condition, "L1", 20, 42)))
                save(screening_path, screening)
        selected = [c for c in CONDITIONS if screening[c]["passed"]]
        status("screening complete", selected=selected)
        for condition in selected:
            for seed in (42, 7):
                for model, limit in MATRIX:
                    execute(job(condition, model, limit, seed))
        status("evaluations complete", selected=selected)
        print("COMPLETE", flush=True)
    except BaseException as error:
        status("stopped with error", error=repr(error))
        raise


if __name__ == "__main__":
    main()
