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
"""Validate L12, train only to 2k, then evaluate walking before further work.

No automatic next hypothesis, no running evaluation, and no automatic 6k extension.
The service exits after the walking screen (pass or fail) for causal review.
"""
import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from run_wide_sequence import ROOT, checkpoint_epoch, run, save_json, summarize
from report_torque_html import rank_candidates
from phase_screen import screen_records

OUT = ROOT / "results/llp_L12_paired_walk_sequence"
NAME = "llp_L12_paired_walk"
SOURCE = ROOT / "results/llp_L11_opposition_sequence/comparison.json"
WEIGHT = -0.021958557930324075


def training_command(name, num_envs, batch_size, epochs):
    return [sys.executable, "protomotions/train_agent.py", "--robot-name", "hip_pendulum",
            "--simulator", "newton", "--experiment-path",
            "examples/experiments/assist_pendulum/mlp_paired_tracking.py",
            "--experiment-name", name, "--motion-file", "none", "--num-envs", num_envs,
            "--batch-size", batch_size, "--training-max-steps", epochs*num_envs*32,
            "--seed", 0, "--human-cost", "both", "--opposition-weight", WEIGHT,
            "--opposition-torque-floor", 10]


def status(stage):
    save_json(OUT/"status.json", dict(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat()))
    print(stage, flush=True)


def snapshot():
    paths = [
        "protomotions/envs/base_env/paired_assist_env.py",
        "protomotions/envs/base_env/assist_env.py",
        "protomotions/envs/context_views.py", "protomotions/envs/rewards/assist.py",
        "protomotions/envs/control/assist_target_control.py",
        "protomotions/envs/control/assist_motion_control.py",
        "protomotions/envs/control/assist_disturbance_control.py",
        "protomotions/envs/action/action_functions.py",
        "protomotions/assist/controller.py", "protomotions/assist/llp_policy.py",
        "protomotions/maniflow/hybrid_control.py",
        "examples/experiments/assist_skeleton/run_system.py",
        "examples/experiments/assist_skeleton/phase_screen.py",
        "examples/experiments/assist_skeleton/report_torque_html.py",
        "examples/experiments/assist_skeleton/analysis.py",
        "examples/experiments/assist_pendulum/run_paired_walk_sequence.py",
        "examples/experiments/assist_pendulum/verify_paired_tracking.py",
    ]
    paths += [str(p.relative_to(ROOT)) for p in
              (ROOT/"examples/experiments/assist_pendulum").glob("mlp*.py")]
    hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    saved = OUT/"source_sha256.json"
    if saved.exists():
        if hashes != json.loads(saved.read_text()):
            raise RuntimeError("Source changed since sequence start; review before resuming")
    else:
        for p in paths:
            dest = OUT/"source_snapshot"/p
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/p, dest)
        save_json(saved, hashes)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    hlp = Path.home()/"Projects/ManiFlow_Policy/ManiFlow/data/outputs/hip_flexion_seed-maniflow_lowdim_policy_walking-flexion100-h15-run01_seed42/checkpoints/epoch=0006-val_loss=0.000762.ckpt"
    paths = [SOURCE, hlp, ROOT/"checkpoints/W1_wearer_human/last.ckpt",
             ROOT/"results/llp_L11_opposition/epoch_2000.ckpt",
             ROOT/"data/motion_for_trackers/skeleton_torque_suit_gait.pt"]
    ready = dict(cuda_available=torch.cuda.is_available(),
                 missing_files=[str(p) for p in paths if not p.is_file()],
                 train_command=[str(x) for x in training_command(NAME, 4096, 16384, 2000)])
    save_json(OUT/"readiness.json", ready)
    if not ready["cuda_available"] or ready["missing_files"]:
        status("blocked before simulation: CUDA unavailable or required files missing")
        print(json.dumps(ready, indent=2))
        raise SystemExit(2)
    if args.check_only:
        print(json.dumps(ready, indent=2))
        return
    snapshot()
    save_json(OUT/"protocol.json", dict(
        reference="L11", hypothesis="paired non-degradation tracking only",
        epochs=2000, num_envs=4096, training_seed=0, evaluation_seed=42,
        evaluation_motion=0, seconds=40, assist_scale=1, final_limit_Nm=100,
        next_step="Stop for analysis after walking; no automatic 6k/running/next hypothesis",
        caveat="Reference-only terminations truncate B; common-survival selection bias is logged",
        screen="Existing per-motion L1 screen unchanged; walking success is not generalization"))
    for n, steps in ((16, 350), (4096, 350)):
        dest = OUT/f"pair_verify_{n}.json"
        if not dest.exists() or not json.loads(dest.read_text()).get("pass", False):
            status(f"validating paired physics, {n} environments")
            run([sys.executable, "examples/experiments/assist_pendulum/verify_paired_tracking.py",
                 "--num-envs", n, "--steps", steps, "--out", dest], OUT/f"pair_verify_{n}.log")
        if not json.loads(dest.read_text())["pass"]:
            raise RuntimeError("Paired physics verification failed")
    small = ROOT/"results/llp_L12_paired_ppo_check/last.ckpt"
    if not small.exists():
        status("validating four real PPO epochs")
        run(training_command("llp_L12_paired_ppo_check", 64, 256, 4), OUT/"ppo_check.log")
    if checkpoint_epoch(small) != 4:
        raise RuntimeError("Small PPO check did not complete four epochs")
    last = ROOT/"results"/NAME/"last.ckpt"
    if not last.exists() or checkpoint_epoch(last) < 2000:
        status("training to epoch 2000")
        run(training_command(NAME, 4096, 16384, 2000), OUT/"train.log")
    if checkpoint_epoch(last) != 2000:
        raise RuntimeError("Unexpected final epoch")
    records = json.loads(SOURCE.read_text())
    key = "eval_llp_L12_paired_walk_e2000_m0"
    done = OUT/f"{key}.json"
    if not done.exists():
        status("evaluating walking only")
        tag = "L12_paired_e2000_m0"
        eval_root = OUT/"evaluations"
        completed = sorted(p for p in eval_root.glob(f"*_{tag}")
                           if (p/"meta.json").is_file() and (p/"traces.npz").is_file())
        if not completed:
            run([sys.executable, "examples/experiments/assist_skeleton/run_system.py",
                 "--human", "tracker", "--tracker-checkpoint", paths[2],
                 "--llp-checkpoint", last, "--hlp-checkpoint", hlp,
                 "--motion-file", paths[4], "--motion-id", 0, "--seconds", 40,
                 "--llp-dt", .01, "--calibration", "wearer", "--assist-scale", 1,
                 "--torque-limit", 100, "--seed", 42, "--out", eval_root, "--tag", tag],
                OUT/"walk_eval.log")
            completed = sorted(p for p in eval_root.glob(f"*_{tag}")
                               if (p/"meta.json").is_file() and (p/"traces.npz").is_file())
        if not completed:
            raise RuntimeError("No completed walking output")
        save_json(done, summarize(completed[-1]))
    records[key] = json.loads(done.read_text())
    save_json(OUT/"comparison.json", records)
    _, diagnostics = rank_candidates(records)
    screen = screen_records(records, diagnostics["evaluations"])
    save_json(OUT/"phase_screen.json", screen)
    save_json(OUT/"walking_result.json", dict(
        metrics=records[key], diagnostic=diagnostics["evaluations"][key],
        walking_screen=screen["evaluations"][key],
        interpretation="Single-seed development result only; stop here for analysis"))
    status("complete: walking result ready for analysis; no further training scheduled")


if __name__ == "__main__":
    main()
