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
"""Run the authorized pendulum ablations and a fixed evaluation matrix serially.

No W1 training or automatic candidate promotion. Existing complete checkpoints
are reused; interrupted runs resume their saved config only after compatibility
checks. An exclusive file lock prevents duplicate sequence runners.
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

import torch

ROOT = Path(__file__).resolve().parents[3]
VARIANTS = {"C0": [], "S1": ["--structured-disturbance-prob", "0.5"],
            "A1": ["--shared-joint-actor"], "T1": ["--human-target-period", "0.05"]}


def checkpoint_epoch(path):
    return int(torch.load(path, map_location="cpu", weights_only=False)["epoch"]) if path.exists() else -1


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="20261014_nativeff")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    p.add_argument("--wait-pid", type=int, default=None, help="An already running baseline training PID")
    p.add_argument("--out", default="results/pendulum_research_nativeff_20261014")
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    if args.epochs < 500 or args.epochs % 500:
        p.error("epochs must be a positive multiple of 500")
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    seed_suffix = f"_s{args.seed}" if args.seed else ""
    lock = (out / f"sequence{seed_suffix}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    names = {v: f"llp_pendulum20_{v}_{args.tag}{seed_suffix}" for v in VARIANTS}
    baseline = ROOT / "results" / names["C0"] / f"epoch_{args.epochs}.ckpt"
    environment = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    status_path = out / f"sequence_status{seed_suffix}.json"

    def status(stage, **extra):
        write_json(status_path, dict(stage=stage, pid=os.getpid(),
            updated_utc=datetime.now(timezone.utc).isoformat(), **extra))
        print(stage, flush=True)

    def run(command, log_path):
        with log_path.open("a") as log:
            log.write("\n" + json.dumps(command) + "\n")
            log.flush()
            subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                           stderr=subprocess.STDOUT, check=True)

    commands = {}
    for variant in args.variants:
        commands[variant] = [sys.executable, "protomotions/train_agent.py",
            "--robot-name", "hip_pendulum", "--simulator", "newton",
            "--experiment-path", "examples/experiments/assist_pendulum/mlp_research.py",
            "--experiment-name", names[variant], "--motion-file", "none",
            "--num-envs", "4096", "--batch-size", "16384", "--seed", str(args.seed),
            "--training-max-steps", str(args.epochs*4096*32), "--combined-human-limit", *VARIANTS[variant],
            "--overrides", "agent.save_epoch_checkpoint_every=500"]
    if args.check_only:
        print(json.dumps(dict(training_commands=commands, baseline=str(baseline)), indent=2))
        return
    try:
        if args.wait_pid:
            proc = Path(f"/proc/{args.wait_pid}/cmdline")
            if proc.exists():
                expected = proc.read_bytes()
                if b"protomotions/train_agent.py" not in expected or names["C0"].encode() not in expected:
                    raise ValueError("wait-pid does not identify this baseline training run")
                status("waiting for existing baseline", wait_pid=args.wait_pid)
                while proc.exists() and proc.read_bytes() == expected:
                    time.sleep(10)
            if checkpoint_epoch(baseline) != args.epochs:
                raise RuntimeError("Existing baseline exited before its planned checkpoint; inspect its log")
        manifest = {}
        source_dir = out / f"sources{seed_suffix}"
        for name in ("protomotions/agents/common/shared_joint_mlp.py",
                     "protomotions/envs/base_env/assist_env.py",
                     "protomotions/envs/control/pendulum_research_control.py",
                     "examples/experiments/assist_pendulum/mlp_research.py",
                     "examples/experiments/assist_pendulum/evaluate_research.py",
                     "examples/experiments/assist_pendulum/runtime_utils.py"):
            data = (ROOT / name).read_bytes()
            destination = source_dir / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and destination.read_bytes() != data:
                raise ValueError("Source changed since the sequence started; use a new output directory")
            destination.write_bytes(data)
            manifest[name] = hashlib.sha256(data).hexdigest()
        write_json(out / f"sequence_manifest{seed_suffix}.json", dict(args=vars(args), source_sha256=manifest))
        records = {}
        for variant in args.variants:
            directory = ROOT / "results" / names[variant]
            final = directory / f"epoch_{args.epochs}.ckpt"
            resolved_path = directory / "resolved_configs.pt"
            if resolved_path.exists():
                resolved = torch.load(resolved_path, map_location="cpu", weights_only=False)
                cc = resolved["env"].control_components["assist"]
                saved_args = json.loads((directory / "config.yaml").read_text())
                assert saved_args["seed"] == args.seed
                assert not resolved["env"].clamp_combined_injection
                assert resolved["env"].human_feedforward_inside_pd
                assert resolved["env"].assist_torque_limit == 20
                assert resolved["agent"].training_max_steps == args.epochs*4096*32
                assert cc.structured_probability == (.5 if variant == "S1" else 0.)
                assert cc.human_target_hold_steps == (5 if variant == "T1" else 1)
                assert ("SharedJointMLP" in resolved["agent"].model.actor.mu_model._target_) == (variant == "A1")
            if checkpoint_epoch(final) != args.epochs:
                status("training " + variant, command=commands[variant])
                run(commands[variant], out / f"{variant}{seed_suffix}_train.log")
            if checkpoint_epoch(final) != args.epochs:
                raise RuntimeError(f"{variant} did not reach the planned checkpoint")
            for epoch in sorted(set([args.epochs // 2, args.epochs])):
                checkpoint = directory / f"epoch_{epoch}.ckpt"
                if not checkpoint.exists():
                    continue
                for condition in ("legacy", "structured", "hold50"):
                    label = f"eval_{variant}{seed_suffix}_e{epoch}_{condition}"
                    destination = out / label
                    summary = destination / "summary.json"
                    if not summary.exists():
                        status("evaluating " + label)
                        run([sys.executable, "examples/experiments/assist_pendulum/evaluate_research.py",
                             "--base-checkpoint", str(baseline), "--checkpoint", str(checkpoint),
                             "--independent-limits", "--combined-human-limit",
                             "--condition", condition, "--out", str(destination)],
                            out / f"{label}.log")
                    report = json.loads(summary.read_text())
                    assert report["protocol"]["independent_limits"]
                    assert report["protocol"]["combined_human_limit"]
                    assert report["injection_error_max_Nm"] == 0
                    records[label] = report
                    write_json(out / f"comparison{seed_suffix}.json", records)
        status("initial matrix complete; scientific review and second-seed confirmation remain",
               comparison=str(out / f"comparison{seed_suffix}.json"))
    except BaseException as exc:
        status("stopped", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
