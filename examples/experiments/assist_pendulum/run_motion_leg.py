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
"""LLP 20 N·m 다리별·동작 의도 학습 실행기 (2026-10-21).

다리마다 한 줄(lane)로, 네 조건을 사용자가 정한 순서로 학습한다:
A·Δ 없음 → A·Δ 모두 → A만 → Δ만. 오른쪽은 학습 seed 0, 왼쪽은 seed 1로 고정해 같은 다리 안의
조건 비교는 초기값·환경 난수가 같은 짝 비교가 된다. 모델은 마지막 epoch로 고정한다.
시뮬레이터가 SIGSEGV/SIGABRT로 죽으면 같은 실험 이름으로 다시 실행해 last.ckpt에서 이어 학습한다.

    python examples/experiments/assist_pendulum/run_motion_leg.py prepare
    python examples/experiments/assist_pendulum/run_motion_leg.py train --lane right
    python examples/experiments/assist_pendulum/run_motion_leg.py train --lane left
"""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
PYTHON = sys.executable
STUDY = ROOT / "results/llp20_motion_20261021"
RECIPE = "examples/experiments/assist_pendulum/mlp_motion_leg.py"
EPOCHS = 6000
NUM_ENVS, BATCH, HORIZON = 4096, 16384, 32
# (이름, 설명, 인자) — 학습 순서 그대로
CONDITIONS = (
    ("A0D0", "A·Δ 없음", ["--assist-effort-weight=0", "--action-rate-weight=0"]),
    ("A1D1", "A·Δ 모두 (R0 값)", ["--assist-effort-weight=-2e-5", "--action-rate-weight=-0.005"]),
    ("A1D0", "A만", ["--assist-effort-weight=-2e-5", "--action-rate-weight=0"]),
    ("A0D1", "Δ만", ["--assist-effort-weight=0", "--action-rate-weight=-0.005"]),
)
LANES = {"right": 0, "left": 1}  # 다리 → 학습 seed
TRAINING_SOURCES = (
    RECIPE,
    "examples/experiments/assist_pendulum/run_motion_leg.py",
    "examples/experiments/assist_pendulum/mlp_relative.py",
    "examples/experiments/assist_pendulum/mlp_v4.py",
    "examples/experiments/assist_pendulum/mlp_v3.py",
    "examples/experiments/assist_pendulum/mlp.py",
    "protomotions/train_agent.py",
    "protomotions/envs/base_env/assist_env.py",
    "protomotions/envs/control/assist_target_control.py",
    "protomotions/envs/control/assist_motion_control.py",
    "protomotions/envs/control/assist_disturbance_control.py",
    "protomotions/envs/control/motion_intent.py",
    "protomotions/envs/rewards/assist.py",
    "protomotions/envs/terminations/assist.py",
    "protomotions/envs/obs/assist_obs.py",
    "protomotions/envs/context_views.py",
    "protomotions/agents/common/shared_joint_mlp.py",
    "protomotions/agents/common/mlp.py",
    "protomotions/agents/ppo/agent.py",
    "protomotions/agents/ppo/model.py",
    "protomotions/agents/base_agent/agent.py",
    "protomotions/robot_configs/hip_pendulum.py",
    "protomotions/tests/test_llp_motion_leg.py",
)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_name(condition, leg):
    return f"llp20_motion_{condition}_{leg}_s{LANES[leg]}_20261021"


def train_command(condition, leg, extra):
    return [PYTHON, "-u", "protomotions/train_agent.py", "--robot-name", "hip_pendulum",
            "--simulator", "newton", "--experiment-path", RECIPE,
            "--experiment-name", run_name(condition, leg), "--motion-file", "none",
            "--num-envs", str(NUM_ENVS), "--batch-size", str(BATCH), "--seed", str(LANES[leg]),
            "--training-max-steps", str(EPOCHS * NUM_ENVS * HORIZON), "--leg", leg, *extra]


def prepare():
    STUDY.mkdir(parents=True, exist_ok=True)
    manifest = STUDY / "manifest.json"
    sources = {name: sha(ROOT / name) for name in TRAINING_SOURCES}
    if manifest.exists():
        old = json.loads(manifest.read_text())
        assert old["training_sources"] == sources, "training sources changed after prepare"
        print("manifest exists, unchanged")
        return
    sys.path.insert(0, str(ROOT / "examples/experiments/assist_pendulum"))
    import mlp_motion_leg as recipe

    plan = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        purpose=("LLP 20 N·m (20·tanh) pendulum training with per-leg models and natural-motion "
                 "intents covering running-level hip speeds; 2×2 comparison of the assist-torque "
                 "squared cost A and the command-change cost Δ"),
        requested_order=[c[0] for c in CONDITIONS],
        conditions={c[0]: dict(label=c[1], arguments=c[2]) for c in CONDITIONS},
        lanes={leg: dict(seed=seed, runs=[run_name(c[0], leg) for c in CONDITIONS])
               for leg, seed in LANES.items()},
        epochs=EPOCHS, num_envs=NUM_ENVS, batch_size=BATCH,
        transitions_per_model=EPOCHS * NUM_ENVS * HORIZON,
        recipe=dict(
            base="R0 (mlp_relative.py, paired weight 0): corrected torque path, mlp_v4 plant and disturbance randomization",
            action="20*tanh(u), assist limit 20 N·m, deployment scale 1",
            per_leg="separate run per leg; only that joint assisted; rewards/terminations count only that joint; actor reads only that joint",
            intent_pools=[list(p) for p in recipe.INTENT_POOLS],
            motion_groups=[list(g) for g in recipe.MOTION_GROUPS],
            motion_accel_frac=recipe.MOTION_ACCEL_FRAC,
            motion_data=dict(zarr=recipe.SEED_ZARR, metadata=recipe.SEED_METADATA,
                             split="train (H1 training split; H1 validation clips stay unseen)"),
        ),
        selection="Fixed final epoch for every run; no checkpoint chosen by evaluation outcomes",
        training_sources=sources,
    )
    save(manifest, plan)
    for name in TRAINING_SOURCES:
        dest = STUDY / "sources" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    save(STUDY / "initial_git_status.json",
         subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines())
    (STUDY / "initial_nvidia_smi.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    print("prepared", manifest)


def train_lane(leg):
    manifest = json.loads((STUDY / "manifest.json").read_text())
    sources = manifest["training_sources"]
    lock = (STUDY / f"worker_{leg}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    records_path = STUDY / f"execution_{leg}.json"
    records = json.loads(records_path.read_text()) if records_path.exists() else {}
    errors_path = STUDY / f"runtime_failures_{leg}.json"
    errors = json.loads(errors_path.read_text()) if errors_path.exists() else []
    environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONFAULTHANDLER="1")
    status_path = STUDY / f"status_{leg}.json"

    def status(stage, **extra):
        save(status_path, dict(stage=stage, lane=leg, pid=os.getpid(),
                               updated_utc=datetime.now(timezone.utc).isoformat(),
                               completed=sorted(records), **extra))

    for condition, _, extra in CONDITIONS:
        label = f"{condition}_{leg}"
        name = run_name(condition, leg)
        checkpoint = ROOT / "results" / name / f"epoch_{EPOCHS}.ckpt"
        if label not in records:
            for path, digest in sources.items():
                assert sha(ROOT / path) == digest, f"frozen training source changed: {path}"
            command = train_command(condition, leg, extra)
            for attempt in range(1, 4):
                log = STUDY / "logs" / f"{label}_attempt{attempt}.log"
                log.parent.mkdir(exist_ok=True)
                started = time.time()
                with log.open("a") as stream:
                    stream.write(json.dumps(command) + "\n")
                    stream.flush()
                    child = subprocess.Popen(command, cwd=ROOT, env=environment,
                                             stdout=stream, stderr=subprocess.STDOUT)
                    while child.poll() is None:
                        with log.open("rb") as inp:
                            inp.seek(max(0, log.stat().st_size - 4000))
                            epochs = re.findall(r"Epoch (\d+)", inp.read().decode(errors="replace"))
                        status("training", current=label, attempt=attempt, child_pid=child.pid,
                               job_elapsed_s=time.time() - started,
                               observed_epoch=int(epochs[-1]) if epochs else None)
                        time.sleep(15)
                outcome = dict(command=command, returncode=child.returncode, attempt=attempt,
                               started_unix=started, elapsed_s=time.time() - started,
                               log=str(log.relative_to(ROOT)))
                if child.returncode == 0:
                    break
                errors.append(dict(label=label, **outcome))
                save(errors_path, errors)
                if child.returncode not in (-11, -6) or attempt == 3:
                    status("stopped with error", current=label, returncode=child.returncode)
                    raise RuntimeError(f"{label} failed: {child.returncode}")
                print(f"RETRY {label}: {child.returncode}", flush=True)
            import torch

            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            assert ckpt["epoch"] == EPOCHS, ckpt["epoch"]
            assert all(not torch.is_tensor(v) or torch.isfinite(v).all() for v in ckpt["model"].values())
            env_cfg = torch.load(checkpoint.parent / "resolved_configs.pt", map_location="cpu",
                                 weights_only=False)["env"]
            joint = LANES_JOINT[leg]
            assert tuple(env_cfg.assist_active_joints) == (joint,)
            assert float(env_cfg.assist_torque_limit) == 20.0
            assert float(env_cfg.action_config["torque_scale"]) == 20.0
            assert env_cfg.clamp_combined_injection is False and env_cfg.human_feedforward_inside_pd is True
            weights = {k: float(env_cfg.reward_components[k].static_params["weight"])
                       for k in ("assist_effort", "action_rate")}
            expected = {"assist_effort": -2e-5 if condition[1] == "1" else 0.0,
                        "action_rate": -0.005 if condition[3] == "1" else 0.0}
            assert weights == expected, (weights, expected)
            records[label] = dict(**outcome, run=name, checkpoint=str(checkpoint.relative_to(ROOT)),
                                  checkpoint_sha256=sha(checkpoint), reward_weights=weights)
            save(records_path, records)
            print(f"DONE {label} {outcome['elapsed_s']:.0f}s", flush=True)
    status("lane complete")
    print("LANE COMPLETE", leg, flush=True)


LANES_JOINT = {"right": 0, "left": 1}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    t = sub.add_parser("train")
    t.add_argument("--lane", choices=sorted(LANES), required=True)
    args = p.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        train_lane(args.lane)


if __name__ == "__main__":
    main()
