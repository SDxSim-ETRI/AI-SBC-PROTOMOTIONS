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
"""Monitor a finite pendulum study through evaluation and independent confirmation.

The plan names already launched training jobs. Checkpoints are evaluated as they
arrive. Real-H1 performance selects candidates, with stability, tracking and work
guards. A qualifying candidate gets a second training seed, a separate test set,
then frozen-W1 transfer. This script never trains W1 or promotes deployed weights.
Every stage and rejection is recorded, including unsuccessful studies.
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
import torch

ROOT = Path(__file__).resolve().parents[3]


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


def improves_reference(report):
    """A failed or merely slowed-down plant cannot qualify as an improvement."""
    m = report["metrics"]["all"]
    return (report["failures"]["assisted"] == 0
            and report["injection_error_max_Nm"] == 0
            and m["assisted_tau_sq_Nm2"] < m["reference_tau_sq_Nm2"]
            and m["assisted_tracking_rmse_deg"] <= m["reference_tracking_rmse_deg"]
            and m["assisted_abs_work_W"] <= m["reference_abs_work_W"])


def improves_baseline(report, baseline):
    a, b = report["metrics"]["all"], baseline["metrics"]["all"]
    return (improves_reference(report)
            and a["assisted_tau_sq_Nm2"] < b["assisted_tau_sq_Nm2"]
            and a["assisted_tracking_rmse_deg"] <= b["assisted_tracking_rmse_deg"]
            and a["assisted_abs_work_W"] <= b["assisted_abs_work_W"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "followup.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    environment = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    records = {}
    state = dict(stage="preflight", pid=os.getpid(), plan=plan, completed_evaluations=0)

    def status(stage, **extra):
        state.update(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
        write_json(out / "followup_status.json", state)
        lines = ["# 진자 후속 실험 — 현재 상태", "", f"상태: {stage}",
                 f"갱신: {state['updated_utc']} (UTC)", "",
                 "완료된 평가만 아래에 표시한다. W1은 고정 평가용이며 학습하지 않는다.", "",
                 "| 평가 | 사람 τ² [N·m²/관절] | 추종 RMSE [°] | 보조 실패 |",
                 "|---|---:|---:|---:|"]
        for label, report in records.items():
            m = report["metrics"]["all"]
            lines.append(f"| {label} | {m['assisted_tau_sq_Nm2']:.1f} | "
                         f"{m['assisted_tracking_rmse_deg']:.3f} | "
                         f"{report['failures']['assisted']}/{report['total_episodes']} |")
        lines += ["", "부담·추종은 공통 생존 구간의 조건부 평균이다. 실패가 있는 후보는 선정하지 않는다.",
                  "실제 H1 조건에서 L1보다 부담·추종·일률이 개선된 후보만 두 번째 학습 seed와 미사용 평가 조건으로 확인한다.",
                  "이는 이 유한 실험 계획의 판정이며 전역 최적 성능이나 배포 성공을 뜻하지 않는다.",
                  "", "원본: `followup_comparison.json`, `candidate_review.json`, `transfer_review.json`."]
        (out / "followup_report.md").write_text("\n".join(lines) + "\n")
        print(stage, flush=True)

    def run(command, label):
        for name, expected in hashes.items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, (
                "Study source changed during execution; start a new followup directory")
        with (out / f"{label}.log").open("a") as log:
            log.write("\n" + json.dumps(command) + "\n")
            log.flush()
            subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                           stderr=subprocess.STDOUT, check=True)

    def evaluate(checkpoint, label, condition="legacy", real_h1=False, scale=1., test=False):
        destination = out / label
        summary = destination / "summary.json"
        command = [sys.executable, "examples/experiments/assist_pendulum/evaluate_research.py",
                   "--base-checkpoint", plan["base_checkpoint"], "--checkpoint", str(checkpoint),
                   "--independent-limits", "--combined-human-limit", "--condition", condition,
                   "--output-scale", str(scale), "--out", str(destination)]
        if real_h1:
            command += ["--hlp-checkpoint", plan["hlp_checkpoint"]]
        if test:
            command += ["--plant-seed", "63031", "--trajectory-seed", "52021", "--hlp-seed", "991"]
        if not summary.exists():
            status("evaluating " + label)
            run(command, label)
        report = json.loads(summary.read_text())
        protocol = report["protocol"]
        assert Path(protocol["checkpoint"]).resolve() == Path(checkpoint).resolve()
        assert protocol["condition"] == condition and protocol["output_scale"] == scale
        assert bool(protocol["hlp_checkpoint"]) == real_h1
        assert protocol["plant_seed"] == (63031 if test else 34011)
        assert protocol["trajectory_seed"] == (52021 if test else 41011)
        assert protocol["independent_limits"] and protocol["combined_human_limit"]
        records[label] = report
        write_json(out / "followup_comparison.json", records)
        state["completed_evaluations"] = len(records)
        return report

    def train_seed(job, epochs):
        name = f"llp_pendulum20_{job['label']}_followup_s1_e{epochs}"
        num_envs = job.get("num_envs", 4096)
        directory = ROOT / "results" / name
        checkpoint = directory / f"epoch_{epochs}.ckpt"
        command = [sys.executable, "-u", "protomotions/train_agent.py",
                   "--robot-name", "hip_pendulum", "--simulator", "newton",
                   "--experiment-path", job.get("experiment_path", "examples/experiments/assist_pendulum/mlp_research.py"),
                   "--experiment-name", name, "--motion-file", "none",
                   "--num-envs", str(num_envs), "--batch-size", str(job.get("batch_size", 16384)), "--seed", "1",
                   "--training-max-steps", str(epochs * num_envs * 32),
                   "--combined-human-limit", *job["variant_args"],
                   "--overrides", "agent.save_epoch_checkpoint_every=500", *job["overrides"]]
        if (directory / "resolved_configs.pt").exists():
            cfg = torch.load(directory / "resolved_configs.pt", map_location="cpu", weights_only=False)
            saved_args = json.loads((directory / "config.yaml").read_text())
            assert saved_args["seed"] == 1
            assert cfg["agent"].training_max_steps == epochs * num_envs * 32
            assert cfg["env"].human_feedforward_inside_pd and not cfg["env"].clamp_combined_injection
            assert ("SharedJointMLP" in cfg["agent"].model.actor.mu_model._target_) == (
                "--shared-joint-actor" in job["variant_args"])
        if not checkpoint.exists():
            status("independent training seed", command=command)
            run(command, name)
        assert torch.load(checkpoint, map_location="cpu", weights_only=False)["epoch"] == epochs
        return checkpoint

    # Refuse to reuse an output directory with different source or study plans.
    hashes = {}
    for name in ("protomotions/agents/common/shared_joint_mlp.py",
                 "protomotions/envs/base_env/assist_env.py",
                 "protomotions/envs/control/pendulum_research_control.py",
                 "examples/experiments/assist_pendulum/mlp_research.py",
                 "examples/experiments/assist_pendulum/evaluate_research.py",
                 "examples/experiments/assist_pendulum/runtime_utils.py",
                 "examples/experiments/assist_pendulum/run_research_followup.py",
                 "examples/experiments/assist_pendulum/mlp_pendulum_adapt.py",
                 "protomotions/agents/ppo/pendulum_adapt.py",
                 "protomotions/envs/control/pendulum_h1_control.py",
                 "examples/experiments/assist_skeleton/run_system.py",
                 "protomotions/assist/controller.py"):
        data = (ROOT / name).read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
        target = out / "followup_sources" / name
        if target.exists():
            assert target.read_bytes() == data, "Source changed; use a new followup directory"
        if not args.check_only:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    for job in plan["jobs"]:
        cfg = torch.load(ROOT / job["directory"] / "resolved_configs.pt",
                         map_location="cpu", weights_only=False)
        assert cfg["env"].human_feedforward_inside_pd
        assert not cfg["env"].clamp_combined_injection
        assert cfg["agent"].training_max_steps == max(job["epochs"]) * job.get("num_envs", 4096) * 32
        assert Path(job["launch"]).exists()
    if args.check_only:
        print(json.dumps(dict(validated=True, jobs=plan["jobs"], source_sha256=hashes), indent=2))
        return
    manifest = dict(plan=plan, source_sha256=hashes)
    manifest_path = out / "followup_manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, "Plan changed; use a new directory"
    write_json(manifest_path, manifest)
    try:
        baseline = evaluate("checkpoints/L1_llp_pendulum/last.ckpt", "eval_L1_H1_legacy",
                            real_h1=True, scale=.5)
        pending = [(job, epoch) for job in plan["jobs"] for epoch in job["epochs"]]
        candidates = []
        while pending:
            ready = False
            for job, epoch in list(pending):
                checkpoint = ROOT / job["directory"] / f"epoch_{epoch}.ckpt"
                launch = json.loads(Path(job["launch"]).read_text())
                proc = Path(f"/proc/{launch['pid']}/cmdline")
                alive = proc.exists() and job["directory"].split("/")[-1].encode() in proc.read_bytes()
                if not checkpoint.exists():
                    if not alive:
                        raise RuntimeError(f"Training exited without {checkpoint}")
                    continue
                # Checkpoints may be observed while torch.save is still writing.
                try:
                    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                except (EOFError, RuntimeError):
                    if not alive:
                        raise
                    continue
                assert saved["epoch"] == epoch
                label = f"eval_{job['label']}_e{epoch}"
                reports = [evaluate(checkpoint, f"{label}_{condition}", condition)
                           for condition in ("legacy", "structured", "hold50")]
                h1 = evaluate(checkpoint, f"{label}_H1_legacy", real_h1=True)
                accepted = all(improves_reference(r) for r in reports) and improves_baseline(h1, baseline)
                candidates.append(dict(job=job, epoch=epoch, checkpoint=str(checkpoint),
                    qualifies=accepted, H1_tau2=h1["metrics"]["all"]["assisted_tau_sq_Nm2"]))
                write_json(out / "candidate_review.json", candidates)
                pending.remove((job, epoch))
                ready = True
                status("checkpoint reviewed", last_candidate=candidates[-1],
                       remaining_checkpoints=len(pending))
            if not ready:
                status("waiting for training checkpoints", remaining_checkpoints=len(pending))
                time.sleep(15)
        accepted = [c for c in candidates if c["qualifies"]]
        if not accepted:
            status("complete: no new candidate passed; retain L1", selected=None)
            return
        best = min(accepted, key=lambda c: c["H1_tau2"])
        status("candidate selected for independent confirmation", selected=best)
        second = train_seed(best["job"], best["epoch"])
        confirmed = []
        for condition in ("legacy", "structured", "hold50"):
            heldout_base = evaluate("checkpoints/L1_llp_pendulum/last.ckpt",
                                   f"test_L1_H1_{condition}", condition,
                                   real_h1=True, scale=.5, test=True)
            for seed, cp in enumerate((best["checkpoint"], second)):
                report = evaluate(cp, f"test_selected_s{seed}_H1_{condition}", condition,
                                  real_h1=True, test=True)
                confirmed.append(improves_baseline(report, heldout_base))
        if not all(confirmed):
            status("complete: independent confirmation failed; retain L1", selected=best)
            return
        # W1 remains frozen. These are transfer diagnostics, never training targets.
        transfers = {}
        for label, checkpoint, scale in (("L1", "checkpoints/L1_llp_pendulum/last.ckpt", .5),
                                          ("candidate", best["checkpoint"], 1.)):
            for motion in (0, 1):
                for seed in (42, 7):
                    tag = f"{label}_motion{motion}_s{seed}"
                    status("frozen W1 transfer " + tag)
                    existing = sorted((out / "W1_transfer").glob(f"*_{tag}/metrics.json"))
                    if not existing:
                        run([sys.executable, "examples/experiments/assist_skeleton/run_system.py",
                         "--llp-checkpoint", checkpoint, "--assist-scale", str(scale),
                         "--torque-limit", str(20 / scale), "--seconds", "40", "--motion-id", str(motion),
                         "--seed", str(seed), "--out", str(out / "W1_transfer"), "--tag", tag,
                         "--hlp-checkpoint", plan["hlp_checkpoint"], "--fullbody-diagnostic"], tag)
                    path = sorted((out / "W1_transfer").glob(f"*_{tag}/metrics.json"))[-1]
                    meta = json.loads((path.parent / "meta.json").read_text())
                    assert meta["assist_scale"] == scale and meta["torque_limit"] == 20 / scale
                    full_meta = json.loads((path.parent / "fullbody_meta.json").read_text())
                    with np.load(path.parent / "fullbody.npz") as full:
                        tau = full["control_tau_pd_prev"] + full["control_tau_ff_prev"]
                    # Each record ends the preceding 10 ms torque window. Use
                    # the same post-warmup window for every joint and both plants.
                    tau = tau[int(round(meta["warmup"] / meta["llp_dt"])):]
                    tau2 = np.mean(tau * tau, axis=0)
                    transfers[tag] = dict(path=str(path), metrics=json.loads(path.read_text()),
                        all_joint_tau2_Nm2={name: dict(reference=float(tau2[0, j]), assisted=float(tau2[1, j]))
                            for j, name in enumerate(full_meta["dof_names_common"])})
                    write_json(out / "transfer_review.json", transfers)
        status("complete: pendulum confirmation and frozen W1 transfer recorded; deployment unchanged",
               selected=best, second_seed_checkpoint=str(second))
    except BaseException as exc:
        status("stopped with error", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
