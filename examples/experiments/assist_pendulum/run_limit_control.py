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
"""Matched 20/40 Nm training in historical and corrected pendulum torque paths.

All arms start from random weights, use 4096 environments and 6000 PPO epochs,
and are evaluated at an effective 20 Nm. The final epoch is fixed in advance.
Historical means reconstructing the saved L1 torque settings in current code;
it does not imply a bitwise reproduction of the original L1 training run.
"""

import argparse
import dataclasses
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
EPOCHS = 6000
NUM_ENVS = 4096
BASE = "results/llp_pendulum20_C0_20261014_nativeff/epoch_2000.ckpt"
RECIPE = "examples/experiments/assist_pendulum/mlp_research.py"
SOURCES = (
    "examples/experiments/assist_pendulum/run_limit_control.py",
    RECIPE,
    "examples/experiments/assist_pendulum/mlp_v4.py",
    "examples/experiments/assist_pendulum/mlp_v3.py",
    "examples/experiments/assist_pendulum/mlp.py",
    "examples/experiments/assist_pendulum/evaluate_research.py",
    "examples/experiments/assist_pendulum/runtime_utils.py",
    "protomotions/envs/base_env/assist_env.py",
    "protomotions/envs/control/assist_target_control.py",
    "protomotions/envs/control/assist_motion_control.py",
    "protomotions/envs/control/assist_disturbance_control.py",
    "protomotions/envs/control/pendulum_research_control.py",
    "protomotions/envs/rewards/assist.py",
    "protomotions/envs/rewards/regularization.py",
)


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def canonical(value):
    if dataclasses.is_dataclass(value):
        return {f.name: canonical(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [canonical(v) for v in value]
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if callable(value):
        return value.__module__ + "." + value.__qualname__
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "compute_func"):
        return {k: canonical(v) for k, v in vars(value).items() if not k.startswith("_")}
    return str(value)


def differences(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        found = []
        for key in sorted(a.keys() | b.keys()):
            found += differences(a.get(key), b.get(key), f"{prefix}.{key}".lstrip("."))
        return found
    return [] if a == b else [dict(field=prefix, left=a, right=b)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/pendulum_limit_control_20261015")
    parser.add_argument("--tag", default="20261015")
    parser.add_argument("--hlp-checkpoint", required=True)
    parser.add_argument("--wait-launch", nargs="*", default=[])
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "study.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    records = json.loads((out / "comparison.json").read_text()) if (out / "comparison.json").exists() else {}
    environment = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    jobs = []
    for seed in (0, 1):
        for regime in ("historical", "corrected"):
            for limit in (20, 40):
                name = f"llp_limit_{regime}_{limit}_s{seed}_{args.tag}"
                command = [sys.executable, "-u", "protomotions/train_agent.py",
                    "--robot-name", "hip_pendulum", "--simulator", "newton",
                    "--experiment-path", RECIPE, "--experiment-name", name,
                    "--motion-file", "none", "--num-envs", str(NUM_ENVS),
                    "--batch-size", "16384", "--seed", str(seed),
                    "--training-max-steps", str(EPOCHS * NUM_ENVS * 32),
                    "--assist-limit", str(limit),
                    "--external-human-ff" if regime == "historical" else "--combined-human-limit",
                    "--overrides", "agent.save_epoch_checkpoint_every=2000",
                    f"env.clamp_combined_injection={'True' if regime == 'historical' else 'False'}"]
                jobs.append(dict(name=name, regime=regime, limit=limit, seed=seed,
                                 directory=f"results/{name}", command=command))
    plan = dict(jobs=jobs, epochs=EPOCHS, num_envs=NUM_ENVS, effective_evaluation_limit=20,
                hlp_checkpoint=args.hlp_checkpoint, wait_launch=args.wait_launch,
                primary="Compare fixed epoch 6000 within each torque path, paired training seeds 0/1. No best-checkpoint selection.")
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
    manifest = dict(plan=plan, source_sha256=hashes)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, "Study changed; use a new output directory"
    else:
        save_json(manifest_path, manifest)
        for name in SOURCES:
            p = out / "sources" / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes((ROOT / name).read_bytes())
    save_json(out / "plan.json", plan)

    def status(stage, **extra):
        state = dict(stage=stage, pid=os.getpid(), updated_utc=datetime.now(timezone.utc).isoformat(),
                     completed_evaluations=len(records), **extra)
        save_json(out / "status.json", state)
        lines = ["# 진자 학습 한계 대조 실험", "", f"상태: {stage}", "",
            "기존 L1과 C0의 비교는 학습 한계의 인과 효과를 분리하지 못한다.",
            "이 실험은 기존/수정 토크 처리 각각에서 20/40 N·m만 바꾸어 처음부터 학습한다.",
            "4,096환경·6,000 epoch·seed 0/1, 고정 마지막 체크포인트. 평가 적용 한계는 모두 20 N·m.",
            "기존 처리 경로는 저장 설정의 현재 코드 재구성이며 당시 실행의 비트 단위 재현이 아니다.", "",
            "| 평가 | 사람 τ² [N·m²/관절] | RMSE [°] | 보조 실패 / 256 |", "|---|---:|---:|---:|"]
        for key, value in records.items():
            m = value["metrics"]["all"]
            lines.append(f"| {key} | {m['assisted_tau_sq_Nm2']:.1f} | {m['assisted_tracking_rmse_deg']:.3f} | {value['failures']['assisted']} |")
        lines += ["", "완료되지 않은 조건은 결과가 없다. 실패가 있는 모델의 생존 구간 평균은 우위 판정에 쓰지 않는다.",
                  "같은 seed의 20/40 차이와 두 seed를 각각 보고하며, 두 번의 실행만으로 통계적 우월성을 확정하지 않는다."]
        (out / "report.md").write_text("\n".join(lines) + "\n")
        print(stage, flush=True)

    def run(command, label):
        assert all(hashlib.sha256((ROOT / n).read_bytes()).hexdigest() == h for n, h in hashes.items()), "Study source changed"
        with (out / f"{label}.log").open("a") as stream:
            stream.write(json.dumps(command) + "\n")
            stream.flush()
            subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)

    try:
        status("preflight: generate and compare all saved configurations")
        configs = {}
        for job in jobs:
            path = ROOT / job["directory"] / "resolved_configs.pt"
            if not path.exists():
                run(job["command"] + ["--create-config-only"], job["name"] + "_config")
            configs[job["name"]] = canonical(torch.load(path, map_location="cpu", weights_only=False))
        audit = []
        for left, right in zip(jobs[::2], jobs[1::2]):
            delta = differences(configs[left["name"]], configs[right["name"]])
            fields = {d["field"] for d in delta}
            assert fields == {"simulator.experiment_name", "env.assist_torque_limit", "env.action_config.torque_scale"}, delta
            audit.append(dict(left=left["name"], right=right["name"], differences=delta))
        save_json(out / "config_audit.json", dict(passed=True, comparisons=audit))
        if args.preflight_only:
            status("preflight passed; training has not started")
            return
        while True:
            waiting = []
            for name in args.wait_launch:
                launch = json.loads((ROOT / name).read_text())
                proc = Path(f"/proc/{launch['pid']}/cmdline")
                if proc.exists() and b"train_agent.py" in proc.read_bytes():
                    waiting.append(launch["pid"])
            if not waiting:
                break
            status("queued: waiting for current adaptation training", waiting_pids=waiting)
            time.sleep(30)
        for job in jobs:
            checkpoint = ROOT / job["directory"] / f"epoch_{EPOCHS}.ckpt"
            if not checkpoint.exists():
                status("training " + job["name"], active_job=job)
                run(job["command"], job["name"] + "_train")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            assert saved["epoch"] == EPOCHS
            del saved
            for condition, real_h1 in [("legacy", False), ("structured", False), ("hold50", False), ("legacy", True)]:
                label = job["name"] + ("_H1_" if real_h1 else "_") + condition
                destination = out / label
                summary = destination / "summary.json"
                command = [sys.executable, "examples/experiments/assist_pendulum/evaluate_research.py",
                    "--base-checkpoint", BASE, "--checkpoint", str(checkpoint),
                    "--independent-limits", "--combined-human-limit", "--condition", condition,
                    "--output-scale", str(20 / job["limit"]), "--out", str(destination)]
                if real_h1:
                    command += ["--hlp-checkpoint", args.hlp_checkpoint]
                if not summary.exists():
                    status("evaluating " + label)
                    run(command, label)
                report = json.loads(summary.read_text())
                protocol = report["protocol"]
                assert Path(protocol["checkpoint"]).resolve() == checkpoint.resolve()
                assert protocol["output_scale"] == 20 / job["limit"]
                assert protocol["condition"] == condition
                assert bool(protocol["hlp_checkpoint"]) == real_h1
                assert protocol["combined_human_limit"] and protocol["independent_limits"]
                records[label] = report
                save_json(out / "comparison.json", records)
            status("completed " + job["name"])
        status("complete: all matched training arms evaluated; no deployment changes")
    except BaseException as error:
        status("stopped with error", error=repr(error))
        raise


if __name__ == "__main__":
    main()
