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
"""L8 fresh training and gain-1 W1 evaluation, with a phase-aware shortlist.

Run --check-only before launching through a user systemd service. A failed GPU
check exits before training, and does not claim that training has started.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from run_wide_sequence import ROOT, checkpoint_epoch, run, save_json, summarize
from report_torque_html import build_report


def commands(args):
    name = f"llp_{args.label}_opposition"
    recipe = "mlp_relative_opposition.py" if args.relative_opposition else "mlp_opposition.py"
    if args.soft_impedance:
        recipe = "mlp_soft_impedance.py"
    if args.mixed_feedforward:
        recipe = "mlp_mixed_feedforward.py"
    train = [sys.executable, "protomotions/train_agent.py", "--robot-name", "hip_pendulum",
             "--simulator", "newton", "--experiment-path", f"examples/experiments/assist_pendulum/{recipe}",
             "--experiment-name", name, "--motion-file", "none", "--num-envs", "4096",
             "--batch-size", "16384", "--training-max-steps", str(6000*4096*32),
             "--seed", str(args.seed), "--human-cost", "both", "--opposition-weight", str(args.opposition_weight)]
    if args.relative_opposition:
        train += ["--opposition-torque-floor", str(args.opposition_torque_floor)]
    return name, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="L8")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opposition-weight", type=float, default=-1e-4)
    parser.add_argument("--relative-opposition", action="store_true")
    parser.add_argument("--soft-impedance", action="store_true")
    parser.add_argument("--mixed-feedforward", action="store_true")
    parser.add_argument("--opposition-torque-floor", type=float, default=10.0)
    parser.add_argument("--comparison-source", default="results/llp_wide100_sequence/comparison.json")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.soft_impedance and not args.relative_opposition:
        parser.error("--soft-impedance requires --relative-opposition")
    if args.mixed_feedforward and (not args.relative_opposition or args.soft_impedance):
        parser.error("--mixed-feedforward requires relative opposition without soft impedance")
    name, train = commands(args)
    out = ROOT / "results" / f"{name}_sequence"
    out.mkdir(parents=True, exist_ok=True)
    protocol = {"label": args.label, "seed": args.seed, "opposition_weight": args.opposition_weight,
                "epochs": 6000, "milestones": [2000,4000,6000], "assist_scale": 1, "limit_Nm": 100}
    if args.relative_opposition:
        protocol.update(opposition_form="log1p_product", torque_floor_Nm=args.opposition_torque_floor)
    if args.soft_impedance:
        protocol.update(omega_n_range=[6.0, 24.0], human_stiffness_floor_Nm_per_rad=60.0)
    if args.mixed_feedforward:
        protocol.update(human_feedforward_disable_prob=0.5, reference_recipe="L9")
    if args.comparison_source != "results/llp_wide100_sequence/comparison.json":
        protocol["comparison_source"] = str((ROOT / args.comparison_source).resolve())
    if (out / "protocol.json").exists() and json.loads((out / "protocol.json").read_text()) != protocol:
        raise ValueError("Protocol mismatch: use a distinct label for a changed experiment")
    save_json(out / "protocol.json", protocol)
    paths = ["checkpoints/W1_wearer_human/last.ckpt", "data/motion_for_trackers/skeleton_torque_suit_gait.pt",
             args.comparison_source]
    hlp = Path.home() / "Projects/ManiFlow_Policy/ManiFlow/data/outputs/hip_flexion_seed-maniflow_lowdim_policy_walking-flexion100-h15-run01_seed42/checkpoints/epoch=0006-val_loss=0.000762.ckpt"
    missing = [p for p in paths if not (ROOT / p).is_file()]
    if not hlp.is_file(): missing.append(str(hlp))
    ready = {"cuda_available": torch.cuda.is_available(), "missing_files": missing, "training_command": train}
    save_json(out / "readiness.json", ready)
    def status(stage):
        save_json(out / "status.json", {"stage": stage, "updated_utc": datetime.now(timezone.utc).isoformat()})
        print(stage, flush=True)
    if not ready["cuda_available"] or missing:
        status("blocked: CUDA unavailable or required files missing; no training launched")
        print(json.dumps(ready, ensure_ascii=False, indent=2), flush=True)
        raise SystemExit(2)
    if args.check_only:
        print(json.dumps(ready, ensure_ascii=False, indent=2))
        return
    train_dir = ROOT / "results" / name
    last = train_dir / "last.ckpt"
    status("training")
    if not last.exists() or checkpoint_epoch(last) < 6000:
        run(train, out / "train.log")
    if checkpoint_epoch(last) != 6000:
        raise RuntimeError("Training did not reach epoch 6000")
    records = json.loads((ROOT / args.comparison_source).read_text())
    for epoch in (2000,4000,6000):
        ckpt = last if epoch == 6000 else train_dir / f"epoch_{epoch}.ckpt"
        if checkpoint_epoch(ckpt) != epoch:
            raise RuntimeError(f"Unexpected checkpoint epoch: {ckpt}")
        for motion in (0,1):
            tag = f"{name}_e{epoch}_m{motion}"
            done = out / f"eval_{tag}.json"
            if not done.exists():
                status(f"evaluating epoch {epoch}, motion {motion}")
                eval_root = out / "evaluations"
                candidates = sorted(p for p in eval_root.glob(f"*_{tag}") if (p/"meta.json").exists() and (p/"traces.npz").exists())
                if not candidates:
                    run([sys.executable, "examples/experiments/assist_skeleton/run_system.py", "--human", "tracker",
                         "--tracker-checkpoint", paths[0], "--llp-checkpoint", ckpt, "--hlp-checkpoint", hlp,
                         "--motion-file", paths[1], "--motion-id", motion, "--seconds", 40, "--llp-dt", .01,
                         "--calibration", "wearer", "--assist-scale", 1, "--torque-limit", 100, "--seed", 42,
                         "--out", eval_root, "--tag", tag], out / f"eval_{tag}.log")
                    candidates = sorted(p for p in eval_root.glob(f"*_{tag}") if (p/"meta.json").exists() and (p/"traces.npz").exists())
                if not candidates: raise RuntimeError(f"No output for {tag}")
                save_json(done, summarize(candidates[-1]))
            records[f"eval_{tag}"] = json.loads(done.read_text())
            save_json(out / "comparison.json", records)
            build_report(out / "comparison.json", out / "comparison.html")
            build_report(out / "comparison.json", ROOT / "docs/ai_sbc/l6_l7_torque_report.html")
    status("complete; awaiting analysis before the next hypothesis")


if __name__ == "__main__":
    main()
