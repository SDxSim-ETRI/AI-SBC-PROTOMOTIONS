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
"""Train L6 then L7, evaluating saved checkpoints at gain 1 on W1.

Designed for a restarting user systemd service. Training resumes using the
framework's last.ckpt; successful evaluations have atomic completion records.
All training finishes for one variant before its saved checkpoints are evaluated.
"""

import argparse
import html
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/experiments/assist_skeleton"))
from analysis import assist_saturation_pct  # noqa: E402
from compare_runs import metrics  # noqa: E402


def save_json(path, obj):
    def clean(value):
        if isinstance(value, np.generic):
            return clean(value.item())
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(clean(obj), indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def run(command, log):
    print("RUN", " ".join(map(str, command)), flush=True)
    with log.open("a") as stream:
        subprocess.run(list(map(str, command)), cwd=ROOT, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def checkpoint_epoch(path):
    return int(torch.load(path, map_location="cpu", weights_only=False)["epoch"])


def summarize(run_dir):
    meta = json.loads((run_dir / "meta.json").read_text())
    with np.load(run_dir / "traces.npz") as f:
        tr = {key: f[key] for key in f.files}
    valid = tr["t"] >= meta["warmup"]
    # Early falls are valid outcomes; never retry them as execution failures.
    out = metrics(run_dir) if valid.sum() > 1 else {"valid_after_warmup": False}
    out["survived_s"] = meta["survived_s"]
    out["run_dir"] = str(run_dir)
    out["assist_scale"] = meta["assist_scale"]
    if valid.any():
        if "tau_requested" in tr:
            request = tr["tau_requested"][valid, 1]
            out["requested_peak_Nm"] = float(np.abs(request).max())
            out["request_clipped_pct"] = float(100 * (np.abs(request) > meta["policy_torque_limit"]).mean())
        out["assist_sat95_pct"] = assist_saturation_pct(tr["tau_assist"][valid, 1], meta)
    return out


def report(out_dir):
    rows = {p.stem: json.loads(p.read_text()) for p in sorted(out_dir.glob("eval_*.json"))}
    for motion, folder in ((0, "2026-10-01_18-38-39_tracker_L1s05_verified"),
                           (1, "2026-10-01_18-40-24_tracker_L1s05_verified")):
        baseline = ROOT / "results/oracle" / folder
        if (baseline / "meta.json").exists():
            rows[f"reference_L1_x0.5_m{motion}"] = summarize(baseline)
    save_json(out_dir / "comparison.json", rows)
    if rows and all("τ²" in row for row in rows.values()):
        from report_torque_html import build_report

        build_report(out_dir / "comparison.json", out_dir / "comparison.html")
        return
    cols = ["τ²", "|ω|", "body", "survived_s", "assist_rms", "request_clipped_pct", "assist_sat95_pct"]
    def cell(v):
        return f"{v:.2f}" if isinstance(v, (int, float)) else html.escape(str(v))
    body = "".join("<tr><td>" + html.escape(name) + "</td>" +
                   "".join("<td>" + cell(row.get(k, "—")) + "</td>" for k in cols) + "</tr>"
                   for name, row in rows.items())
    (out_dir / "comparison.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>L6/L7 W1 평가</title>'
        '<style>body{font:16px sans-serif;margin:40px;color:#17232c}'
        'table{border-collapse:collapse}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:right}</style>'
        '<h1>L6/L7 · W1 걷기/달리기</h1><p>추론 배율 1, 최종 제한 ±100 N·m. '
        'L1×0.5는 기존 기준선(제한 ±20 N·m). τ²·|ω|·body는 무보조 대비 변화(%). survived_s는 A/B 생존 시간(s). '
        'assist_rms는 N·m, request_clipped_pct는 제한 전 요청의 ±100 초과 비율(%), '
        'assist_sat95_pct는 적용 토크의 한계 95% 초과 비율(%). 짧은 낙상은 지표 없음.</p>'
        '<table><tr><th>체크포인트/동작</th>' + ''.join('<th>' + html.escape(c) + '</th>' for c in cols) +
        '</tr>' + body + '</table>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="llp_wide100")
    parser.add_argument("--output", default="results/llp_wide100_sequence")
    parser.add_argument("--epochs", type=int, default=6000)
    parser.add_argument("--milestones", type=int, nargs="+", default=[2000, 4000, 6000])
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--eval-seconds", type=float, default=40.)
    parser.add_argument("--hlp-checkpoint", default=str(Path.home() / "Projects/ManiFlow_Policy/ManiFlow/data/outputs/hip_flexion_seed-maniflow_lowdim_policy_walking-flexion100-h15-run01_seed42/checkpoints/epoch=0006-val_loss=0.000762.ckpt"))
    args = parser.parse_args()
    if not Path(args.hlp_checkpoint).is_file():
        raise FileNotFoundError(args.hlp_checkpoint)
    if any(e > args.epochs or (e != args.epochs and e % 2000) for e in args.milestones):
        raise ValueError("Milestones must be saved 2000-epoch checkpoints or the final epoch")
    out = (ROOT / args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    protocol = vars(args)
    protocol_file = out / "protocol.json"
    if protocol_file.exists() and json.loads(protocol_file.read_text()) != protocol:
        raise ValueError("Existing output has a different protocol; choose a new output directory")
    save_json(protocol_file, protocol)
    def status(stage):
        save_json(out / "status.json", {"stage": stage, "updated_utc": datetime.now(timezone.utc).isoformat()})
        print(stage, flush=True)
    py = sys.executable
    for label, cost in (("L6", "both"), ("L7", "torque_sq")):
        name = f"{args.prefix}_{label}"
        run_dir = ROOT / "results" / name
        last = run_dir / "last.ckpt"
        status(f"{label}: training to epoch {args.epochs}")
        if not last.exists() or checkpoint_epoch(last) < args.epochs:
            run([py, "protomotions/train_agent.py", "--robot-name", "hip_pendulum",
                 "--simulator", "newton", "--experiment-path", "examples/experiments/assist_pendulum/mlp_wide.py",
                 "--experiment-name", name, "--motion-file", "none", "--num-envs", args.num_envs,
                 "--batch-size", args.batch_size, "--training-max-steps", args.epochs * args.num_envs * 32,
                 "--seed", 0, "--human-cost", cost], out / f"train_{label}.log")
        if checkpoint_epoch(last) != args.epochs:
            raise RuntimeError(f"Unexpected final epoch in {last}")
        for epoch in args.milestones:
            ckpt = last if epoch == args.epochs else run_dir / f"epoch_{epoch}.ckpt"
            if checkpoint_epoch(ckpt) != epoch:
                raise RuntimeError(f"Wrong checkpoint: {ckpt}")
            for motion in (0, 1):
                tag = f"{name}_e{epoch}_m{motion}"
                done = out / f"eval_{tag}.json"
                if done.exists():
                    continue
                status(f"{label}: evaluating epoch {epoch}, motion {motion}")
                eval_root = out / "evaluations"
                candidates = sorted(p for p in eval_root.glob(f"*_{tag}") if (p / "meta.json").exists() and (p / "traces.npz").exists())
                if not candidates:
                    run([py, "examples/experiments/assist_skeleton/run_system.py", "--human", "tracker",
                     "--tracker-checkpoint", "checkpoints/W1_wearer_human/last.ckpt",
                     "--llp-checkpoint", ckpt, "--hlp-checkpoint", args.hlp_checkpoint,
                     "--motion-file", "data/motion_for_trackers/skeleton_torque_suit_gait.pt",
                     "--motion-id", motion, "--seconds", args.eval_seconds, "--llp-dt", .01,
                     "--calibration", "wearer", "--assist-scale", 1, "--torque-limit", 100,
                     "--seed", 42, "--out", eval_root, "--tag", tag], out / f"eval_{tag}.log")
                candidates = sorted(eval_root.glob(f"*_{tag}"))
                if not candidates:
                    raise RuntimeError(f"No evaluation output for {tag}")
                save_json(done, summarize(candidates[-1]))
                report(out)
    status("complete")


if __name__ == "__main__":
    main()
