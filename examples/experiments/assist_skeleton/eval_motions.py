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
"""여러 모션 클립에 대해 통합 시스템 A/B 평가를 돌리고 표로 모은다.

`run_system.py --human tracker`를 모션 id마다 별도 프로세스로 실행(Newton은 한
프로세스에서 env를 다시 만들지 않음)하고 각 `metrics.json`을 읽어 마크다운 표를
만든다. LLP의 일반화(학습 모션 vs 미학습 모션, walk 학습 LLP vs 15모션 학습 LLP)
비교용.

실행 (sbc env):
    python examples/experiments/assist_skeleton/eval_motions.py \\
        --llp-checkpoint results/llp_skel_motions15_v4/last.ckpt \\
        --tracker-checkpoint checkpoints/v31_newton_suit_passive_motions15/last.ckpt \\
        --motion-file "data/motion_for_trackers/skeleton_torque_suit_motions_11+koo_4.pt" \\
        --calibration motion_all --seconds 20 --out results/assist_skeleton_motions/v4
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def motion_names(motion_file: str):
    m = torch.load(motion_file, map_location="cpu", weights_only=False)
    names = [os.path.basename(f).replace(".npz", "") for f in m["motion_files"]]
    lengths = [float(x) for x in m["motion_lengths"]]
    return names, lengths


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--llp-checkpoint", required=True)
    p.add_argument("--tracker-checkpoint", required=True)
    p.add_argument("--motion-file", required=True)
    p.add_argument("--motion-ids", default="all", help="'all' 또는 '0,5,10'")
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--calibration", default="motion_all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    p.add_argument("--extra", default="", help="run_system.py에 그대로 넘길 추가 인자")
    args = p.parse_args()

    names, lengths = motion_names(args.motion_file)
    ids = list(range(len(names))) if args.motion_ids == "all" else [int(v) for v in args.motion_ids.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for mid in ids:
        # 클립이 요청 길이보다 짧으면 클립 길이의 90 %만 (tracker의 clip 종료 리셋을 피한다)
        seconds = min(args.seconds, max(3.0, 0.9 * lengths[mid]))
        tag = f"m{mid:02d}_{names[mid]}"
        cmd = [
            sys.executable, str(HERE / "run_system.py"), "--human", "tracker",
            "--tracker-checkpoint", args.tracker_checkpoint, "--llp-checkpoint", args.llp_checkpoint,
            "--motion-file", args.motion_file, "--motion-id", str(mid), "--seconds", f"{seconds:.2f}",
            "--calibration", args.calibration, "--seed", str(args.seed), "--out", str(out), "--tag", tag,
        ] + (args.extra.split() if args.extra else [])
        print(f"[eval_motions] {tag} ({seconds:.1f} s) …", flush=True)
        res = subprocess.run(cmd, capture_output=True, text=True)
        run_dirs = sorted(out.glob(f"*_{tag}"))
        if res.returncode != 0 or not run_dirs:
            print(f"  FAILED (rc={res.returncode})\n" + "\n".join(res.stderr.splitlines()[-8:]))
            rows.append({"id": mid, "name": names[mid], "failed": True})
            continue
        metrics = json.loads((run_dirs[-1] / "metrics.json").read_text())
        meta = json.loads((run_dirs[-1] / "meta.json").read_text())
        s = metrics["summary"]
        rows.append({
            "id": mid, "name": names[mid], "seconds": seconds,
            "power_A": s["human_power_W_A"], "power_B": s["human_power_W_B"],
            "reduction_pct": s["human_power_reduction_pct"], "corr": s["assist_vs_agentA_corr"],
            "assist_rms": s["assist_rms_Nm"], "track_A": s["tracking_err_deg_A"], "track_B": s["tracking_err_deg_B"],
            "survived_s": meta.get("survived_s"), "hlp_150ms": metrics["hlp_lead_mae_deg"].get("150ms"),
        })
        print(f"  A {s['human_power_W_A']:.1f} W → B {s['human_power_W_B']:.1f} W ({s['human_power_reduction_pct']:+.1f} %), corr {s['assist_vs_agentA_corr']:.2f}, fall {meta.get('survived_s')}")

    ok = [r for r in rows if not r.get("failed")]
    lines = [
        f"# 모션별 A/B 평가 — LLP `{args.llp_checkpoint}`",
        "",
        f"tracker `{args.tracker_checkpoint}`, {args.seconds:.0f} s/클립(짧은 클립은 90 %), calibration {args.calibration}, seed {args.seed}",
        "",
        "| id | 모션 | 착용자 파워 A→B [W] | 절감 | corr | assist RMS | hip 오차 A→B [deg] | HLP 150 ms MAE | 낙상 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("failed"):
            lines.append(f"| {r['id']} | {r['name']} | 실패 | | | | | | |")
            continue
        fall = "-" if r["survived_s"] is None else f"{r['survived_s']:.1f} s"
        hlp = "-" if r["hlp_150ms"] is None else f"{r['hlp_150ms']:.1f}°"
        lines.append(
            f"| {r['id']} | {r['name']} | {r['power_A']:.1f} → {r['power_B']:.1f} | **{r['reduction_pct']:+.1f} %** | {r['corr']:.2f} | "
            f"{r['assist_rms']:.1f} | {r['track_A']:.1f} → {r['track_B']:.1f} | {hlp} | {fall} |"
        )
    if ok:
        pa = sum(r["power_A"] * r["seconds"] for r in ok) / sum(r["seconds"] for r in ok)
        pb = sum(r["power_B"] * r["seconds"] for r in ok) / sum(r["seconds"] for r in ok)
        lines += ["", f"시간 가중 평균: {pa:.1f} → {pb:.1f} W (**{100*(1-pb/pa):+.1f} %**), 낙상 {sum(1 for r in ok if r['survived_s'] is not None)}/{len(ok)}"]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    (out / "summary.json").write_text(json.dumps(rows, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
