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
"""Render and compare the assist removal experiment without changing its poses."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def comparison(folder, sources, labels, name, switches):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-filter_complex_threads", "1"]
    for source in sources:
        cmd += ["-i", str(folder / source)]
    filters = []
    for i, (label, switch) in enumerate(zip(labels, switches)):
        prefix = (f"[{i}:v]scale=1600:900,crop=960:900:(iw-960)/2:0,"
                  "pad=960:1080:0:90:color=white,"
                  "drawbox=x=0:y=0:w=iw:h=75:color=white@0.94:t=fill,"
                  f"drawtext=fontfile={FONT}:text='{label}':fontcolor=0x253545:fontsize=27:x=25:y=24,")
        on = (f"drawtext=fontfile={FONT}:text='ASSIST ON':fontcolor=0x12866e:fontsize=25:x=w-tw-25:y=25")
        if switch:
            on += ":enable='lt(t,5)'"
            on += (f",drawtext=fontfile={FONT}:text='ASSIST OFF':fontcolor=0xb45349:fontsize=25:"
                   "x=w-tw-25:y=25:enable='gte(t,5)'")
        filters.append(prefix + on + f"[v{i}]")
    filters.append("[v0][v1]hstack=inputs=2[out]")
    cmd += ["-filter_complex", ";".join(filters), "-map", "[out]", "-an", "-c:v", "libx264",
            "-preset", "fast", "-crf", "18", "-threads", "4", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(folder/name)]
    subprocess.run(cmd, check=True)


def render_record(study, key, filename):
    runs = json.loads((study/"runs.json").read_text())
    source = Path(runs[key]["path"])
    out = study/"deliverables"/filename
    command = [sys.executable, "examples/experiments/assist_skeleton/presentation_render.py",
               "--run", str(source), "--out", str(out), "--env", "1", "--camera-env", "1",
               "--camera-height-mode", "body-center",
               "--start", "5", "--duration", "15"]
    subprocess.run(command, check=True)
    info = json.loads(out.with_suffix(".json").read_text())
    info["experiment"] = key
    info["display_switch_s"] = 5
    info["simulation_switch_s"] = 10
    info["wearer_torque_replay"] = "replay" in key
    info["assistance_removed"] = not key.endswith("replay_on")
    info.pop("policy_or_physics_modified", None)
    info["rendering_changes_dynamics"] = False
    info["policy_weights_changed"] = False
    info["experiment_intervention"] = (
        "Recorded 400 Hz whole-body wearer torque replaces PD feedback at simulation 10s"
        if "replay" in key else "Assist torque becomes zero at simulation 10s; W1 feedback retained"
    )
    out.with_suffix(".json").write_text(json.dumps(info,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("results/presentation_off_20261023"))
    parser.add_argument("--only", help="One run key for progressive rendering")
    parser.add_argument("--compose-only", action="store_true")
    args = parser.parse_args()
    records = [
        ("L1_feedback", "01_L1_OFF_feedback.mp4"),
        ("R0_feedback", "02_R0_OFF_feedback.mp4"),
        ("L1_replay_off", "03_L1_OFF_fixed_wearer.mp4"),
        ("R0_replay_off", "04_R0_OFF_fixed_wearer.mp4"),
        ("L1_replay_on", "05_L1_ON_fixed_wearer_control.mp4"),
        ("R0_replay_on", "06_R0_ON_fixed_wearer_control.mp4"),
    ]
    if not args.compose_only:
        for key, name in records:
            if args.only is None or key == args.only:
                render_record(args.study, key, name)
    if args.only:
        return
    folder = args.study/"deliverables"
    comparison(folder, [records[0][1], records[1][1]], ["L1 | W1 feedback", "R0 | W1 feedback"],
               "07_OFF_feedback_comparison.mp4", [True, True])
    comparison(folder, [records[2][1], records[3][1]], ["L1 | Torque replay", "R0 | Torque replay"],
               "08_OFF_fixed_wearer_comparison.mp4", [True, True])
    for model, off, on in [("L1", records[2][1], records[4][1]), ("R0", records[3][1], records[5][1])]:
        comparison(folder, [on, off], [model+" | Torque replay", model+" | Torque replay"],
                   f"09_{model}_replay_control_ON_vs_OFF.mp4", [False, True])


if __name__ == "__main__":
    main()
