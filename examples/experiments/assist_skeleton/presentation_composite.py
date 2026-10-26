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
"""Synchronized inference comparisons; labels only, no narration or subtitles."""

import argparse
import json
from pathlib import Path
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from report_output_scale import align

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def compose(folder, sources, labels, name, short=False):
    command = ["ffmpeg", "-y", "-loglevel", "error", "-filter_complex_threads", "1"]
    for source in sources:
        command.extend(["-i", str(folder / source)])
    n = len(sources)
    width, height = 1920 // n, 720 if short else 1080
    filters = []
    for i, label in enumerate(labels):
        transform = f"scale=1280:720,crop={width}:720:(iw-{width})/2:0" if short else f"crop={width}:1080:(iw-{width})/2:0"
        filters.append(
            f"[{i}:v]{transform},drawbox=x=0:y=0:w=iw:h=70:color=white@0.93:t=fill,"
            f"drawtext=fontfile={FONT}:text='{label}':fontcolor=0x233547:fontsize=29:x=(w-tw)/2:y=22[v{i}]"
        )
    layout = "|".join(f"{i*width}_0" for i in range(n))
    filters.append("".join(f"[v{i}]" for i in range(n)) + f"xstack=inputs={n}:layout={layout}[out]")
    output = folder / name
    command.extend(["-filter_complex", ";".join(filters), "-map", "[out]", "-an", "-c:v", "libx264",
                    "-preset", "fast", "-crf", "18", "-threads", "4", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)])
    subprocess.run(command, check=True)
    return output


def synchronized_torques(study, source, output):
    data = {}
    for label in ("L1_20", "R0_20"):
        run = next((study / "inference").glob(f"*_{label}_seed42"))
        with np.load(run / "traces.npz") as traces:
            data[label] = align(dict(traces))
    colors = {"No assist": "#777f89", "L1": "#4274b8", "R0": "#12866e"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(19.2, 3.6), dpi=100)
    figure.subplots_adjust(left=.065, right=.98, bottom=.19, top=.85, wspace=.22)
    specifications = [
        [("No assist", "L1_20", "A"), ("L1", "L1_20", "B"), ("R0", "R0_20", "B")],
        [("L1", "L1_20", "assist"), ("R0", "R0_20", "assist")],
    ]
    lines = []
    for ax, specs, title, limits in zip(axes, specifications,
                                      ["Right hip | Wearer torque [N m]", "Right hip | Assist torque [N m]"],
                                      [(-190, 120), (-23, 23)]):
        current = []
        for label, key, channel in specs:
            line, = ax.plot([], [], color=colors[label], lw=1.8, label=label)
            current.append((line, key, channel))
        ax.set(title=title, xlabel="Time relative to current frame [s]", xlim=(-2.5, 0), ylim=limits)
        ax.grid(alpha=.2)
        ax.axhline(0, lw=.7, color="#aab3bd")
        ax.legend(loc="lower left", frameon=False, ncol=len(specs), fontsize=10)
        lines.append(current)
    decoder = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(source), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    encoder = subprocess.Popen([
        "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1920x1080",
        "-r", "60", "-i", "-", "-an", "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-threads", "4", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ], stdin=subprocess.PIPE)
    bytes_per_frame = 1920 * 720 * 3
    frames = 0
    try:
        for i in range(1200):
            raw = decoder.stdout.read(bytes_per_frame)
            if len(raw) != bytes_per_frame:
                raise RuntimeError(f"Video ended at frame {i}")
            stamp = 5 + i / 60
            for specs in lines:
                for line, key, channel in specs:
                    values = data[key]
                    selected = (values["t"] >= stamp - 2.5) & (values["t"] <= stamp)
                    line.set_data(values["t"][selected] - stamp, values[channel][selected, 0])
            figure.canvas.draw()
            plot = np.asarray(figure.canvas.buffer_rgba())[:, :, :3]
            top = np.frombuffer(raw, dtype=np.uint8).reshape(720, 1920, 3)
            frame = np.concatenate([top, plot], axis=0)
            encoder.stdin.write(frame.tobytes())
            if i in (0, 600, 1199):
                Image.fromarray(frame).save(output.with_name(f"{output.stem}_frame{i}.png"))
            if i % 120 == 0:
                print(f"Torque comparison {i/60:.0f}/20s", flush=True)
            frames += 1
    finally:
        decoder.stdout.close()
        encoder.stdin.close()
        decode_status = decoder.wait()
        encode_status = encoder.wait()
        plt.close(figure)
    assert decode_status == 0 and encode_status == 0 and frames == 1200


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("results/presentation_walk_20261022"))
    args = parser.parse_args()
    folder = args.study / "deliverables"
    sources = ["00_unassisted_clean.mp4", "01_L1_20Nm_clean.mp4", "02_R0_20Nm_clean.mp4"]
    labels = ["No assist", "L1 | +/-20 N m", "R0 | +/-20 N m"]
    compose(folder, sources, labels, "04_comparison_same_limit.mp4")
    compose(folder, ["03_L1_10Nm_clean.mp4", "02_R0_20Nm_clean.mp4"],
            ["L1 | +/-10 N m", "R0 | +/-20 N m"], "05_comparison_similar_assist_rms.mp4")
    temporary = compose(folder, sources, labels, "_comparison_top.mp4", short=True)
    synchronized_torques(args.study, temporary, folder / "06_comparison_with_torques.mp4")
    temporary.unlink()
    (folder / "comparison_provenance.json").write_text(json.dumps(dict(
        source_start_s=5, source_end_s=25, real_time=True, fps=60,
        no_narration=True, no_explanatory_subtitles=True,
        labels="Model names and applied torque limits only",
        torque_strip="Right hip, previous 2.5 seconds, raw aligned 100 Hz samples; no smoothing",
        unassisted_display="Environment A from L1 +/-20 run; each policy has its own paired reference in evidence.json",
    ), indent=2))


if __name__ == "__main__":
    main()
