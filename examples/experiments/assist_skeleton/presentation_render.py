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
"""Render measured inference poses with the original suit skeleton visual asset.

No new physics is run. Translation interpolation and shortest-path quaternion
SLERP resample the saved 100 Hz poses to the requested presentation frame rate.
Camera tracking uses the unassisted root for both members of a paired run.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

import newton
import newton.viewer
import numpy as np
from PIL import Image
import warp as wp


def interpolate_pose(positions, rotations, index, fraction):
    following = min(index + 1, len(positions) - 1)
    position = positions[index] * (1 - fraction) + positions[following] * fraction
    first = rotations[index]
    second = rotations[following].copy()
    dot = np.sum(first * second, axis=-1, keepdims=True)
    second = np.where(dot < 0, -second, second)
    dot = np.clip(np.abs(dot), 0, 1)
    angle = np.arccos(dot)
    sine = np.sin(angle)
    linear = sine < 1e-5
    denominator = np.maximum(sine, 1e-8)
    a = np.where(linear, 1 - fraction, np.sin((1 - fraction) * angle) / denominator)
    b = np.where(linear, fraction, np.sin(fraction * angle) / denominator)
    rotation = a * first + b * second
    rotation /= np.linalg.norm(rotation, axis=-1, keepdims=True)
    return position, rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--env", type=int, default=1)
    parser.add_argument("--camera-env", type=int, default=0,
                        help="Environment whose pelvis the camera follows")
    parser.add_argument("--camera-height-mode", choices=["fixed", "body-center"], default="fixed",
                        help="Use body-center to keep a fallen body inside the frame")
    parser.add_argument("--start", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--view", choices=["oblique", "side"], default="oblique")
    parser.add_argument("--preview-only", action="store_true")
    args = parser.parse_args()
    data = np.load(args.run / "traces.npz")
    names = data["presentation_body_names"].tolist()
    positions = data["presentation_body_pos"]
    rotations = data["presentation_body_rot"]
    times = data["t"]
    dt = float(np.median(np.diff(times)))
    assert 0 <= args.env < positions.shape[1]
    assert 0 <= args.camera_env < positions.shape[1]
    assert args.start >= times[0] and args.start + args.duration <= times[-1] + dt + 1e-5
    assert np.isfinite(positions).all() and np.isfinite(rotations).all()

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.add_mjcf(
        str(data["presentation_asset"]), floating=True,
        collapse_fixed_joints=False, parse_visuals=True,
        ignore_names=["floor", "ground"],
    )
    labels = [name.rsplit("/", 1)[-1] for name in builder.body_label]
    assert set(labels) == set(names), (labels, names)
    mapping = np.asarray([names.index(name) for name in labels])
    builder.add_ground_plane(color=(0.30, 0.34, 0.40))
    model = builder.finalize(device="cuda:0")
    state = model.state()
    viewer = newton.viewer.ViewerGL(width=args.width, height=args.height, vsync=False, headless=True)
    viewer.show_ui = False
    viewer.show_joints = False
    viewer.show_contacts = False
    viewer.show_collision = False
    viewer.show_visual = True
    viewer.set_model(model)
    viewer.renderer.draw_fps = False
    viewer.renderer.draw_sky = True
    viewer.renderer.sky_upper = (0.89, 0.92, 0.96)
    viewer.renderer.sky_lower = (0.63, 0.69, 0.77)
    viewer.renderer.spotlight_enabled = False
    viewer.camera.fov = 40.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    poses = np.zeros((len(mapping), 7), dtype=np.float32)
    pelvis = names.index("pelvis")
    count = 1 if args.preview_only else int(round(args.duration * args.fps))
    encoder = None
    if not args.preview_only:
        command = [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{args.width}x{args.height}", "-r", str(args.fps),
            "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.out),
        ]
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for frame_index in range(count):
            stamp = args.start + frame_index / args.fps
            index = int(np.searchsorted(times, stamp, side="right") - 1)
            fraction = min(max((stamp - times[index]) / dt, 0), 1)
            position, rotation = interpolate_pose(positions, rotations, index, fraction)
            poses[:, :3] = position[args.env, mapping]
            poses[:, 3:] = rotation[args.env, mapping]
            state.body_q.assign(poses)
            target = position[args.camera_env, pelvis].copy()
            target[2] = 0.94
            if args.camera_height_mode == "body-center":
                heights = position[args.camera_env, :, 2]
                target[2] = max(0.25, float(0.5 * (heights.min() + heights.max())))
            offset = np.array([1.5, -2.85, 0.45] if args.view == "oblique" else [0.0, -3.2, 0.30])
            camera = target + offset
            direction = -offset / np.linalg.norm(offset)
            viewer.set_camera(wp.vec3(camera.tolist()),
                              float(np.degrees(np.arcsin(direction[2]))),
                              float(np.degrees(np.arctan2(direction[1], direction[0]))))
            viewer.begin_frame(stamp)
            viewer.log_state(state)
            viewer.end_frame()
            image = viewer.get_frame(render_ui=False).numpy()
            assert image.shape == (args.height, args.width, 3), image.shape
            if frame_index == 0:
                Image.fromarray(image).save(args.out.with_suffix(".png"))
            if encoder:
                encoder.stdin.write(image.tobytes())
            if frame_index % args.fps == 0:
                print(f"render {args.out.name}: {frame_index / args.fps:.0f}/{args.duration:g}s", flush=True)
    finally:
        if encoder:
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError("Video encoder failed")
        viewer.close()
    record = dict(
        source_run=str(args.run.resolve()), environment=args.env,
        asset=str(data["presentation_asset"]), start_s=args.start,
        duration_s=args.duration, fps=args.fps, size=[args.width, args.height],
        pose_source_hz=1 / dt, rendering="Original visual meshes; measured inference poses; SLERP at real time",
        camera=f"{args.view}; follows environment {args.camera_env} root; {args.camera_height_mode} height; fixed distance",
        captions=False, policy_or_physics_modified=False,
    )
    args.out.with_suffix(".json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
