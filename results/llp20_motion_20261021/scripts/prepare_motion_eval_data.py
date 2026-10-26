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
"""진자 평가용 ETRI 슈트 14동작 사본 (2026-10-21).

ETRI 기록은 학습에 쓰지 않는다 (W1 평가 걷기·달리기와 같은 출처). 평가용으로만
* 4채널을 SEED 영점으로 맞추고 (`channel_mean_offset`, HLP 입력 규약),
* 고관절 2채널을 진자 관절 범위 안(±86°)으로 자른다. 한계를 넘는 클립(높이 들기·런지 등)을
  통째로 빼지 않기 위해서이며, 잘린 프레임 비율을 동작별로 기록한다.
* 같은 기록의 복제본(walk_koo = walk)은 뺀다.
동작 이름을 동작 종류로 쓰는 metadata parquet도 함께 만든다.

    python examples/experiments/assist_pendulum/prepare_motion_eval_data.py
"""

import argparse
import json
from pathlib import Path

import numpy as np

from protomotions.envs.control.motion_intent import channel_mean_offset

ROOT = Path(__file__).resolve().parents[3]
DATA = Path("/home/bak/Projects/ManiFlow_Policy/ManiFlow/data")
CLAMP_RAD = float(np.radians(86.0))
DUPLICATES = {"walk_koo.npz"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--etri", default=str(DATA / "hip-flexion-etri-suit14-100fps.zarr"))
    p.add_argument("--seed-ref", default=str(DATA / "hip-flexion-seed-locomotion-100fps.zarr"))
    p.add_argument("--out", default=str(ROOT / "results/llp20_motion_20261021/eval_data"))
    args = p.parse_args()
    import pandas as pd
    import zarr

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    offset = np.asarray(channel_mean_offset(args.seed_ref, args.etri), dtype=np.float32)
    src = zarr.open(args.etri, mode="r")
    names = list(src.attrs["motion_names"])
    ends = np.asarray(src["meta/episode_ends"][:], dtype=np.int64)
    starts = np.concatenate([[0], ends[:-1]])
    state = np.asarray(src["data/state"][:], dtype=np.float32) + offset[None, :]
    clips, kept, info = [], [], {}
    for name, a, b in zip(names, starts, ends):
        if name in DUPLICATES:
            continue
        x = state[a:b].copy()
        clipped = np.abs(x[:, :2]) > CLAMP_RAD
        x[:, :2] = np.clip(x[:, :2], -CLAMP_RAD, CLAMP_RAD)
        clips.append(x)
        kept.append(name)
        info[name] = dict(seconds=float((b - a) / 100.0),
                          clamped_frame_fraction=float(clipped.any(axis=1).mean()),
                          peak_flexion_deg=float(np.degrees(np.abs(state[a:b, :2]).max())))
    zarr.config.set({"default_zarr_format": 2})
    path = out / "etri_suit13_seedzero_clamp86_100fps.zarr"
    root = zarr.open(str(path), mode="w")
    data = np.concatenate(clips).astype(np.float32)
    new_ends = np.cumsum([len(c) for c in clips]).astype(np.int64)
    root.create_group("data").create_array("state", shape=data.shape, dtype="float32")[:] = data
    root.create_group("meta").create_array("episode_ends", shape=new_ends.shape, dtype="int64")[:] = new_ends
    for key in ("features", "obs_spec", "action_dof_names", "convention", "embodiment"):
        if key in src.attrs:
            root.attrs[key] = src.attrs[key]
    root.attrs["fps"] = 100.0
    root.attrs["motion_names"] = kept
    root.attrs["note"] = "ETRI suit 13 motions, SEED neutral offset applied, hip channels clipped to ±86 deg; evaluation only"
    stems = [Path(n).stem for n in kept]
    pd.DataFrame({"filename": stems, "content_type_of_movement": stems}).to_parquet(out / "etri_suit13_metadata.parquet")
    report = dict(source=args.etri, offset_rad=offset.tolist(), clamp_deg=86.0, motions=info,
                  zarr=str(path.relative_to(ROOT)), metadata=str((out / "etri_suit13_metadata.parquet").relative_to(ROOT)))
    (out / "etri_eval_data.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
