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
"""모션 묶음에서 보행 계열 클립만 골라 새 motion 파일을 만든다.

LLP를 "걷는 동작"에만 집중시키기 위한 학습·평가용 부분집합. 기본값은
`skeleton_torque_suit_motions_11+koo_4.pt`의 walk(=walk_koo와 동일 데이터)와 run이며,
스쿼트·제자리 걸음·한 걸음·런지처럼 주기적 보행이 아닌 클립은 제외한다.
샘플링 비중은 걷기 위주(walk 0.75 / run 0.25).

실행 (sbc env):
    python examples/experiments/assist_skeleton/make_gait_motions.py
"""

import argparse
import os
from pathlib import Path

import torch

PER_MOTION = ("length_starts", "motion_num_frames", "motion_lengths", "motion_dt", "motion_weights")
PER_FRAME = ("gts", "grs", "gvs", "gavs", "dps", "dvs", "contacts")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="data/motion_for_trackers/skeleton_torque_suit_motions_11+koo_4.pt")
    p.add_argument("--names", default="walk,run", help="고를 클립 이름 (쉼표)")
    p.add_argument("--weights", default="0.75,0.25", help="샘플링 비중 (쉼표, --names와 같은 순서)")
    p.add_argument("--out", default="data/motion_for_trackers/skeleton_torque_suit_gait.pt")
    args = p.parse_args()

    src = torch.load(args.source, map_location="cpu", weights_only=False)
    names = [os.path.basename(str(f)).replace(".npz", "") for f in src["motion_files"]]
    want = [w.strip() for w in args.names.split(",")]
    weights = [float(w) for w in args.weights.split(",")]
    assert len(want) == len(weights), "names와 weights 길이가 다르다"
    ids = [names.index(w) for w in want]

    out = {k: [] for k in PER_FRAME}
    starts, frames, lengths, dts = [], [], [], []
    cursor = 0
    for mid in ids:
        s, n = int(src["length_starts"][mid]), int(src["motion_num_frames"][mid])
        for k in PER_FRAME:
            out[k].append(src[k][s : s + n])
        starts.append(cursor)
        frames.append(n)
        lengths.append(float(src["motion_lengths"][mid]))
        dts.append(float(src["motion_dt"][mid]))
        cursor += n

    new = {k: torch.cat(v, dim=0) for k, v in out.items()}
    new["motion_files"] = tuple(src["motion_files"][m] for m in ids)
    new["length_starts"] = torch.tensor(starts, dtype=src["length_starts"].dtype)
    new["motion_num_frames"] = torch.tensor(frames, dtype=src["motion_num_frames"].dtype)
    new["motion_lengths"] = torch.tensor(lengths, dtype=src["motion_lengths"].dtype)
    new["motion_dt"] = torch.tensor(dts, dtype=src["motion_dt"].dtype)
    new["motion_weights"] = torch.tensor(weights, dtype=src["motion_weights"].dtype)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(new, args.out)
    print(f"saved {args.out}: " + ", ".join(f"{n} ({l:.0f} s, w={w})" for n, l, w in zip(want, lengths, weights)))


if __name__ == "__main__":
    main()
