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
"""멀티 모션 MotionLib .pt에서 단일 클립을 추출해 별도 .pt로 저장합니다.

이 태스크의 play/record/collect 스크립트가 참조하는
``data/motion_for_trackers/skeleton_torque_suit_walk.pt`` (walk 1클립)가 없는
머신에서, 학습에 사용된 멀티 모션 파일
(``skeleton_torque_suit_motions_11+koo_4.pt``)로부터 walk 클립을 잘라내
복원하는 용도입니다.

MotionLib 패키지 포맷: 프레임 축으로 이어붙인 텐서
(gts/grs/gvs/gavs/dps/dvs/contacts)와 O(1) 인덱싱용 length_starts 및 모션별
메타데이터(motion_files/num_frames/lengths/dt/weights). 슬라이스 후 메타데이터를
단일 모션 기준으로 재구성합니다.

실행:
  python tasks/mimic_suit_active_cable_walk_23dof/extract_walk_motion.py
  python tasks/mimic_suit_active_cable_walk_23dof/extract_walk_motion.py \
      --motion-name walk_koo --output data/motion_for_trackers/..._walk_koo.pt
"""

import argparse
import os

import torch

FRAME_FIELDS = ["gts", "grs", "gvs", "gavs", "dps", "dvs", "contacts"]
PER_MOTION_FIELDS = ["motion_num_frames", "motion_lengths", "motion_dt", "motion_weights"]

DEFAULT_SOURCE = "data/motion_for_trackers/skeleton_torque_suit_motions_11+koo_4.pt"
DEFAULT_OUTPUT = "data/motion_for_trackers/skeleton_torque_suit_walk.pt"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", default=DEFAULT_SOURCE)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--motion-name", default="walk",
                   help="추출할 모션 (motion_files의 파일명 stem과 정확히 일치)")
    p.add_argument("--force", action="store_true", help="기존 출력 파일 덮어쓰기")
    args = p.parse_args()

    if os.path.exists(args.output) and not args.force:
        raise SystemExit(f"이미 존재: {args.output} (덮어쓰려면 --force)")

    data = torch.load(args.source, map_location="cpu", weights_only=False)

    stems = [os.path.splitext(os.path.basename(f))[0] for f in data["motion_files"]]
    if args.motion_name not in stems:
        raise SystemExit(f"'{args.motion_name}' 없음. 가능한 모션: {stems}")
    idx = stems.index(args.motion_name)

    start = int(data["length_starts"][idx])
    n = int(data["motion_num_frames"][idx])
    end = start + n
    print(f"추출: [{idx}] {data['motion_files'][idx]}  frames {start}:{end} ({n})")

    out = {}
    for k in FRAME_FIELDS:
        if k in data and data[k] is not None:
            out[k] = data[k][start:end].clone()
    for k in PER_MOTION_FIELDS:
        out[k] = data[k][idx:idx + 1].clone()
    out["motion_weights"] = torch.ones_like(out["motion_weights"])
    out["length_starts"] = torch.zeros_like(data["length_starts"][:1])
    out["motion_files"] = [data["motion_files"][idx]]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(out, args.output)
    print(f"저장: {args.output}")
    for k, v in out.items():
        shape = tuple(v.shape) if torch.is_tensor(v) else v
        print(f"  {k}: {shape}")


if __name__ == "__main__":
    main()
