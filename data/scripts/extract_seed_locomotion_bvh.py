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
"""bones-seed tar.gz 아카이브에서 목록에 있는 멤버만 스트리밍 추출.

seed의 soma_uniform.tar.gz(45GB)는 전 패키지 142k 모션이 한 아카이브라서,
locomotion(74,488개)만 뽑으려면 전체를 풀지 않고 한 번의 순차 패스로
목록 매칭 파일만 디스크에 쓴다 (tarfile r|gz 스트리밍 — 랜덤 액세스 불가
포맷이므로 이 방식이 최선).

사용 예:
  python data/scripts/extract_seed_locomotion_bvh.py \
      --archive data/seed/soma_uniform.tar.gz \
      --file-list data/seed/locomotion_soma_uniform_files.txt \
      --output-dir data/seed
"""

import argparse
import os
import tarfile
import time


def main():
    parser = argparse.ArgumentParser(description="tar.gz에서 목록 멤버만 스트리밍 추출")
    parser.add_argument("--archive", required=True, help="입력 tar.gz")
    parser.add_argument("--file-list", required=True,
                        help="추출할 멤버 경로 목록 (한 줄당 하나, 아카이브 내 경로)")
    parser.add_argument("--output-dir", required=True, help="추출 루트 디렉토리")
    args = parser.parse_args()

    with open(args.file_list) as f:
        wanted = {line.strip() for line in f if line.strip()}
    # 아카이브 멤버가 './' 접두사로 저장된 경우도 매칭
    wanted |= {f"./{p}" for p in list(wanted)}
    print(f"want {len(wanted) // 2} members")

    n_done = 0
    n_seen = 0
    t0 = time.time()
    with tarfile.open(args.archive, mode="r|gz") as tar:
        for member in tar:
            n_seen += 1
            if member.name in wanted:
                tar.extract(member, path=args.output_dir, filter="data")
                n_done += 1
                if n_done % 2000 == 0:
                    el = time.time() - t0
                    print(f"  {n_done} extracted / {n_seen} scanned "
                          f"({el / 60:.1f} min)", flush=True)

    missing = len(wanted) // 2 - n_done
    print(f"done: {n_done} extracted, {n_seen} scanned, "
          f"{missing} missing, {(time.time() - t0) / 60:.1f} min")
    if missing:
        print("WARNING: 일부 목록 파일이 아카이브에 없습니다 — 목록/아카이브 버전 확인")


if __name__ == "__main__":
    main()
