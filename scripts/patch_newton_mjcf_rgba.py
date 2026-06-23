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
"""Newton import_mjcf.py rgba 색상 패치.

MJCF에서 rgba 속성을 읽어 primitive shape(box, cylinder, sphere 등)에
색상을 전달하지 않는 Newton 버그를 수정합니다.

Newton 1.2.x / 1.3.x 모두 지원.

사용법:
    python scripts/patch_newton_mjcf_rgba.py          # 패치 적용
    python scripts/patch_newton_mjcf_rgba.py --check  # 적용 여부만 확인
    python scripts/patch_newton_mjcf_rgba.py --revert # 패치 되돌리기
"""

import argparse
import pathlib
import sys


def _find_mjcf_file() -> pathlib.Path:
    try:
        import newton._src.utils.import_mjcf as m  # type: ignore[import]
    except ImportError:
        sys.exit("ERROR: newton 패키지를 찾을 수 없습니다. venv/환경을 확인하세요.")
    return pathlib.Path(m.__file__)


# 패치 삽입 위치: material_color 계산 직후, texture 처리 직전
_ANCHOR = '            texture = None'

_PATCH = (
    '            # Pass rgba color to primitive shapes (box, cylinder, sphere, etc.)\n'
    '            if material_color is not None:\n'
    '                shape_kwargs["color"] = material_color\n'
    '\n'
)

_MARKER = 'shape_kwargs["color"] = material_color'


def check(fpath: pathlib.Path) -> bool:
    return _MARKER in fpath.read_text()


def apply(fpath: pathlib.Path) -> None:
    txt = fpath.read_text()
    if _MARKER in txt:
        print(f"이미 패치됨: {fpath}")
        return
    if _ANCHOR not in txt:
        sys.exit(
            f"ERROR: 삽입 위치를 찾을 수 없습니다 (Newton 버전이 다를 수 있음).\n"
            f"  파일: {fpath}\n"
            f"  찾는 문자열: {_ANCHOR!r}"
        )
    patched = txt.replace(_ANCHOR, _PATCH + _ANCHOR, 1)
    fpath.write_text(patched)
    print(f"패치 완료: {fpath}")


def revert(fpath: pathlib.Path) -> None:
    txt = fpath.read_text()
    if _MARKER not in txt:
        print(f"패치 없음 (되돌릴 내용 없음): {fpath}")
        return
    reverted = txt.replace(_PATCH + _ANCHOR, _ANCHOR, 1)
    if reverted == txt:
        sys.exit("ERROR: 패치 블록을 정확히 찾지 못했습니다. 수동으로 확인하세요.")
    fpath.write_text(reverted)
    print(f"패치 되돌림 완료: {fpath}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="패치 적용 여부만 확인")
    parser.add_argument("--revert", action="store_true", help="패치 되돌리기")
    args = parser.parse_args()

    fpath = _find_mjcf_file()
    print(f"대상 파일: {fpath}")

    if args.check:
        if check(fpath):
            print("상태: 패치 적용됨 ✓")
        else:
            print("상태: 패치 없음 ✗  →  python scripts/patch_newton_mjcf_rgba.py 로 적용하세요.")
        return

    if args.revert:
        revert(fpath)
        return

    apply(fpath)


if __name__ == "__main__":
    main()
