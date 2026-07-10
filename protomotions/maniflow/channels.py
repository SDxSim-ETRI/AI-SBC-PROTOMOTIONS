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
"""ManiFlow 추정기 action 채널 정의 — 수집/학습/inference 공용 단일 소스.

공통(COMMON) DOF 순서는 kinematic tree 순서(오른다리 체인 → 왼다리 체인 → …)
이므로 skeleton_torque_suit* 로봇에서 앞 6개를 자르면 hip이 아니라
[hip_flexion_r, hip_adduction_r, hip_rotation_r, knee_angle_r, ankle_angle_r,
hip_flexion_l]이 나옵니다(과거 수집 스크립트의 버그). 순수 hip 6개는 공통 DOF
[0, 1, 2, 5, 6, 7]이며, 반드시 이 모듈의 이름 기반 파생 함수를 통해서만
인덱스를 얻어야 합니다.
"""

from typing import List, Sequence

# 순수 hip 6채널 (오른쪽 3 + 왼쪽 3). skeleton_torque_suit* 공통 DOF 기준
# [0, 1, 2, 5, 6, 7]에 해당하며, 인덱스는 항상 hip_dof_indices()로 파생할 것.
HIP_DOF_NAMES: List[str] = [
    "hip_flexion_r",
    "hip_adduction_r",
    "hip_rotation_r",
    "hip_flexion_l",
    "hip_adduction_l",
    "hip_rotation_l",
]


def hip_dof_indices(dof_names: Sequence[str]) -> List[int]:
    """공통 DOF 이름 목록에서 hip 6채널의 인덱스를 이름으로 파생합니다.

    Args:
        dof_names: ``robot_config.kinematic_info.dof_names`` (COMMON ordering).

    Returns:
        ``HIP_DOF_NAMES`` 순서의 인덱스 목록 (skeleton_torque_suit* 로봇에서는
        ``[0, 1, 2, 5, 6, 7]``).

    Raises:
        ValueError: hip DOF가 하나라도 없으면 (다른 로봇/이름 규약).
    """
    names = list(dof_names)
    missing = [n for n in HIP_DOF_NAMES if n not in names]
    if missing:
        raise ValueError(
            f"hip DOFs not found in dof_names: {missing} (got {names[:12]}...)"
        )
    return [names.index(n) for n in HIP_DOF_NAMES]


def resolve_action_dofs(mode: str, dof_names: Sequence[str]) -> List[int]:
    """스크립트 공용 ``--action-dofs`` 해석기.

    Args:
        mode: ``"hips"`` — 순수 hip 6채널(신규 계약, 기본값).
              ``"first6"`` — 공통 DOF 0-5 (과거 잘못 수집된 모델과의 비교 전용:
              오른다리 전체 + 왼쪽 hip flexion).
        dof_names: COMMON ordering DOF 이름 목록.
    """
    if mode == "hips":
        return hip_dof_indices(dof_names)
    if mode == "first6":
        return list(range(6))
    raise ValueError(f"unknown action-dofs mode: {mode!r} (hips | first6)")
