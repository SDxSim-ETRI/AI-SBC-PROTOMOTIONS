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
"""`protomotions.assist` 단위 테스트 (CPU, 체크포인트 불필요한 부분).

전체 등가성(진자 env ↔ 독립 제어기)은 GPU·체크포인트가 필요하므로
`python -m protomotions.assist.verify_controller`로 따로 돌린다.
"""

import math

import pytest
import torch

from protomotions.assist.skeleton_features import (
    SUIT_LUMBAR_AXES,
    hip_flexion_indices,
    hlp_channels,
    lumbar_indices,
    quat_xyzw_to_matrix,
    torso_rotation,
    trunk_pitch_roll,
)

DOF_NAMES = [
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r", "knee_angle_r", "ankle_angle_r",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l", "knee_angle_l", "ankle_angle_l",
    "lumbar_extension", "lumbar_bending", "lumbar_rotation",
]


def _quat_about(axis, angle):
    ax = torch.tensor(axis, dtype=torch.float32)
    ax = ax / ax.norm()
    s = math.sin(angle / 2)
    return torch.tensor([ax[0] * s, ax[1] * s, ax[2] * s, math.cos(angle / 2)])


def test_quat_to_matrix_identity_and_rotation():
    eye = quat_xyzw_to_matrix(torch.tensor([[0.0, 0.0, 0.0, 1.0]]))
    assert torch.allclose(eye[0], torch.eye(3), atol=1e-6)
    # +90° about z maps x → y
    R = quat_xyzw_to_matrix(_quat_about((0, 0, 1), math.pi / 2)[None])[0]
    assert torch.allclose(R @ torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)


def test_trunk_pitch_roll_signs():
    upright = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    zero_lumbar = torch.zeros(1, 3)
    pr = trunk_pitch_roll(torso_rotation(upright, zero_lumbar))
    assert torch.allclose(pr, torch.zeros(1, 2), atol=1e-6)

    # 앞으로 숙임(+pitch): skeleton 규약 x-앞, y-왼, z-위 → +y축 회전이 몸통 위끝을 +x로 보낸다
    lean = _quat_about((0, 1, 0), math.radians(20))[None]
    pr = trunk_pitch_roll(torso_rotation(lean, zero_lumbar))
    assert pr[0, 0].item() == pytest.approx(math.radians(20), abs=1e-5)
    assert abs(pr[0, 1].item()) < 1e-5

    # 오른쪽 기울임(+roll): +x축 회전
    tilt = _quat_about((1, 0, 0), math.radians(10))[None]
    pr = trunk_pitch_roll(torso_rotation(tilt, zero_lumbar))
    assert pr[0, 1].item() == pytest.approx(math.radians(10), abs=1e-5)

    # lumbar_extension 힌지축 (0,−1,0): 양의 각 = 뒤로 젖힘(신전) → pitch 감소
    pr = trunk_pitch_roll(torso_rotation(upright, torch.tensor([[math.radians(15), 0.0, 0.0]])))
    assert pr[0, 0].item() == pytest.approx(-math.radians(15), abs=1e-5)
    assert SUIT_LUMBAR_AXES[0] == (0.0, -1.0, 0.0)


def test_hlp_channels_layout():
    dof = torch.zeros(2, len(DOF_NAMES))
    dof[:, 0] = 0.3  # hip_flexion_r
    dof[:, 5] = -0.1  # hip_flexion_l
    pelvis = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(2, -1)
    ch = hlp_channels(dof, pelvis, hip_flexion_indices(DOF_NAMES), lumbar_indices(DOF_NAMES))
    assert ch.shape == (2, 4)
    assert torch.allclose(ch[:, 0], torch.full((2,), 0.3))
    assert torch.allclose(ch[:, 1], torch.full((2,), -0.1))
    assert torch.allclose(ch[:, 2:], torch.zeros(2, 2), atol=1e-6)


def test_chunk_interpolation_matches_env_rule():
    """AssistController._interp_chunk를 인스턴스 없이 검증 (같은 코드 경로를 흉내)."""
    from protomotions.assist.controller import AssistController

    class _Stub:
        num_knots = 7
        hlp_dt = 0.025
        layout = type("L", (), {"num_dofs": 2})()
        _chunk = torch.arange(7, dtype=torch.float32).view(1, 7, 1).expand(1, 7, 2).clone()
        _chunk_t0 = torch.zeros(1)

    tt = torch.tensor([[0.0, 0.0125, 0.05, 0.15, 0.2]])  # 범위 밖은 마지막 knot로 ZOH
    out = AssistController._interp_chunk(_Stub(), tt)
    expected = torch.tensor([0.0, 0.5, 2.0, 6.0, 6.0])
    assert torch.allclose(out[0, :, 0], expected, atol=1e-6)
    assert torch.allclose(out[0, :, 1], expected, atol=1e-6)
