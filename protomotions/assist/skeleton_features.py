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
"""스켈레톤(suit) 상태 → HLP 관측 4ch 규약 — 학습 데이터 추출기와 동일 계산.

HLP(ManiFlow flexion)는 `data/scripts/extract_soma23_hip_imu_zarr.py
--features flexion_trunk`가 만든 `[hip_flexion_r, hip_flexion_l, trunk_pitch,
trunk_roll]`(rad)로 학습됐다. suit 데이터 경로에서

- hip flexion = MJCF hinge 각도 그대로 (+ = 굴곡)
- trunk = pelvis 자세 × lumbar 3힌지 FK로 torso 프레임을 만든 뒤, 그 프레임에서
  본 중력 방향으로 pitch(앞숙임 +)/roll(오른쪽 기울임 +)

이 모듈은 같은 계산을 시뮬레이터 상태(pelvis quat xyzw + dof_pos)에 적용한다.
실기기에서는 몸통 IMU 융합각이 이 자리에 들어온다.

영점 보정: SEED와 ETRI suit 규약은 채널별 상수만큼 다르다(프레임워크 문서
§3.1). 온라인에서는 재생 중인 클립의 채널 평균을 SEED 학습 데이터 평균에
맞추는 오프셋(`calibration_offset_from_motion`)을 HLP 관측에 가산한다 —
Phase C 평가(eval_abc `--align-ref-zarr`)와 같은 정렬 방식.
"""

from typing import Sequence, Tuple

import torch
from torch import Tensor

# extract_soma23_hip_imu_zarr.py의 SUIT_LUMBAR_AXES (skeleton_torque_suit.xml 선언 순서)
SUIT_LUMBAR_AXES = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
LUMBAR_DOF_NAMES = ("lumbar_extension", "lumbar_bending", "lumbar_rotation")
HIP_FLEXION_DOF_NAMES = ("hip_flexion_r", "hip_flexion_l")
HLP_CHANNELS = ("hip_flexion_r", "hip_flexion_l", "trunk_pitch", "trunk_roll")

# HLP 학습 데이터(hip-flexion-seed-locomotion-40fps.zarr, 20,657,709프레임) 채널 평균
# [deg] — 2026-09-12 전체 스트리밍 평균. `--seed-zarr`로 재계산 가능.
SEED_CHANNEL_MEAN_DEG = (10.41, 10.42, 7.39, -0.58)


def hip_flexion_indices(dof_names: Sequence[str]) -> Tuple[int, int]:
    return tuple(list(dof_names).index(n) for n in HIP_FLEXION_DOF_NAMES)


def lumbar_indices(dof_names: Sequence[str]) -> Tuple[int, int, int]:
    return tuple(list(dof_names).index(n) for n in LUMBAR_DOF_NAMES)


def quat_xyzw_to_matrix(q: Tensor) -> Tensor:
    """(…, 4) xyzw 단위 쿼터니언 → (…, 3, 3) 회전행렬."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z, w = q.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    row0 = torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1)
    row1 = torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1)
    row2 = torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _axis_rot(axis, theta: Tensor) -> Tensor:
    """고정축 회전행렬 배치 (N, 3, 3) — Rodrigues (추출기 `_axis_rot`와 동일)."""
    a = torch.tensor(axis, dtype=theta.dtype, device=theta.device)
    K = torch.tensor(
        [[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]],
        dtype=theta.dtype,
        device=theta.device,
    )
    s = torch.sin(theta)[:, None, None]
    c = torch.cos(theta)[:, None, None]
    eye = torch.eye(3, dtype=theta.dtype, device=theta.device).expand(theta.shape[0], 3, 3)
    return eye + s * K + (1 - c) * (K @ K)


def torso_rotation(pelvis_quat_xyzw: Tensor, lumbar: Tensor) -> Tensor:
    """pelvis 월드 자세 × lumbar 3힌지 → torso 월드 회전 (N, 3, 3)."""
    R = quat_xyzw_to_matrix(pelvis_quat_xyzw.float())
    for ax, th in zip(SUIT_LUMBAR_AXES, lumbar.float().unbind(-1)):
        R = R @ _axis_rot(ax, th)
    return R


def trunk_pitch_roll(R_torso: Tensor) -> Tensor:
    """torso 프레임에서 본 중력 → (pitch, roll) [rad]. 앞숙임 +, 오른쪽 기울임 +."""
    down = torch.tensor([0.0, 0.0, -1.0], dtype=R_torso.dtype, device=R_torso.device)
    g = torch.einsum("nij,j->ni", R_torso.transpose(-1, -2), down)
    pitch = torch.atan2(g[:, 0], -g[:, 2])
    roll = torch.atan2(-g[:, 1], -g[:, 2])
    return torch.stack([pitch, roll], dim=-1)


def hlp_channels(
    dof_pos: Tensor,
    pelvis_quat_xyzw: Tensor,
    hip_idx: Sequence[int],
    lumbar_idx: Sequence[int],
) -> Tensor:
    """(N, D) dof_pos + (N, 4) pelvis quat → HLP 관측 4ch (N, 4) [rad]."""
    flex = dof_pos[:, list(hip_idx)].float()
    trunk = trunk_pitch_roll(torso_rotation(pelvis_quat_xyzw, dof_pos[:, list(lumbar_idx)]))
    return torch.cat([flex, trunk], dim=-1)


@torch.no_grad()
def motion_channel_mean(
    motion_file: str,
    dof_names: Sequence[str],
    fixed_root: bool = False,
    motion_index: int = 0,
) -> Tensor:
    """mocap 클립 전체의 HLP 4ch 평균 [rad] — 런타임과 같은 함수로 계산.

    Args:
        motion_file: ProtoMotions motion .pt (dps / grs 포함).
        fixed_root: True면 pelvis 자세를 identity로 둔다 (골반 고정 대역은 루트가
            직립 고정이라 trunk pitch에 mocap 골반 기울기가 들어오지 않는다).
        motion_index: 여러 클립 파일이면 몇 번째 클립인지.
    """
    m = torch.load(motion_file, map_location="cpu", weights_only=False)
    dps, grs = m["dps"].float(), m["grs"].float()
    starts = m["length_starts"]
    n_frames = m["motion_num_frames"]
    s = int(starts[motion_index])
    e = s + int(n_frames[motion_index])
    dps, grs = dps[s:e], grs[s:e]
    pelvis = grs[:, 0]
    if fixed_root:
        pelvis = torch.zeros_like(pelvis)
        pelvis[:, 3] = 1.0
    ch = hlp_channels(dps, pelvis, hip_flexion_indices(dof_names), lumbar_indices(dof_names))
    return ch.mean(dim=0)


def calibration_offset_from_motion(
    motion_file: str,
    dof_names: Sequence[str],
    fixed_root: bool = False,
    seed_mean_deg: Sequence[float] = SEED_CHANNEL_MEAN_DEG,
    motion_index: int = 0,
) -> Tensor:
    """HLP 관측 영점 오프셋 [rad] = SEED 평균 − 클립 평균 (관측에 가산)."""
    seed_mean = torch.deg2rad(torch.tensor(seed_mean_deg, dtype=torch.float32))
    return seed_mean - motion_channel_mean(motion_file, dof_names, fixed_root, motion_index)
