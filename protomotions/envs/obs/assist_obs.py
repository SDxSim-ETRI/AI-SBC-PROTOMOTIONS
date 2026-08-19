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
"""AI-SBC LLP(assist torque RL) 관측 커널.

actor 관측은 실기기(엑소슈트)에서 측정 가능한 값만 사용:
  * 관절 각/각속도 + 히스토리 (encoder/IMU)
  * action 히스토리 (자체 이력)
  * HLP 예측 chunk의 interpolation 창 (통신으로 수신)

critic 전용 privileged 관측은 시뮬레이션에서만 아는 값(사람 토크, 깨끗한
의도 궤적, 사람 게인 배율)을 포함한다.
"""

import torch
from torch import Tensor


def compute_assist_proprio_obs(
    dof_pos: Tensor,
    dof_vel: Tensor,
    historical_dof_pos: Tensor,
    historical_dof_vel: Tensor,
    dof_vel_scale: float = 0.25,
) -> Tensor:
    """관절 상태 + 히스토리 관측.

    Args:
        dof_pos: 현재 관절 각 [num_envs, num_dofs].
        dof_vel: 현재 관절 각속도 [num_envs, num_dofs].
        historical_dof_pos: 관절 각 히스토리 [num_envs, H, num_dofs].
        historical_dof_vel: 관절 각속도 히스토리 [num_envs, H, num_dofs].
        dof_vel_scale: 각속도 스케일 (각도와 크기 정렬용).

    Returns:
        [num_envs, num_dofs * (1 + H) * 2]
    """
    num_envs = dof_pos.shape[0]
    return torch.cat(
        [
            dof_pos,
            dof_vel * dof_vel_scale,
            historical_dof_pos.reshape(num_envs, -1),
            (historical_dof_vel * dof_vel_scale).reshape(num_envs, -1),
        ],
        dim=-1,
    )


def compute_assist_target_obs(
    theta_d_future: Tensor,
    dof_pos: Tensor,
    chunk_age: Tensor,
) -> Tensor:
    """HLP 목표 창 관측 (절대값 + 현재 관절 대비 상대값 + chunk 경과시간).

    Args:
        theta_d_future: interpolation된 미래 목표 [num_envs, S, num_dofs].
        dof_pos: 현재 관절 각 [num_envs, num_dofs].
        chunk_age: 마지막 chunk 갱신 후 경과 시간 [num_envs, 1].

    Returns:
        [num_envs, S * num_dofs * 2 + 1]
    """
    num_envs = dof_pos.shape[0]
    rel = theta_d_future - dof_pos.unsqueeze(1)
    return torch.cat(
        [
            theta_d_future.reshape(num_envs, -1),
            rel.reshape(num_envs, -1),
            chunk_age,
        ],
        dim=-1,
    )


def compute_assist_privileged_obs(
    tau_agent: Tensor,
    tau_assist: Tensor,
    theta_g: Tensor,
    dof_pos: Tensor,
    gain_scale: Tensor,
    torque_scale: float = 0.01,
) -> Tensor:
    """critic 전용 privileged 관측 (실기기 측정 불가 값들).

    Args:
        tau_agent: 사람 PD 토크 계측 [num_envs, num_dofs].
        tau_assist: 인가된 assist 토크 [num_envs, num_dofs].
        theta_g: 깨끗한 의도 궤적 목표 [num_envs, num_dofs].
        dof_pos: 현재 관절 각 [num_envs, num_dofs].
        gain_scale: env별 사람 게인 배율 [num_envs, num_dofs].
        torque_scale: 토크 정규화 스케일 (1/N·m).

    Returns:
        [num_envs, num_dofs * 4]
    """
    return torch.cat(
        [
            tau_agent * torque_scale,
            tau_assist * torque_scale,
            theta_g - dof_pos,
            gain_scale,
        ],
        dim=-1,
    )
