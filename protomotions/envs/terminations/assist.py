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
"""AI-SBC LLP(assist torque RL) 종료 조건 커널.

주의: 파라미터 이름에 "threshold"를 쓰지 않는다 — MdpComponent의 metadata
키(evaluation용)와 겹쳐 compute 함수에 전달되기 전에 걸러지기 때문.
"""

from torch import Tensor


def compute_tracking_failure_term(
    dof_pos: Tensor,
    theta_g: Tensor,
    max_err: float = 1.0,
) -> Tensor:
    """추종 실패 종료: 관절 오차 최대값이 max_err 초과.

    assist가 사람 PD를 이겨서 궤적을 크게 이탈시키는 (또는 발산하는) 경우를
    조기 종료한다.

    Args:
        dof_pos: 현재 관절 각 [num_envs, num_dofs].
        theta_g: 의도 궤적 목표 [num_envs, num_dofs].
        max_err: 허용 최대 오차 (rad).

    Returns:
        [num_envs] bool — True면 종료.
    """
    return (dof_pos - theta_g).abs().amax(dim=-1) > max_err


def compute_dof_vel_failure_term(
    dof_vel: Tensor,
    max_vel: float = 25.0,
) -> Tensor:
    """과속 종료: 관절 각속도 최대값이 max_vel 초과 (발산/공진 방지).

    Args:
        dof_vel: 관절 각속도 [num_envs, num_dofs].
        max_vel: 허용 최대 각속도 (rad/s).

    Returns:
        [num_envs] bool — True면 종료.
    """
    return dof_vel.abs().amax(dim=-1) > max_vel
