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
"""AI-SBC LLP(assist torque RL) 보상 커널.

핵심 목표 (모델 설명 자료 10p): τ_assist는 사람(agent)의 에너지
E = ∫|τ_agent·θ̇| dt 를 최소화하는 방향으로 학습하되, 목표 궤적 추종이
깨지지 않아야 한다 (추종 보상이 "사람을 멈춰 세워 파워 0" 같은 퇴화 해를
차단).
"""

import torch
from torch import Tensor


def compute_human_power_rew(
    tau_agent: Tensor,
    dof_vel: Tensor,
    use_torque_squared: bool = False,
) -> Tensor:
    """사람 기계적 파워 |τ_agent·θ̇| (음수 weight로 사용).

    Args:
        tau_agent: 사람 PD 토크 계측 [num_envs, num_dofs].
        dof_vel: 관절 각속도 [num_envs, num_dofs].
        use_torque_squared: True면 τ² (근활성 프록시)로 대체.

    Returns:
        [num_envs] 파워 합 (항상 >= 0).
    """
    if use_torque_squared:
        return (tau_agent * tau_agent).sum(dim=-1)
    return (tau_agent * dof_vel).abs().sum(dim=-1)


def compute_target_tracking_rew(
    dof_pos: Tensor,
    theta_g: Tensor,
    tracking_coef: float = 40.0,
) -> Tensor:
    """의도 궤적 추종 보상 exp(-k * mean(err²)).

    Args:
        dof_pos: 현재 관절 각 [num_envs, num_dofs].
        theta_g: 의도 궤적 목표 [num_envs, num_dofs].
        tracking_coef: 오차 감쇠 계수 k.

    Returns:
        [num_envs] (0, 1] 보상.
    """
    err2 = (dof_pos - theta_g).square().mean(dim=-1)
    return torch.exp(-tracking_coef * err2)


def compute_assist_effort_rew(tau_assist: Tensor) -> Tensor:
    """assist 토크 사용량 Στ² (음수 weight로 사용 — 배터리/현실성 정칙화).

    Args:
        tau_assist: 인가된 assist 토크 [num_envs, num_dofs].

    Returns:
        [num_envs] 토크 제곱 합.
    """
    return (tau_assist * tau_assist).sum(dim=-1)


# =============================================================================
# 평가 지표 커널 (BaseEvaluator evaluation_components용)
# =============================================================================


def compute_tracking_error_eval(dof_pos: Tensor, theta_g: Tensor) -> Tensor:
    """평균 추종 오차 |θ - θ_g| (rad). threshold/fail_above와 함께 사용.

    Args:
        dof_pos: 현재 관절 각 [num_envs, num_dofs].
        theta_g: 의도 궤적 목표 [num_envs, num_dofs].

    Returns:
        [num_envs] 평균 절대 오차.
    """
    return (dof_pos - theta_g).abs().mean(dim=-1)


def compute_human_power_eval(tau_agent: Tensor, dof_vel: Tensor) -> Tensor:
    """사람 기계적 파워 지표 (W). 리포팅용 — threshold는 매우 크게 설정.

    Args:
        tau_agent: 사람 PD 토크 계측 [num_envs, num_dofs].
        dof_vel: 관절 각속도 [num_envs, num_dofs].

    Returns:
        [num_envs] 파워 합.
    """
    return (tau_agent * dof_vel).abs().sum(dim=-1)
