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

from typing import Optional, Sequence

import torch
from torch import Tensor


def _joint_weights(like: Tensor, joint_mask: Sequence[float]) -> Optional[Tensor]:
    """관절 마스크 (num_dofs,) 텐서. 비어 있으면 None (전 관절)."""
    if not joint_mask:
        return None
    w = torch.as_tensor(tuple(joint_mask), dtype=like.dtype, device=like.device)
    if w.numel() != like.shape[-1]:
        raise ValueError(f"joint_mask has {w.numel()} entries for {like.shape[-1]} joints")
    return w


def compute_human_power_rew(
    tau_agent: Tensor,
    dof_vel: Tensor,
    use_torque_squared: bool = False,
    joint_mask: Sequence[float] = (),
) -> Tensor:
    """사람 기계적 파워 |τ_agent·θ̇| (음수 weight로 사용).

    Args:
        tau_agent: 사람 PD 토크 계측 [num_envs, num_dofs].
        dof_vel: 관절 각속도 [num_envs, num_dofs].
        use_torque_squared: True면 τ² (근활성 프록시)로 대체.
        joint_mask: 관절별 0/1 (비면 전 관절). 다리별 학습에서 담당 관절만 센다.

    Returns:
        [num_envs] 파워 합 (항상 >= 0).
    """
    value = tau_agent * tau_agent if use_torque_squared else (tau_agent * dof_vel).abs()
    w = _joint_weights(value, joint_mask)
    return value.sum(dim=-1) if w is None else (value * w).sum(dim=-1)


def compute_assist_opposition_cost(tau_agent: Tensor, tau_assist: Tensor) -> Tensor:
    """Penalize opposing wearer/device torques within the same control window.

    Per-joint rectification prevents useful assistance on one hip from cancelling
    opposition on the other. The output has torque-squared units; use a negative
    reward weight. Reward-only privileged torques never enter actor observations.

    This is a coordination proxy, NOT a counterfactual no-assistance comparison.
    It can also penalize transient opposition needed during a movement change;
    tracking and held-out evaluation must decide whether it helps in practice.
    """
    return torch.clamp(-tau_agent * tau_assist, min=0.0).sum(dim=-1)


def compute_relative_assist_opposition_cost(
    tau_agent: Tensor, tau_assist: Tensor, torque_floor: float = 10.0
) -> Tensor:
    """Dimensionless log cost reduces domination by rare large-torque samples.

    A fixed torque scale regularizes zero without putting either actor's torque
    in the denominator. The cost must never shrink when opposing wearer OR
    device torque grows. Log compression keeps that monotonicity while giving
    moderate repeated opposition more relative weight than a quadratic product.
    This remains a coordination proxy, not a counterfactual burden measurement.
    """
    if not 0.0 < torque_floor < float("inf"):
        raise ValueError("torque_floor must be finite and positive")
    opposition = torch.clamp(-tau_agent * tau_assist, min=0.0)
    return torch.log1p(opposition / torque_floor**2).sum(dim=-1)


def compute_target_tracking_rew(
    dof_pos: Tensor,
    theta_g: Tensor,
    tracking_coef: float = 40.0,
    joint_mask: Sequence[float] = (),
) -> Tensor:
    """의도 궤적 추종 보상 exp(-k * mean(err²)).

    Args:
        dof_pos: 현재 관절 각 [num_envs, num_dofs].
        theta_g: 의도 궤적 목표 [num_envs, num_dofs].
        tracking_coef: 오차 감쇠 계수 k.
        joint_mask: 관절별 0/1 (비면 전 관절) — 평균을 담당 관절에서만 낸다.

    Returns:
        [num_envs] (0, 1] 보상.
    """
    sq = (dof_pos - theta_g).square()
    w = _joint_weights(sq, joint_mask)
    err2 = sq.mean(dim=-1) if w is None else (sq * w).sum(dim=-1) / w.sum()
    return torch.exp(-tracking_coef * err2)


def compute_assist_effort_rew(tau_assist: Tensor, joint_mask: Sequence[float] = ()) -> Tensor:
    """assist 토크 사용량 Στ² (음수 weight로 사용 — 배터리/현실성 정칙화).

    Args:
        tau_assist: 인가된 assist 토크 [num_envs, num_dofs].
        joint_mask: 관절별 0/1 (비면 전 관절).

    Returns:
        [num_envs] 토크 제곱 합.
    """
    sq = tau_assist * tau_assist
    w = _joint_weights(sq, joint_mask)
    return sq.sum(dim=-1) if w is None else (sq * w).sum(dim=-1)


def compute_paired_tracking_rew(
    dof_pos: Tensor,
    reference_dof_pos: Tensor,
    theta_g: Tensor,
    tracking_coef: float = 40.0,
) -> Tensor:
    """Keep tracking no worse than a separately integrated unassisted wearer.

    Rectify each joint before averaging: improvement of one hip cannot pay for
    worsening the other. Beating the reference earns no additional reward.
    The reference is privileged, reward-only data; it is NOT the assisted state
    with the instantaneous assistance subtracted from its torque.
    """
    excess = ((dof_pos - theta_g).square()
              - (reference_dof_pos.detach() - theta_g).square()).clamp_min(0.0)
    return torch.exp(-tracking_coef * excess.mean(dim=-1))


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

def compute_body_error_hinge_rew(
    current_rigid_body_pos: Tensor,
    ref_rigid_body_pos: Tensor,
    margin: float = 0.09,
) -> Tensor:
    """착용자 몸체 추종 오차가 무보조 수준(margin)을 넘는 만큼만 벌점 (보행 착용자 LLP용).

    보조가 착용자의 보행 과제를 **깨뜨리지 않게** 하되, 추종을 "더 좋게" 만드는 데
    보상을 주지는 않는다 — exp 형 추종 보상을 그대로 쓰면 정책이 최대 토크로 착용자를
    mocap 쪽으로 끌어당기는 해(파워 절감 없음)로 수렴했다 (llp_skel_walk_v1, 2026-09-12).

    Args:
        current_rigid_body_pos: [num_envs, num_bodies, 3].
        ref_rigid_body_pos: 레퍼런스 몸체 위치 [num_envs, num_bodies, 3].
        margin: 벌점이 시작되는 평균 몸체 오차 [m] (단독 tracker ≈ 0.08).

    Returns:
        [num_envs] ≤ 0. weight 곱은 MdpComponent가 한다.
    """
    err = (current_rigid_body_pos - ref_rigid_body_pos).norm(dim=-1).mean(dim=-1)
    return -(err - margin).clamp(min=0.0)
