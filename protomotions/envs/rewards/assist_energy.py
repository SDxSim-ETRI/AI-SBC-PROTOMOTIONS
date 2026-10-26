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
"""Optional mechanical absorption cost, without phase labels or action gates."""
import torch
from torch import Tensor


def compute_assist_absorbed_power(tau_assist: Tensor, dof_vel: Tensor) -> Tensor:
    """Sum max(-tau*omega, 0) [W] per joint; negative reward weight.

    Same post-step velocity convention as the existing human power reward.
    This is a preference, not a guarantee of useful assistance: braking can be
    necessary, so held-out tracking and stopping behavior must also be assessed.
    """
    return torch.clamp(-tau_assist*dof_vel, min=0.).sum(dim=-1)


def compute_paired_excess_burden(
    tau_agent: Tensor,
    reference_tau_agent: Tensor,
    reference_tau_sq_window: Tensor,
    floor: float = 10.0,
) -> Tensor:
    """Scale-free per-joint harm against an unassisted twin; negative reward weight.

    excess_j = relu(tau_B_j^2 - tau_A_j^2) / (mean_window(tau_A_j^2) + floor^2),
    averaged over joints. A is the independently integrated unassisted wearer
    (same intent, plant and exogenous loads), B the assisted wearer of the same
    step. Reduction below A earns nothing here; the absolute wearer costs keep
    that incentive. The causal window and torque floor bound the ratio when the
    unassisted need crosses zero. Reward-only: A never enters actor inputs.
    The experiment file validates floor > 0; the kernel stays compile-friendly.
    """
    harm = torch.clamp(tau_agent.square() - reference_tau_agent.detach().square(), min=0.0)
    return (harm / (reference_tau_sq_window.detach() + floor * floor)).mean(dim=-1)
