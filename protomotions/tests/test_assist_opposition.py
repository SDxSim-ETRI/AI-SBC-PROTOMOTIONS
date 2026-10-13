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
"""Coordination cost must preserve joint-local opposition and finite gradients."""

import torch

from protomotions.envs.rewards.assist import (
    compute_assist_opposition_cost,
    compute_relative_assist_opposition_cost,
)


def test_opposite_hip_does_not_cancel_helpful_hip():
    wearer = torch.tensor([[10., 10.], [-10., 10.], [0., 10.]])
    assist = torch.tensor([[-5., 5.], [-5., 5.], [100., 0.]])
    torch.testing.assert_close(compute_assist_opposition_cost(wearer, assist), torch.tensor([50., 0., 0.]))
    torch.testing.assert_close(compute_assist_opposition_cost(-wearer, -assist), torch.tensor([50., 0., 0.]))


def test_more_opposition_costs_more_without_nonfinite_gradient():
    wearer = torch.tensor([[20., -20.]])
    assist = torch.tensor([[-3., 3.]], requires_grad=True)
    cost = compute_assist_opposition_cost(wearer, assist)
    torch.testing.assert_close(cost, torch.tensor([120.]))
    torch.testing.assert_close(compute_assist_opposition_cost(wearer, 2*assist), 2*cost)
    cost.sum().backward()
    torch.testing.assert_close(assist.grad, torch.tensor([[-20., 20.]]))


def test_relative_cost_preserves_local_sign_and_limits_scale_dominance():
    wearer = torch.tensor([[10., 10.], [100., 100.]])
    assist = torch.tensor([[-10., 10.], [-100., 100.]])
    raw = compute_assist_opposition_cost(wearer, assist)
    relative = compute_relative_assist_opposition_cost(wearer, assist)
    assert raw[1] / raw[0] == 100
    assert 1 < relative[1] / relative[0] < 10
    torch.testing.assert_close(relative, compute_relative_assist_opposition_cost(-wearer, -assist))


def test_relative_cost_cannot_shrink_by_increasing_opposing_assistance():
    wearer = torch.tensor([[0., 10., -10.]])
    assist = torch.tensor([[100., -5., 5.]], requires_grad=True)
    cost = compute_relative_assist_opposition_cost(wearer, assist)
    assert (compute_relative_assist_opposition_cost(wearer, 10*assist) > cost).all()
    assert (compute_relative_assist_opposition_cost(10*wearer, assist) > cost).all()
    cost.sum().backward()
    assert torch.isfinite(assist.grad).all()
    assert assist.grad[0, 0] == 0
    assert assist.grad[0, 1] < 0 < assist.grad[0, 2]
