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
"""L13 policy transfer and paired objective invariants."""

import torch

from protomotions.agents.ppo.assist_finetune import mirror_actor_state
from protomotions.envs.base_env.paired_skeleton_assist_env import paired_burden_terms


def test_mirrored_policy_is_same_physical_controller():
    torch.manual_seed(5)
    n = 13
    signs = -torch.ones(n)
    signs[10] = 1  # chunk age
    state = {
        "mu.norm.running_obs_norm.mean": torch.randn(n),
        "mu.norm.running_obs_norm.var": torch.rand(n) + .2,
        "mu.mlp.0.weight": torch.randn(7, n),
        "mu.mlp.0.bias": torch.randn(7),
        "mu.mlp.2.weight": torch.randn(2, 7),
        "mu.mlp.2.bias": torch.randn(2),
        "logstd": torch.full((2,), -2.),
    }
    copied = {k: v.clone() for k, v in state.items()}
    converted = mirror_actor_state(state, signs)
    def evaluate(s, x):
        x = ((x - s["mu.norm.running_obs_norm.mean"]) / s["mu.norm.running_obs_norm.var"].sqrt()).clamp(-5, 5)
        x = (x @ s["mu.mlp.0.weight"].T + s["mu.mlp.0.bias"]).relu()
        return x @ s["mu.mlp.2.weight"].T + s["mu.mlp.2.bias"]
    x = torch.randn(123, n) * 5
    torch.testing.assert_close(evaluate(converted, x), -evaluate(state, x * signs))
    for k in state:
        torch.testing.assert_close(state[k], copied[k], rtol=0, atol=0)


def test_local_harm_cannot_cancel_between_hips():
    a = torch.tensor([[10., 10.]])
    b = torch.tensor([[0., 14.]])
    reduction, excess = paired_burden_terms(a, b, 100.)
    assert reduction.item() > 0  # net improvement hides the harmed hip
    assert (reduction - excess).item() < 0
    zero, harm = paired_burden_terms(a, a, 100.)
    assert zero.item() == harm.item() == 0


def test_wrong_sign_assistance_is_not_rewarded_as_work_saving():
    # Squared wearer torque cannot be reduced by just reversing its sign.
    a = torch.tensor([[10., -20.]])
    reduction, excess = paired_burden_terms(a, -a, 100.)
    assert reduction.item() == excess.item() == 0


def test_relative_harm_protects_low_burden_with_finite_zero_reference():
    a = torch.tensor([[0.], [10.], [50.]])
    b = (a.square() + 100).sqrt()
    reduction, excess = paired_burden_terms(a, b, 1600., floor=10.)
    torch.testing.assert_close(reduction, torch.full((3,), -100 / 1600))
    assert torch.isfinite(excess).all()
    assert excess[0] > excess[1] > excess[2]
