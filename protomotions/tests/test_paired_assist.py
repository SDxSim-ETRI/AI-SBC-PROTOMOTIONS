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
"""Causal-reference reward, common RNG and truncation contracts for L12."""

import random

import numpy as np
import pytest
import torch

from protomotions.envs.base_env.paired_assist_env import (
    RandomState, combine_pair_endings, paired_call,
)
from protomotions.envs.rewards.assist import compute_paired_tracking_rew


def test_tracking_no_reward_for_beating_reference():
    target = torch.zeros(3, 2)
    reference = torch.full((3, 2), 0.2)
    assisted = torch.tensor([[0.2, -0.2], [0.1, -0.1], [0.0, 0.0]])
    torch.testing.assert_close(compute_paired_tracking_rew(assisted, reference, target), torch.ones(3))


def test_each_hip_cannot_hide_other_hip_degradation():
    reference = torch.tensor([[0.3, 0.0]])
    assisted = torch.tensor([[0.0, 0.2]], requires_grad=True)
    target = torch.zeros_like(assisted)
    reward = compute_paired_tracking_rew(assisted, reference, target)
    torch.testing.assert_close(reward, torch.exp(torch.tensor([-40 * 0.04 / 2])))
    reward.sum().backward()
    assert torch.isfinite(assisted.grad).all()
    assert assisted.grad[0, 0] == 0
    assert assisted.grad[0, 1] < 0
    assert compute_paired_tracking_rew(assisted * 2, reference, target) < reward


def test_reference_is_not_differentiated():
    a = torch.tensor([[0.1, 0.1]], requires_grad=True)
    b = torch.tensor([[0.2, 0.2]], requires_grad=True)
    compute_paired_tracking_rew(b, a, torch.zeros_like(b)).sum().backward()
    assert a.grad is None
    assert torch.isfinite(b.grad).all()


def test_rng_replay_advances_global_stream_once():
    def draw():
        return random.random(), np.random.rand(3), torch.rand(2)
    before = RandomState.capture('cpu')
    expected = draw()
    after = RandomState.capture('cpu')
    before.restore()
    a, b = paired_call(draw, draw, 'cpu')
    assert RandomState.capture('cpu').matches(after)
    for out in (a, b):
        assert out[0] == expected[0]
        np.testing.assert_array_equal(out[1], expected[1])
        torch.testing.assert_close(out[2], expected[2], rtol=0, atol=0)


def test_pairing_fails_if_random_consumption_diverges():
    with pytest.raises(RuntimeError, match='different random numbers'):
        paired_call(lambda: torch.rand(1), lambda: torch.rand(2), 'cpu')


def test_reference_failure_bootstraps_b_and_never_erases_b_failure():
    b_done = torch.tensor([False, True, True, False])
    b_term = torch.tensor([False, True, False, False])
    a_done = torch.tensor([True, True, False, False])
    done, term = combine_pair_endings(b_done, b_term, a_done)
    assert done.tolist() == [True, True, True, False]
    assert term.tolist() == [False, True, False, False]
    # Mirrors PPO's next_value * (1-terminated), independent of the GAE reset.
    next_value = torch.full((4,), 7.) * (1 - term.float())
    assert next_value.tolist() == [7., 0., 7., 7.]


def test_l12_preserves_model_inputs_and_all_other_rewards():
    import argparse
    import dataclasses
    from examples.experiments.assist_pendulum import mlp_mixed_feedforward as old
    from examples.experiments.assist_pendulum import mlp_paired_tracking as new
    from protomotions.robot_configs.hip_pendulum import HipPendulumRobotConfig
    from protomotions.envs.mdp_component import MdpComponent

    parser = argparse.ArgumentParser()
    old.additional_experiment_arguments(parser)
    args = parser.parse_args(["--opposition-weight=-0.021958557930324075"])
    robot = HipPendulumRobotConfig()
    a, b = old.env_config(robot, args), new.env_config(robot, args)
    def plain(x):
        if isinstance(x, MdpComponent):
            return {k: plain(getattr(x, k)) for k in ("compute_func", "dynamic_vars", "static_params")}
        if dataclasses.is_dataclass(x):
            return {f.name: plain(getattr(x, f.name)) for f in dataclasses.fields(x)}
        if isinstance(x, dict):
            return {k: plain(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [plain(v) for v in x]
        return x
    assert plain(a.observation_components) == plain(b.observation_components)
    for key in a.reward_components:
        if key != "tracking":
            assert plain(a.reward_components[key]) == plain(b.reward_components[key])
    assert a.reward_components["tracking"].static_params == b.reward_components["tracking"].static_params
    b._target_ = a._target_
    b.reward_components["tracking"] = a.reward_components["tracking"]
    assert plain(a) == plain(b)
