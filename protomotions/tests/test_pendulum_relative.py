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
"""Paired relative burden cost: kernel properties, twin window buffer and config contrast."""
import argparse
import sys
from dataclasses import fields
from pathlib import Path

import pytest
import torch

from protomotions.envs.base_env.paired_assist_env import PairedAssistEnv
from protomotions.envs.rewards.assist_energy import compute_paired_excess_burden
from protomotions.robot_configs.hip_pendulum import HipPendulumRobotConfig

EXPERIMENTS = Path(__file__).resolve().parents[2] / "examples/experiments/assist_pendulum"


def test_excess_is_zero_without_harm_and_rectified_per_joint():
    a = torch.tensor([[10., 10.], [0., 30.]])
    assert torch.equal(compute_paired_excess_burden(a, a, a.square()), torch.zeros(2))
    better = torch.tensor([[5., 5.], [0., 10.]])
    assert torch.equal(compute_paired_excess_burden(better, a, a.square()), torch.zeros(2))
    mixed = torch.tensor([[0., 20.]])  # one hip helped, the other harmed
    cost = compute_paired_excess_burden(mixed, a[:1], a[:1].square(), floor=10.)
    expected = 0.5 * (400. - 100.) / (100. + 100.)
    torch.testing.assert_close(cost, torch.tensor([expected]))


def test_excess_is_scale_free_and_bounded_by_floor():
    zero = torch.zeros(1, 2)
    small_harm = compute_paired_excess_burden(torch.full((1, 2), 10.), zero, zero, floor=10.)
    torch.testing.assert_close(small_harm, torch.tensor([1.]))
    # The same 10 N·m of extra torque at a 100 N·m need is a tiny relative harm.
    heavy = torch.full((1, 2), 100.)
    heavy_harm = compute_paired_excess_burden(torch.full((1, 2), 110.), heavy, heavy.square(), floor=10.)
    assert heavy_harm.item() < 0.25 * small_harm.item()
    assert torch.isfinite(compute_paired_excess_burden(torch.full((1, 2), 150.), zero, zero)).all()


def test_reference_is_not_differentiated():
    a = torch.tensor([[5., 5.]], requires_grad=True)
    b = torch.tensor([[12., 12.]], requires_grad=True)
    compute_paired_excess_burden(b, a, a.square()).sum().backward()
    assert a.grad is None and torch.isfinite(b.grad).all() and (b.grad > 0).all()


def test_twin_window_is_causal_mean_and_reset_locally():
    env = PairedAssistEnv.__new__(PairedAssistEnv)
    env._reference_window_steps = 5
    env._reference_tau_sq_ring = None
    env._reference_ring_count = None

    class Twin:
        pass
    env.reference_env = Twin()
    env.reference_env.assist_component = Twin()
    values = [1., 2., 3., 4., 5., 6., 7.]
    for k, v in enumerate(values):
        env.reference_env.assist_component._tau_agent = torch.full((3, 2), v)
        env._push_reference_torque()
        window = env.reference_tau_sq_window()
        recent = [x * x for x in values[max(0, k - 4):k + 1]]
        torch.testing.assert_close(window, torch.full((3, 2), sum(recent) / len(recent)))
    env._reference_tau_sq_ring[torch.tensor([1])] = 0.
    env._reference_ring_count[torch.tensor([1])] = 0
    env.reference_env.assist_component._tau_agent = torch.full((3, 2), 2.)
    env._push_reference_torque()
    window = env.reference_tau_sq_window()
    torch.testing.assert_close(window[1], torch.full((2,), 4.))
    assert (window[0] > 4.).all()


def _env_config(module_name, extra):
    sys.path.insert(0, str(EXPERIMENTS))
    module = __import__(module_name)
    parser = argparse.ArgumentParser()
    module.additional_experiment_arguments(parser)
    return module.env_config(HipPendulumRobotConfig(), parser.parse_args(extra))


def _component_signature(component):
    return (component.compute_func.__name__,
            {k: v.path for k, v in component.dynamic_vars.items()}, dict(component.static_params))


def _contrast(a, b, ignore):
    changed = {}
    for f in fields(a):
        if f.name in ignore:
            continue
        va, vb = getattr(a, f.name), getattr(b, f.name)
        if f.name.endswith("_components"):
            va = {k: _component_signature(c) for k, c in va.items()} if f.name != "control_components" else va
            vb = {k: _component_signature(c) for k, c in vb.items()} if f.name != "control_components" else vb
        if va != vb:
            changed[f.name] = (va, vb)
    return changed


def test_r0_differs_from_l1_only_in_torque_path_and_r1_adds_one_reward():
    l1 = _env_config("mlp_v4", [])
    r0 = _env_config("mlp_relative", ["--paired-excess-weight=0.0"])
    r1 = _env_config("mlp_relative", ["--paired-excess-weight=-0.5"])
    assert r0._target_ == l1._target_ and r1._target_.endswith("PairedAssistEnv")
    changed = _contrast(l1, r0, ignore=("paired_reference_window_steps",))
    assert set(changed) == {"clamp_combined_injection", "human_feedforward_inside_pd"}, changed
    assert r0.clamp_combined_injection is False and r0.human_feedforward_inside_pd is True
    changed = _contrast(r0, r1, ignore=("_target_",))
    assert set(changed) == {"reward_components"}, changed
    old, new = (dict(x) for x in changed["reward_components"])
    assert set(new) - set(old) == {"paired_excess"} and all(new[k] == old[k] for k in old)
    name, paths, params = new["paired_excess"]
    assert name == "compute_paired_excess_burden"
    assert paths == {"tau_agent": "assist.tau_agent", "reference_tau_agent": "assist.reference_tau_agent",
                     "reference_tau_sq_window": "assist.reference_tau_sq_window"}
    assert params == {"floor": 10., "weight": -.5, "min_value": -3., "zero_during_grace_period": True}
    assert not any("reference" in p for c in r1.observation_components.values() for p in _component_signature(c)[1].values())
    with pytest.raises(ValueError):
        _env_config("mlp_relative", ["--paired-excess-weight=0.5"])
    with pytest.raises(ValueError):
        _env_config("mlp_relative", ["--paired-excess-weight=-0.5", "--excess-floor=0"])


def test_reference_twin_config_drops_paired_terms_only():
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.context_views import EnvContext
    components = {"tracking": MdpComponent(compute_func=lambda dof_pos: dof_pos, dynamic_vars={"dof_pos": EnvContext.current.dof_pos}),
                  "paired_excess": MdpComponent(compute_func=compute_paired_excess_burden,
                      dynamic_vars={"tau_agent": EnvContext.assist.tau_agent,
                                    "reference_tau_agent": EnvContext.assist.reference_tau_agent,
                                    "reference_tau_sq_window": EnvContext.assist.reference_tau_sq_window})}
    kept = {name: c for name, c in components.items()
            if not any(getattr(v, "path", "").startswith("assist.reference") for v in c.dynamic_vars.values())}
    assert set(kept) == {"tracking"}
