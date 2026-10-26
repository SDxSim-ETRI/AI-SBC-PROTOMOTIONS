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
"""Joint independence, command timing and bounded passive pendulum disturbances."""

from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from protomotions.agents.common.config import MLPLayerConfig
from protomotions.agents.common.shared_joint_mlp import SharedJointMLP, SharedJointMLPConfig
from protomotions.envs.control.assist_disturbance_control import (
    DisturbedAssistTargetControl, DisturbedAssistTargetControlConfig,
)
from protomotions.envs.control.pendulum_research_control import (
    PendulumResearchControl, PendulumResearchControlConfig,
)
from protomotions.robot_configs.hip_pendulum import HipPendulumRobotConfig


def control(config, cls=PendulumResearchControl):
    env = SimpleNamespace(num_envs=8, dt=.01, device=torch.device("cpu"),
                          robot_config=HipPendulumRobotConfig(),
                          progress_buf=torch.zeros(8, dtype=torch.long),
                          state_history=SimpleNamespace(dof_pos=torch.zeros(8, 1, 2),
                                                        dof_vel=torch.zeros(8, 1, 2)))
    return cls(config, env)


def test_shared_actor_equivariance_independence_and_reload():
    cfg = SharedJointMLPConfig(num_out=2, normalize_obs=True,
        in_keys=["assist_proprio", "assist_target", "previous_actions"],
        out_keys=["mu"], layers=[MLPLayerConfig(units=16)])
    model = SharedJointMLP(cfg).eval()
    obs = TensorDict({"assist_proprio": torch.randn(4, 104),
                      "assist_target": torch.randn(4, 13),
                      "previous_actions": torch.randn(4, 20)}, batch_size=[4])
    mu = model(obs.clone())["mu"]
    swapped, changed = obs.clone(), obs.clone()
    for key in cfg.in_keys:
        stop = -1 if key == "assist_target" else None
        swapped[key][..., :stop] = obs[key][..., :stop].reshape(4, -1, 2).flip(-1).flatten(1)
        changed[key][..., :stop][..., 0::2] += 3
    torch.testing.assert_close(model(swapped)["mu"], mu.flip(-1))
    torch.testing.assert_close(model(changed)["mu"][:, 1], mu[:, 1], rtol=0, atol=0)
    copy = SharedJointMLP(cfg).eval()
    copy(obs.clone())
    copy.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(copy(obs.clone())["mu"], mu, rtol=0, atol=0)
    local = model.joint_observations(obs)
    torch.testing.assert_close(local[:, 0, 58], obs["assist_target"][:, -1])
    torch.testing.assert_close(local[:, 1, 58], obs["assist_target"][:, -1])
    model(obs.clone())["mu"].sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_disabled_ablation_preserves_old_rng_and_trajectory():
    cfg = PendulumResearchControlConfig()
    old_cfg = DisturbedAssistTargetControlConfig(**{
        f.name: getattr(cfg, f.name) for f in fields(DisturbedAssistTargetControlConfig)
        if f.name != "_target_"})
    torch.manual_seed(21)
    old = control(old_cfg, DisturbedAssistTargetControl)
    after = torch.get_rng_state()
    torch.manual_seed(21)
    new = control(cfg)
    assert torch.equal(torch.get_rng_state(), after)
    for step in range(300):
        old.env.progress_buf[:] = new.env.progress_buf[:] = step
        before = torch.get_rng_state()
        old.step()
        after = torch.get_rng_state()
        torch.set_rng_state(before)
        new.step()
        assert torch.equal(torch.get_rng_state(), after)
        torch.testing.assert_close(old.external_torque(), new.external_torque(), rtol=0, atol=0)
        for a, b in zip(old.get_pd_targets_and_vel(), new.get_pd_targets_and_vel()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_target_hold_keeps_command_but_not_intent_frozen():
    obj = control(PendulumResearchControlConfig(human_target_hold_steps=5))
    obj.env.progress_buf[:] = 20
    first = tuple(x.clone() for x in obj.get_pd_targets_and_vel())
    obj.env.progress_buf[:] = 24
    for a, b in zip(first, obj.get_pd_targets_and_vel()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not torch.equal(first[0], obj._target_at(obj._episode_time().unsqueeze(1)).squeeze(1))
    obj.env.progress_buf[:] = 25
    assert not torch.equal(first[0], obj.get_pd_targets())
    obj.reset(torch.tensor([0, 3]))
    obj.env.progress_buf[[0, 3]] = 0
    obj.get_pd_targets()
    assert obj._hold_step.tolist() == [0, 25, 25, 0, 25, 25, 25, 25]


def test_damping_is_passive_bounded_and_state_independent_rng():
    cfg = PendulumResearchControlConfig(structured_probability=1.0)
    torch.manual_seed(7)
    a = control(cfg)
    torch.manual_seed(7)
    b = control(cfg)
    a.env.state_history.dof_vel[:] = 4
    b.env.state_history.dof_vel[:] = -4
    b.env.state_history.dof_pos[:] = .4
    active = 0
    for step in range(400):
        a.env.progress_buf[:] = b.env.progress_buf[:] = step
        before = torch.get_rng_state()
        a.step()
        after = torch.get_rng_state()
        torch.set_rng_state(before)
        b.step()
        assert torch.equal(after, torch.get_rng_state())
        for obj in (a, b):
            assert (obj._tau_damping * obj.env.state_history.dof_vel[:, 0] <= 0).all()
            assert obj._tau_damping.abs().max() <= cfg.damping_torque_limit
            assert (obj._tau_field.abs() <= obj._ev_load.abs() + 1e-5).all()
            assert torch.isfinite(obj.external_torque()).all()
        active += int(a._tau_damping.count_nonzero())
    assert active > 0


def test_invalid_refresh_period_is_rejected():
    import argparse
    from examples.experiments.assist_pendulum import mlp_research
    parser = argparse.ArgumentParser()
    mlp_research.additional_experiment_arguments(parser)
    args = parser.parse_args(["--human-target-period", "0.015"])
    with pytest.raises(ValueError, match="multiple"):
        mlp_research.env_config(HipPendulumRobotConfig(), args)
