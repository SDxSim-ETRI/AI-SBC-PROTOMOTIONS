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
"""Generalization event passivity, bounds, pairing and backwards compatibility."""
from dataclasses import fields
import torch
from protomotions.tests.test_pendulum_research import control
from protomotions.envs.control.pendulum_research_control import PendulumResearchControlConfig
from protomotions.envs.control.pendulum_general_control import PendulumGeneralControl, PendulumGeneralControlConfig


def test_disabled_general_control_preserves_rng_and_old_dynamics():
    old_config = PendulumResearchControlConfig(structured_probability=.5)
    cfg = PendulumGeneralControlConfig(**{f.name: getattr(old_config, f.name) for f in fields(old_config) if f.name != "_target_"})
    torch.manual_seed(42)
    a = control(old_config)
    after = torch.get_rng_state()
    torch.manual_seed(42)
    b = control(cfg, PendulumGeneralControl)
    assert torch.equal(after, torch.get_rng_state())
    for step in range(400):
        a.env.progress_buf[:] = b.env.progress_buf[:] = step
        before = torch.get_rng_state()
        a.step()
        after = torch.get_rng_state()
        torch.set_rng_state(before)
        b.step()
        assert torch.equal(after, torch.get_rng_state())
        torch.testing.assert_close(a.external_torque(), b.external_torque(), rtol=0, atol=0)


def test_phase_events_are_paired_passive_bounded_and_reset_locally():
    cfg = PendulumGeneralControlConfig(phase_probability=1., structured_probability=.5)
    torch.manual_seed(7)
    a = control(cfg, PendulumGeneralControl)
    torch.manual_seed(7)
    b = control(cfg, PendulumGeneralControl)
    a.env.state_history.dof_vel[:] = 4.
    b.env.state_history.dof_vel[:] = -3.
    counts = {"swing": 0, "shock": 0}
    for step in range(1000):
        a.env.progress_buf[:] = b.env.progress_buf[:] = step
        before = torch.get_rng_state()
        a.step()
        after = torch.get_rng_state()
        torch.set_rng_state(before)
        b.step()
        assert torch.equal(after, torch.get_rng_state())
        for event in counts:
            torch.testing.assert_close(getattr(a, f"_{event}_start"), getattr(b, f"_{event}_start"), rtol=0, atol=0)
            counts[event] += int(getattr(a, f"_tau_{event}").count_nonzero())
            for obj in (a, b):
                torque = getattr(obj, f"_tau_{event}")
                assert (torque * obj.env.state_history.dof_vel[:, 0] <= 0).all()
                assert torque.abs().max() <= cfg.phase_torque_limit
    assert all(counts.values()), counts
    other = a._swing_start[1:].clone()
    a.reset(torch.tensor([0]))
    assert (a._swing_start[0] == -1).all()
    torch.testing.assert_close(other, a._swing_start[1:], rtol=0, atol=0)
    assert not a._tau_swing[0].any()


def test_fresh_recipe_23nm_and_cost_multipliers_keep_goal_contract():
    import argparse
    from examples.experiments.assist_pendulum import mlp_general
    from protomotions.robot_configs.hip_pendulum import HipPendulumRobotConfig
    from protomotions.envs.action.action_functions import normalized_torque_action

    parser = argparse.ArgumentParser()
    mlp_general.additional_experiment_arguments(parser)
    base = mlp_general.env_config(HipPendulumRobotConfig(), parser.parse_args([]))
    changed = mlp_general.env_config(HipPendulumRobotConfig(), parser.parse_args([
        '--human-cost-multiplier', '2', '--regularization-multiplier', '4']))
    assert base.assist_torque_limit == base.action_config['torque_scale'] == 23
    control_config = base.control_components['assist']
    assert control_config.chunk_anchor == 'intent'
    assert control_config.obs_future_samples == 3
    assert control_config.obs_future_dt == .01
    for name in ('human_power', 'human_torque_sq'):
        assert changed.reward_components[name].static_params['weight'] == 2 * base.reward_components[name].static_params['weight']
    for name in ('assist_effort', 'action_rate'):
        assert changed.reward_components[name].static_params['weight'] == 4 * base.reward_components[name].static_params['weight']
    action = torch.tensor([[-1e6, 1e6], [-1., 1.]])
    value = normalized_torque_action(action, torque_scale=23.)['processed_action']
    assert value.abs().max() == 23
    torch.testing.assert_close(value, 23 * action.tanh())


def test_joint_control_arm_and_shared_baseline_use_same_other_settings():
    import argparse
    from examples.experiments.assist_pendulum import mlp_general
    from protomotions.robot_configs.hip_pendulum import HipPendulumRobotConfig
    parser = argparse.ArgumentParser()
    mlp_general.additional_experiment_arguments(parser)
    robot = HipPendulumRobotConfig()
    targets = []
    for argv in (['--joint-actor'], []):
        args = parser.parse_args(argv)
        args.batch_size, args.training_max_steps = 16384, 393216000
        env = mlp_general.env_config(robot, args)
        agent = mlp_general.agent_config(robot, env, args)
        targets.append(agent.model.actor.mu_model._target_)
        assert agent.bounds_loss_coef == .001
        assert env.action_config['torque_scale'] == env.assist_torque_limit == 23
    assert targets == ['protomotions.agents.common.mlp.MLPWithConcat',
                       'protomotions.agents.common.shared_joint_mlp.SharedJointMLP']
