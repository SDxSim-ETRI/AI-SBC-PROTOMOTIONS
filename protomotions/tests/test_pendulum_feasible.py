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
"""Physical intention bounds, continuity, reward direction and config contrast."""
import math
import torch
from protomotions.tests.test_pendulum_research import control
from protomotions.envs.control.pendulum_feasible_control import (
    PendulumFeasibleControl, PendulumFeasibleControlConfig,
    quintic_blend, transition_duration,
)
from protomotions.envs.rewards.assist_energy import compute_assist_absorbed_power


def test_blend_has_zero_first_and_second_endpoint_derivatives():
    x = torch.tensor([0., 1.], dtype=torch.float64, requires_grad=True)
    y = quintic_blend(x)
    dy = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    ddy = torch.autograd.grad(dy.sum(), x)[0]
    torch.testing.assert_close(y, torch.tensor([0., 1.], dtype=x.dtype))
    assert dy.abs().max() < 1e-12 and ddy.abs().max() < 1e-12


def test_duration_bounds_blend_inertial_demand_across_random_parameters():
    torch.manual_seed(17003)
    dq, dv = torch.rand(500)*2, torch.rand(500)*12
    inertia = .418+torch.rand(500)*3.2
    duration = transition_duration(dq, dv, inertia, 40.)
    demand = inertia*(3.75*dv/duration+(10/math.sqrt(3))*dq/duration.square())
    assert demand.max() <= 40.00002
    longer = transition_duration(dq, dv, inertia*2, 40.)
    assert (longer >= duration).all()


def test_actual_sampled_curvature_is_bounded_and_partial_reset_is_local():
    cfg = PendulumFeasibleControlConfig(chunk_refresh_steps=1, blend_range=(.25, 2.5))
    c = control(cfg, PendulumFeasibleControl)
    inertia = torch.linspace(.418,3.576,16).reshape(8,2)
    c.set_wearer_gains(torch.ones(8,2)*500,torch.ones(8,2)*30,torch.ones(8,2)*150,inertia)
    # Numerical acceleration within blends; segment inertia <=75 plus extra<=40.
    for step in range(0,1000):
        c.env.progress_buf[:] = step
        c.step()
        if step % 7:
            continue
        t = c._episode_time()[:,None].double()
        delta = .001
        # Promote parameters to float64 for trustworthy second differences.
        original = c._params
        c._params = {k:v.double() for k,v in original.items()}
        accel = (c._target_at(t+delta)-2*c._target_at(t)+c._target_at(t-delta))/delta**2
        c._params = original
        assert (accel[:,0].abs()*inertia).max() < 115.1
    old_duration=c._blend_dur[1:].clone()
    c.reset(torch.tensor([0]))
    torch.testing.assert_close(c._blend_dur[1:],old_duration,rtol=0,atol=0)


def test_absorption_cost_cannot_cancel_between_hips():
    torque=torch.tensor([[10.,-10.],[-10.,-10.],[0.,1.]])
    velocity=torch.tensor([[2.,2.],[-2.,2.],[2.,0.]])
    torch.testing.assert_close(compute_assist_absorbed_power(torque,velocity),torch.tensor([20.,20.,0.]))


def test_recipes_differ_only_by_one_reward_component():
    import argparse
    from examples.experiments.assist_pendulum import mlp_feasible
    from protomotions.robot_configs.hip_pendulum import HipPendulumRobotConfig
    p=argparse.ArgumentParser(); mlp_feasible.additional_experiment_arguments(p)
    robot=HipPendulumRobotConfig()
    a=mlp_feasible.env_config(robot,p.parse_args([]))
    b=mlp_feasible.env_config(robot,p.parse_args(['--absorbed-power-weight=-0.0032']))
    b.reward_components.pop('assist_absorbed_power')
    from dataclasses import fields, is_dataclass
    def canonical(value):
        if is_dataclass(value):
            return {f.name: canonical(getattr(value,f.name)) for f in fields(value)}
        if isinstance(value, dict):
            return {k:canonical(v) for k,v in value.items()}
        if isinstance(value, (tuple,list)):
            return [canonical(v) for v in value]
        if hasattr(value, '__dict__') and not callable(value):
            return canonical(vars(value))
        return value
    assert canonical(a) == canonical(b)
    assert a.assist_torque_limit == a.action_config['torque_scale'] == 40
    assert a.human_feedforward_inside_pd and not a.clamp_combined_injection
