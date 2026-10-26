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
"""Torque-unit and additive-source regression tests; no simulator required."""

from types import SimpleNamespace

import torch

from protomotions.assist.llp_policy import LLPPolicy
from protomotions.maniflow.hybrid_control import JointTorqueOverride


def policy(transform, limit):
    obj = LLPPolicy.__new__(LLPPolicy)
    obj.layout = SimpleNamespace(action_transform=transform, torque_scale=40., torque_limit=limit)
    return obj


def test_linear_request_and_independent_final_limit():
    raw = torch.tensor([[-4., -.5, .5, 2.5, 4.]])
    p = policy(None, 100.)
    assert torch.equal(p.requested_torque(raw), torch.tensor([[-160., -20., 20., 100., 160.]]))
    assert torch.equal(p.to_torque(raw), torch.tensor([[-100., -20., 20., 100., 100.]]))
    # Changing only the limit must not amplify commands below either limit.
    assert torch.equal(p.to_torque(raw)[:, 1:3], policy(None, 200.).to_torque(raw)[:, 1:3])


def test_legacy_l1_mapping_unchanged():
    raw = torch.linspace(-10, 10, 201)
    assert torch.equal(policy("tanh", 40.).to_torque(raw), 40. * raw.tanh())


def test_independently_bounded_sources_do_not_share_wearer_limit():
    inj = JointTorqueOverride.__new__(JointTorqueOverride)
    inj._engaged = True
    inj.device = torch.device("cpu")
    inj.torque_limits = torch.tensor([150., 150.])
    inj.env_ids = torch.tensor([0])
    inj.sim_dof_indices = torch.tensor([0, 1])
    inj._torques_sim = torch.zeros(1, 1, 2)
    inj._torques_wp = inj._torques_sim
    writes = []
    inj.sim = SimpleNamespace(control=None, robot_view=SimpleNamespace(
        set_dof_forces=lambda control, values: writes.append(values.clone())))
    combined = torch.tensor([[170., -190.]])
    # Legacy clients retain exactly the previous clamp.
    assert torch.equal(inj.set_torques(combined), torch.tensor([[150., -150.]]))
    assert torch.equal(inj.set_torques(combined, clamp_to_effort_limits=False), combined)
    assert torch.equal(writes[-1][:, 0], combined)


def test_combined_human_actuator_limit_and_source_accounting():
    from protomotions.envs.base_env.assist_env import AssistEnv

    # Row 0 is unsaturated. Row 1 needs net braking even though FF alone is
    # positive and large: separately saturating PD and FF would cancel it.
    env = AssistEnv.__new__(AssistEnv)
    env.config = SimpleNamespace(assist_torque_limit=20., clamp_combined_injection=False,
        assist_component_name="assist", human_velocity_feedforward=True,
        human_feedforward_inside_pd=True)
    q = torch.tensor([[0.1, 0.1], [1., 1.]])
    velocity = torch.zeros_like(q)
    target = torch.zeros_like(q)
    target_velocity = torch.tensor([[1., 1.], [20., 20.]])
    env._human_kp = torch.full_like(q, 300.)
    env._human_kd = torch.full_like(q, 10.)
    env._effort_limit = torch.full_like(q, 150.)
    env._human_ff_enabled = None
    env._gain_scheduled = False
    env._assist_limit = None
    env._current_raw_action = torch.zeros_like(q)
    env._current_processed_action = torch.zeros_like(q)
    comp = SimpleNamespace(get_pd_targets_and_vel=lambda: (target, target_velocity),
                           external_torque=lambda: torch.full_like(q, 180.),
                           _tau_agent=torch.zeros_like(q), _tau_assist=torch.zeros_like(q))
    env.control_manager = SimpleNamespace(components={"assist": comp})
    env._lazy_init_assist = lambda: None
    env._build_global_context = lambda: SimpleNamespace()
    env._process_action = lambda action, context: {"processed_action": action*20}
    writes, human = [], []
    def inject(torque, clamp_to_effort_limits):
        assert not clamp_to_effort_limits
        writes.append(torque.clone())
        return torque
    def simulate(pd_target, markers_callback):
        human.append((env._human_kp*(pd_target-q)-env._human_kd*velocity).clamp(-150,150))
    env._torque_override = SimpleNamespace(set_torques=inject)
    env.simulator = SimpleNamespace(step=simulate, user_requested_reset=False,
        get_substep_mean_dof_forces=lambda: human[-1])
    env.post_physics_step = lambda: None
    env.get_obs = lambda: {}
    env.rew_buf = torch.zeros(2)
    env.reset_buf = env.terminate_buf = torch.zeros(2,dtype=torch.bool)
    AssistEnv.step(env, torch.ones_like(q))
    torch.testing.assert_close(comp._tau_agent, torch.tensor([[-20.,-20.],[-100.,-100.]]))
    torch.testing.assert_close(writes[-1], torch.full_like(q, 200.))
    torch.testing.assert_close(comp._tau_assist, torch.full_like(q, 20.))
    assert env.extras["assist/injection_error_max_Nm"] == 0
