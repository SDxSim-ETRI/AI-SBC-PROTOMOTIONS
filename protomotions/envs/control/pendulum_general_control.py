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
"""Synthetic swing deceleration and reversal shock, never actor phase gates.

Events depend on synthetic intent only: paired plants consume identical RNG even
when assistance changes actual kinematics. Inertia stays fixed. A reversal shock
is a bounded dissipative torque pulse, NOT a contact or variable-inertia model.
"""

from dataclasses import dataclass
import torch
from protomotions.envs.control.pendulum_research_control import (
    PendulumResearchControl, PendulumResearchControlConfig,
)


@dataclass
class PendulumGeneralControlConfig(PendulumResearchControlConfig):
    _target_: str = "protomotions.envs.control.pendulum_general_control.PendulumGeneralControl"
    phase_probability: float = 0.0
    swing_damping_max: float = 20.0
    swing_zero_probability: float = 0.25
    swing_duration_range: tuple = (0.06, 0.18)
    shock_duration_range: tuple = (0.02, 0.05)
    shock_damping_max: float = 35.0
    phase_torque_limit: float = 60.0


class PendulumGeneralControl(PendulumResearchControl):
    def __init__(self, config, env):
        if not 0 <= config.phase_probability <= 1:
            raise ValueError("phase_probability must be in [0, 1]")
        if not 0 <= config.swing_zero_probability <= 1:
            raise ValueError("swing_zero_probability must be in [0, 1]")
        for name in ("swing_damping_max", "shock_damping_max", "phase_torque_limit"):
            if not 0 <= getattr(config, name) < float("inf"):
                raise ValueError(f"Invalid {name}")
        for name in ("swing_duration_range", "shock_duration_range"):
            lo, hi = getattr(config, name)
            if not env.dt <= lo <= hi < float("inf"):
                raise ValueError(f"Invalid {name}")
        shape = (env.num_envs, env.robot_config.number_of_actions)
        for name in ("_phase_enabled", "_decelerating"):
            setattr(self, name, torch.zeros(shape, dtype=torch.bool, device=env.device))
        for name in ("_previous_intent_velocity", "_swing_start", "_swing_end",
                     "_swing_c", "_shock_start", "_shock_end", "_shock_c",
                     "_tau_swing", "_tau_shock"):
            setattr(self, name, torch.zeros(shape, device=env.device))
        super().__init__(config, env)

    def reset(self, env_ids):
        super().reset(env_ids)
        self._phase_enabled[env_ids] = False
        self._decelerating[env_ids] = False
        for name in ("_previous_intent_velocity", "_swing_c", "_shock_c", "_tau_swing", "_tau_shock"):
            getattr(self, name)[env_ids] = 0
        for name in ("_swing_start", "_swing_end", "_shock_start", "_shock_end"):
            getattr(self, name)[env_ids] = -1
        if self.config.phase_probability > 0 and len(env_ids):
            self._phase_enabled[env_ids] = self._u(0, 1, len(env_ids), self.n_dofs) < self.config.phase_probability

    def step(self):
        super().step()
        cfg = self.config
        if not cfg.disturbance_enabled or cfg.phase_probability == 0:
            return
        time = self._episode_time()
        dt = self.env.dt
        q = self._target_at(torch.stack((time-dt, time, time+dt), dim=1))
        velocity = (q[:, 2] - q[:, 0]) / (2*dt)
        acceleration = (q[:, 2] - 2*q[:, 1] + q[:, 0]) / dt**2
        t = time[:, None].expand_as(velocity)
        eligible = self._phase_enabled & (t > cfg.start_blend)
        decelerating = (velocity * acceleration < 0) & (velocity.abs() > .3)
        onset = eligible & decelerating & ~self._decelerating & (t > self._swing_end)
        if onset.any():
            count = int(onset.sum())
            self._swing_start[onset] = t[onset]
            self._swing_end[onset] = t[onset] + self._u(*cfg.swing_duration_range, count)
            self._swing_c[onset] = self._u(0, cfg.swing_damping_max, count) * (
                self._u(0, 1, count) >= cfg.swing_zero_probability)
        reversal = eligible & (velocity * self._previous_intent_velocity < 0) & (t > self._shock_end)
        if reversal.any():
            count = int(reversal.sum())
            self._shock_start[reversal] = t[reversal]
            self._shock_end[reversal] = t[reversal] + self._u(*cfg.shock_duration_range, count)
            self._shock_c[reversal] = self._u(0, cfg.shock_damping_max, count)
        actual_velocity = self.env.state_history.dof_vel[:, 0]
        for event, ramp in (("swing", .02), ("shock", .01)):
            gate = self._gate(t, getattr(self, f"_{event}_start"), getattr(self, f"_{event}_end"), ramp)
            torque = (-gate * getattr(self, f"_{event}_c") * actual_velocity).clamp(
                -cfg.phase_torque_limit, cfg.phase_torque_limit)
            setattr(self, f"_tau_{event}", torque)
            self._tau_ext = self._tau_ext + torque
        self._decelerating.copy_(decelerating)
        self._previous_intent_velocity.copy_(velocity)
