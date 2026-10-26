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
"""Generic pendulum force fields and independent human-command sample/hold.

There are no walking phases, contact labels or W1 torque targets. Event sampling
depends only on elapsed episode time, so paired plants consume identical random
numbers even when their actual angles and resulting external torques differ.
"""

from dataclasses import dataclass

import torch

from protomotions.envs.control.assist_disturbance_control import (
    DisturbedAssistTargetControl,
    DisturbedAssistTargetControlConfig,
)


@dataclass
class PendulumResearchControlConfig(DisturbedAssistTargetControlConfig):
    _target_: str = "protomotions.envs.control.pendulum_research_control.PendulumResearchControl"
    structured_probability: float = 0.0
    damping_probability: float = 0.75
    damping_max: float = 30.0
    damping_torque_limit: float = 80.0
    human_target_hold_steps: int = 1


class PendulumResearchControl(DisturbedAssistTargetControl):
    def __init__(self, config, env):
        if not 0 <= config.structured_probability <= 1:
            raise ValueError("structured_probability must be in [0, 1]")
        if not 0 <= config.damping_probability <= 1:
            raise ValueError("damping_probability must be in [0, 1]")
        if min(config.damping_max, config.damping_torque_limit) < 0:
            raise ValueError("Damping coefficient and torque limit must be nonnegative")
        if config.human_target_hold_steps < 1:
            raise ValueError("human_target_hold_steps must be positive")
        n, d, dev = env.num_envs, env.robot_config.number_of_actions, env.device
        self._structured = torch.zeros(n, 1, dtype=torch.bool, device=dev)
        for name in ("_field_center", "_field_slope", "_field_bias", "_damp_start",
                     "_damp_end", "_damp_next", "_damp_c", "_tau_field", "_tau_damping",
                     "_held_target", "_held_velocity"):
            setattr(self, name, torch.zeros(n, d, device=dev))
        self._hold_step = torch.full((n,), -1, dtype=torch.long, device=dev)
        super().__init__(config, env)

    def reset(self, env_ids):
        super().reset(env_ids)
        self._hold_step[env_ids] = -1
        self._tau_field[env_ids] = 0
        self._tau_damping[env_ids] = 0
        cfg = self.config
        if not len(env_ids) or cfg.structured_probability == 0:
            return
        m, d = len(env_ids), self.n_dofs
        self._structured[env_ids] = self._u(0, 1, m, 1) < cfg.structured_probability
        self._field_center[env_ids] = self._u(*cfg.offset_range, m, d)
        # Dimensionless factors scale with each already-bounded load event.
        self._field_slope[env_ids] = self._u(-1, 1, m, d) / self._u(0.15, 0.6, m, d)
        self._field_bias[env_ids] = self._u(-0.25, 0.25, m, d)
        self._damp_start[env_ids] = -1
        self._damp_end[env_ids] = -1
        self._damp_c[env_ids] = 0
        self._damp_next[env_ids] = cfg.start_blend + self._u(*cfg.stance_gap_range, m, d)

    def step(self):
        super().step()
        cfg = self.config
        if not cfg.disturbance_enabled or cfg.structured_probability == 0:
            return
        t = self._episode_time().unsqueeze(1).expand(-1, self.n_dofs)
        due = (t >= self._damp_next) & (t > self._damp_end)
        if due.any():
            count = int(due.sum())
            duration = self._u(*cfg.stance_dur_range, count)
            self._damp_start[due] = t[due]
            self._damp_end[due] = t[due] + duration
            self._damp_next[due] = self._damp_end[due] + self._u(*cfg.stance_gap_range, count)
            self._damp_c[due] = self._u(0, cfg.damping_max, count) * (
                self._u(0, 1, count) < cfg.damping_probability)
        hist = self.env.state_history
        q, velocity = hist.dof_pos[:, 0], hist.dof_vel[:, 0]
        amplitude = self._ev_load.abs()
        slope_limit = 0.5 * self._kp_base * self._kp_scale
        slope = (amplitude * self._field_slope).clamp(-slope_limit, slope_limit)
        field = (slope * (q - self._field_center) + self._field_bias * amplitude).clamp(
            -amplitude, amplitude)
        load_gate = self._gate(t, self._ev_start, self._ev_end, cfg.stance_ramp)
        self._tau_field = self._structured * load_gate * field
        damping_gate = self._gate(t, self._damp_start, self._damp_end, cfg.stance_ramp)
        self._tau_damping = self._structured * (-damping_gate * self._damp_c * velocity).clamp(
            -cfg.damping_torque_limit, cfg.damping_torque_limit)
        # Replace, rather than double, the old load; preserve its gain schedule
        # and perturbations. Unstructured episodes retain the original force.
        self._tau_ext = (self._tau_ext - self._structured * load_gate * self._ev_load
                         + self._tau_field + self._tau_damping)

    def get_pd_targets_and_vel(self):
        if self.config.human_target_hold_steps == 1:
            return super().get_pd_targets_and_vel()
        step = self.env.progress_buf
        due = (self._hold_step < 0) | (
            step - self._hold_step >= self.config.human_target_hold_steps)
        if due.any():
            target, velocity = super().get_pd_targets_and_vel()
            self._held_target[due] = target[due]
            self._held_velocity[due] = velocity[due]
            self._hold_step[due] = step[due]
        return self._held_target, self._held_velocity

    def get_pd_targets(self):
        if self.config.human_target_hold_steps == 1:
            return super().get_pd_targets()
        return self.get_pd_targets_and_vel()[0]
