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
"""Inertia-aware, C2-continuous blending of the existing synthetic intentions.

Steady segment amplitudes/frequencies and exogenous loads stay unchanged. Only
transition duration increases when its extra inertial demand exceeds the device
budget. This bounds intention inertia, not full inverse dynamics under loads.
"""
from dataclasses import dataclass
import math

import torch

from protomotions.envs.control.assist_disturbance_control import (
    DisturbedAssistTargetControl, DisturbedAssistTargetControlConfig,
)


def quintic_blend(x):
    x = x.clamp(0., 1.)
    return x**3 * (10. + x * (-15. + 6. * x))


def transition_duration(position_gap_bound, velocity_gap_bound, inertia, torque_budget):
    """Solve I*(3.75*Dv/T + (10/sqrt(3))*Dq/T^2) <= budget."""
    acceleration_budget = torque_budget / inertia
    linear = 3.75 * velocity_gap_bound
    constant = (10. / math.sqrt(3.)) * position_gap_bound
    return (linear + torch.sqrt(linear.square() + 4.*acceleration_budget*constant)) / (2.*acceleration_budget)


@dataclass
class PendulumFeasibleControlConfig(DisturbedAssistTargetControlConfig):
    _target_: str = "protomotions.envs.control.pendulum_feasible_control.PendulumFeasibleControl"
    transition_torque_budget: float = 40.0


class PendulumFeasibleControl(DisturbedAssistTargetControl):
    def __init__(self, config, env):
        if not 0 < config.transition_torque_budget < float("inf"):
            raise ValueError("transition_torque_budget must be finite and positive")
        if config.intent_source != "synthetic" or config.chunk_refresh_steps != 1:
            raise ValueError("This study requires synthetic intentions and 100 Hz refresh")
        self._limited_start = torch.full((env.num_envs,), float("nan"), device=env.device)
        self._requested_blend = torch.zeros(env.num_envs, device=env.device)
        self._transition_extensions = torch.zeros(env.num_envs, device=env.device)
        super().__init__(config, env)

    def reset(self, env_ids):
        self._limited_start[env_ids] = float("nan")
        super().reset(env_ids)

    def set_wearer_gains(self, kp, kd, effort_limit=None, inertia=None):
        super().set_wearer_gains(kp, kd, effort_limit, inertia)
        ids = torch.arange(self.env.num_envs, device=self.env.device)
        # Initialization changes I after control construction; no simulation has
        # occurred yet. Recalculate against the original sampled durations.
        self._blend_dur[:] = self._requested_blend
        self._limited_start[:] = float("nan")
        self._limit_new_transitions(ids)

    def _limit_new_transitions(self, env_ids):
        ids = env_ids[self._limited_start[env_ids] != self._seg_start[env_ids]]
        if not len(ids):
            return
        p = self._params
        amp = p["amp"][:, ids].abs()
        amp2 = p["amp2"][:, ids].abs()
        frequency = p["freq"][:, ids].abs()
        dq = (p["offset"][1, ids]-p["offset"][0, ids]).abs() + (amp+amp2).sum(0)
        dv = (2.*math.pi*frequency*(amp+2.*amp2)).sum(0)
        minimum = transition_duration(dq, dv, self._inertia[ids], self.config.transition_torque_budget).amax(-1)
        requested = self._blend_dur[ids].clone()
        self._requested_blend[ids] = requested
        self._blend_dur[ids] = torch.maximum(requested, minimum)
        self._transition_extensions[ids] += (minimum > requested).float()
        self._limited_start[ids] = self._seg_start[ids]

    def _refresh_chunk(self, env_ids, t_now):
        # Parent samples duration before this hook; limit before any goal or
        # forecast is evaluated. No extra RNG draw or duplicate refresh.
        self._limit_new_transitions(env_ids)
        super()._refresh_chunk(env_ids, t_now)

    def _target_at(self, t, env_ids=None):
        start = self._seg_start if env_ids is None else self._seg_start[env_ids]
        duration = self._blend_dur if env_ids is None else self._blend_dur[env_ids]
        mix = quintic_blend((t-start[:, None])/duration[:, None])
        a, b = self._eval_segment(0, t, env_ids), self._eval_segment(1, t, env_ids)
        return (a+(b-a)*mix[..., None]).clamp(self._target_low, self._target_high)
