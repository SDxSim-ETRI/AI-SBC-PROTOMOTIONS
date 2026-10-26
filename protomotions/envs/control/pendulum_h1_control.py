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
"""Frozen H1 forecasts from pendulum observations, with isolated predictor RNG."""

from dataclasses import dataclass

import torch

from protomotions.envs.control.pendulum_research_control import (
    PendulumResearchControl, PendulumResearchControlConfig,
)
from protomotions.maniflow.chunk_source import ManiFlowChunkSource


@dataclass
class PendulumH1ControlConfig(PendulumResearchControlConfig):
    _target_: str = "protomotions.envs.control.pendulum_h1_control.PendulumH1Control"
    predictor_seed: int = 17001


class PendulumH1Control(PendulumResearchControl):
    def __init__(self, config, env):
        # Keep the synthetic source active to consume exactly its usual random
        # draws. Only the actor's forecast changes; intent/physics do not.
        assert config.chunk_source == "emulated" and config.intent_source == "synthetic"
        self._h1_device = torch.device(env.device)
        with torch.random.fork_rng(devices=[self._h1_device]):
            self._h1_source = ManiFlowChunkSource.from_config(config, env)
        self._h1_source.estimator.policy.eval().requires_grad_(False)
        assert abs(self._h1_source.knot_dt - config.hlp_dt) < 1e-6
        assert self._h1_source.num_knots == config.chunk_knots
        assert config.chunk_refresh_steps == 1
        self._h1_cpu_rng = torch.Generator().manual_seed(config.predictor_seed).get_state()
        self._h1_cuda_rng = torch.Generator(device=self._h1_device).manual_seed(
            config.predictor_seed).get_state()
        super().__init__(config, env)

    def reset(self, env_ids):
        self._h1_source.reset(env_ids, self.env.state_history.dof_pos[env_ids, 0], trunk=None)
        super().reset(env_ids)

    def step(self):
        self._h1_source.observe(self.env.state_history.dof_pos[:, 0], trunk=None)
        super().step()

    def _refresh_chunk(self, env_ids, t_now):
        super()._refresh_chunk(env_ids, t_now)
        with torch.random.fork_rng(devices=[self._h1_device]), torch.no_grad():
            torch.set_rng_state(self._h1_cpu_rng)
            torch.cuda.set_rng_state(self._h1_cuda_rng, self._h1_device)
            knots, t0 = self._h1_source.predict_knots(env_ids, t_now)
            self._h1_cpu_rng = torch.get_rng_state()
            self._h1_cuda_rng = torch.cuda.get_rng_state(self._h1_device)
        self._chunk[env_ids] = knots.clamp(self._target_low, self._target_high)
        self._chunk_t0[env_ids] = t0
