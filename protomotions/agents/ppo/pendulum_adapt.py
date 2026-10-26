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
"""Actor-only pendulum adaptation with an unchanged raw-action convention."""

from dataclasses import dataclass
from pathlib import Path

import torch

from protomotions.agents.ppo.agent import PPO
from protomotions.agents.ppo.config import PPOAgentConfig


@dataclass
class PendulumAdaptConfig(PPOAgentConfig):
    _target_: str = "protomotions.agents.ppo.pendulum_adapt.PendulumAdaptPPO"
    initial_actor_checkpoint: str = ""


class PendulumAdaptPPO(PPO):
    def setup(self):
        super().setup()
        path = Path(self.config.initial_actor_checkpoint)
        cfg = torch.load(path.parent / "resolved_configs_inference.pt",
                         map_location="cpu", weights_only=False)
        source = cfg["env"].control_components["assist"]
        target = self.env.config.control_components["assist"]
        assert source.flexion_sign == target.flexion_sign == -1.0
        assert cfg["agent"].model.actor.in_keys == self.config.model.actor.in_keys
        saved = torch.load(path, map_location=self.device, weights_only=False)
        state = {key.removeprefix("_actor."): value
                 for key, value in saved["model"].items() if key.startswith("_actor.")}
        self.model._actor.load_state_dict(state, strict=True)
        # New critic/optimizers/epoch. The new environment applies 20*tanh(a),
        # exactly the source L1's 40*tanh(a) at output scale 0.5.
        print(f"Pendulum actor-only initialization: {path}; fresh critic/optimizers", flush=True)

    def load(self, checkpoint, load_env=True):
        super().load(checkpoint, load_env=load_env)
        if self.config.evaluator.eval_metrics_every is None:
            self.just_loaded_checkpoint_should_evaluate = False
