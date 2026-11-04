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
"""Actor-only initialization from a pendulum LLP with an exact change of hip coordinate signs."""

from dataclasses import dataclass
from pathlib import Path

import torch

from protomotions.agents.ppo.agent import PPO
from protomotions.agents.ppo.config import PPOAgentConfig


@dataclass
class AssistFinetuneConfig(PPOAgentConfig):
    _target_: str = "protomotions.agents.ppo.assist_finetune.AssistFinetunePPO"
    initial_actor_checkpoint: str = ""


def mirror_actor_state(state, input_signs):
    """Return mu_new(x) = -mu_old(S*x), including normalized observation means.

S negates q, qdot, predictions and raw action history; chunk age is unchanged.
Output log standard deviation and observation variances stay unchanged.
"""
    state = {k: v.clone() for k, v in state.items()}
    state["mu.norm.running_obs_norm.mean"].mul_(input_signs)
    state["mu.mlp.0.weight"].mul_(input_signs.unsqueeze(0))
    indices = [int(k.split('.')[2]) for k in state if k.startswith("mu.mlp.") and k.endswith(".weight")]
    last = max(indices)
    state[f"mu.mlp.{last}.weight"].neg_()
    state[f"mu.mlp.{last}.bias"].neg_()
    return state


class AssistFinetunePPO(PPO):
    def load(self, checkpoint, load_env=True):
        super().load(checkpoint, load_env=load_env)
        if self.config.evaluator.eval_metrics_every is None:
            # This recipe uses independent run_system A/B evaluations. BaseAgent
            # otherwise forces its inherited pendulum evaluator after a resume,
            # even when periodic evaluation has explicitly been disabled.
            self.just_loaded_checkpoint_should_evaluate = False

    def setup(self):
        super().setup()
        path = self.config.initial_actor_checkpoint
        if not path:
            return
        source = torch.load(path, map_location=self.device, weights_only=False)
        state = {k.removeprefix("_actor."): v for k, v in source["model"].items() if k.startswith("_actor.")}
        obs = self.env.get_obs()
        keys = self.config.model.actor.in_keys
        signs = torch.cat([torch.full((obs[k].shape[-1],), -1., device=self.device) for k in keys])
        age = sum(obs[k].shape[-1] for k in keys[:keys.index("assist_target") + 1]) - 1
        signs[age] = 1
        resolved = torch.load(Path(path).parent / "resolved_configs_inference.pt",
                              map_location="cpu", weights_only=False)
        sign = resolved["env"].control_components["assist"].flexion_sign
        if sign == -1:
            state = mirror_actor_state(state, signs)
        elif sign != 1:
            raise ValueError(f"Unsupported source hip coordinate sign: {sign}")
        self.model._actor.load_state_dict(state, strict=True)
        # Critic, optimizers, return/advantage statistics and epoch remain fresh.
        print(f"Actor-only warm start from {path}; hip sign {sign:+g} -> +1", flush=True)
