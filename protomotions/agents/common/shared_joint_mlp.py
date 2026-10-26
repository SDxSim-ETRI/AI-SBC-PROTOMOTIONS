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
"""One shared single-joint actor behind the existing bilateral LLP interface."""

from dataclasses import dataclass, replace

import torch
from tensordict.nn import TensorDictModuleBase

from protomotions.agents.common.config import MLPWithConcatConfig
from protomotions.agents.common.mlp import MLPWithConcat


@dataclass
class SharedJointMLPConfig(MLPWithConcatConfig):
    _target_: str = "protomotions.agents.common.shared_joint_mlp.SharedJointMLP"
    num_joints: int = 2


class SharedJointMLP(TensorDictModuleBase):
    """Apply the same MLP and observation statistics to each joint independently.

    The three public observation keys retain their deployed shapes. Joint fields
    are interleaved in each key; only assist_target's final chunk-age scalar is
    shared. No opposite-joint observation enters a joint's deterministic action.
    """

    def __init__(self, config: SharedJointMLPConfig):
        super().__init__()
        if set(config.in_keys) != {"assist_proprio", "assist_target", "previous_actions"}:
            raise ValueError("SharedJointMLP requires the three standard LLP observation keys")
        if config.num_out != config.num_joints or len(config.out_keys) != 1:
            raise ValueError("SharedJointMLP requires one action per joint and one output key")
        self.config = config
        self.in_keys, self.out_keys = config.in_keys, config.out_keys
        self.joint_model = MLPWithConcat(replace(
            config, in_keys=["joint_obs"], out_keys=["joint_action"], num_out=1,
        ))

    def joint_observations(self, tensordict):
        pieces = []
        for key in self.in_keys:
            value = tensordict[key]
            fields = value[..., :-1] if key == "assist_target" else value
            if fields.shape[-1] % self.config.num_joints:
                raise ValueError(f"{key} must contain interleaved joint fields")
            fields = fields.reshape(*fields.shape[:-1], -1, self.config.num_joints)
            pieces.append(fields.transpose(-1, -2))
            if key == "assist_target":
                pieces.append(value[..., -1:].unsqueeze(-2).expand(
                    *value.shape[:-1], self.config.num_joints, 1,
                ))
        return torch.cat(pieces, dim=-1)

    def forward(self, tensordict):
        from tensordict import TensorDict

        obs = self.joint_observations(tensordict)
        local = TensorDict({"joint_obs": obs.reshape(-1, obs.shape[-1])},
                           batch_size=[obs.numel() // obs.shape[-1]])
        action = self.joint_model(local)["joint_action"]
        tensordict[self.out_keys[0]] = action.reshape(*obs.shape[:-1])
        return tensordict


@dataclass
class SingleJointMLPConfig(SharedJointMLPConfig):
    _target_: str = "protomotions.agents.common.shared_joint_mlp.SingleJointMLP"
    joint_index: int = 0


class SingleJointMLP(SharedJointMLP):
    """One leg's actor: reads only that joint's observation fields.

    The bilateral action layout is kept so the deployed controller and the
    environment need no new interface. The other joint's mean action is a
    constant zero (no parameters), and the environment applies no assistance
    there (`AssistEnv.assist_active_joints`). Left and right are trained as
    separate runs and combined at deployment (`LLPLegPair`).
    """

    def __init__(self, config: SingleJointMLPConfig):
        super().__init__(config)
        if not 0 <= config.joint_index < config.num_joints:
            raise ValueError("joint_index must select one of the joints")

    def forward(self, tensordict):
        from tensordict import TensorDict

        obs = self.joint_observations(tensordict)[..., self.config.joint_index, :]
        local = TensorDict({"joint_obs": obs}, batch_size=[obs.shape[0]])
        joint = self.joint_model(local)["joint_action"]
        action = torch.zeros(*obs.shape[:-1], self.config.num_joints,
                             dtype=joint.dtype, device=joint.device)
        action[..., self.config.joint_index] = joint[..., 0]
        tensordict[self.out_keys[0]] = action
        return tensordict
