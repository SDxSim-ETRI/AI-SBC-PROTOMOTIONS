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
"""단일 관절 LLP를 양쪽 다리에 쓰는 두 방식 (2026-10-21).

* `LLPSharedLeg` (사용 중): 단일 관절 진자로 학습한 정책 하나를 양쪽 다리에 같은 가중치로 적용.
* `LLPLegPair`: 다리마다 따로 학습한 두 모델을 묶음. 오른쪽 모델(관절 0)과 왼쪽 모델(관절 1)은
  각자 자기 관절의 관측만 읽고 (`SingleJointMLP`), 다른 관절의 출력은 0이다.
관측 레이아웃·토크 변환은 기존 `LLPPolicy`와 같다.
"""

from dataclasses import fields
from pathlib import Path

import torch
from torch import Tensor

from protomotions.assist.llp_policy import LLPObsLayout, LLPPolicy


def _active_joints(policy: LLPPolicy) -> tuple:
    return tuple(int(j) for j in getattr(policy.resolved["env"], "assist_active_joints", ()) or ())


class LLPLegPair:
    """`LLPPolicy`와 같은 인터페이스(layout, act, requested_torque, to_torque, epoch)."""

    def __init__(self, right_checkpoint: str, left_checkpoint: str, device: torch.device):
        self.right = LLPPolicy(right_checkpoint, device)
        self.left = LLPPolicy(left_checkpoint, device)
        if _active_joints(self.right) != (0,) or _active_joints(self.left) != (1,):
            raise ValueError(
                f"leg checkpoints must assist joint 0 (right) and joint 1 (left): "
                f"{_active_joints(self.right)} / {_active_joints(self.left)}"
            )
        for f in fields(LLPObsLayout):
            a, b = getattr(self.right.layout, f.name), getattr(self.left.layout, f.name)
            same = torch.equal(a.cpu(), b.cpu()) if torch.is_tensor(a) else a == b
            if not same:
                raise ValueError(f"leg checkpoints differ in layout field {f.name}: {a} / {b}")
        self.layout = self.right.layout
        self.checkpoint = Path(right_checkpoint)
        self.checkpoints = (Path(right_checkpoint), Path(left_checkpoint))
        self.epochs = (self.right.epoch, self.left.epoch)
        self.epoch = self.right.epoch if self.right.epoch == self.left.epoch else -1
        self.resolved = self.right.resolved

    @torch.no_grad()
    def act(self, obs) -> Tensor:
        """양측 관측 → raw mean action (N, 2): [오른쪽 모델의 관절 0, 왼쪽 모델의 관절 1]."""
        right = self.right.act(obs)
        left = self.left.act(obs)
        return torch.stack([right[:, 0], left[:, 1]], dim=-1)

    def requested_torque(self, raw_action: Tensor) -> Tensor:
        return self.right.requested_torque(raw_action)

    def to_torque(self, raw_action: Tensor) -> Tensor:
        return self.right.to_torque(raw_action)


class LLPSharedLeg(LLPPolicy):
    """단일 관절 진자로 학습한 정책 하나를 양쪽 다리에 같은 가중치로 적용한다 (가중치 공유).

    `SingleJointMLP`(한쪽 관절로 학습) 또는 `SharedJointMLP` 체크포인트의 관절 모델을 다리마다
    그 다리의 관측(각도·속도 히스토리, 미래 목표, 이전 명령)으로 한 번씩 돌린다. 두 다리는 서로의
    관측을 보지 않는다.
    """

    def __init__(self, checkpoint: str, device: torch.device):
        super().__init__(checkpoint, device)
        from protomotions.agents.common.shared_joint_mlp import SharedJointMLP

        if not isinstance(self.actor.mu, SharedJointMLP):
            raise ValueError(f"weight sharing needs a per-joint actor, got {type(self.actor.mu).__name__}")
        self.checkpoints = (Path(checkpoint),)

    @torch.no_grad()
    def act(self, obs) -> Tensor:
        from tensordict import TensorDict

        from protomotions.agents.common.shared_joint_mlp import SharedJointMLP

        batch = obs[self.layout.in_keys[0]].shape[0]
        td = TensorDict({k: obs[k].to(self.device) for k in self.layout.in_keys},
                        batch_size=batch, device=self.device)
        # SingleJointMLP.forward는 학습 관절 하나만 내므로, 같은 관절 모델을 두 관절에 돌리는
        # SharedJointMLP.forward를 쓴다.
        return SharedJointMLP.forward(self.actor.mu, td)[self.mu_key]
