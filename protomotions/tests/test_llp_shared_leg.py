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
"""단일 관절 LLP의 가중치 공유 추론 (2026-10-21).

    pytest protomotions/tests/test_llp_shared_leg.py
"""

from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from protomotions.agents.common.config import MLPLayerConfig
from protomotions.agents.common.shared_joint_mlp import (
    SharedJointMLP,
    SingleJointMLP,
    SingleJointMLPConfig,
)

KEYS = ["assist_proprio", "assist_target", "previous_actions"]
SMOKE = Path(__file__).resolve().parents[2] / "results/smoke_llp20_motion_right/last.ckpt"


def _mirror(obs):
    """좌우 관절 필드를 맞바꾼 관측 (chunk age 스칼라는 그대로)."""
    out = obs.clone()
    for key in KEYS:
        stop = -1 if key == "assist_target" else None
        fields = obs[key][..., :stop]
        out[key][..., :stop] = fields.reshape(*fields.shape[:-1], -1, 2).flip(-1).flatten(-2)
    return out


def test_shared_forward_applies_the_trained_joint_model_to_both_legs():
    cfg = SingleJointMLPConfig(num_out=2, normalize_obs=True, joint_index=0, in_keys=KEYS,
                               out_keys=["mu"], layers=[MLPLayerConfig(units=16)])
    model = SingleJointMLP(cfg).eval()
    obs = TensorDict({"assist_proprio": torch.randn(5, 104), "assist_target": torch.randn(5, 13),
                      "previous_actions": torch.randn(5, 20)}, batch_size=[5])
    trained = model(obs.clone())["mu"]
    shared = SharedJointMLP.forward(model, obs.clone())["mu"]
    torch.testing.assert_close(shared[:, 0], trained[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(shared[:, 1], model(_mirror(obs))["mu"][:, 0], rtol=0, atol=0)
    # 한 다리의 출력은 다른 다리의 관측에 의존하지 않는다
    other = obs.clone()
    for key in KEYS:
        stop = -1 if key == "assist_target" else None
        other[key][..., :stop][..., 1::2] += 4.0
    torch.testing.assert_close(SharedJointMLP.forward(model, other)["mu"][:, 0], shared[:, 0], rtol=0, atol=0)


@pytest.mark.skipif(not SMOKE.exists(), reason="smoke checkpoint not present")
def test_llp_shared_leg_matches_single_policy_on_its_training_joint():
    from protomotions.assist.llp_policy import LLPPolicy

    single = LLPPolicy(str(SMOKE), torch.device("cpu"))
    shared = LLPPolicy(str(SMOKE), torch.device("cpu"), share_legs=True)
    obs = {k: torch.randn(3, n) for k, n in single.obs_dims().items()}
    a, b = single.act(obs), shared.act(obs)
    assert torch.all(a[:, 1] == 0) and torch.all(b[:, 1] != 0)
    torch.testing.assert_close(b[:, 0], a[:, 0])
    mirrored = _mirror(TensorDict(obs, batch_size=[3]))
    torch.testing.assert_close(b[:, 1], single.act({k: mirrored[k] for k in obs})[:, 0])
