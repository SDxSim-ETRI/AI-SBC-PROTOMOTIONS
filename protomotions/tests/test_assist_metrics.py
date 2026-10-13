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
"""Regression checks for torque bounds used in the assist reports."""

import numpy as np
import pytest

from examples.experiments.assist_skeleton.analysis import (
    assist_output_limit,
    assist_saturation_pct,
)


def test_scaled_policy_saturation_uses_applied_torque_units():
    meta = {"torque_limit": 40.0, "assist_scale": 0.5}
    torque = np.array([0.0, 10.0, 19.5, -20.0])
    assert assist_output_limit(meta) == 20.0
    assert assist_saturation_pct(torque, meta) == 50.0
    # Scaling both the signal and its bound must preserve the statistic.
    assert assist_saturation_pct(torque * 2, {"torque_limit": 40.0}) == 50.0


@pytest.mark.parametrize("oracle", ["beta", "phase"])
def test_oracle_uses_its_own_limit_before_scale(oracle):
    meta = {"torque_limit": 20.0, "policy_torque_limit": 10.0,
            "assist_scale": 0.5, "oracle": oracle, "oracle_limit": 60.0}
    assert assist_output_limit(meta) == 30.0
    assert assist_saturation_pct(np.array([29.0, -30.0]), meta) == 100.0
    meta["oracle_limit"] = None
    assert assist_output_limit(meta) == 10.0


def test_policy_cannot_exceed_its_checkpoint_limit():
    meta = {"torque_limit": 80.0, "policy_torque_limit": 40.0, "assist_scale": 0.5}
    assert assist_output_limit(meta) == 20.0


def test_disabled_assistance_is_not_saturated():
    meta = {"torque_limit": 40.0, "assist_scale": 0.0}
    assert assist_output_limit(meta) == 0.0
    assert assist_saturation_pct(np.zeros((10, 2)), meta) == 0.0
