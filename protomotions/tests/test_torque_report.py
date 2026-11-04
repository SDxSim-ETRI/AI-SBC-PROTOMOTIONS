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
"""Check physical time alignment and phase aggregation used by `report.py finetune`."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/experiments/assist_skeleton"))
from analysis import aligned_torques, phase_profile


def test_previous_window_alignment_and_phase_wrap():
    tr = {"t": np.array([0., .01, .02]),
          "gait_phase": np.array([[.99, .99], [.01, .01], [.03, .03]]),
          "tau_pd_prev": np.full((3, 2, 2), 10.),
          "tau_ff_prev": np.full((3, 2, 2), 2.),
          "tau_assist": np.array([[[0., 0.], [7., 7.]], [[0., 0.], [9., 9.]], [[0., 0.], [99., 99.]]])}
    d = aligned_torques(tr, {"warmup": 0.})
    np.testing.assert_allclose(d["assist_B"], [[7., 7.], [9., 9.]])
    np.testing.assert_allclose(d["total_B"], [[19., 19.], [21., 21.]])
    np.testing.assert_allclose(d["t"], [.005, .015])
    np.testing.assert_allclose(d["phase"], [[0., 0.], [.02, .02]], atol=1e-10)


def test_rms_preserves_opposite_sign_load_and_weighted_contribution():
    a = np.array([[10., -10.], [20., -20.]])
    b = a / 2
    d = {"phase": np.array([[.1, .1], [.6, .6]]), "agent_A": a, "agent_B": b,
         "assist_A": np.zeros_like(a), "assist_B": a-b, "total_B": a}
    p = phase_profile(d, 2)
    assert p[0]["agent_A"] == 0  # signed means cancel; load must not disappear
    assert p[0]["agent_A_rms"] == 10
    assert p[0]["rms_reduction_Nm"] == 5
    assert p[0]["tau2_change_pct"] == -75
    assert sum(r["tau2_contribution_Nm2"] for r in p) == pytest.approx(float(np.mean(np.sum(b*b-a*a, axis=1))))


