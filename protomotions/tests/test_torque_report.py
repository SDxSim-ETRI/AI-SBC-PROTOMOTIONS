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
"""Check physical time alignment and phase aggregation of torque reports."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/experiments/assist_skeleton"))
from report_torque_html import aligned_torques, phase_profile


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


def test_shortlist_uses_both_gaits_and_does_not_rank_falls_by_good_mean(tmp_path):
    import json
    from report_torque_html import rank_candidates

    records = {}
    for label, peak, fell, change in (("reference", 11, False, -10), ("L8", 10, False, -30),
                                      ("L9", 11, False, -40), ("L10", 12, False, -50),
                                      ("L11", 13, False, -60), ("L12", 9, True, -70),
                                      ("L13", 9, False, 1)):
        for motion in (0, 1):
            key = f"{label}_e6000_m{motion}"
            run = tmp_path / key
            run.mkdir()
            t = np.arange(40, dtype=float)
            pd = np.full((40, 2, 2), 10.)
            pd[::2, 1] = peak
            pd[1::2, 1] = 1.
            np.savez(run / "traces.npz", t=t, gait_phase=np.tile((t%10*.1+.02)[:, None],(1,2)),
                     tau_pd_prev=pd, tau_ff_prev=np.zeros_like(pd), tau_assist=np.zeros_like(pd))
            (run / "meta.json").write_text(json.dumps({"motion_id":motion,"warmup":0,
                "survived_s":5 if fell and motion==1 else None,"seconds":40,"llp_dt":1}))
            records[key] = {"run_dir":str(run), "τ²":change}
    selected, diag = rank_candidates(records)
    assert diag["selected_groups"] == ["L8_e6000", "L9_e6000", "L10_e6000"]
    assert len(selected) == 8  # baseline and three candidates, two motions each
    assert not any(k.startswith(("L11_", "L12_", "L13_")) for k in selected)
