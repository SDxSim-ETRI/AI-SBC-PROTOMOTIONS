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
"""A favorable average must never hide a failing phase or failed completion."""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/experiments/assist_skeleton"))
from phase_screen import screen_motion


def inputs():
    row = {"τ²":-25., "body_A":.06, "body_B":.0605, "|ω|":-4., "speed":.01}
    ref = {"τ²":-20., "|ω|":-3.}
    diag = {"phase":[{"samples":100,"rms_reduction_Nm":2.} for _ in range(10)],
            "worst_phase_rms_increase_Nm":0.,
            "intervals":{k:{"agent_A_rms_Nm":20.,"agent_B_rms_Nm":18.} for k in ("20-40","70-90")}}
    return row,ref,diag,copy.deepcopy(diag)


def test_one_bad_phase_fails_despite_better_average():
    row,ref,diag,base=inputs()
    assert screen_motion(row,ref,diag,base,True,1.,100.)["pass"]
    diag["phase"][8]["rms_reduction_Nm"]=-10.
    result=screen_motion(row,ref,diag,base,True,1.,100.)
    assert not result["pass"]
    assert any(c["name"]=="phase_80_90_RMS_regression_Nm" and not c["pass"] for c in result["checks"])


def test_completion_and_action_contract_are_required():
    args=inputs()
    assert not screen_motion(*args,False,1.,100.)["pass"]
    assert not screen_motion(*args,True,.5,100.)["pass"]
    assert not screen_motion(*args,True,1.,40.)["pass"]


def test_missing_phase_and_nonfinite_metric_never_pass():
    row,ref,diag,base=inputs()
    diag["phase"].pop()
    row["τ²"]=float("nan")
    result=screen_motion(row,ref,diag,base,True,1.,100.)
    assert not result["pass"]
    assert next(c for c in result["checks"] if c["name"]=="global_tau2_change_pct")["value"] is None
