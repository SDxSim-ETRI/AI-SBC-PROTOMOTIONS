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
"""Checks against window shifts and treatment-dependent diagnostic selection."""

import numpy as np
import pytest

from examples.experiments.assist_skeleton.analysis import (
    align,
    analyze_traces,
    masks_from_reference,
)


def traces():
    time = np.arange(300, dtype=np.float64) * .01
    velocity = np.broadcast_to((8-2*time)[:, None, None], (len(time), 2, 2)).copy()
    return dict(t=time, tau_pd_prev=np.full_like(velocity, 2.),
                tau_ff_prev=np.zeros_like(velocity), tau_assist=np.zeros_like(velocity),
                hip_vel=velocity)


def test_previous_window_torque_is_compared_with_its_response():
    data = traces()
    data["tau_assist"][100, 1] = 12.
    data["tau_pd_prev"][101, 1] = -10.
    aligned = align(data)
    row = np.flatnonzero(aligned["index"] == 101)[0]
    np.testing.assert_array_equal(aligned["assist"][row], [12., 12.])
    np.testing.assert_array_equal(aligned["B"][row], [-10., -10.])
    np.testing.assert_array_equal(aligned["assist"][row+1], [0., 0.])


def test_masks_do_not_depend_on_assisted_motion_or_torque():
    reference = traces()
    original = masks_from_reference(reference)
    changed = {key: value.copy() for key, value in reference.items()}
    changed["hip_vel"][:, 1] *= -100
    changed["tau_pd_prev"][:, 1] = 200
    changed["tau_assist"][:, 1] = 20
    modified = masks_from_reference(changed)
    for name in original["masks"]:
        np.testing.assert_array_equal(original["masks"][name], modified["masks"][name])
    a = analyze_traces(reference, original)
    b = analyze_traces(changed, original)
    assert a["low_decel_5"]["joint_samples"] == b["low_decel_5"]["joint_samples"] > 0
    assert a["low_decel_5"]["rms_increase"] == 0
    assert b["low_decel_5"]["rms_increase"] == 198


def test_low_burden_uses_rms_not_canceling_signed_mean():
    data = traces()
    data["tau_pd_prev"][:, 0] = (20 * (-1.)**np.arange(300))[:, None]
    reference = masks_from_reference(data)
    assert reference["masks"]["deceleration"].any()
    assert not reference["masks"]["low_decel_10"].any()


def test_time_mismatch_is_rejected_instead_of_reusing_wrong_masks():
    data = traces()
    reference = masks_from_reference(data)
    data["t"] = data["t"] + .005
    with pytest.raises(ValueError, match="different control times"):
        analyze_traces(data, reference)
