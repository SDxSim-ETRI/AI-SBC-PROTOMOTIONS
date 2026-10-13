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
"""Independent geometry tests for the SOMA23 -> suit coordinate contract."""

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from data.scripts.retarget_soma23_to_suit import (
    BODY_MAP, M_SOMA, SOMA23_BODY_NAMES, foot_surfaces, load_model, retarget,
)


def source_from_known_suit(model):
    data = mujoco.MjData(model)
    data.qpos[:3] = [1.0, 2.0, 1.0]
    data.qpos[3:7] = Rotation.from_euler("z", 0.7).as_quat()[[3, 0, 1, 2]]
    for name, angle in {"hip_flexion_r": 0.3, "hip_flexion_l": -0.15,
                        "knee_angle_r": -0.4, "knee_angle_l": -0.2,
                        "ankle_angle_r": 0.15, "lumbar_extension": -0.1,
                        "elbow_flex_r": 0.3, "elbow_flex_l": 0.25}.items():
        data.qpos[model.joint(name).qposadr[0]] = angle
    mujoco.mj_kinematics(model, data)
    known = data.qpos.copy()
    # Construct source rotations by expressing measured MuJoCo body frames in
    # SOMA's x-left/y-back T-pose convention. No ProtoMotions FK in the fixture.
    source_rot = np.tile(np.eye(3), (23, 1, 1))
    mapping = M_SOMA.numpy()
    source_rot[0] = data.xmat[1].reshape(3, 3) @ mapping.T
    for suit, soma in BODY_MAP.items():
        correction = mapping
        if suit.startswith(("humerus", "ulna", "hand")):
            correction = mapping @ Rotation.from_euler("x", -np.pi/2 if suit.endswith("_r") else np.pi/2).as_matrix()
        source_rot[SOMA23_BODY_NAMES.index(soma)] = data.xmat[model.body(suit).id].reshape(3, 3) @ correction.T
    pos = np.zeros((3, 23, 3))
    pos[:, 0] = np.array([[1,2,1],[1.025,2,1],[1.05,2,1]])
    source = {"fps": 40, "rigid_body_pos": torch.tensor(pos, dtype=torch.float32),
              "rigid_body_rot": torch.tensor(np.tile(Rotation.from_matrix(source_rot).as_quat(),(3,1,1)),dtype=torch.float32),
              "dof_pos": torch.zeros(3,66), "rigid_body_contacts": torch.zeros(3,23,dtype=torch.bool)}
    return source, known


def test_native_timing_and_suit_coordinate_round_trip():
    model = load_model()
    source, known = source_from_known_suit(model)
    motion, audit, qpos, surfaces = retarget(source, model)
    np.testing.assert_allclose(qpos[:,7:20],np.tile(known[7:20],(3,1)),atol=5e-4)
    np.testing.assert_allclose(qpos[:,3:7],np.tile(known[3:7],(3,1)),atol=2e-7)
    np.testing.assert_allclose(motion.rigid_body_vel[:,0,0],1.0,atol=1e-5)
    assert audit["duration_s"] == 0.05
    assert audit["joint_limit_violation_max_rad"] == 0
    assert audit["mujoco_fk_max_position_error_m"] < 1e-5
    assert surfaces.min() >= -1e-10
    assert motion.dof_pos.shape == (3,27)
    assert motion.rigid_body_pos.shape == (3,28,3)
    assert torch.count_nonzero(motion.dof_pos[:,23:]) == 0


def test_foot_surface_uses_geometry_not_body_origin():
    model = load_model()
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model,data)
    before = foot_surfaces(model,data)
    data.qpos[2] += 0.123
    mujoco.mj_kinematics(model,data)
    np.testing.assert_allclose(foot_surfaces(model,data)-before,0.123,atol=1e-10)
    assert foot_surfaces(model,data)[0] < data.xpos[model.body("talus_r").id,2]


def test_rejects_invalid_timebase():
    model = load_model()
    source,_ = source_from_known_suit(model)
    source["fps"] = 0
    with pytest.raises(ValueError,match="fps"):
        retarget(source,model)
