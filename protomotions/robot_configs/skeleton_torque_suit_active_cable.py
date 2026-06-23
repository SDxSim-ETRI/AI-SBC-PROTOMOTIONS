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
"""Suit robot config with active cable joints controlled by DOFC.

The 4 suit cable/slide joints (slide1–4, DOF indices 23–26) are set to
stiffness=50 and damping=5 (same as the original skeleton_torque_suit).
The DOFC rule-based controller overrides cable DOF actions at each step,
so the PPO policy only effectively controls body DOFs 0–22.

  Phase A (passive cable): v18 — zero torque on cables
  Phase B (active cable):  v20 — DOFC A버전 rule-based cable control

DOFC cable mapping:
  slide1 (DOF 23): unused (force=0)
  slide2 (DOF 24): right hip flexion assist
  slide3 (DOF 25): unused (force=0)
  slide4 (DOF 26): left hip flexion assist
"""
from dataclasses import dataclass, field

from protomotions.components.pose_lib import ControlInfo
from protomotions.robot_configs.skeleton_torque_suit import (
    SkeletonTorqueSuitRobotConfig,
)
from protomotions.robot_configs.base import ControlConfig, ControlType


@dataclass
class SkeletonTorqueSuitActiveCableRobotConfig(SkeletonTorqueSuitRobotConfig):
    """Suit robot with active cable joints.

    Identical to SkeletonTorqueSuitRobotConfig — cable DOFs (slide1–4)
    keep stiffness=50/damping=5 so the DOFC-computed target positions
    generate real cable forces. The DOFC A버전 controller overrides
    PPO output for DOFs 23–26 at each environment step.
    """

    control: ControlConfig = field(
        default_factory=lambda: ControlConfig(
            control_type=ControlType.BUILT_IN_PD,
            override_control_info={
                "hip_(flexion|adduction)_[rl]": ControlInfo(
                    stiffness=200.0, damping=20.0, effort_limit=600, velocity_limit=10
                ),
                "hip_rotation_[rl]": ControlInfo(
                    stiffness=200.0, damping=20.0, effort_limit=600, velocity_limit=10
                ),
                "knee_angle_[rl]": ControlInfo(
                    stiffness=200.0, damping=20.0, effort_limit=600, velocity_limit=10
                ),
                "ankle_angle_[rl]": ControlInfo(
                    stiffness=100.0, damping=10.0, effort_limit=500, velocity_limit=10
                ),
                "lumbar_(extension|bending|rotation)": ControlInfo(
                    stiffness=100.0, damping=10.0, effort_limit=160, velocity_limit=10
                ),
                "arm_(flex|add|rot)_[rl]": ControlInfo(
                    stiffness=100.0, damping=10.0, effort_limit=250, velocity_limit=10
                ),
                "(elbow_flex|pro_sup)_[rl]": ControlInfo(
                    stiffness=50.0, damping=5.0, effort_limit=250, velocity_limit=10
                ),
                # Cable joints — active, controlled by DOFC A버전
                "slide[1-4]": ControlInfo(
                    stiffness=50.0, damping=5.0, effort_limit=500, velocity_limit=2
                ),
            },
        )
    )
