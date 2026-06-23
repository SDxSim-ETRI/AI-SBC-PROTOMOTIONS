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
"""Suit robot config with passive cable joints (zero torque).

The 4 suit cable/slide joints (slide1–4, DOF indices 23–26) are set to
stiffness=0 and damping=0, so the controller never applies force on them.
The cables move freely under physical forces (gravity, contact) only.

This is the baseline for comparison with a future suit controller policy:

  Phase A (this config): skeleton policy trained with passive cables
  Phase B (future):      skeleton policy + suit cable controller, compared against Phase A

Policy output is still 27-dimensional (same network shape as
skeleton_torque_suit) — the last 4 values are simply ignored at runtime.
"""
from dataclasses import dataclass, field

from protomotions.components.pose_lib import ControlInfo
from protomotions.robot_configs.skeleton_torque_suit import (
    SkeletonTorqueSuitRobotConfig,
)
from protomotions.robot_configs.base import ControlConfig, ControlType


@dataclass
class SkeletonTorqueSuitPassiveCableRobotConfig(SkeletonTorqueSuitRobotConfig):
    """Suit robot with passive (zero-torque) cable joints.

    Identical to SkeletonTorqueSuitRobotConfig except slide1–4 have
    stiffness=0 and damping=0 so no actuator force is applied on the cables.
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
                # Cable joints — fully passive, zero controller torque
                # effort_limit must be > 0 for MuJoCo actfrcrange validation;
                # stiffness=0/damping=0 ensures torque = kp*(t-q) - kd*dq = 0.
                "slide[1-4]": ControlInfo(
                    stiffness=0.0, damping=0.0, effort_limit=500, velocity_limit=2
                ),
            },
        )
    )
