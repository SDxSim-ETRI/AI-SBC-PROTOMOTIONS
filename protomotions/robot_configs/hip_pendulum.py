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
from protomotions.robot_configs.base import (
    RobotConfig,
    RobotAssetConfig,
    ControlConfig,
    ControlType,
    SimulatorParams,
)
from protomotions.simulator.newton.config import NewtonSimParams
from protomotions.simulator.mujoco.config import MujocoSimParams
from protomotions.components.pose_lib import ControlInfo
from typing import List, Dict
from dataclasses import dataclass, field


@dataclass
class HipPendulumRobotConfig(RobotConfig):
    """AI-SBC LLP 1단계용 2-DOF hip pendulum (고정 골반 + 좌/우 대퇴 hinge).

    구조:
        pelvis (월드 고정, fix_base_link=True)
        ├── right_hip_pitch (hinge, Y축) — right_thigh (~7 kg 캡슐)
        └── left_hip_pitch  (hinge, Y축) — left_thigh  (~7 kg 캡슐)

    DOF ordering (COMMON): [right_hip_pitch, left_hip_pitch]

    제어: BUILT_IN_PD가 "사람" 모델 (θ_g 추종 PD, 매 substep implicit 평가),
    RL의 assist torque는 JointTorqueOverride(qfrc_applied)로 additive 주입.
    사람 게인 근사: Kp=300 Nm/rad, Kd=15, 최대 토크 150 Nm (성인 hip flexion).
    """

    common_naming_to_robot_body_names: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "all_left_foot_bodies": ["left_thigh"],
            "all_right_foot_bodies": ["right_thigh"],
            "all_left_hand_bodies": ["left_thigh"],
            "all_right_hand_bodies": ["right_thigh"],
            "head_body_name": ["pelvis"],
            "torso_body_name": ["pelvis"],
        }
    )

    trackable_bodies_subset: List[str] = field(
        default_factory=lambda: ["pelvis", "right_thigh", "left_thigh"]
    )

    default_root_height: float = 1.0

    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_file_name="mjcf/hip_pendulum.xml",
            fix_base_link=True,
            self_collisions=False,
        )
    )

    control: ControlConfig = field(
        default_factory=lambda: ControlConfig(
            control_type=ControlType.BUILT_IN_PD,
            override_control_info={
                "(left|right)_hip_pitch": ControlInfo(
                    stiffness=300.0,
                    damping=15.0,
                    effort_limit=150.0,
                    velocity_limit=30.0,
                ),
            },
        )
    )

    simulation_params: SimulatorParams = field(
        default_factory=lambda: SimulatorParams(
            newton=NewtonSimParams(
                fps=400,
                decimation=4,  # 물리 400 Hz, policy 100 Hz
            ),
            mujoco=MujocoSimParams(
                fps=400,
                decimation=4,
            ),
        )
    )
