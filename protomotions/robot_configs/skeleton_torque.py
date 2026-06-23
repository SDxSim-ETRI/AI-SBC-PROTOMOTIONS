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
from protomotions.simulator.isaacgym.config import IsaacGymSimParams
from protomotions.simulator.isaaclab.config import IsaacLabSimParams
from protomotions.simulator.genesis.config import GenesisSimParams
from protomotions.simulator.newton.config import NewtonSimParams
from protomotions.simulator.mujoco.config import MujocoSimParams
from protomotions.components.pose_lib import ControlInfo
from typing import List, Dict
from dataclasses import dataclass, field


@dataclass
class SkeletonTorqueRobotConfig(RobotConfig):
    """
    Biomechanical human skeleton model (GZ_DW_sub01, Hamner2010 OpenSim).
    21 bodies (world + 20), 23 active hinge DOFs.

    DOF ordering:
        hip_flexion_r/adduction_r/rotation_r  knee_angle_r  ankle_angle_r  (5)
        hip_flexion_l/adduction_l/rotation_l  knee_angle_l  ankle_angle_l  (5)
        lumbar_extension/bending/rotation                                   (3)
        arm_flex_r/add_r/rot_r  elbow_flex_r  pro_sup_r                    (5)
        arm_flex_l/add_l/rot_l  elbow_flex_l  pro_sup_l                    (5)

    See mjcf/skeleton_torque.xml for full definition.
    """

    common_naming_to_robot_body_names: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "all_left_foot_bodies": ["calcn_l", "toes_l"],
            "all_right_foot_bodies": ["calcn_r", "toes_r"],
            "all_left_hand_bodies": ["hand_l"],
            "all_right_hand_bodies": ["hand_r"],
            "head_body_name": ["torso"],
            "torso_body_name": ["torso"],
        }
    )

    trackable_bodies_subset: List[str] = field(
        default_factory=lambda: [
            "torso",
            "calcn_r",
            "calcn_l",
            "hand_r",
            "hand_l",
        ]
    )

    contact_bodies: List[str] = field(
        default_factory=lambda: [
            "calcn_r",   # 오른발 뒤꿈치+발바닥 (calcaneus)
            "toes_r",    # 오른발 발가락
            "calcn_l",   # 왼발 뒤꿈치+발바닥
            "toes_l",    # 왼발 발가락
        ]
    )

    default_root_height: float = 0.975

    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_file_name="mjcf/skeleton_torque.xml",
            usd_asset_file_name="usd/skeleton_torque/skeleton_torque.usda",
            usd_bodies_root_prim_path="/World/envs/env_.*/Robot/pelvis/",
        )
    )

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
            },
        )
    )

    simulation_params: SimulatorParams = field(
        default_factory=lambda: SimulatorParams(
            isaacgym=IsaacGymSimParams(
                fps=40,
                decimation=2,
                substeps=2,
            ),
            isaaclab=IsaacLabSimParams(
                fps=120,
                decimation=6,
            ),
            genesis=GenesisSimParams(
                fps=60,
                decimation=3,
                substeps=2,
            ),
            newton=NewtonSimParams(
                fps=120,
                decimation=6,
            ),
            mujoco=MujocoSimParams(
                fps=120,
                decimation=6,
            ),
        )
    )
