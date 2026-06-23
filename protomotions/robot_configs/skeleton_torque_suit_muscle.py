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
class SkeletonTorqueSuitMuscleRobotConfig(RobotConfig):
    """
    Biomechanical muscle skeleton + ETRI exoskeleton suit — visualization config.
    28 bodies (world excluded), 35 active DOFs.

    Body ordering matches skeleton_torque_suit (28 bodies):
      pelvis(0)
      femur_r(1) tibia_r(2) talus_r(3) calcn_r(4) toes_r(5)
      RH_dump(6) RH_dump2(7)
      femur_l(8) tibia_l(9) talus_l(10) calcn_l(11) toes_l(12)
      LH_dump(13) LH_dump2(14)
      torso(15)
      humerus_r(16) ulna_r(17) radius_r(18) hand_r(19)
      humerus_l(20) ulna_l(21) radius_l(22) hand_l(23)
      slider1(24) slider2(25) slider3(26) slider4(27)

    DOF ordering (31 DOFs):
      hip_flexion_r/adduction_r/rotation_r  knee_angle_r  ankle_angle_r
      subtalar_angle_r  mtp_angle_r                                       (7)
      hip_flexion_l/adduction_l/rotation_l  knee_angle_l  ankle_angle_l
      subtalar_angle_l  mtp_angle_l                                       (7)
      lumbar_extension/bending/rotation                                   (3)
      arm_flex_r/add_r/rot_r  elbow_flex_r  pro_sup_r                    (5)
      arm_flex_l/add_l/rot_l  elbow_flex_l  pro_sup_l                    (5)
      slide1  slide2  slide3  slide4                                      (4)

    subtalar/mtp DOFs are equality-constrained to 0 in the MJCF.
    Wrist joints removed (also locked to 0, violate pose_lib 1-or-3 rule).
    Use skeleton_torque_suit_muscle_motions.pt (31 DOFs) as motion file.
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

    default_root_height: float = 0.975

    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_file_name="mjcf/skeleton_torque_suit_muscle.xml",
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
                "(subtalar_angle|mtp_angle)_[rl]": ControlInfo(
                    stiffness=50.0, damping=5.0, effort_limit=50, velocity_limit=5
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
                "slide[1-4]": ControlInfo(
                    stiffness=50.0, damping=5.0, effort_limit=500, velocity_limit=2
                ),
            },
        )
    )

    simulation_params: SimulatorParams = field(
        default_factory=lambda: SimulatorParams(
            isaacgym=IsaacGymSimParams(fps=40, decimation=2, substeps=2),
            isaaclab=IsaacLabSimParams(fps=120, decimation=6),
            genesis=GenesisSimParams(fps=60, decimation=3, substeps=2),
            newton=NewtonSimParams(fps=120, decimation=6),
            mujoco=MujocoSimParams(fps=40, decimation=2),
        )
    )
