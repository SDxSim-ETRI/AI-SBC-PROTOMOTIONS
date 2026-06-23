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
from protomotions.robot_configs.base import RobotConfig


def robot_config(robot_name: str, **updates) -> RobotConfig:
    """Factory function to create robot configuration based on robot type.

    Args:
        robot_name: Name of the robot type (smpl, smplx, amp, g1, h1_2, soma23, rigv1, skeleton_torque)
        **updates: Optional field updates to apply to the robot config

    Returns:
        RobotConfig: Robot configuration object

    Raises:
        ValueError: If robot_name is not recognized
    """
    if robot_name == "smpl":
        from protomotions.robot_configs.smpl import SmplRobotConfig

        config = SmplRobotConfig()
    elif robot_name == "smplx":
        from protomotions.robot_configs.smplx import SMPLXRobotConfig

        config = SMPLXRobotConfig()
    elif robot_name == "amp":
        from protomotions.robot_configs.amp import AMPRobotConfig

        config = AMPRobotConfig()
    elif robot_name == "g1":
        from protomotions.robot_configs.g1 import G1RobotConfig

        config = G1RobotConfig()
    elif robot_name == "h1_2":
        from protomotions.robot_configs.h1_2 import H1_2RobotConfig

        config = H1_2RobotConfig()
    elif robot_name == "rigv1":
        from protomotions.robot_configs.rigv1 import Rigv1RobotConfig

        config = Rigv1RobotConfig()
    elif robot_name == "soma23":
        from protomotions.robot_configs.soma23 import Soma23RobotConfig

        config = Soma23RobotConfig()
    elif robot_name == "skeleton_torque":
        from protomotions.robot_configs.skeleton_torque import SkeletonTorqueRobotConfig

        config = SkeletonTorqueRobotConfig()
    elif robot_name == "skeleton_torque_suit":
        from protomotions.robot_configs.skeleton_torque_suit import SkeletonTorqueSuitRobotConfig

        config = SkeletonTorqueSuitRobotConfig()
    elif robot_name == "skeleton_torque_suit_muscle":
        from protomotions.robot_configs.skeleton_torque_suit_muscle import SkeletonTorqueSuitMuscleRobotConfig

        config = SkeletonTorqueSuitMuscleRobotConfig()
    elif robot_name == "skeleton_torque_suit_passive_cable":
        from protomotions.robot_configs.skeleton_torque_suit_passive_cable import SkeletonTorqueSuitPassiveCableRobotConfig

        config = SkeletonTorqueSuitPassiveCableRobotConfig()
    elif robot_name == "skeleton_torque_suit_active_cable":
        from protomotions.robot_configs.skeleton_torque_suit_active_cable import SkeletonTorqueSuitActiveCableRobotConfig

        config = SkeletonTorqueSuitActiveCableRobotConfig()
    elif robot_name in ("skeleton_torque_31dof", "skeleton"):
        from protomotions.robot_configs.skeleton import SkeletonRobotConfig

        config = SkeletonRobotConfig()
    elif robot_name in ("skeleton_torque_suit_31dof", "etrisuit"):
        from protomotions.robot_configs.etrisuit import EtriSuitRobotConfig

        config = EtriSuitRobotConfig()
    elif robot_name == "etrisuit_active_cable":
        from protomotions.robot_configs.etrisuit_active_cable import EtriSuitActiveCableRobotConfig

        config = EtriSuitActiveCableRobotConfig()
    else:
        raise ValueError(f"Invalid robot name: {robot_name}")

    # Apply any updates
    if updates:
        config.update_fields(**updates)

    return config
