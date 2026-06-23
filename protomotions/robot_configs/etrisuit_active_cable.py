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
"""35 DOF suit robot config with active cable joints controlled by DOFC A버전.

Extends SkeletonTorqueSuit31DofRobotConfig (35 DOF = 31 skeleton + 4 cables).
Cable DOFs (slide1–4, DOF indices 31–34) retain stiffness=50/damping=5 so
DOFC-computed target positions generate real cable forces.

DOFC cable mapping (35 DOF):
  slide1 (DOF 31): unused (force=0)
  slide2 (DOF 32): right hip flexion assist
  slide3 (DOF 33): unused (force=0)
  slide4 (DOF 34): left hip flexion assist

Hip DOF indices (35 DOF):
  hip_flexion_r → DOF 0  (same as 27 DOF)
  hip_flexion_l → DOF 7  (+2 vs 27 DOF due to subtalar_r + mtp_r)

Use with ActiveCableEnv31Dof (_target_ in experiment config).

Warm-start chain:
  v29_isaaclab_active_cable_rough27 (27 DOF)
    → zero-pad 27→35 DOF (tools/expand_checkpoint_dof.py)
    → newton_suit_active_cable_walk_flat (this config)
"""
from dataclasses import dataclass

from protomotions.robot_configs.etrisuit import EtriSuitRobotConfig


@dataclass
class EtriSuitActiveCableRobotConfig(EtriSuitRobotConfig):
    """ETRI 소프트슈트 35 DOF with active cable joints (DOFC A버전).

    Identical to SkeletonTorqueSuit31DofRobotConfig — cable DOFs (slide1–4)
    already have stiffness=50/damping=5 for DOFC-computed target positions.
    The ActiveCableEnv31Dof overrides PPO output for DOFs 31–34 at each step.
    """
    # Control config inherited unchanged from SkeletonTorqueSuit31DofRobotConfig.
    # slide[1-4] already set to stiffness=50, damping=5 — correct for active cable.
    pass
