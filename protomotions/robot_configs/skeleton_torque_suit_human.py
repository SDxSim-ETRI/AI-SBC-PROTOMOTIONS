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
"""사람 범위 근력의 착용자 — 슈트 관절 수동(토크 0) + **hip 토크 한계 200 N·m**.

`skeleton_torque_suit_passive_cable`과 같되 고관절(굴곡·내전·회전)의 effort_limit만 600 → 200이다.

왜: 착용자 v32(hip 한계 600)는 tracker가 50 ms마다 내는 PD 목표각을 토크 손잡이로 써서 착지마다
−332 N·m(3.7 N·m/kg)를 낸다 — 문헌의 보행 hip 모멘트 피크는 약 1 N·m/kg다(`docs/NEXT_SESSION.md`
§2-E). 보조 정책의 τ² 절감률이 이 "사람보다 3.7배 센" 착용자 기준으로 재어지고, 진자 학습 분포
(하중 ≤120 N·m)와도 어긋난다. 200 N·m(2.2 N·m/kg)는 달리기까지 덮으면서 슬램은 막는 값이다.

무릎·발목·허리·팔·슈트 slide 관절은 그대로. LLP 관측·보조 주입 경로도 그대로(hip qfrc).
tracker 재학습 레시피: `examples/experiments/assist_skeleton/tracker_human.py` (v33).
"""
from dataclasses import dataclass, field

from protomotions.components.pose_lib import ControlInfo
from protomotions.robot_configs.base import ControlConfig, ControlType
from protomotions.robot_configs.skeleton_torque_suit_passive_cable import (
    SkeletonTorqueSuitPassiveCableRobotConfig,
)

HIP_EFFORT_LIMIT_NM = 200.0


@dataclass
class SkeletonTorqueSuitHumanRobotConfig(SkeletonTorqueSuitPassiveCableRobotConfig):
    """수동 슈트 관절 + hip effort_limit 200 N·m (사람 범위)."""

    control: ControlConfig = field(
        default_factory=lambda: ControlConfig(
            control_type=ControlType.BUILT_IN_PD,
            override_control_info={
                "hip_(flexion|adduction)_[rl]": ControlInfo(
                    stiffness=200.0, damping=20.0, effort_limit=HIP_EFFORT_LIMIT_NM, velocity_limit=10
                ),
                "hip_rotation_[rl]": ControlInfo(
                    stiffness=200.0, damping=20.0, effort_limit=HIP_EFFORT_LIMIT_NM, velocity_limit=10
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
                # 슈트 slide 관절 — 수동(토크 0). effort_limit은 MuJoCo actfrcrange 검증용으로만 > 0.
                "slide[1-4]": ControlInfo(
                    stiffness=0.0, damping=0.0, effort_limit=500, velocity_limit=2
                ),
            },
        )
    )
