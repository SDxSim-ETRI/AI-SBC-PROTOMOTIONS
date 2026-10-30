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
"""2026-10-14/15 진자 연구 checkpoint(C0·S1·A1·T1·AB1·PA 계열)의 저장 설정을 읽기 위한 호환 이름.

당시 ``PendulumResearchControl``은 ``DisturbedAssistTargetControl``에 착용자 명령 sample-and-hold
(``human_target_hold_steps``)만 더한 변형이었고, 2026-10-26 정리(ec95c44)에서 그 기능이 본체로 흡수됐다.
저장된 ``resolved_configs*.pt``는 이 모듈 경로를 pickle로 가리키므로, 같은 클래스를 이 이름으로도 내보낸다.
새 코드는 ``assist_disturbance_control``을 직접 쓴다.
"""

from protomotions.envs.control.assist_disturbance_control import (
    DisturbedAssistTargetControl,
    DisturbedAssistTargetControlConfig,
)

PendulumResearchControlConfig = DisturbedAssistTargetControlConfig
PendulumResearchControl = DisturbedAssistTargetControl

__all__ = ["PendulumResearchControl", "PendulumResearchControlConfig"]
