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
"""AI-SBC 통합 제어기 — HLP(ManiFlow) + LLP(assist RL)를 env 없이 실행.

- `LLPPolicy`: assist_pendulum 체크포인트에서 actor + 관측 레이아웃 복원
- `AssistController`: 관절각·몸통자세 스트림 → assist 토크 (실기기/시뮬 공용)
- `verify_controller`: 진자 env와의 step 단위 등가성 검증
"""

from protomotions.assist.controller import AssistController, AssistControllerConfig
from protomotions.assist.llp_policy import LLPObsLayout, LLPPolicy

__all__ = ["AssistController", "AssistControllerConfig", "LLPObsLayout", "LLPPolicy"]
