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
"""ActiveCableEnv: BaseEnv subclass that injects DOFC A버전 cable control.

DOFC A버전 (simplified — no LPF, no bias removal, no delay):
  y = sin(-hip_r) - sin(-hip_l)
  tau_right = -kappa * y
  tau_left  = +kappa * y
  force = tau * ext_gain / pulley_radius   (N)
  target_pos = force / stiffness           (m, for BUILT_IN_PD)

Cable DOF mapping (skeleton_torque_suit, 27 DOFs):
  slide1 (DOF 23): unused
  slide2 (DOF 24): right hip flexion assist
  slide3 (DOF 25): unused
  slide4 (DOF 26): left hip flexion assist

Hip DOF indices:
  hip_flexion_r → DOF 0
  hip_flexion_l → DOF 5
"""
import math

import torch
from torch import Tensor

from protomotions.envs.base_env.env import BaseEnv

# ── DOFC A버전 파라미터 ──────────────────────────────────────────────────────
_KAPPA = 1.1
_EXT_GAIN = 0.8
_PULLEY_RADIUS = 0.042  # m
_STIFFNESS = 50.0       # N/m (slide joint stiffness)

# DOF indices
_HIP_R = 0
_HIP_L = 5
_SLIDE2 = 24  # right cable
_SLIDE4 = 26  # left cable


def _dofc_a_target_pos(dof_pos: Tensor):
    """DOFC A버전: 즉각 피드백, LPF/bias/delay 없음.

    Args:
        dof_pos: [num_envs, num_dofs] joint positions in radians

    Returns:
        (target_slide2, target_slide4): target positions in meters [num_envs]
    """
    hip_r = dof_pos[:, _HIP_R]
    hip_l = dof_pos[:, _HIP_L]

    # DOFC 부호 규칙: -hip (굴곡 양수 → sin 양수)
    y = torch.sin(-hip_r) - torch.sin(-hip_l)

    tau_right = -_KAPPA * y
    tau_left = +_KAPPA * y

    # 토크 → 케이블 힘 → PD 타겟 위치 (stiffness=50 → target=force/stiffness)
    target_slide2 = (-tau_right * _EXT_GAIN / _PULLEY_RADIUS) / _STIFFNESS
    target_slide4 = (-tau_left * _EXT_GAIN / _PULLEY_RADIUS) / _STIFFNESS

    return target_slide2, target_slide4


class ActiveCableEnv(BaseEnv):
    """BaseEnv with DOFC A버전 cable control injected at each step.

    PPO controls body DOFs (0–22). Cable DOFs (slide2=24, slide4=26) are
    overridden by DOFC A버전 rule-based targets. slide1/3 are zeroed.
    """

    def step(self, action: Tensor):
        self.extras = {}
        self._current_context = None
        self._current_noisy_obs = None
        self._current_raw_action[:] = action

        # Process PPO action (this also rebuilds self.context)
        action_dict = self._process_action(action, self.context)
        processed_action = action_dict["processed_action"].clone()

        # DOFC A버전: override cable DOF target positions
        # 케이블은 단방향(당김만 가능) → 음수 타겟을 0으로 클램핑
        dof_pos = self.context.current.dof_pos
        s2, s4 = _dofc_a_target_pos(dof_pos)
        s2 = s2.clamp(min=0.0, max=0.51)
        s4 = s4.clamp(min=0.0, max=0.51)
        processed_action[:, _SLIDE2] = s2
        processed_action[:, _SLIDE4] = s4
        # slide1/3 are unused — zero displacement
        processed_action[:, 23] = 0.0
        processed_action[:, 25] = 0.0

        self._current_processed_action[:] = processed_action
        self.simulator.step(processed_action, markers_callback=self.get_markers_state)
        self.post_physics_step()

        if self.simulator.user_requested_reset:
            self.user_reset()

        obs = self.get_obs()
        return obs, self.rew_buf, self.reset_buf, self.terminate_buf, self.extras
