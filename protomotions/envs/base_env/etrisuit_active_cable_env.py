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
"""ActiveCableEnv31Dof: DOFC A버전 cable control for 35 DOF suit (31 skeleton + 4 cables).

DOFC A버전 (simplified — no LPF, no bias removal, no delay):
  y = sin(-hip_r) - sin(-hip_l)
  tau_right = -kappa * y
  tau_left  = +kappa * y
  force = tau * ext_gain / pulley_radius   (N)
  target_pos = force / stiffness           (m, for BUILT_IN_PD)

Cable DOF mapping (skeleton_torque_suit_31dof, 35 DOFs):
  slide1 (DOF 31): unused
  slide2 (DOF 32): right hip flexion assist
  slide3 (DOF 33): unused
  slide4 (DOF 34): left hip flexion assist

Hip DOF indices (35 DOF):
  hip_flexion_r → DOF 0
  hip_flexion_l → DOF 7  (+2 vs 27 DOF: subtalar_r + mtp_r inserted before left leg)
"""
import torch
from torch import Tensor

from protomotions.envs.base_env.active_cable_env import _KAPPA, _EXT_GAIN, _PULLEY_RADIUS, _STIFFNESS
from protomotions.envs.base_env.env import BaseEnv

# DOF indices for 35 DOF robot (31 skeleton + 4 cables)
_HIP_R_31 = 0
_HIP_L_31 = 7   # was 5 in 27 DOF; +2 because subtalar_r(5), mtp_r(6) inserted
_SLIDE1_31 = 31  # unused
_SLIDE2_31 = 32  # right cable
_SLIDE3_31 = 33  # unused
_SLIDE4_31 = 34  # left cable


def _dofc_a_target_pos_31dof(dof_pos: Tensor):
    """DOFC A버전 for 35 DOF: 즉각 피드백, LPF/bias/delay 없음."""
    hip_r = dof_pos[:, _HIP_R_31]
    hip_l = dof_pos[:, _HIP_L_31]

    y = torch.sin(-hip_r) - torch.sin(-hip_l)
    tau_right = -_KAPPA * y
    tau_left = +_KAPPA * y

    target_slide2 = (-tau_right * _EXT_GAIN / _PULLEY_RADIUS) / _STIFFNESS
    target_slide4 = (-tau_left * _EXT_GAIN / _PULLEY_RADIUS) / _STIFFNESS

    return target_slide2, target_slide4


class ActiveCableEnv31Dof(BaseEnv):
    """BaseEnv with DOFC A버전 cable control for 35 DOF suit.

    PPO controls body DOFs (0–30). Cable DOFs (slide2=32, slide4=34) are
    overridden by DOFC A버전 rule-based targets. slide1/3 are zeroed.
    """

    def step(self, action: Tensor):
        self.extras = {}
        self._current_context = None
        self._current_noisy_obs = None
        self._current_raw_action[:] = action

        action_dict = self._process_action(action, self.context)
        processed_action = action_dict["processed_action"].clone()

        # DOFC A버전: override cable DOF target positions
        dof_pos = self.context.current.dof_pos
        s2, s4 = _dofc_a_target_pos_31dof(dof_pos)
        s2 = s2.clamp(min=0.0, max=0.51)
        s4 = s4.clamp(min=0.0, max=0.51)
        processed_action[:, _SLIDE2_31] = s2
        processed_action[:, _SLIDE4_31] = s4
        # slide1/3 are unused — zero displacement
        processed_action[:, _SLIDE1_31] = 0.0
        processed_action[:, _SLIDE3_31] = 0.0

        self._current_processed_action[:] = processed_action
        self.simulator.step(processed_action, markers_callback=self.get_markers_state)
        self.post_physics_step()

        if self.simulator.user_requested_reset:
            self.user_reset()

        obs = self.get_obs()
        return obs, self.rew_buf, self.reset_buf, self.terminate_buf, self.extras
