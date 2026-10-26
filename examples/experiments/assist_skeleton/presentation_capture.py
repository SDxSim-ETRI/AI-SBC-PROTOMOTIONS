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
"""Read-only 100 Hz pose capture around the standard paired inference runner.

Accepts run_system.py arguments. Rendering is performed separately from saved
poses, so presentation cameras and frame rates cannot affect the physics.
"""

from pathlib import Path

import run_system

import numpy as np
import torch


class PresentationHook(run_system.SystemHook):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        count = len(self.log["t"])
        kin = self.env.robot_config.kinematic_info
        shape = (count, self.sim.num_envs, len(kin.body_names))
        self.presentation_pos = torch.zeros(*shape, 3, device=self.env.device)
        self.presentation_rot = torch.zeros(*shape, 4, device=self.env.device)
        self.presentation_tau = torch.zeros(
            count, self.sim.num_envs, self.n_dofs, device=self.env.device
        )

    def __call__(self, sim, window):
        super().__call__(sim, window)
        index = self.i - 1
        state = sim._get_simulator_bodies_state()
        common = sim.data_conversion.body_convert_to_common
        self.presentation_pos[index].copy_(state.rigid_body_pos[:, common])
        self.presentation_rot[index].copy_(state.rigid_body_rot[:, common])
        self.presentation_tau[index].copy_(sim.window_tau_pd_common)

    def traces(self):
        traces = super().traces()
        kin = self.env.robot_config.kinematic_info
        asset = self.env.robot_config.asset
        traces.update(
            presentation_body_pos=self.presentation_pos[:self.i].cpu().numpy(),
            presentation_body_rot=self.presentation_rot[:self.i].cpu().numpy(),
            presentation_human_tau_prev=self.presentation_tau[:self.i].cpu().numpy(),
            presentation_body_names=np.asarray(kin.body_names),
            presentation_dof_names=np.asarray(kin.dof_names),
            presentation_asset=np.asarray(str((Path(asset.asset_root) / asset.asset_file_name).resolve())),
        )
        return traces


if __name__ == "__main__":
    run_system.SystemHook = PresentationHook
    run_system.main()
