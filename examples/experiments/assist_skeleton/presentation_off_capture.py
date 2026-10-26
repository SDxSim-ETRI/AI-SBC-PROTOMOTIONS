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
"""Assist removal demos with an optional full-body wearer-torque replay control.

Run through the standard inference runner. Only this process changes behavior.
Record 400 Hz actual forces, so replay does not replace substep feedback with a
100 Hz averaged torque. The replay-on condition checks replay-induced drift.
"""

import argparse
import json
from pathlib import Path
import sys

import presentation_capture
import run_system
import numpy as np
import torch
import warp as wp


class RemovalHook(presentation_capture.PresentationHook):
    def __init__(self, *args, protocol, **kwargs):
        super().__init__(*args, **kwargs)
        self.protocol = protocol
        self.step_index = 0
        self.active = False
        self.replay_active = False
        self.steps_per_window = self.sim.hook_every
        self.switch_index = int(round(protocol.off_at / self.llp_dt))
        count = self.n_calls * self.steps_per_window
        shape = (count, 2, self.n_dofs)
        self.physics_human = torch.zeros(shape, device=self.env.device)
        self.physics_assist = torch.zeros_like(self.physics_human)
        self.physics_pd = torch.zeros_like(self.physics_human)
        self.physics_applied = torch.zeros_like(self.physics_human)
        self.source = None
        if protocol.reference:
            with np.load(protocol.reference / "traces.npz") as tr:
                self.source = {k: torch.as_tensor(tr[k], device=self.env.device)
                               for k in ("physics_human_tau", "physics_assist_tau")}
            meta = json.loads((protocol.reference / "meta.json").read_text())
            assert Path(meta["llp_checkpoint"]).resolve() == Path(self.args.llp_checkpoint).resolve()
            assert meta["physics_fps"] == self.args.physics_fps
            assert len(self.source["physics_human_tau"]) >= count
        if protocol.mode.startswith("replay") and self.source is None:
            raise ValueError("Replay requires the matching continuous-ON reference")
        self.disable_pd = run_system.JointTorqueOverride(
            self.sim, env_ids=[1], common_dof_indices=range(self.n_dofs)
        )
        mapping = self.sim.solver.mjc_dof_to_newton_dof.numpy()[0]
        common = self.sim.data_conversion.dof_convert_to_common.cpu().numpy()
        ids = [int(np.flatnonzero(mapping == i + self.sim._pd_qd_dof_start)[0]) for i in common]
        self.mj_ids = torch.tensor(ids, device=self.env.device)
        self.original_step = self.sim.solver.step
        self.sim.solver.step = self.physics_step

    @torch.no_grad()
    def __call__(self, sim, window):
        index = self.i
        if index >= self.n_calls:
            return
        super().__call__(sim, window)
        self.active = True
        if index >= self.switch_index:
            if self.protocol.mode == "feedback":
                self.override.set_torques(torch.zeros(2, self.n_dofs, device=self.env.device))
                self.log["tau_assist"][index].zero_()
            elif self.protocol.mode.startswith("replay"):
                if not self.replay_active:
                    self.disable_pd.engage(gain_scale=0.0)
                    self.replay_active = True
                assist = self.source["physics_assist_tau"][self.step_index, 1]
                if self.protocol.mode == "replay_off":
                    assist = torch.zeros_like(assist)
                self.log["tau_assist"][index, 1] = assist[self.hip_idx_t]
        if self.step_index:
            human = self.physics_human[self.step_index-self.steps_per_window:self.step_index].mean(dim=0)
            self.presentation_tau[index] = human
            self.log["tau_pd_prev"][index] = human[:, self.hip_idx_t]

    @torch.no_grad()
    def physics_step(self, state_in, state_out, control, contacts, dt):
        if not self.active:
            return self.original_step(state_in, state_out, control, contacts, dt)
        i = self.step_index
        assert i < len(self.physics_human)
        assist = torch.zeros(2, self.n_dofs, device=self.env.device)
        assist[:, self.hip_idx_t] = self.log["tau_assist"][self.i-1]
        if self.replay_active:
            assist[1] = self.source["physics_assist_tau"][i, 1]
            if self.protocol.mode == "replay_off":
                assist[1].zero_()
            injected = assist.clone()
            injected[1] += self.source["physics_human_tau"][i, 1]
            self.override.set_torques(injected, clamp_to_effort_limits=False)
        self.original_step(state_in, state_out, control, contacts, dt)
        data = self.sim.solver.mjw_data
        pd = wp.to_torch(data.qfrc_actuator)[:, self.mj_ids]
        applied = wp.to_torch(data.qfrc_applied)[:, self.mj_ids]
        self.physics_pd[i] = pd
        self.physics_applied[i] = applied
        self.physics_assist[i] = assist
        self.physics_human[i] = pd + applied - assist
        self.step_index += 1

    def traces(self):
        result = super().traces()
        assert self.step_index == self.i * self.steps_per_window
        result.update(
            physics_human_tau=self.physics_human[:self.step_index].cpu().numpy(),
            physics_assist_tau=self.physics_assist[:self.step_index].cpu().numpy(),
            physics_raw_pd_tau=self.physics_pd[:self.step_index].cpu().numpy(),
            physics_applied_tau=self.physics_applied[:self.step_index].cpu().numpy(),
            removal_mode=np.asarray(self.protocol.mode),
            removal_time_s=np.asarray(self.protocol.off_at),
        )
        return result


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--removal-mode", dest="mode", choices=["reference", "feedback", "replay_on", "replay_off"], required=True)
    parser.add_argument("--off-at", type=float, default=10.0)
    parser.add_argument("--replay-reference", dest="reference", type=Path)
    protocol, remaining = parser.parse_known_args()
    # The unchanged runner owns environment creation, frozen policies and output.
    run_system.parse_args = lambda: base_parser(remaining)
    run_system.SystemHook = lambda *a, **kw: RemovalHook(*a, protocol=protocol, **kw)
    run_system.main()


base_parser = run_system.parse_args
if __name__ == "__main__":
    main()
