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
"""Limit wearer hip effort to measured assisted peaks; retain native PD feedback."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import warp as wp
from newton.solvers import SolverNotifyFlags

import presentation_off_capture as removal
import run_system


class CappedHook(removal.RemovalHook):
    def __init__(self, *args, cap_protocol, **kwargs):
        super().__init__(*args, **kwargs)
        self.cap_protocol = cap_protocol
        self.cap_active = False
        with np.load(cap_protocol.cap_reference / "traces.npz") as source:
            assert source["presentation_dof_names"].tolist() == self.env.robot_config.kinematic_info.dof_names
            fps = self.args.physics_fps
            lo, hi = round(cap_protocol.cap_start * fps), round(cap_protocol.cap_end * fps)
            values = source["physics_human_tau"][lo:hi, 1][:, self.hip_idx]
            self.caps = torch.as_tensor(np.abs(values).max(axis=0), device=self.env.device)
        self.cap_index = round(cap_protocol.limit_at / self.llp_dt)
        self.cap_audit = {}

    def apply_caps(self):
        sim = self.sim
        gains = {name: wp.to_torch(getattr(sim.model, name)).clone()
                 for name in ("joint_target_ke", "joint_target_kd")}
        limits_wp = sim.robot_view.get_attribute("joint_effort_limit", sim.model)
        limits = wp.to_torch(limits_wp).view(sim.num_envs, -1)
        old = limits.clone()
        indices = sim.data_conversion.dof_convert_to_common[self.hip_idx_t]
        limits[1, indices] = self.caps
        sim.robot_view.set_attribute("joint_effort_limit", sim.model, limits_wp)
        sim.solver.notify_model_changed(SolverNotifyFlags.JOINT_DOF_PROPERTIES)
        for name, before in gains.items():
            assert torch.equal(wp.to_torch(getattr(sim.model, name)), before), name
        assert int((old != limits).sum()) == 2
        # Joint-level force limits cap the summed native PD actuator contribution.
        # External assistance remains independently injected via qfrc_applied.
        mapping = sim.solver.mjc_jnt_to_newton_dof.numpy()
        dofs_per_world = sim.model.joint_dof_count // sim.num_envs
        joint_ids = [int(np.flatnonzero(mapping[1] == dofs_per_world + sim._pd_qd_dof_start + int(j))[0])
                     for j in indices]
        ranges = wp.to_torch(sim.solver.mjw_model.jnt_actfrcrange)[1, joint_ids]
        assert torch.allclose(ranges[:, 0], -self.caps)
        assert torch.allclose(ranges[:, 1], self.caps)
        self.cap_audit = {"PD_gains_unchanged": True, "changed_effort_limits": 2,
                          "original_caps_Nm": old[1, indices].cpu().tolist(),
                          "caps_Nm": self.caps.cpu().tolist(), "joint_ids": joint_ids}
        print("HIP_CAP_AUDIT " + json.dumps(self.cap_audit), flush=True)
        self.cap_active = True

    @torch.no_grad()
    def __call__(self, sim, window):
        if self.i >= self.cap_index and not self.cap_active:
            self.apply_caps()
        super().__call__(sim, window)

    def traces(self):
        result = super().traces()
        assert self.cap_active
        measured = self.physics_human[self.cap_index*self.steps_per_window:self.step_index, 1][:, self.hip_idx_t]
        excess = (measured.abs() - self.caps).max().item()
        assert excess < 1e-3, f"Hip effort limit not enforced: {excess}"
        result.update(hip_effort_caps=self.caps.cpu().numpy(), hip_effort_cap_time_s=np.asarray(self.cap_protocol.limit_at),
                      hip_effort_cap_reference=np.asarray(str(self.cap_protocol.cap_reference)),
                      hip_effort_cap_audit=np.asarray(json.dumps(self.cap_audit)))
        return result


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cap-reference", type=Path, required=True)
    parser.add_argument("--cap-start", type=float, default=5.0)
    parser.add_argument("--cap-end", type=float, default=20.0)
    parser.add_argument("--limit-at", type=float, default=5.0)
    parser.add_argument("--condition", choices=["on", "off"], required=True)
    parser.add_argument("--off-at", type=float, default=10.0)
    cap, remaining = parser.parse_known_args()
    source_meta = json.loads((cap.cap_reference / "meta.json").read_text())
    args = run_system.parse_args(remaining)
    assert Path(source_meta["llp_checkpoint"]).resolve() == Path(args.llp_checkpoint).resolve()
    assert source_meta["physics_fps"] == args.physics_fps
    protocol = argparse.Namespace(mode="reference" if cap.condition == "on" else "feedback", off_at=cap.off_at, reference=None)
    run_system.parse_args = lambda: args
    run_system.SystemHook = lambda *a, **kw: CappedHook(*a, protocol=protocol, cap_protocol=cap, **kw)
    run_system.main()


if __name__ == "__main__":
    main()
