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
"""Opt-in readback of whole-body walking dynamics, outside policy observations.

Control records are at the START of each 10 ms window. Dynamics/contact records
are sampled on EACH completed 2.5 ms solver step, then grouped by their window.
MuJoCo qacc is the force-balance acceleration, not implicit-integrator delta-v/dt.
No targets, torques, model properties, solver options or RNG streams are changed.
"""

import json
from pathlib import Path

import mujoco
import numpy as np
import torch
import warp as wp
from newton.sensors import SensorContact


class FullBodyDiagnostic:
    def __init__(self, sim, n_calls, out_dir):
        self.sim = sim
        self.out_dir = Path(out_dir)
        self.calls = 0
        self.steps = 0
        self.every = sim.hook_every
        solver = sim.solver
        self.model = solver.mj_model
        self.data = solver.mjw_data
        if solver.use_mujoco_cpu or not solver._use_mujoco_contacts:
            raise ValueError("Full-body readback requires the GPU MuJoCo contact path")
        n, nv = sim.num_envs, self.model.nv
        self.names = list(sim.robot_config.kinematic_info.dof_names)
        self.body_names = list(sim.robot_config.kinematic_info.body_names)
        self.contact_names = list(sim.robot_config.contact_bodies)
        joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
                       for j in range(self.model.njnt)]
        # Use the solver's coordinate map: Newton merges three-axis joints and
        # renames them in MuJoCo, so matching human-readable joint names is unsafe.
        mj_to_newton = solver.mjc_dof_to_newton_dof.numpy()[0]
        common_to_newton = sim.data_conversion.dof_convert_to_common.cpu().numpy() + sim._pd_qd_dof_start
        self.dof_ids = []
        for name, newton_id in zip(self.names, common_to_newton):
            matches = np.flatnonzero(mj_to_newton == newton_id)
            if len(matches) != 1:
                raise ValueError(f"Cannot map {name} into MuJoCo joints: {joint_names}")
            self.dof_ids.append(int(matches[0]))
        self.hips = [self.dof_ids[self.names.index('hip_flexion_' + side)] for side in ('r', 'l')]
        self.hips_t = torch.tensor(self.hips, device=sim.device)
        self.common = sim.data_conversion.dof_convert_to_common
        self.contacts = sim.model.contacts()
        self.sensors = {}
        for name in self.contact_names:
            sensor = SensorContact(sim.model, sensing_obj_bodies=sim._get_contact_sensor_body_patterns(name),
                                   verbose=False)
            sim._validate_contact_sensor_match(name, sim._get_contact_sensor_body_patterns(name),
                                              sim._count_sensor_sensing_objects(sensor))
            self.sensors[name] = sensor

        # qM can be dense or packed along each DOF's ancestor chain.
        self.sparse = self.data.qM.shape[1] == 1
        if self.sparse:
            addresses = np.full((nv, nv), -1, dtype=np.int64)
            for row in range(nv):
                col, address = row, int(self.model.dof_Madr[row])
                while col >= 0:
                    addresses[row, col] = addresses[col, row] = address
                    col = int(self.model.dof_parentid[col])
                    address += 1
            addresses = addresses[self.hips]
            self.mass_valid = torch.tensor(addresses >= 0, device=sim.device)
            self.mass_addresses = torch.tensor(np.maximum(addresses, 0), device=sim.device)

        def zeros(count, *shape):
            return torch.zeros(count, n, *shape, device=sim.device)

        self.control_log = {k: zeros(n_calls, len(self.names)) for k in
                            ('dof_pos', 'dof_vel', 'pd_target', 'pd_target_velocity', 'reference_pos',
                             'reference_vel', 'tau_pd_prev', 'tau_ff_prev')}
        for key, width in (('body_pos', 3), ('body_rot', 4), ('body_vel', 3), ('body_ang_vel', 3)):
            self.control_log[key] = zeros(n_calls, len(self.body_names), width)
        count = n_calls * self.every
        self.physics_log = {k: zeros(count, nv) for k in
                            ('qvel_before', 'qvel_after', 'qacc', 'qfrc_bias', 'qfrc_passive',
                             'qfrc_actuator', 'qfrc_applied', 'qfrc_constraint')}
        self.physics_log['qpos_before'] = zeros(count, self.model.nq)
        self.physics_log['hip_mass_rows'] = zeros(count, 2, nv)
        self.physics_log['contact_force_world'] = zeros(count, len(self.contact_names), 3)
        self.constraint_dense = self.data.efc.J.shape[1] > 1
        self.physics_log['hip_constraint_contact'] = zeros(count, 2)
        self.physics_log['hip_constraint_other'] = zeros(count, 2)
        self.constraint_rows = torch.arange(self.data.efc.force.shape[1], device=sim.device)[None, :]
        self.jac_columns = torch.arange(nv, device=sim.device)[None, None, :]
        self.original_step = solver.step
        solver.step = self._step
        mujoco.mj_saveModel(self.model, str(self.out_dir / 'diagnostic_model.mjb'), None)
        gains = {}
        for name in ('joint_target_ke', 'joint_target_kd'):
            value = wp.to_torch(sim.robot_view.get_attribute(name, sim.model)).reshape(n, -1)
            gains[name] = value[:, self.common].cpu().tolist()
        self.meta = dict(
            dof_names_common=self.names, body_names_common=self.body_names,
            contact_body_names=self.contact_names, mj_joint_names=joint_names,
            common_dof_to_mj_dof=self.dof_ids, hip_mj_dofs=self.hips,
            mj_nv=nv, mj_nq=self.model.nq, sim_dt=sim.sim_dt, substeps_per_window=self.every,
            qM_sparse=self.sparse, gains=gains,
            constraint_type_split=True,
            root_mj_dofs=[int(np.flatnonzero(mj_to_newton == i)[0]) for i in range(sim._pd_qd_dof_start)],
            timing='control=start of 10ms window; physics=each solver substep; qacc=force-balance acceleration',
            contacts='independent SensorContact objects; world force on sensing foot body; each physics substep',
            constraint_caveat='qfrc_constraint includes contacts, joint limits, friction and equality constraints',
            changes_policy_inputs=False,
        )
        print(f'[fullbody] enabled: {nv} velocity DOFs, hips {self.hips}, {count} substeps', flush=True)

    @torch.no_grad()
    def _step(self, state_in, state_out, control, contacts, dt):
        i = self.steps
        if i >= len(self.physics_log['qacc']):
            raise RuntimeError('Full-body diagnostic buffer exhausted')
        log = self.physics_log
        log['qpos_before'][i] = wp.to_torch(self.data.qpos)
        log['qvel_before'][i] = wp.to_torch(self.data.qvel)
        self.original_step(state_in, state_out, control, contacts, dt)
        log['qvel_after'][i] = wp.to_torch(self.data.qvel)
        for key in ('qacc', 'qfrc_bias', 'qfrc_passive', 'qfrc_actuator', 'qfrc_applied', 'qfrc_constraint'):
            log[key][i] = wp.to_torch(getattr(self.data, key))
        force = wp.to_torch(self.data.efc.force)
        valid = self.constraint_rows < wp.to_torch(self.data.nefc)[:, None]
        kind = wp.to_torch(self.data.efc.type)
        contact_kind = kind >= int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS)
        if self.constraint_dense:
            jac = wp.to_torch(self.data.efc.J)[:, :force.shape[1], self.hips_t]
        else:
            rowadr = wp.to_torch(self.data.efc.J_rowadr).long()
            rownnz = wp.to_torch(self.data.efc.J_rownnz)
            packed = wp.to_torch(self.data.efc.J)[:, 0]
            addresses = (rowadr[:, :, None] + self.jac_columns).clamp(0, packed.shape[1] - 1)
            values = packed.gather(1, addresses.flatten(1)).reshape_as(addresses)
            columns = wp.to_torch(self.data.efc.J_colind)[:, 0].gather(1, addresses.flatten(1)).reshape_as(addresses)
            values = values * (self.jac_columns < rownnz[:, :, None])
            jac = torch.stack([(values * (columns == hip)).sum(dim=-1) for hip in self.hips], dim=-1)
        for label, mask in (('contact', contact_kind), ('other', ~contact_kind)):
            log['hip_constraint_' + label][i] = (jac * (force * (valid & mask))[:, :, None]).sum(dim=1)
        mass = wp.to_torch(self.data.qM)
        if self.sparse:
            log['hip_mass_rows'][i] = mass[:, 0, self.mass_addresses] * self.mass_valid
        else:
            log['hip_mass_rows'][i] = mass[:, self.hips_t, :]
        # Convert into a separate Contacts buffer; don't change simulator contacts.
        self.sim.solver.update_contacts(self.contacts, state_out)
        for j, sensor in enumerate(self.sensors.values()):
            sensor.update(state_out, self.contacts)
            log['contact_force_world'][i, :, j] = wp.to_torch(sensor.net_force).sum(dim=-2).reshape(self.sim.num_envs, 3)
        self.steps += 1

    @torch.no_grad()
    def record_control(self, dof_pos, dof_vel, reference, tau_ff_prev):
        i = self.calls
        log = self.control_log
        log['dof_pos'][i] = dof_pos
        log['dof_vel'][i] = dof_vel
        log['reference_pos'][i] = reference.dof_pos
        log['reference_vel'][i] = reference.dof_vel
        log['tau_pd_prev'][i] = self.sim.window_tau_pd_common
        log['tau_ff_prev'][i] = tau_ff_prev
        for key, field in (('pd_target', 'joint_target_pos'), ('pd_target_velocity', 'joint_target_vel')):
            value = wp.to_torch(self.sim.robot_view.get_attribute(field, self.sim.control)).reshape(self.sim.num_envs, -1)
            log[key][i] = value[:, self.common]
        state = self.sim.get_bodies_state()
        for key, field in (('body_pos', 'rigid_body_pos'), ('body_rot', 'rigid_body_rot'),
                           ('body_vel', 'rigid_body_vel'), ('body_ang_vel', 'rigid_body_ang_vel')):
            log[key][i] = getattr(state, field)
        self.calls += 1

    def save(self):
        if self.steps != self.calls * self.every:
            raise RuntimeError(f'Incomplete physics windows: {self.steps} vs {self.calls}*{self.every}')
        arrays = {'control_' + k: v[:self.calls].cpu().numpy() for k, v in self.control_log.items()}
        arrays.update({'physics_' + k: v[:self.steps].cpu().numpy() for k, v in self.physics_log.items()})
        np.savez_compressed(self.out_dir / 'fullbody.npz', **arrays)
        self.meta.update(control_samples=self.calls, physics_samples=self.steps)
        (self.out_dir / 'fullbody_meta.json').write_text(json.dumps(self.meta, indent=2))
        self.sim.solver.step = self.original_step
        print(f'[fullbody] saved {self.calls} windows / {self.steps} physics samples', flush=True)
