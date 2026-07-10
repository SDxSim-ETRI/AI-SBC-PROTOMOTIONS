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
"""Per-env / per-DOF torque override on top of Newton BUILT_IN_PD.

Lets selected envs drive selected DOFs with externally computed torques
(e.g. ManiFlow predictions) while every other env/DOF keeps the simulator's
built-in (implicit) PD — the exact actuation the RL policy was trained with.

Mechanism (Newton + SolverMuJoCo only):
  1. ``engage()`` zeroes ``joint_target_ke/kd`` for the overridden (env, DOF)
     pairs and calls ``notify_model_changed(JOINT_DOF_PROPERTIES)``. MuJoCo
     Warp stores actuator gains per world (``actuator_gainprm/biasprm`` are
     world-expanded), so this disables the implicit PD only where requested.
  2. ``set_torques()`` writes the desired torques into ``control.joint_f``
     via ``ArticulationView.set_dof_forces``. SolverMuJoCo copies
     ``joint_f`` into ``qfrc_applied`` every substep — an applied
     generalized force, additive with (and independent of) actuator forces.
     Non-overridden entries are always written as 0 so PD-only envs are
     unaffected. Writes are in-place scatters → CUDA-graph safe.

Notes:
  - Torques are held constant across the decimated substeps of one control
    step, i.e. standard torque control with decimation.
  - ``qfrc_actuator`` readback (``robot_state.dof_forces`` under
    BUILT_IN_PD) reports ~0 for overridden DOFs: applied force goes through
    ``qfrc_applied``, which MuJoCo does not include in ``qfrc_actuator``.
    The commanded torque itself is the ground truth for those DOFs.
"""

from typing import List, Optional, Sequence

import torch

from protomotions.robot_configs.base import ControlType


class JointTorqueOverride:
    """Torque override for a subset of envs/DOFs on a Newton simulator.

    Args:
        simulator: A ``NewtonSimulator`` running ``ControlType.BUILT_IN_PD``.
        env_ids: Env indices whose DOFs get torque control (e.g. ``[1]``).
        common_dof_indices: DOF indices in COMMON ordering to override
            (e.g. ``range(6)`` for the channels a ManiFlow policy predicts).
    """

    def __init__(
        self,
        simulator,
        env_ids: Sequence[int],
        common_dof_indices: Sequence[int],
    ):
        from protomotions.simulator.newton.simulator import NewtonSimulator

        assert isinstance(simulator, NewtonSimulator), (
            "JointTorqueOverride requires the Newton simulator "
            f"(got {type(simulator).__name__})"
        )
        assert simulator.control_type == ControlType.BUILT_IN_PD, (
            "JointTorqueOverride composes with BUILT_IN_PD only; "
            f"simulator runs {simulator.control_type.name}"
        )

        self.sim = simulator
        self.device = simulator.device
        self.num_envs = simulator.num_envs
        self.env_ids = torch.tensor(
            list(env_ids), dtype=torch.long, device=self.device
        )
        self.common_dof_indices = torch.tensor(
            list(common_dof_indices), dtype=torch.long, device=self.device
        )
        # Positions of the overridden common DOFs inside SIM-ordered arrays
        self.sim_dof_indices = simulator.data_conversion.dof_convert_to_common[
            self.common_dof_indices
        ]

        num_dofs = simulator.robot_config.number_of_actions
        self.dof_names: List[str] = [
            simulator.robot_config.kinematic_info.dof_names[i]
            for i in common_dof_indices
        ]
        self.torque_limits = simulator._torque_limits_common[self.common_dof_indices]

        # Persistent (num_envs, 1, num_dofs) SIM-ordered buffer aliased by a
        # warp array; set_dof_forces scatters it into control.joint_f.
        self._torques_sim = torch.zeros(
            self.num_envs, 1, num_dofs, device=self.device, dtype=torch.float32
        )
        import warp as wp

        self._torques_wp = wp.from_torch(
            self._torques_sim, dtype=wp.float32, requires_grad=False
        )

        self._engaged = False
        self._saved_gains: Optional[tuple] = None

    def engage(self) -> None:
        """Disable built-in PD on the overridden (env, DOF) pairs."""
        if self._engaged:
            return
        import warp as wp
        from newton.solvers import SolverNotifyFlags

        sim = self.sim
        ke_wp = sim.robot_view.get_attribute("joint_target_ke", sim.model)
        kd_wp = sim.robot_view.get_attribute("joint_target_kd", sim.model)
        ke = wp.to_torch(ke_wp).view(self.num_envs, -1)
        kd = wp.to_torch(kd_wp).view(self.num_envs, -1)
        self._saved_gains = (
            ke[self.env_ids[:, None], self.sim_dof_indices[None, :]].clone(),
            kd[self.env_ids[:, None], self.sim_dof_indices[None, :]].clone(),
        )
        ke[self.env_ids[:, None], self.sim_dof_indices[None, :]] = 0.0
        kd[self.env_ids[:, None], self.sim_dof_indices[None, :]] = 0.0
        sim.robot_view.set_attribute("joint_target_ke", sim.model, ke_wp)
        sim.robot_view.set_attribute("joint_target_kd", sim.model, kd_wp)
        sim.solver.notify_model_changed(SolverNotifyFlags.JOINT_DOF_PROPERTIES)
        self._engaged = True

    def restore(self) -> None:
        """Re-enable the original PD gains and clear injected torques."""
        if not self._engaged:
            return
        import warp as wp
        from newton.solvers import SolverNotifyFlags

        sim = self.sim
        ke_wp = sim.robot_view.get_attribute("joint_target_ke", sim.model)
        kd_wp = sim.robot_view.get_attribute("joint_target_kd", sim.model)
        ke = wp.to_torch(ke_wp).view(self.num_envs, -1)
        kd = wp.to_torch(kd_wp).view(self.num_envs, -1)
        saved_ke, saved_kd = self._saved_gains
        ke[self.env_ids[:, None], self.sim_dof_indices[None, :]] = saved_ke
        kd[self.env_ids[:, None], self.sim_dof_indices[None, :]] = saved_kd
        sim.robot_view.set_attribute("joint_target_ke", sim.model, ke_wp)
        sim.robot_view.set_attribute("joint_target_kd", sim.model, kd_wp)
        sim.solver.notify_model_changed(SolverNotifyFlags.JOINT_DOF_PROPERTIES)
        self.zero()
        self._engaged = False

    def set_torques(self, torques: torch.Tensor) -> torch.Tensor:
        """Set the torques applied on the next control step(s).

        Args:
            torques: (len(env_ids), len(common_dof_indices)) raw torques in
                N·m, COMMON DOF ordering of the overridden channels. NaNs are
                treated as 0; values are clamped to the per-DOF effort limit.

        Returns:
            The clamped torques actually written, same shape as the input.
        """
        assert self._engaged, "call engage() before set_torques()"
        torques = torch.nan_to_num(
            torques.to(device=self.device, dtype=torch.float32), nan=0.0
        )
        torques = torch.clamp(torques, -self.torque_limits, self.torque_limits)

        self._torques_sim.zero_()
        self._torques_sim[self.env_ids[:, None], 0, self.sim_dof_indices[None, :]] = (
            torques
        )
        self.sim.robot_view.set_dof_forces(self.sim.control, self._torques_wp)
        return torques

    def zero(self) -> None:
        """Clear all injected torques (call after env resets)."""
        self._torques_sim.zero_()
        self.sim.robot_view.set_dof_forces(self.sim.control, self._torques_wp)
