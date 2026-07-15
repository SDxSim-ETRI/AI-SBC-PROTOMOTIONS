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
"""Closed-loop hip-torque estimation with a ManiFlow lowdim policy.

Wraps a trained ``ManiFlowLowdimPolicy`` (sensor-state-only inverse-dynamics
estimator) for use inside a ProtoMotions simulation loop. Consumes the
simulator-agnostic ``RobotState`` (COMMON body/DOF ordering) and reproduces
the exact observation layout the policy was trained on — the one written by
``tasks/mimic_suit_active_cable_walk_23dof/collect_walk_zarr.py``:

    obs = concat(dof_pos, dof_vel, root_pos, root_vel, contacts)
          (27)     (27)     (3)       (3)      (28)     = 88 dims

where root_pos/root_vel are the pelvis (body 0) world position / linear
velocity and contacts are binary rigid-body contact flags. Because
``Simulator.get_robot_state()`` already converts every field to COMMON
ordering, the same estimator works unchanged across Newton / IsaacLab /
IsaacGym backends.

NOTE on action channels: the estimator itself only returns the policy's raw
6-dim action — which COMMON DOFs those channels map to is a training-data
contract, resolved by the caller via ``protomotions.maniflow.channels``.
Current contract (``hips``): the six hip DOFs, COMMON [0, 1, 2, 5, 6, 7] =
[hip_flexion_r, hip_adduction_r, hip_rotation_r, hip_flexion_l,
hip_adduction_l, hip_rotation_l]. (Legacy checkpoints trained before
2026-07-09 used COMMON DOFs 0-5 instead — right leg + left hip flexion, a
collection-script bug; the last such model was deleted on 2026-07-14.)
Never hard-code channel names — derive them from
``robot_config.kinematic_info.dof_names`` via ``hip_dof_indices()``.
"""

from typing import Optional

import torch

from protomotions.simulator.base_simulator.simulator_state import RobotState


class ManiFlowTorqueEstimator:
    """Receding-horizon torque predictor over a rolling observation history.

    Call :meth:`observe` once per policy step (after ``env.step``), then
    :meth:`predict` whenever a new action chunk is needed — every
    ``n_action_steps`` steps for receding-horizon replay, or every step if
    only the first predicted action is consumed.

    Envs that were just (re)set must be flagged via :meth:`reset` so their
    history is re-primed: the first observation after a reset is repeated
    across the whole window, matching the ``pad_before`` edge-padding the
    policy saw during training.
    """

    ROOT_BODY_IDX = 0  # pelvis in COMMON body ordering

    def __init__(self, policy, num_envs: int, device: torch.device):
        self.policy = policy
        self.num_envs = num_envs
        self.device = torch.device(device)

        self.n_obs_steps = int(policy.n_obs_steps)
        self.n_action_steps = int(policy.n_action_steps)
        self.action_dim = int(policy.action_dim)
        self.obs_key = policy.obs_encoder.state_key  # 'agent_pos'
        self.obs_dim = int(policy.obs_encoder.state_shape[0])

        self._history = torch.zeros(
            num_envs, self.n_obs_steps, self.obs_dim, device=self.device
        )
        self._primed = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str,
        num_envs: int,
        device: torch.device,
        use_ema: Optional[bool] = None,
        maniflow_root: Optional[str] = None,
    ) -> "ManiFlowTorqueEstimator":
        from protomotions.maniflow.loader import load_maniflow_policy

        policy, _cfg, _info = load_maniflow_policy(
            ckpt_path, device=str(device), use_ema=use_ema, maniflow_root=maniflow_root
        )
        return cls(policy, num_envs=num_envs, device=device)

    def obs_from_robot_state(self, robot_state: RobotState) -> torch.Tensor:
        """Build the (num_envs, obs_dim) observation the policy was trained on."""
        obs = torch.cat(
            [
                robot_state.dof_pos,
                robot_state.dof_vel,
                robot_state.rigid_body_pos[:, self.ROOT_BODY_IDX],
                robot_state.rigid_body_vel[:, self.ROOT_BODY_IDX],
                robot_state.rigid_body_contacts.float(),
            ],
            dim=-1,
        ).to(device=self.device, dtype=torch.float32)
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(
                f"Observation is {obs.shape[-1]}-dim but the policy expects "
                f"{self.obs_dim}. RobotState fields do not match the layout "
                "this policy was trained on."
            )
        return obs

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        """Mark envs as freshly reset so their history gets re-primed."""
        if env_ids is None:
            self._primed[:] = False
        else:
            self._primed[env_ids] = False

    def observe(self, robot_state: RobotState) -> torch.Tensor:
        """Push the current simulator state into the rolling history."""
        obs = self.obs_from_robot_state(robot_state)
        rolled = torch.cat([self._history[:, 1:], obs[:, None]], dim=1)
        primed_full = obs[:, None].expand(-1, self.n_obs_steps, -1)
        self._history = torch.where(self._primed[:, None, None], rolled, primed_full)
        self._primed[:] = True
        return obs

    @torch.no_grad()
    def predict(self) -> torch.Tensor:
        """Predict the next hip-torque chunk from the current history.

        Returns:
            (num_envs, n_action_steps, action_dim) torque in raw units. The
            first step corresponds to the timestep of the latest
            :meth:`observe` call, the rest are its successors.
        """
        if not bool(self._primed.all()):
            raise RuntimeError("predict() called before observe() primed all envs")
        result = self.policy.predict_action({self.obs_key: self._history})
        return result["action"].detach()
