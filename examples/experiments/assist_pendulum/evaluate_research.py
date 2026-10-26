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
"""Fixed-cohort paired pendulum evaluation, without failure-triggered resampling.

Build all evaluation plants from one baseline config, independently of the actor.
Both plants run the full scheduled horizon. Metrics use common survival up to
first failure, INCLUDING that failure's transition; first failures and censored
exposure are reported separately. This conditional metric is not an unconditional
performance guarantee. No failure can silently start a different random episode.
"""

from dataclasses import fields
import hashlib
import json
from pathlib import Path
import random

from protomotions.utils.simulator_imports import import_simulator_before_torch

import_simulator_before_torch("newton")

import numpy as np  # noqa: E402
import torch  # noqa: E402


class IsolatedH1Forecast:
    """Frozen H1 observations with a private RNG stream, separate from the plant.

    The emulated control component still runs in both paired plants. Only the
    assisted actor's target observation is replaced, so H1 cannot change the
    synthetic intention or the external disturbance random sequence.
    """

    def __init__(self, checkpoint, env):
        from protomotions.maniflow.angle_estimator import ManiFlowAngleEstimator
        from protomotions.maniflow.chunk_source import ManiFlowChunkSource

        self.device = torch.device(env.device)
        estimator = ManiFlowAngleEstimator.from_checkpoint(
            checkpoint, num_envs=env.num_envs, device=self.device, denoise_steps=1,
        )
        self.source = ManiFlowChunkSource(
            estimator, env.num_envs, env.dt, self.device, flexion_sign=-1.,
        )
        assert abs(self.source.knot_dt - env.dt) < 1e-6
        assert self.source.num_knots >= 3
        self.env_ids = torch.arange(env.num_envs, device=self.device)
        cfg = env.assist_component.config
        assert cfg.obs_future_samples == 3 and abs(cfg.obs_future_dt - env.dt) < 1e-6
        assert cfg.chunk_refresh_steps == 1
        kin = env.robot_config.kinematic_info
        self.low = kin.dof_limits_lower.to(self.device) * cfg.dof_limit_margin
        self.high = kin.dof_limits_upper.to(self.device) * cfg.dof_limit_margin

    def reset(self, q, seed):
        self.source.reset(self.env_ids, q, trunk=None)
        self.cpu_rng = torch.Generator().manual_seed(seed).get_state()
        self.cuda_rng = torch.Generator(device=self.device).manual_seed(seed).get_state()

    def predict(self, q, time_s, observe=True):
        if observe:
            self.source.observe(q, trunk=None)
        with torch.random.fork_rng(devices=[self.device]):
            torch.set_rng_state(self.cpu_rng)
            torch.cuda.set_rng_state(self.cuda_rng, self.device)
            knots, _ = self.source.predict_knots(
                self.env_ids, torch.full((len(q),), time_s, device=self.device),
            )
            self.cpu_rng = torch.get_rng_state()
            self.cuda_rng = torch.cuda.get_rng_state(self.device)
        return knots[:, :3].clamp(self.low, self.high)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def summarize(arrays, dt, warmup):
    valid = arrays["valid"] & (arrays["time_s"] >= warmup)
    # Masks refer to the unassisted member only, to avoid action-dependent bins.
    moving = np.abs(arrays["reference_velocity"]) >= .3
    groups = {"all": np.ones_like(moving, dtype=bool),
              "moving": moving, "near_stationary": ~moving,
              "heavy_load": arrays["reference_load_event"],
              "outside_load": ~arrays["reference_load_event"],
              "damping": arrays["reference_damping"] > 0}
    result = {}
    for name, mask in groups.items():
        mask = mask & valid[..., None]
        count = int(mask.sum())
        if not count:
            continue
        a, b = arrays["reference_tau"][mask], arrays["assisted_tau"][mask]
        ea = (arrays["reference_q"] - arrays["intent"])[mask]
        eb = (arrays["assisted_q"] - arrays["intent"])[mask]
        assist = arrays["assist_tau"][mask]
        am, bm = float(np.mean(a*a)), float(np.mean(b*b))
        result[name] = dict(joint_samples=count, exposure_joint_s=count*dt,
            reference_tau_sq_Nm2=am, assisted_tau_sq_Nm2=bm,
            torque_sq_change_percent=100*(bm/am-1),
            reference_abs_work_W=float(np.mean(np.abs(a*arrays["reference_velocity"][mask]))),
            assisted_abs_work_W=float(np.mean(np.abs(b*arrays["assisted_velocity"][mask]))),
            reference_tracking_rmse_deg=float(np.sqrt(np.mean(ea*ea))*180/np.pi),
            assisted_tracking_rmse_deg=float(np.sqrt(np.mean(eb*eb))*180/np.pi),
            assist_rms_Nm=float(np.sqrt(np.mean(assist*assist))),
            assist_above_19Nm_fraction=float(np.mean(np.abs(assist)>19)))
    return result
