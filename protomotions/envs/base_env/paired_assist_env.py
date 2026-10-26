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
"""Paired synthetic wearers for reward-only unassisted tracking references.

Both plants are integrated. Common random numbers are replayed across construction,
lazy initialization, step and partial reset; the global stream advances only once.
A reference failure truncates B (bootstrap from B's final state), while B's own
failure remains terminal. This censors both rollouts at either failure and therefore
introduces a common-survival selection bias, which is logged explicitly.

This environment supports the synthetic, emulated-HLP fixed-base recipe only.
It is a training/diagnostic environment; deployment still uses AssistController.
"""

import copy
import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from protomotions.envs.base_env.assist_env import AssistEnv
from protomotions.envs.context_views import EnvContext
from protomotions.envs.rewards.assist import compute_target_tracking_rew
from protomotions.utils.component_builder import build_simulator_from_config


@dataclass
class RandomState:
    python: object
    numpy: tuple
    cpu: torch.Tensor
    cuda: object
    device: torch.device

    @classmethod
    def capture(cls, device):
        device = torch.device(device)
        return cls(random.getstate(), np.random.get_state(), torch.get_rng_state(),
                   torch.cuda.get_rng_state(device) if device.type == "cuda" else None, device)

    def restore(self):
        random.setstate(self.python)
        np.random.set_state(self.numpy)
        torch.set_rng_state(self.cpu)
        if self.cuda is not None:
            torch.cuda.set_rng_state(self.cuda, self.device)

    def matches(self, other):
        return (self.python == other.python
                and self.numpy[0] == other.numpy[0]
                and np.array_equal(self.numpy[1], other.numpy[1])
                and self.numpy[2:] == other.numpy[2:]
                and torch.equal(self.cpu, other.cpu)
                and (self.cuda is None if other.cuda is None
                     else self.cuda is not None and torch.equal(self.cuda, other.cuda)))


def paired_call(reference: Callable, assisted: Callable, device):
    """Replay global RNG, and fail closed if the two paths consume different draws."""
    before = RandomState.capture(device)
    try:
        a = reference()
        after_a = RandomState.capture(device)
    finally:
        before.restore()
    b = assisted()
    if not after_a.matches(RandomState.capture(device)):
        raise RuntimeError("Paired wearers consumed different random numbers")
    return a, b


def combine_pair_endings(assisted_done, assisted_terminal, reference_done):
    """A-only failure is truncation; never erase B failure or mark A failure as B's."""
    return assisted_done | reference_done, assisted_terminal.clone()


class PairedAssistEnv(AssistEnv):
    def __init__(self, config, robot_config, device, terrain, simulator,
                 scene_lib, motion_lib, *args, **kwargs):
        cc = config.control_components[config.assist_component_name]
        # Motion-data intents (2026-10-21) draw clips from their own generator seeded
        # identically in both twins, so they stay paired; evaluators assert θ_g equality
        # every step. A real HLP inside the component would consume global RNG.
        if (getattr(cc, "intent_source", "synthetic") not in ("synthetic", "motion_data")
                or getattr(cc, "chunk_source", "emulated") != "emulated"):
            raise ValueError("PairedAssistEnv requires synthetic or motion-data intent and emulated HLP")
        if motion_lib.num_motions() or scene_lib.num_scenes() or not simulator.headless:
            raise ValueError("PairedAssistEnv requires headless fixed-base plants without motions/scenes")
        if not robot_config.asset.fix_base_link or not terrain.is_flat():
            raise ValueError("PairedAssistEnv requires fixed-base robots on a flat terrain")
        if getattr(cc, "reversal_brake_prob", 0) != 0:
            raise ValueError("State-dependent reversal brakes require a separate pairing design")
        self.reference_env = None
        ref_config = copy.deepcopy(config)
        ref_config._target_ = "protomotions.envs.base_env.assist_env.AssistEnv"
        ref_config.assist_torque_limit = 0.0
        # Keep the same limit distribution/random draws. The reference action is
        # exactly zero, so sampled per-env bounds cannot introduce assistance.
        tracking = ref_config.reward_components["tracking"]
        tracking.compute_func = compute_target_tracking_rew
        tracking.dynamic_vars = {"dof_pos": EnvContext.current.dof_pos,
                                 "theta_g": EnvContext.assist.theta_g}
        # The terrain and empty libraries are immutable/shared; simulator, model,
        # state, control components, config and history are all separate objects.
        before = RandomState.capture(device)
        ref_robot = copy.deepcopy(robot_config)
        ref_sim = build_simulator_from_config(copy.deepcopy(simulator.config),
                                             ref_robot, terrain, scene_lib, device)
        before.restore()

        def build_reference():
            self.reference_env = AssistEnv(
                config=ref_config, robot_config=ref_robot, device=device,
                terrain=terrain, simulator=ref_sim, scene_lib=scene_lib,
                motion_lib=motion_lib, *args, **kwargs)

        paired_call(build_reference,
                    lambda: super(PairedAssistEnv, self).__init__(
                        config=config, robot_config=robot_config, device=device,
                        terrain=terrain, simulator=simulator, scene_lib=scene_lib,
                        motion_lib=motion_lib, *args, **kwargs), device)
        self._current_context = None

    def _lazy_init_assist(self):
        if self._torque_override is not None:
            return
        paired_call(self.reference_env._lazy_init_assist,
                    lambda: super(PairedAssistEnv, self)._lazy_init_assist(), self.device)

    def _build_global_context(self):
        ctx = super()._build_global_context()
        if self.reference_env is not None:
            ctx.assist.reference_dof_pos = self.reference_env.simulator.get_robot_state().dof_pos.detach()
        return ctx

    def reset(self, env_ids=None, *args, **kwargs):
        self._lazy_init_assist()
        _, result = paired_call(lambda: self.reference_env.reset(env_ids, *args, **kwargs),
                                lambda: super(PairedAssistEnv, self).reset(env_ids, *args, **kwargs), self.device)
        return result

    def step(self, action):
        self._lazy_init_assist()
        # PPO can update this via its episode-length curriculum each epoch.
        self.reference_env.max_episode_length = self.max_episode_length
        _, result = paired_call(lambda: self.reference_env.step(torch.zeros_like(action)),
                                lambda: super(PairedAssistEnv, self).step(action), self.device)
        own_done = self.reset_buf.clone()
        self.reset_buf[:], self.terminate_buf[:] = combine_pair_endings(
            own_done, self.terminate_buf, self.reference_env.reset_buf)
        self.extras["paired/reference_done_fraction"] = self.reference_env.reset_buf.float().mean()
        self.extras["paired/reference_only_truncation_fraction"] = (
            self.reference_env.reset_buf & ~own_done).float().mean()
        self.extras["paired/assisted_failure_fraction"] = self.terminate_buf.float().mean()
        # Return the same buffers, now with the joint reset decision. Reference
        # information is never added to obs or to actor/critic input definitions.
        return result
