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
"""L13: independent, synchronized unassisted whole-body reference for LLP learning.

Reference state is reward-only. Neither actor inputs nor policy size changes.
Both wearers keep their own frozen tracker and real HLP; common random numbers
pair motion resets and prediction noise. A-only failure truncates the pair.
"""

import copy
from dataclasses import dataclass

import torch

from protomotions.envs.base_env.paired_assist_env import (
    RandomState, combine_pair_endings, paired_call,
)
from protomotions.envs.base_env.skeleton_assist_env import (
    SkeletonAssistEnv, SkeletonAssistEnvConfig,
)
from protomotions.utils.component_builder import build_simulator_from_config
from protomotions.envs.motion_manager.config import MimicMotionManagerConfig
from protomotions.envs.motion_manager.mimic_motion_manager import MimicMotionManager


@dataclass
class WalkingSegmentConfig(MimicMotionManagerConfig):
    _target_: str = "protomotions.envs.base_env.paired_skeleton_assist_env.WalkingSegmentManager"
    start_min_s: float = 0.0
    start_max_s: float = 20.0


class WalkingSegmentManager(MimicMotionManager):
    def sample_time(self, motion_ids, truncate_time=None):
        # With 8s episodes, LLP adaptation sees 0-28s, leaving 60s+ for testing.
        lo, hi = self.config.start_min_s, self.config.start_max_s
        lengths = self.motion_lib.motion_lengths[motion_ids]
        assert torch.all(lengths > hi + 8.5), "walk clip is too short for the declared split"
        return lo + (hi - lo) * torch.rand(len(motion_ids), device=self.device)


@dataclass
class PairedSkeletonAssistEnvConfig(SkeletonAssistEnvConfig):
    _target_: str = "protomotions.envs.base_env.paired_skeleton_assist_env.PairedSkeletonAssistEnv"
    warmup_steps: int = 100
    burden_scale: float = 1600.0
    local_excess_weight: float = 1.0
    local_excess_floor_Nm: float = 0.0
    body_margin: float = 0.005
    body_weight: float = 20.0
    hip_error_margin_sq: float = 0.0003
    hip_error_weight: float = 5.0
    speed_margin: float = 0.03
    speed_weight: float = 5.0
    motion_window: int = 120
    motion_retention: float = 0.90
    motion_weight: float = 2.0
    effort_weight: float = 0.000001
    rate_weight: float = 0.005
    failure_penalty: float = 5.0


def paired_burden_terms(tau_a, tau_b, scale, floor=0.0):
    """Separate per-joint local harm from net burden reduction, in fixed units."""
    difference = tau_b.square() - tau_a.square()
    excess = (difference.relu() / (tau_a.square() + floor**2)).mean(-1) if floor > 0 else difference.relu().mean(-1) / scale
    return -difference.mean(-1) / scale, excess


class PairedSkeletonAssistEnv(SkeletonAssistEnv):
    def __init__(self, config, robot_config, device, terrain, simulator,
                 scene_lib, motion_lib, *args, **kwargs):
        if not simulator.headless or not terrain.is_flat() or scene_lib.num_scenes():
            raise ValueError("L13 pilot supports headless flat walking without scenes")
        self.reference_env = None
        ref_config = copy.deepcopy(config)
        ref_config._target_ = "protomotions.envs.base_env.skeleton_assist_env.SkeletonAssistEnv"
        ref_config.assist_torque_limit = 0.0
        ref_config.reward_components = {}
        before = RandomState.capture(device)
        ref_robot = copy.deepcopy(robot_config)
        ref_sim = build_simulator_from_config(copy.deepcopy(simulator.config),
                                             ref_robot, terrain, scene_lib, device)
        before.restore()
        # BaseEnv smooths reference contact labels in place during construction.
        # Each wearer needs its own library to avoid smoothing the same data twice.
        ref_motion_lib = copy.deepcopy(motion_lib)

        def build_reference():
            self.reference_env = SkeletonAssistEnv(
                config=ref_config, robot_config=ref_robot, device=device,
                terrain=terrain, simulator=ref_sim, scene_lib=scene_lib,
                motion_lib=ref_motion_lib, *args, **kwargs)

        paired_call(build_reference,
                    lambda: super(PairedSkeletonAssistEnv, self).__init__(
                        config=config, robot_config=robot_config, device=device,
                        terrain=terrain, simulator=simulator, scene_lib=scene_lib,
                        motion_lib=motion_lib, *args, **kwargs), device)
        self._motion_a = torch.zeros(self.num_envs, config.motion_window, 5, device=device)
        self._motion_b = torch.zeros_like(self._motion_a)
        self._previous_action = torch.zeros(self.num_envs, 2, device=device)
        self._current_context = None

    def _lazy_init_assist(self):
        if self._torque_override is None:
            paired_call(self.reference_env._lazy_init_assist,
                        lambda: super(PairedSkeletonAssistEnv, self)._lazy_init_assist(), self.device)

    def reset(self, env_ids=None, **kwargs):
        if env_ids is not None and len(env_ids) == 0:
            return self.get_obs(), {}
        self._lazy_init_assist()
        _, result = paired_call(lambda: self.reference_env.reset(env_ids, **kwargs),
                                lambda: super(PairedSkeletonAssistEnv, self).reset(env_ids, **kwargs),
                                self.device)
        ids = slice(None) if env_ids is None else env_ids
        self._motion_a[ids] = 0
        self._motion_b[ids] = 0
        self._previous_action[ids] = 0
        self.check_pairing()
        return result

    def check_pairing(self):
        a, b = self.reference_env.motion_manager, self.motion_manager
        assert torch.equal(a.motion_ids, b.motion_ids), "paired motion IDs differ"
        assert torch.equal(a.motion_times, b.motion_times), "paired motion clocks differ"
        assert torch.equal(self.reference_env.progress_buf, self.progress_buf)
        assert torch.count_nonzero(self.reference_env.assist_component._tau_assist) == 0

    def compute_reward(self, context):
        # The reference has already advanced to exactly the same motion time.
        a, b = self.reference_env.context, context
        cfg = self.config
        reduction, excess = paired_burden_terms(a.assist.tau_agent, b.assist.tau_agent,
                                               cfg.burden_scale, cfg.local_excess_floor_Nm)
        body_a = (a.current.rigid_body_pos - a.mimic.ref_state.rigid_body_pos).norm(dim=-1).mean(-1)
        body_b = (b.current.rigid_body_pos - b.mimic.ref_state.rigid_body_pos).norm(dim=-1).mean(-1)
        hip_a = (a.assist.dof_pos - a.assist.theta_g).square()
        hip_b = (b.assist.dof_pos - b.assist.theta_g).square()
        for ring, ctx in ((self._motion_a, a), (self._motion_b, b)):
            slot = self.progress_buf % cfg.motion_window
            ids = torch.arange(self.num_envs, device=self.device)
            speed = ctx.current.root_vel[:, :2].norm(dim=-1, keepdim=True)
            ring[ids, slot] = torch.cat((ctx.assist.dof_pos, ctx.assist.dof_vel, speed), -1)
        # Only use complete windows. This discourages slowing/flattening the hip
        # while allowing small phase changes, unlike pointwise velocity matching.
        motion_ready = self.progress_buf >= cfg.warmup_steps + cfg.motion_window
        amp_a = self._motion_a[:, :, :2].amax(1) - self._motion_a[:, :, :2].amin(1)
        amp_b = self._motion_b[:, :, :2].amax(1) - self._motion_b[:, :, :2].amin(1)
        vel_a = self._motion_a[:, :, 2:4].square().mean(1)
        vel_b = self._motion_b[:, :, 2:4].square().mean(1)
        retention = ((cfg.motion_retention * amp_a - amp_b).relu() / amp_a.clamp_min(.1)).mean(-1)
        retention += ((cfg.motion_retention * vel_a - vel_b).relu() / vel_a.clamp_min(.25)).mean(-1)
        speed_loss = (self._motion_a[:, :, 4].mean(1) - self._motion_b[:, :, 4].mean(1)).abs()
        terms = {
            "burden_reduction": reduction,
            "local_excess": -cfg.local_excess_weight * excess,
            "body_nondegradation": -cfg.body_weight * (body_b - body_a - cfg.body_margin).relu(),
            "hip_nondegradation": -cfg.hip_error_weight * (hip_b - hip_a - cfg.hip_error_margin_sq).relu().mean(-1),
            "speed_nondegradation": -cfg.speed_weight * (speed_loss - cfg.speed_margin).relu() * motion_ready,
            "motion_retention": -cfg.motion_weight * retention * motion_ready,
            "assist_effort": -cfg.effort_weight * b.assist.tau_assist.square().mean(-1),
            "action_rate": -cfg.rate_weight * (self._current_raw_action - self._previous_action).square().mean(-1),
        }
        active = self.progress_buf > cfg.warmup_steps
        self.rew_buf[:] = sum(terms.values()) * active
        self._previous_action.copy_(self._current_raw_action)
        for key, value in terms.items():
            self.extras[f"scaled_r/{key}"] = value * active
        self.extras["total_env_reward"] = self.rew_buf.clone()

    def step(self, action):
        self._lazy_init_assist()
        self.reference_env.max_episode_length = self.max_episode_length
        action = action.clone()
        action[self.progress_buf < self.config.warmup_steps] = 0
        _, result = paired_call(lambda: self.reference_env.step(torch.zeros_like(action)),
                                lambda: super(PairedSkeletonAssistEnv, self).step(action), self.device)
        own_done = self.reset_buf.clone()
        self.reset_buf[:], self.terminate_buf[:] = combine_pair_endings(
            own_done, self.terminate_buf, self.reference_env.reset_buf)
        self.rew_buf.sub_(self.config.failure_penalty * self.terminate_buf)
        self.extras["paired/reference_only_truncation"] = (self.reference_env.reset_buf & ~own_done).float()
        self.extras["paired/assisted_failure"] = self.terminate_buf.float()
        self.extras["total_env_reward"] = self.rew_buf.clone()
        return result
