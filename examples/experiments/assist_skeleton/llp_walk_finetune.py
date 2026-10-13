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
"""L13 walking pilot: L9 actor, W1 + real HLP, paired whole-body rewards."""

import copy
import sys
from dataclasses import fields
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llp_walk import (
    additional_experiment_arguments as legacy_arguments,
    env_config as legacy_env_config,
    motion_lib_config, scene_lib_config, terrain_config,
)
from protomotions.agents.ppo.assist_finetune import AssistFinetuneConfig
from protomotions.envs.base_env.paired_skeleton_assist_env import PairedSkeletonAssistEnvConfig, WalkingSegmentConfig


def additional_experiment_arguments(parser):
    legacy_arguments(parser)
    parser.add_argument("--initial-actor-checkpoint", default="results/llp_L9_opposition/epoch_2000.ckpt")
    parser.set_defaults(
        tracker_checkpoint="checkpoints/W1_wearer_human/last.ckpt", decimation=4,
        assist_limit=100., human_cost="torque_sq", hlp_denoise=1,
        hlp_checkpoint="/home/bak/Projects/ManiFlow_Policy/ManiFlow/data/outputs/hip_flexion_seed-maniflow_lowdim_policy_walking-flexion100-h15-run01_seed42/checkpoints/epoch=0006-val_loss=0.000762.ckpt",
        hlp_calibration="4.598162323314862,1.40271338852458,-17.06335958453886,2.3705996455698113")


def _source(args):
    return torch.load(Path(args.initial_actor_checkpoint).parent / "resolved_configs.pt",
                      map_location="cpu", weights_only=False)


def configure_robot_and_simulator(robot_cfg, simulator_cfg, args):
    wearer = torch.load(Path(args.tracker_checkpoint).parent / "resolved_configs.pt",
                        map_location="cpu", weights_only=False)
    robot_cfg.__dict__.update(copy.deepcopy(wearer["robot"].__dict__))
    robot_cfg.update_fields(contact_bodies=["all_left_foot_bodies", "all_right_foot_bodies"])
    simulator_cfg.sim = copy.deepcopy(wearer["simulator"].sim)
    simulator_cfg.sim.fps = 400
    simulator_cfg.sim.decimation = 4


def env_config(robot_cfg, args):
    source = _source(args)["env"]
    old = legacy_env_config(robot_cfg, args)
    cfg = PairedSkeletonAssistEnvConfig(**{f.name: getattr(old, f.name) for f in fields(old) if f.name != "_target_"})
    cfg.num_state_history_steps = source.num_state_history_steps
    cc = cfg.control_components["assist"]
    cc.hlp_dt = .01
    cc.chunk_knots = 6
    cc.target_low_override = (-1.52, -1.52)
    cc.target_high_override = (1.52, 1.52)
    for key in ("assist_proprio", "assist_target", "previous_actions", "assist_privileged"):
        cfg.observation_components[key] = copy.deepcopy(source.observation_components[key])
    # The source proprio/critic references full pendulum DOFs; select the two hips.
    from protomotions.envs.context_views import EnvContext
    for key in ("assist_proprio", "assist_target", "assist_privileged"):
        obs = cfg.observation_components[key]
        for name in ("dof_pos", "dof_vel", "historical_dof_pos", "historical_dof_vel"):
            if name in obs.dynamic_vars:
                obs.dynamic_vars[name] = getattr(EnvContext.assist, name)
    cfg.action_config = copy.deepcopy(source.action_config)
    cfg.assist_torque_limit = source.assist_torque_limit
    cfg.reward_components = {}  # Paired reward is computed after both physics steps.
    cfg.motion_manager = WalkingSegmentConfig(**{f.name: copy.deepcopy(getattr(cfg.motion_manager, f.name))
                                               for f in fields(cfg.motion_manager) if f.name != "_target_"})
    cfg.motion_manager.exclude_motion_ids = [1]  # gait.pt contains walk=0, run=1
    cfg.motion_manager.init_start_prob = 0.
    cfg.max_episode_length = 800
    return cfg


def agent_config(robot_config, env_config, args):
    source = _source(args)["agent"]
    cfg = AssistFinetuneConfig(**{f.name: copy.deepcopy(getattr(source, f.name)) for f in fields(source) if f.name != "_target_"})
    cfg.initial_actor_checkpoint = str(Path(args.initial_actor_checkpoint).resolve())
    cfg.batch_size = args.batch_size
    cfg.training_max_steps = args.training_max_steps
    cfg.model.actor_optimizer.lr = 3e-5
    cfg.model.critic_optimizer.lr = 5e-4
    cfg.save_epoch_checkpoint_every = 50
    cfg.save_last_checkpoint_every = 10
    cfg.evaluator.eval_metrics_every = None  # Independent run_system A/B after each stage.
    return cfg


def apply_inference_overrides(robot_cfg, simulator_cfg, env_cfg, agent_cfg,
                              terrain_cfg, motion_lib_cfg, scene_lib_cfg, args):
    env_cfg.max_episode_length = 5000
