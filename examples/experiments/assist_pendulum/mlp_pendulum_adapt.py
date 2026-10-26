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
"""Adapt the existing L1 actor in the corrected pendulum, with synthetic or H1 input."""

import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlp_research as baseline  # noqa: E402
from mlp_research import (  # noqa: E402, F401
    apply_inference_overrides, configure_robot_and_simulator,
    motion_lib_config, scene_lib_config, terrain_config,
)
from protomotions.agents.ppo.pendulum_adapt import PendulumAdaptConfig  # noqa: E402


def additional_experiment_arguments(parser):
    baseline.additional_experiment_arguments(parser)
    parser.add_argument("--initial-actor-checkpoint", default="checkpoints/L1_llp_pendulum/last.ckpt")
    parser.add_argument("--training-hlp-checkpoint", default=None)


def env_config(robot_cfg, args):
    cfg = baseline.env_config(robot_cfg, args)
    if args.training_hlp_checkpoint:
        from protomotions.envs.control.pendulum_h1_control import PendulumH1ControlConfig

        old = cfg.control_components["assist"]
        values = {f.name: getattr(old, f.name) for f in fields(old) if f.name != "_target_"}
        values.update(maniflow_ckpt=args.training_hlp_checkpoint, maniflow_denoise_steps=1)
        cfg.control_components["assist"] = PendulumH1ControlConfig(
            **values, predictor_seed=17001 + args.seed)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    if args.shared_joint_actor:
        raise ValueError("L1 actor initialization requires the original bilateral architecture")
    old = baseline.agent_config(robot_cfg, env_cfg, args)
    cfg = PendulumAdaptConfig(**{f.name: getattr(old, f.name) for f in fields(old) if f.name != "_target_"})
    cfg.initial_actor_checkpoint = str(Path(args.initial_actor_checkpoint).resolve())
    cfg.model.actor_optimizer.lr = 3e-5
    cfg.save_epoch_checkpoint_every = 100
    cfg.save_last_checkpoint_every = 50
    cfg.evaluator.eval_metrics_every = None
    return cfg
