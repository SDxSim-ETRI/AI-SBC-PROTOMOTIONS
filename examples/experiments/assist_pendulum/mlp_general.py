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
"""Fresh-initialization pendulum study with independent architecture/cost ablations."""

import sys
from dataclasses import fields
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlp_research as baseline  # noqa: E402
from mlp_research import (  # noqa: E402, F401
    apply_inference_overrides, configure_robot_and_simulator,
    motion_lib_config, scene_lib_config, terrain_config,
)
from protomotions.envs.control.pendulum_general_control import PendulumGeneralControlConfig  # noqa: E402


def additional_experiment_arguments(parser):
    baseline.additional_experiment_arguments(parser)
    parser.set_defaults(assist_limit=23.0, shared_joint_actor=True)
    parser.add_argument("--joint-actor", action="store_false", dest="shared_joint_actor")
    parser.add_argument("--phase-probability", type=float, default=0.0)
    parser.add_argument("--human-cost-multiplier", type=float, default=1.0)
    parser.add_argument("--regularization-multiplier", type=float, default=1.0)


def env_config(robot_cfg, args):
    cfg = baseline.env_config(robot_cfg, args)
    old = cfg.control_components["assist"]
    values = {f.name: getattr(old, f.name) for f in fields(old) if f.name != "_target_"}
    cfg.control_components["assist"] = PendulumGeneralControlConfig(
        **values, phase_probability=args.phase_probability,
    )
    for multiplier, names in (
        (args.human_cost_multiplier, ("human_power", "human_torque_sq")),
        (args.regularization_multiplier, ("assist_effort", "action_rate")),
    ):
        if not 0 < multiplier < float("inf"):
            raise ValueError("Reward multipliers must be finite and positive")
        for name in names:
            if name in cfg.reward_components:
                cfg.reward_components[name].static_params["weight"] *= multiplier
    return cfg


def agent_config(robot_config, env_config, args):
    cfg = baseline.agent_config(robot_config, env_config, args)
    # Same soft action bound in ALL arms; previous shared policies saturated.
    # This is an optimizer regularizer, separate from environment rewards.
    cfg.bounds_loss_coef = .001
    return cfg
