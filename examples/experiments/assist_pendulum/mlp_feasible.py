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
"""Two fresh PPO recipes retaining L1's bilateral actor and 40 Nm action scale.

F1: corrected independent torque accounting + inertia-bounded C2 transitions.
F2: F1 + absorbed assist power cost. No W1 training or phase-dependent gate.
"""
import sys
from dataclasses import fields
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlp_v4 as baseline
from mlp_v4 import (apply_inference_overrides, configure_robot_and_simulator,
                    motion_lib_config, scene_lib_config, terrain_config)
from protomotions.envs.control.pendulum_feasible_control import PendulumFeasibleControlConfig
from protomotions.envs.rewards.assist_energy import compute_assist_absorbed_power
from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent


def additional_experiment_arguments(parser):
    baseline.additional_experiment_arguments(parser)
    parser.add_argument("--absorbed-power-weight", type=float, default=0.)


def env_config(robot_cfg, args):
    cfg = baseline.env_config(robot_cfg, args)
    old = cfg.control_components["assist"]
    cfg.control_components["assist"] = PendulumFeasibleControlConfig(
        **{f.name: getattr(old, f.name) for f in fields(old) if f.name != "_target_"},
        transition_torque_budget=float(args.assist_limit))
    # Common integrity prerequisite: preserve all three additive torque sources;
    # constrain the whole human PD+FF actuator once, not each part separately.
    cfg.clamp_combined_injection = False
    cfg.human_feedforward_inside_pd = True
    if not -float("inf") < args.absorbed_power_weight <= 0:
        raise ValueError("absorbed-power-weight must be finite and non-positive")
    if args.absorbed_power_weight:
        cfg.reward_components["assist_absorbed_power"] = MdpComponent(
            compute_func=compute_assist_absorbed_power,
            dynamic_vars={"tau_assist": EnvContext.assist.tau_assist,
                          "dof_vel": EnvContext.current.dof_vel},
            static_params={"weight": args.absorbed_power_weight})
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    cfg = baseline.agent_config(robot_cfg, env_cfg, args)
    cfg.save_epoch_checkpoint_every = 500
    cfg.save_last_checkpoint_every = 50
    # Fixed final epoch; no test-set or periodic best-checkpoint selection.
    cfg.evaluator.eval_metrics_every = None
    return cfg
