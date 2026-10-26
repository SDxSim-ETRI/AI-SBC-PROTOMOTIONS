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
"""Pendulum-only 20 Nm ablations: shared joint actor, force fields, command hold."""

import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlp_v4 as baseline  # noqa: E402
from mlp_v4 import (  # noqa: E402, F401
    apply_inference_overrides,
    configure_robot_and_simulator,
    motion_lib_config,
    scene_lib_config,
    terrain_config,
)
from protomotions.agents.common.shared_joint_mlp import SharedJointMLPConfig  # noqa: E402
from protomotions.envs.control.pendulum_research_control import (  # noqa: E402
    PendulumResearchControlConfig,
)


def additional_experiment_arguments(parser):
    baseline.additional_experiment_arguments(parser)
    parser.set_defaults(assist_limit=20.0, separate_torque_limits=True)
    group = parser.add_argument_group("Pendulum-only research ablations")
    group.add_argument("--structured-disturbance-prob", type=float, default=0.0)
    group.add_argument("--human-target-period", type=float, default=0.01, help="PD/FF command refresh [s]")
    group.add_argument("--shared-joint-actor", action="store_true")
    group.add_argument("--combined-human-limit", action="store_true", default=True)
    group.add_argument("--external-human-ff", action="store_false", dest="combined_human_limit",
                       help="Historical separately saturated external FF, for diagnostics only")


def env_config(robot_cfg, args):
    cfg = baseline.env_config(robot_cfg, args)
    cfg.human_feedforward_inside_pd = args.combined_human_limit
    steps = round(args.human_target_period / 0.01)
    if steps < 1 or abs(args.human_target_period - steps * 0.01) > 1e-9:
        raise ValueError("human-target-period must be a positive multiple of the 10 ms control step")
    old = cfg.control_components["assist"]
    values = {f.name: getattr(old, f.name) for f in fields(old) if f.name != "_target_"}
    cfg.control_components["assist"] = PendulumResearchControlConfig(
        **values, structured_probability=args.structured_disturbance_prob,
        human_target_hold_steps=steps,
    )
    return cfg


def agent_config(robot_config, env_config, args):
    cfg = baseline.agent_config(robot_config, env_config, args)
    if args.shared_joint_actor:
        old = cfg.model.actor.mu_model
        values = {f.name: getattr(old, f.name) for f in fields(old) if f.name != "_target_"}
        cfg.model.actor.mu_model = SharedJointMLPConfig(**values)
    return cfg
