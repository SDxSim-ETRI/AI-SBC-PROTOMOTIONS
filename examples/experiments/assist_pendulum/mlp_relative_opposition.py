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
"""L9: replace L8's absolute opposition with a mean-matched relative cost.

The weight is calibrated on a synthetic pendulum rollout, never W1 test torques.
All actor inputs, plant, other rewards and the gain-1 torque contract stay fixed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlp_opposition import (  # noqa: E402, F401
    additional_experiment_arguments as _arguments,
    agent_config,
    apply_inference_overrides,
    configure_robot_and_simulator,
    env_config as _env_config,
    motion_lib_config,
    scene_lib_config,
    terrain_config,
)


def additional_experiment_arguments(parser):
    _arguments(parser)
    parser.set_defaults(opposition_weight=None)
    parser.add_argument("--opposition-torque-floor", type=float, default=10.0)


def env_config(robot_cfg, args):
    import math
    from protomotions.envs.rewards.assist import compute_relative_assist_opposition_cost

    if args.opposition_weight is None:
        raise ValueError("Supply a negative --opposition-weight from pendulum calibration")
    floor = args.opposition_torque_floor
    if not math.isfinite(floor) or floor <= 0:
        raise ValueError("opposition-torque-floor must be finite and positive")
    cfg = _env_config(robot_cfg, args)
    cost = cfg.reward_components["assist_opposition"]
    cost.compute_func = compute_relative_assist_opposition_cost
    cost.static_params["torque_floor"] = floor
    return cfg
