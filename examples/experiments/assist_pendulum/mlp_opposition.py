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
"""L8: L6 recipe plus a same-window wearer/device opposition cost.

Fresh training, unchanged actor/plant/observations/action contract. No fixed gait
phase gate and no inference-only scaling. The new signal is a hypothesis about
coordination; it is not a guarantee of lower counterfactual or phase-wise burden.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlp_wide import (  # noqa: E402, F401
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
    parser.add_argument("--opposition-weight", type=float, default=-1e-4,
                        help="Reward weight on sum(relu(-tau_agent*tau_assist)); training only")


def env_config(robot_cfg, args):
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.rewards.assist import compute_assist_opposition_cost

    if not math.isfinite(args.opposition_weight) or args.opposition_weight >= 0:
        raise ValueError("opposition-weight must be finite and negative")
    cfg = _env_config(robot_cfg, args)
    cfg.reward_components["assist_opposition"] = MdpComponent(
        compute_func=compute_assist_opposition_cost,
        dynamic_vars={"tau_agent": EnvContext.assist.tau_agent,
                      "tau_assist": EnvContext.assist.tau_assist},
        static_params={"weight": args.opposition_weight, "min_value": -3.0,
                       "zero_during_grace_period": True},
    )
    return cfg
