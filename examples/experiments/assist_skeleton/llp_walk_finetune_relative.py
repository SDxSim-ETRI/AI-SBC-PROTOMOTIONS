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
"""L13R: mean-matched relative paired harm, to protect low-burden intervals."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llp_walk_finetune import (
    additional_experiment_arguments as _arguments, env_config as _env_config,
    agent_config, configure_robot_and_simulator, motion_lib_config,
    scene_lib_config, terrain_config, apply_inference_overrides,
)


def additional_experiment_arguments(parser):
    _arguments(parser)
    parser.set_defaults(initial_actor_checkpoint="results/llp_L13_fullbody_walk/epoch_100.ckpt")
    parser.add_argument("--local-excess-weight", type=float, required=True)


def env_config(robot_cfg, args):
    cfg = _env_config(robot_cfg, args)
    if not 0 < args.local_excess_weight < 100:
        raise ValueError("Use the positive coefficient measured on training rollouts")
    cfg.local_excess_floor_Nm = 10.0
    cfg.local_excess_weight = args.local_excess_weight
    return cfg
