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
"""L12: L11 with reward-only paired non-degradation tracking.

The sole learning change replaces absolute tracking reward by per-joint excess
squared error over a separately integrated unassisted wearer. No W1 data or
reference states enter actor/critic inputs. All other L11 rewards are preserved.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlp_mixed_feedforward import (  # noqa: E402, F401
    additional_experiment_arguments,
    agent_config,
    apply_inference_overrides,
    configure_robot_and_simulator,
    env_config as _env_config,
    motion_lib_config,
    scene_lib_config,
    terrain_config,
)
from protomotions.envs.context_views import EnvContext  # noqa: E402
from protomotions.envs.rewards.assist import compute_paired_tracking_rew  # noqa: E402


def env_config(robot_cfg, args):
    cfg = _env_config(robot_cfg, args)
    cfg._target_ = "protomotions.envs.base_env.paired_assist_env.PairedAssistEnv"
    tracking = cfg.reward_components["tracking"]
    tracking.compute_func = compute_paired_tracking_rew
    tracking.dynamic_vars["reference_dof_pos"] = EnvContext.assist.reference_dof_pos
    return cfg
