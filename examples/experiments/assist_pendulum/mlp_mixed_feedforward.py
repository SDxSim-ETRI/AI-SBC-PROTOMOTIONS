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
"""L11: L9 plus a 50/50 mixture of position-PD and velocity-FF wearers.

Controller type is sampled once per environment. Both types remain additive,
with identical impedance, inertia, disturbances and rewards. No W1 trajectories
or privileged wearer-controller type are added to the actor observations.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlp_relative_opposition import (  # noqa: E402, F401
    additional_experiment_arguments,
    agent_config,
    apply_inference_overrides,
    configure_robot_and_simulator,
    env_config as _env_config,
    motion_lib_config,
    scene_lib_config,
    terrain_config,
)


def env_config(robot_cfg, args):
    cfg = _env_config(robot_cfg, args)
    cfg.human_feedforward_disable_prob = 0.5
    return cfg
