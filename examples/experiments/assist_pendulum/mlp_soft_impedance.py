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
"""L10: broaden L9's synthetic wearer impedance to include softer heavy legs.

Natural frequency 6..24 rad/s with Kp >= 60 Nm/rad retains the old minimum
stiffness for light legs while adding soft high-inertia cases. Damping is derived
from the final Kp so the same 0.3..0.8 damping-ratio range is preserved. This is a
training-domain hypothesis, not an identification of W1's full-body dynamics.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlp_relative_opposition import (  # noqa: E402, F401
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
    parser.set_defaults(omega_n_range=(6.0, 24.0))


def env_config(robot_cfg, args):
    cfg = _env_config(robot_cfg, args)
    cfg.human_stiffness_floor = 60.0
    return cfg
