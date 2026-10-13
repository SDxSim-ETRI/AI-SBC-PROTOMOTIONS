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
"""L6/L7: linear torque request, independent final +/-100 Nm limit.

Both variants use 40 Nm per raw-action unit (the L1 small-signal scale),
with no tanh and no inference gain. Smoothness already uses applied Nm.
L6: --human-cost both; L7: --human-cost torque_sq. Train both from scratch.
The plant randomization stays at L1 settings. Independent qfrc sources are
summed without the legacy combined wearer-limit clamp; consequently L1
is a reference policy, not a clean action-limit-only ablation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlp_v4 import (  # noqa: E402, F401
    agent_config as _agent_config,
    apply_inference_overrides,
    configure_robot_and_simulator,
    env_config,
    motion_lib_config,
    scene_lib_config,
    terrain_config,
)
from mlp_v4 import additional_experiment_arguments as _arguments  # noqa: E402


def additional_experiment_arguments(parser):
    _arguments(parser)
    parser.set_defaults(
        assist_limit=100.0,
        action_torque_scale=40.0,
        action_transform="none",
        separate_torque_limits=True,
    )


def agent_config(robot_config, env_config, args):
    cfg = _agent_config(robot_config, env_config, args)
    cfg.save_epoch_checkpoint_every = 2000
    cfg.fail_on_bad_grads = True
    return cfg
