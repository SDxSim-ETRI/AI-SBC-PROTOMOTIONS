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
"""Walking tempo curriculum, preserving the full-body relative-harm objective.

Motion ids1/3/5 are training tempos; other tempos stay held out.
"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'examples/experiments/assist_skeleton'))
from llp_walk_finetune_relative import (
    additional_experiment_arguments, env_config as _env_config,
    agent_config, configure_robot_and_simulator, motion_lib_config,
    scene_lib_config, terrain_config, apply_inference_overrides,
)


def env_config(robot_cfg,args):
    cfg=_env_config(robot_cfg,args)
    cfg.motion_manager.exclude_motion_ids=[0,2,4,6]
    # The prior 5mm training slack did not enforce the established 1mm gate.
    cfg.body_margin=.001
    return cfg
