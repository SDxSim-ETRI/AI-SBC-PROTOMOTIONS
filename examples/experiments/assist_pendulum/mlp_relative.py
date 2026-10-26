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
"""R0: the L1 recipe on the corrected torque path.

R0 (control): mlp_v4 (L1) defaults; the only change is the torque accounting — the
    human PD+FF is one actuator limited once and assistance/loads are injected
    independently, as in G0-G4 and F1/F2 (no C2 blending, no absorbed-power cost).
R1 (R0 + paired relative burden cost) was trained once, failed and was removed on 2026-10-26;
    its recipe is archived in results/llp_relative_20261020/scripts/.

Training (sbc env):
    python protomotions/train_agent.py --robot-name hip_pendulum --simulator newton \
        --experiment-path examples/experiments/assist_pendulum/mlp_relative.py \
        --experiment-name llp_relative_R0_s0_20261020 --motion-file none \
        --num-envs 4096 --batch-size 16384 --seed 0 --training-max-steps 786432000
"""
import sys
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


def additional_experiment_arguments(parser):
    baseline.additional_experiment_arguments(parser)


def env_config(robot_cfg, args):
    cfg = baseline.env_config(robot_cfg, args)
    # Common corrected torque path (G/F studies): one human actuator limit, independent sources.
    cfg.clamp_combined_injection = False
    cfg.human_feedforward_inside_pd = True
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    cfg = baseline.agent_config(robot_cfg, env_cfg, args)
    cfg.save_epoch_checkpoint_every = 500
    cfg.save_last_checkpoint_every = 50
    # Fixed final epoch, as F1/F2: no periodic evaluation or best-checkpoint selection.
    cfg.evaluator.eval_metrics_every = None
    return cfg
