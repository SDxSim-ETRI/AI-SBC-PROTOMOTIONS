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
"""R0/R1: the L1 recipe on the corrected torque path, optionally with a paired relative burden cost.

R0 (control): mlp_v4 (L1) defaults; the only change is the torque accounting — the
    human PD+FF is one actuator limited once and assistance/loads are injected
    independently, as in G0-G4 and F1/F2 (no C2 blending, no absorbed-power cost).
R1 (candidate): R0 + a reward-only cost on the wearer torque in excess of an unassisted
    twin integrated with the same intent, plant and exogenous loads:
        -w * mean_j relu(tau_B_j^2 - tau_A_j^2) / (mean_50ms(tau_A_j^2) + floor^2)
    The ratio is dimensionless, so low-need states weigh as much as heavy-load states.
    No new actor inputs, no phase gates, no W1 data, no change to L1's other rewards.

Training (sbc env):
    python protomotions/train_agent.py --robot-name hip_pendulum --simulator newton \
        --experiment-path examples/experiments/assist_pendulum/mlp_relative.py \
        --experiment-name llp_relative_R1_s0_20261020 --motion-file none \
        --num-envs 4096 --batch-size 16384 --seed 0 --training-max-steps 786432000 \
        --paired-excess-weight=-0.5
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
from protomotions.envs.context_views import EnvContext  # noqa: E402
from protomotions.envs.mdp_component import MdpComponent  # noqa: E402
from protomotions.envs.rewards.assist_energy import compute_paired_excess_burden  # noqa: E402

PAIRED_ENV = "protomotions.envs.base_env.paired_assist_env.PairedAssistEnv"


def additional_experiment_arguments(parser):
    baseline.additional_experiment_arguments(parser)
    g = parser.add_argument_group("R0/R1 paired relative burden")
    g.add_argument("--paired-excess-weight", type=float, default=0.0,
                   help="Reward weight (<= 0) of the paired excess burden; 0 = R0 control without a twin. Pass as '=-0.5'")
    g.add_argument("--excess-floor", type=float, default=10.0, help="Torque floor [N·m] added (squared) to the window scale")
    g.add_argument("--excess-window-steps", type=int, default=5, help="Causal window of the twin's tau^2 [policy steps]")
    g.add_argument("--excess-min-value", type=float, default=-3.0, help="Per-step clip of the weighted cost, as the other wearer costs")


def env_config(robot_cfg, args):
    cfg = baseline.env_config(robot_cfg, args)
    # Common corrected torque path (G/F studies): one human actuator limit, independent sources.
    cfg.clamp_combined_injection = False
    cfg.human_feedforward_inside_pd = True
    weight = float(args.paired_excess_weight)
    if not -float("inf") < weight <= 0.0:
        raise ValueError("paired-excess-weight must be finite and non-positive")
    if not 0.0 < float(args.excess_floor) < float("inf") or int(args.excess_window_steps) < 1:
        raise ValueError("excess-floor must be finite positive and excess-window-steps >= 1")
    cfg.paired_reference_window_steps = int(args.excess_window_steps)
    if weight:
        cfg._target_ = PAIRED_ENV
        cfg.reward_components["paired_excess"] = MdpComponent(
            compute_func=compute_paired_excess_burden,
            dynamic_vars={
                "tau_agent": EnvContext.assist.tau_agent,
                "reference_tau_agent": EnvContext.assist.reference_tau_agent,
                "reference_tau_sq_window": EnvContext.assist.reference_tau_sq_window,
            },
            static_params={
                "floor": float(args.excess_floor),
                "weight": weight,
                "min_value": float(args.excess_min_value),
                "zero_during_grace_period": True,
            },
        )
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    cfg = baseline.agent_config(robot_cfg, env_cfg, args)
    cfg.save_epoch_checkpoint_every = 500
    cfg.save_last_checkpoint_every = 50
    # Fixed final epoch, as F1/F2: no periodic evaluation or best-checkpoint selection.
    cfg.evaluator.eval_metrics_every = None
    return cfg
