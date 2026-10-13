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
"""Measure paired harm normalization on training-only walking segments."""

import argparse
import hashlib
import json
from pathlib import Path

from protomotions.utils.simulator_imports import import_simulator_before_torch
import_simulator_before_torch("newton")
import torch

from check_train_env import build_env
from protomotions.assist.llp_policy import LLPPolicy
from protomotions.envs.base_env.paired_skeleton_assist_env import paired_burden_terms


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='results/llp_L13_fullbody_walk/epoch_100.ckpt')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    torch.manual_seed(37)
    path = Path(args.checkpoint)
    env = build_env(path.parent, 32)
    env.max_episode_length = 800
    actor = LLPPolicy(str(path), env.device)
    obs, _ = env.reset()
    old_values, new_values = [], []
    for step in range(800):
        obs, _, done, _, _ = env.step(actor.act(obs))
        a, b = env.reference_env.context.assist, env.context.assist
        active = env.progress_buf > env.config.warmup_steps
        _, old = paired_burden_terms(a.tau_agent, b.tau_agent, 1600.)
        _, new = paired_burden_terms(a.tau_agent, b.tau_agent, 1600., floor=10.)
        old_values.append(old[active].cpu())
        new_values.append(new[active].cpu())
        ids = done.nonzero().flatten()
        if len(ids):
            obs, _ = env.reset(ids)
    old = torch.cat(old_values).mean().item()
    new = torch.cat(new_values).mean().item()
    assert old > 0 and new > 0
    report = dict(checkpoint=str(path), checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  seed=37, num_envs=32, steps=800, active_samples=sum(x.numel() for x in old_values),
                  training_only_start_s=[0,20], old_cost_mean=old, relative_cost_mean=new,
                  relative_torque_floor_Nm=10., mean_matched_weight=old/new,
                  uses_evaluation_torques=False)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
