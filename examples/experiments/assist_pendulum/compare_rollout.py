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
"""assist 정책 vs 어시스트-제로 베이스라인의 paired rollout 비교.

같은 seed로 두 조건을 순차 실행하므로 의도 궤적 θ_g와 HLP chunk 노이즈가
완전히 동일한 paired 비교가 된다 (mean_action은 RNG를 소비하지 않음).

    python examples/experiments/assist_pendulum/compare_rollout.py \
        --checkpoint results/assist_pendulum/last.ckpt \
        --num-envs 8 --num-steps 1000 --out /tmp/assist_compare.npz
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

import_simulator_before_torch("newton")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from runtime_utils import build_env_agent, rollout_recorded  # noqa: E402


def summarize(logs, dt):
    power = (logs["tau_agent"] * logs["dof_vel"]).abs().sum(-1)  # (T, N)
    err = (logs["dof_pos"] - logs["theta_g"]).abs().mean(-1)  # (T, N)
    return {
        "human_power_mean": power.mean().item(),
        "human_work_per_episode": (power.mean(0) * dt * power.shape[0]).mean().item(),
        "tracking_err_mean": err.mean().item(),
        "assist_power_mean": (logs["tau_assist"] * logs["dof_vel"]).abs().sum(-1).mean().item(),
        "assist_torque_rms": logs["tau_assist"].square().mean().sqrt().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="/tmp/assist_compare.npz")
    args = parser.parse_args()

    env, agent = build_env_agent(args.checkpoint, args.num_envs, headless=True)
    dt = env.dt

    print(f"[1/2] assist policy rollout ({args.num_steps} steps x {args.num_envs} envs)")
    logs_policy, resets_p = rollout_recorded(env, agent, args.num_steps, args.seed)

    print("[2/2] zero-assist baseline rollout (same seed -> same trajectories)")
    env.config.assist_torque_limit = 0.0
    logs_base, resets_b = rollout_recorded(env, agent, args.num_steps, args.seed)

    sp, sb = summarize(logs_policy, dt), summarize(logs_base, dt)
    reduction = 100.0 * (1.0 - sp["human_power_mean"] / sb["human_power_mean"])

    print("\n===== paired comparison (identical trajectories) =====")
    print(f"{'metric':28s} {'policy':>10s} {'baseline':>10s}")
    for k in sp:
        print(f"{k:28s} {sp[k]:10.3f} {sb[k]:10.3f}")
    print(f"\nhuman power reduction: {reduction:.1f}%")
    print(f"resets during rollout: policy={resets_p}, baseline={resets_b}")

    np.savez(
        args.out,
        dt=dt,
        **{f"policy_{k}": v.numpy() for k, v in logs_policy.items()},
        **{f"base_{k}": v.numpy() for k, v in logs_base.items()},
    )
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
