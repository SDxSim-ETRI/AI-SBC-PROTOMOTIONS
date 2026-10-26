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
"""`mlp_motion_leg.py` 학습 환경 점검: 실제로 학습에 들어가는 의도·관성 분포와 다리별 계약.

무작위 action으로 학습 env를 돌리며 기록한다 (의도 분포는 정책과 무관).
* 집단별 env 수·hip 관성 범위, 집단·동작 그룹별 시간 비율, 담당 관절 |θ̇_g| > 4 rad/s 시간 비율.
* 실현 가능성: 대체 클립 사용 수, 현재 클립이 상한을 넘는 env 비율.
* 다리별 계약: 비담당 관절 보조 0, 보상 원시값이 담당 관절만으로 계산되는지.

    python examples/experiments/assist_pendulum/audit_motion_leg.py \\
        --checkpoint results/<run>/last.ckpt --num-envs 4096 --steps 2000 --out <json>
"""

import argparse
import json
from pathlib import Path

from protomotions.utils.simulator_imports import import_simulator_before_torch

import_simulator_before_torch("newton")
import numpy as np  # noqa: E402
import torch  # noqa: E402
from runtime_utils import build_env_agent  # noqa: E402

from protomotions.envs.control.motion_intent import group_clip_probabilities  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    env, _ = build_env_agent(args.checkpoint, args.num_envs)
    env._lazy_init_assist()
    comp = env.assist_component
    cfg = comp.config
    joint = int(tuple(env.config.assist_active_joints)[0])
    other = 1 - joint
    intent = comp._intent
    names = [str(g[0]) for g in cfg.motion_groups]
    _, group_index = group_clip_probabilities(
        intent.table.clip_type, intent.table.clip_len.cpu().numpy(), cfg.motion_groups,
        [(names[0], 1.0)], intent.min_frames,
    )
    group_index = torch.as_tensor(group_index, device=env.device)
    pools = comp.env_pool
    motion_env = comp._motion_env
    n_pools = len(comp.pool_summary)
    inertia = env.hip_inertia_eff()[:, joint]

    fast = torch.zeros(n_pools, device=env.device)
    total = torch.zeros(n_pools, device=env.device)
    group_time = torch.zeros(n_pools, len(names) + 1, device=env.device)  # 마지막 = hold/전환
    infeasible = 0.0
    other_assist_max = 0.0
    reward_err = {}
    vel_abs = []
    env.reset()
    with torch.no_grad():
        for _ in range(args.steps):
            action = torch.randn(args.num_envs, 2, device=env.device)
            _, goal_vel = comp.get_pd_targets_and_vel()
            v = goal_vel[:, joint].abs()
            vel_abs.append(v.cpu())
            for k in range(n_pools):
                m = pools == k
                fast[k] += (v[m] > 4.0).float().sum()
                total[k] += m.float().sum()
            clip = intent._clip[0]
            hold = intent._hold[0] | ~motion_env
            g = torch.where(hold, torch.full_like(clip, len(names)), group_index[clip])
            group_time.index_put_((pools, g), torch.ones_like(v), accumulate=True)
            need = intent.table.clip_accel99[clip][:, joint]
            cap = intent._accel_cap[:, joint]
            infeasible += float(((need > cap) & ~hold).float().mean())
            _, _, _, _, extras = env.step(action)
            ctx = env.context
            other_assist_max = max(other_assist_max, float(ctx.assist.tau_assist[:, other].abs().max()))
            tau = ctx.assist.tau_agent
            manual = {
                "human_torque_sq": tau[:, joint] ** 2,
                "human_power": (tau[:, joint] * ctx.current.dof_vel[:, joint]).abs(),
                "assist_effort": ctx.assist.tau_assist[:, joint] ** 2,
            }
            grace = env.progress_buf <= env.config.reset_grace_period
            for name, value in manual.items():
                # 리셋 직후 env는 보상 계산 뒤 상태가 바뀌므로 제외
                raw, value = extras["raw_r/" + name][~grace], value[~grace]
                if raw.numel() == 0:
                    continue
                reward_err[name] = max(reward_err.get(name, 0.0), float((raw - value).abs().max()))

    vel_abs = torch.cat(vel_abs).numpy()
    report = dict(
        checkpoint=str(args.checkpoint), active_joint=joint, num_envs=args.num_envs, steps=args.steps,
        pools=[dict(summary, env_count=int((pools == k).sum()),
                    hip_inertia_kgm2=[float(inertia[pools == k].min()), float(inertia[pools == k].max())],
                    fast_time_fraction=float(fast[k] / total[k]),
                    group_time_fraction={n: float(group_time[k, j] / total[k]) for j, n in enumerate(names)},
                    hold_or_synthetic_time_fraction=float(group_time[k, -1] / total[k]))
               for k, summary in enumerate(comp.pool_summary)],
        overall_fast_time_fraction=float(fast.sum() / total.sum()),
        intent_speed_percentiles_rad_s={str(q): float(np.percentile(vel_abs, q)) for q in (50, 90, 99, 99.9)},
        fallback_clips=int(intent.fallback_count),
        infeasible_current_clip_fraction=infeasible / args.steps,
        other_joint_assist_max_Nm=other_assist_max,
        reward_raw_max_abs_error=reward_err,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    assert other_assist_max == 0.0, "non-assigned joint received assistance"
    assert all(v < 1e-3 for v in reward_err.values()), reward_err


if __name__ == "__main__":
    main()
