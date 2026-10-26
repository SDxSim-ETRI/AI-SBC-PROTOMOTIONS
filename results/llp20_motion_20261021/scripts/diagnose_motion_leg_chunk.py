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
"""진단 (2026-10-22): 학습 분포 진자에서 미래 목표 관측이 에뮬레이션 예측일 때와 실제 H1일 때의 차이.

새 레시피(`mlp_motion_leg.py`)의 학습 환경을 그대로 짝 환경으로 만들고(집단·관성·하중·실현 가능성
같음, H1 학습분할 동작), 보조 관절(오른쪽) 하나만 비교한다. 미래 목표 관측은
* emulated: 학습과 같은 에뮬레이션 예측 (의도 θ_g의 미래 + 잡음·지연),
* h1: 동결 H1이 실제 관절 각도 이력으로 낸 예측 (몸통 입력 없음, 합성 집단 포함 모든 env 동일).
L1·R0는 환경 출력 20·tanh에 그대로 넣어 ×0.5와 같다.

    python examples/experiments/assist_pendulum/diagnose_motion_leg_chunk.py \\
        --config-checkpoint results/llp20_motion_A0D0_right_s0_20261021/epoch_6000.ckpt \\
        --checkpoint <정책> [--hlp-checkpoint <H1>] --out <json>
"""

import argparse
import json
from pathlib import Path

from protomotions.utils.simulator_imports import import_simulator_before_torch

import_simulator_before_torch("newton")
import numpy as np  # noqa: E402
import torch  # noqa: E402
from evaluate_motion_leg import TrunkH1Forecast  # noqa: E402
from evaluate_research import seed_all  # noqa: E402
from runtime_utils import build_env_agent  # noqa: E402

from protomotions.assist.llp_policy import LLPPolicy  # noqa: E402
from protomotions.envs.obs.assist_obs import compute_assist_target_obs  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-checkpoint", required=True, help="학습 환경 설정을 가져올 새 레시피 체크포인트")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--hlp-checkpoint", default=None, help="주면 실제 H1 예측, 없으면 에뮬레이션")
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--plant-seed", type=int, default=123)
    p.add_argument("--trajectory-seed", type=int, default=7)
    p.add_argument("--hlp-seed", type=int, default=991)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    seed_all(args.plant_seed)
    env, _ = build_env_agent(args.config_checkpoint, args.num_envs,
                             overrides={"env._target_": "protomotions.envs.base_env.paired_assist_env.PairedAssistEnv"})
    env._lazy_init_assist()
    ref = env.reference_env
    env.max_episode_length = args.steps + 2
    joint = int(tuple(env.config.assist_active_joints)[0])
    policy = LLPPolicy(args.checkpoint, env.device)
    assert policy.layout.history_steps == env.config.num_state_history_steps
    pools = env.assist_component.env_pool.cpu().numpy()
    names = [s["name"] for s in env.assist_component.pool_summary]
    n = args.num_envs
    logs = {k: [] for k in ("a", "b", "s", "v", "valid")}
    with torch.no_grad():
        seed_all(args.trajectory_seed)
        obs, _ = env.reset()
        h1 = None
        if args.hlp_checkpoint:
            h1 = TrunkH1Forecast(args.hlp_checkpoint, env)
            h1.reset(env.context.current.dof_pos, args.hlp_seed)
        first = torch.full((n,), -1, device=env.device)
        for step in range(args.steps):
            valid = first < 0
            if h1 is not None:
                predicted = h1.predict(env.context.current.dof_pos, step * env.dt, observe=step > 0)
                obs["assist_target"] = compute_assist_target_obs(
                    predicted, env.context.current.dof_pos, torch.zeros(n, 1, device=env.device))
            obs, _, _, terminal, _ = env.step(policy.act(obs))
            failed = (terminal | ref.terminate_buf) & (first < 0)
            first[failed] = step
            if step >= args.warmup_steps:
                logs["a"].append(ref.context.assist.tau_agent[:, joint].cpu().numpy())
                logs["b"].append(env.context.assist.tau_agent[:, joint].cpu().numpy())
                logs["s"].append(env.context.assist.tau_assist[:, joint].cpu().numpy())
                logs["v"].append(ref.context.current.dof_vel[:, joint].cpu().numpy())
                logs["valid"].append(valid.cpu().numpy())
    z = {k: np.stack(v) for k, v in logs.items()}
    result = dict(protocol=vars(args), joint=joint, pools={})
    for k, name in [(k, nm) for k, nm in enumerate(names)] + [(None, "all")]:
        m = z["valid"] & (True if k is None else (pools == k)[None, :])
        a, b, s, v = z["a"][m], z["b"][m], z["s"][m], z["v"][m]
        fast = np.abs(v) > 4.0
        result["pools"][name] = dict(
            reference_tau_sq_Nm2=float(np.mean(a * a)), torque_sq_change_percent=float(100 * (np.mean(b * b) / np.mean(a * a) - 1)),
            assist_rms_Nm=float(np.sqrt(np.mean(s * s))), assist_vs_unassisted_torque_corr=float(np.corrcoef(s, a)[0, 1]),
            fast_torque_sq_change_percent=float(100 * (np.mean(b[fast] ** 2) / np.mean(a[fast] ** 2) - 1)) if fast.sum() > 100 else None)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result["pools"], indent=1))


if __name__ == "__main__":
    main()
