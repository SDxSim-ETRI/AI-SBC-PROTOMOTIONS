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
"""LLP 학습 env(SkeletonAssistEnv) 점검 — 내장 tracker가 걷는지, 신호 스케일은 어떤지.

학습 run 디렉터리의 `resolved_configs_inference.pt`로 env를 만들고 assist=0으로
N step 돌리며 기록한다:

  * 에피소드 종료 수·원인 (mimic 몸체 오차 / 과속 / 클립 종료)
  * 착용자 몸체 위치 오차 (tracker 자체 지표와 비교 — 단독 tracker ≈ 8 cm)
  * θ_g(=mocap hip flexion) − θ 의 크기 → 평가 threshold 설계 근거
  * τ_agent(hip) RMS, 착용자 hip 파워 → human_power 보상 weight 근거
  * HLP chunk lead별 오차 (실 ManiFlow, 보행 tracker 스트림)

실행 (sbc env):
    python examples/experiments/assist_skeleton/check_train_env.py --run results/llp_skel_smoke --num-envs 16 --steps 600
"""

import argparse
import math
import sys
from pathlib import Path

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

import_simulator_before_torch("newton")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from lightning.fabric import Fabric  # noqa: E402

from protomotions.utils.hydra_replacement import get_class  # noqa: E402


def build_env(run_dir: Path, num_envs: int):
    resolved = torch.load(run_dir / "resolved_configs_inference.pt", map_location="cpu", weights_only=False)
    from protomotions.utils.inference_utils import apply_backward_compatibility_fixes

    apply_backward_compatibility_fixes(resolved["robot"], resolved["simulator"], resolved["env"])
    resolved["simulator"].num_envs = num_envs
    resolved["simulator"].headless = True
    fabric = Fabric(accelerator="gpu", devices=1, num_nodes=1, loggers=[], callbacks=[])
    fabric.launch()
    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator
    from protomotions.utils.component_builder import build_all_components

    terrain_cfg, sim_cfg = convert_friction_for_simulator(resolved["terrain"], resolved["simulator"])
    comps = build_all_components(
        terrain_config=terrain_cfg, scene_lib_config=resolved["scene_lib"],
        motion_lib_config=resolved["motion_lib"], simulator_config=sim_cfg,
        robot_config=resolved["robot"], device=fabric.device, save_dir=None,
    )
    EnvClass = get_class(resolved["env"]._target_)
    env = EnvClass(
        config=resolved["env"], robot_config=resolved["robot"], device=fabric.device,
        terrain=comps["terrain"], scene_lib=comps["scene_lib"], motion_lib=comps["motion_lib"],
        simulator=comps["simulator"],
    )
    return env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="results/llp_skel_smoke")
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--assist-scale", type=float, default=0.0, help="0 = 착용자 단독")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = build_env(Path(args.run), args.num_envs)
    comp = env.assist_component
    torch.manual_seed(args.seed)
    obs, _ = env.reset()
    n_act = comp.num_assist_dofs
    hip = comp.assist_dof_indices

    logs = {k: [] for k in ("err_g", "tau_agent", "power", "body_err", "root_z", "hip_pos", "chunk", "t0", "age", "t")}
    resets, term_track, term_vel, done_clip = 0, 0, 0, 0
    for step in range(args.steps):
        action = torch.zeros(args.num_envs, n_act, device=env.device)
        if args.assist_scale != 0.0:
            action = torch.randn_like(action) * args.assist_scale
        obs, rew, done, term, extras = env.step(action)
        ctx = env._current_context
        a = ctx.assist
        logs["err_g"].append((a.dof_pos - a.theta_g).abs().mean(dim=-1).cpu())
        logs["tau_agent"].append(a.tau_agent.cpu())
        logs["power"].append((a.tau_agent * a.dof_vel).abs().sum(dim=-1).cpu())
        body_err = (ctx.current.rigid_body_pos - ctx.mimic.ref_state.rigid_body_pos).norm(dim=-1).mean(dim=-1)
        logs["body_err"].append(body_err.cpu())
        logs["root_z"].append(ctx.current.rigid_body_pos[:, 0, 2].cpu())
        logs["hip_pos"].append(a.dof_pos.cpu())
        logs["chunk"].append(comp._chunk.cpu().clone())
        logs["t0"].append(comp._chunk_t0.cpu().clone())
        logs["age"].append(a.chunk_age[:, 0].cpu())
        logs["t"].append((env.progress_buf.float() * env.dt).cpu())
        # 종료 원인 집계 (termination 컴포넌트 출력이 extras에 있으면 사용)
        if done.any():
            ids = done.nonzero(as_tuple=False).flatten()
            resets += len(ids)
            term_track += int(term[ids].sum())
            env.reset(ids)
            obs = env.get_obs()

    L = {k: torch.stack(v) for k, v in logs.items()}  # (T, N, ...)
    err_g = L["err_g"]
    print("\n=== SkeletonAssistEnv 점검 (assist_scale=%.2f, %d env × %d step = %.1f s) ===" % (
        args.assist_scale, args.num_envs, args.steps, args.steps * env.dt))
    print(f"episodes reset: {resets} (terminated by failure: {term_track})  env dt={env.dt}")
    print(f"body tracking error [m]: mean {L['body_err'].mean():.3f}  p95 {L['body_err'].quantile(0.95):.3f}")
    print(f"root z [m]: mean {L['root_z'].mean():.3f}  min {L['root_z'].min():.3f}")
    print(f"|θ_g − θ| hip (mocap vs actual) [deg]: mean {math.degrees(err_g.mean()):.1f}  p50 {math.degrees(err_g.median()):.1f}  p95 {math.degrees(err_g.quantile(0.95)):.1f}")
    print(f"τ_agent hip RMS [N·m]: {L['tau_agent'].pow(2).mean().sqrt():.1f}   human hip power [W]: mean {L['power'].mean():.1f}")
    w_pow = env.config.reward_components["human_power"].static_params.get("weight", -0.0015)
    print(f"human_power reward/step at this scale: {w_pow * L['power'].mean():.3f} (w {w_pow}); "
          f"reward keys: {list(env.config.reward_components)}")
    # HLP lead 오차
    T, N = L["t"].shape
    K = L["chunk"].shape[2]
    hlp_dt = comp.config.hlp_dt
    errs = [[] for _ in range(K)]
    t = L["t"]
    for n in range(N):
        tn = t[:, n].numpy()
        pos = L["hip_pos"][:, n].numpy()
        # 에피소드 경계(시간 역행) 이후만 보간되도록 단조 구간별 처리
        starts = [0] + [i for i in range(1, T) if tn[i] < tn[i - 1]] + [T]
        for s0, s1 in zip(starts[:-1], starts[1:]):
            seg_t, seg_p = tn[s0:s1], pos[s0:s1]
            if len(seg_t) < 20:
                continue
            fresh = [i for i in range(s0, s1) if L["age"][i, n] < 1e-6 and tn[i] >= 1.0]
            for i in fresh:
                for k in range(K):
                    tk = tn[i] + k * hlp_dt
                    if tk > seg_t[-1]:
                        break
                    actual = np.array([np.interp(tk, seg_t, seg_p[:, d]) for d in range(2)])
                    errs[k].append(np.abs(L["chunk"][i, n, k].numpy() - actual))
    lead = {f"{int(round(k*hlp_dt*1e3))}ms": (math.degrees(np.mean(e)) if e else float('nan')) for k, e in enumerate(errs)}
    print("HLP lead MAE [deg]:", {k: round(v, 2) for k, v in lead.items()})


if __name__ == "__main__":
    main()
