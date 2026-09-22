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
"""v4 진자 환경 점검 — 학습 전에 "스파이크 있는 세상"이 됐는지 무보조로 돌려 본다.

실험 파일(`mlp_v4.py`)로 env를 조립해 assist=0으로 N step 돌리고, 착용자 토크
τ_agent의 모양을 스켈레톤 보행 착용자(§2-E: 피크/RMS 4.3, 범위 −332~+176 N·m,
|θ̈| RMS 48 rad/s²)와 같은 잣대로 잰다. 관성·게인 랜덤화가 실제로 적용됐는지도
Newton 배열에서 직접 읽는다.

실행 (sbc env):
    python examples/experiments/assist_pendulum/check_env_v4.py --num-envs 64 --steps 1500
    python examples/experiments/assist_pendulum/check_env_v4.py --no-disturbance   # 절제 비교
"""

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

import_simulator_before_torch("newton")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from lightning.fabric import Fabric  # noqa: E402

from protomotions.utils.hydra_replacement import get_class  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default=str(HERE / "mlp_v4.py"))
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--steps", type=int, default=1500, help="10 ms step 수 (1500 = 15 s)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="docs/pendulum_v4_env_check.png")
    args, rest = p.parse_known_args()

    spec = importlib.util.spec_from_file_location("exp", args.experiment)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ep = argparse.ArgumentParser()
    mod.additional_experiment_arguments(ep)
    exp_args = ep.parse_args(rest)

    from protomotions.utils.config_builder import build_standard_configs

    ns = SimpleNamespace(
        robot_name="hip_pendulum", simulator="newton", headless=True, num_envs=args.num_envs,
        experiment_name="check_v4", motion_file="none", scenes_file=None, batch_size=1, training_max_steps=1,
        **vars(exp_args),
    )
    cfgs = build_standard_configs(
        args=ns, terrain_config_fn=mod.terrain_config, scene_lib_config_fn=mod.scene_lib_config,
        motion_lib_config_fn=mod.motion_lib_config, env_config_fn=mod.env_config,
        configure_robot_and_simulator_fn=mod.configure_robot_and_simulator, agent_config_fn=mod.agent_config,
    )
    fabric = Fabric(accelerator="gpu", devices=1, loggers=[], callbacks=[])
    fabric.launch()
    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator
    from protomotions.utils.component_builder import build_all_components

    tc, sc = convert_friction_for_simulator(cfgs["terrain"], cfgs["simulator"])
    comps = build_all_components(terrain_config=tc, scene_lib_config=cfgs["scene_lib"], motion_lib_config=cfgs["motion_lib"],
                                 simulator_config=sc, robot_config=cfgs["robot"], device=fabric.device, save_dir=None)
    env = get_class(cfgs["env"]._target_)(config=cfgs["env"], robot_config=cfgs["robot"], device=fabric.device,
                                            terrain=comps["terrain"], scene_lib=comps["scene_lib"],
                                            motion_lib=comps["motion_lib"], simulator=comps["simulator"])
    torch.manual_seed(args.seed)
    env.reset()
    comp = env.assist_component
    n, d = args.num_envs, comp.n_dofs
    zero = torch.zeros(n, d, device=env.device)
    env.step(zero)  # lazy init (관성·게인 랜덤화)

    import warp as wp
    m = wp.to_torch(env.simulator.robot_view.get_attribute("body_mass", env.simulator.model)).view(n, -1)
    ke = wp.to_torch(env._ke_wp).view(n, -1)[:, env._dof_sim_idx]
    kd = wp.to_torch(env._kd_wp).view(n, -1)[:, env._dof_sim_idx]
    print("=== 플랜트 랜덤화 확인 (Newton 배열 직접 읽음) ===")
    I_eff = env.hip_inertia_eff()
    print(f"  대퇴 질량 [kg]: min {m[:, 1:].min():.2f}  max {m[:, 1:].max():.2f}  (고정 6.49 — 중력 13 N·m 유지)")
    print(f"  hip 축 관성 [kg·m²]: min {I_eff.min():.2f}  max {I_eff.max():.2f}  (진자 0.42, 스켈레톤 다리 3.5)   배율 범위 {cfgs['env'].inertia_scale_range}")
    print(f"  Kp [N·m/rad]: min {ke.min():.0f}  max {ke.max():.0f}  |  Kd: min {kd.min():.1f}  max {kd.max():.1f}   (진자 300/15, 스켈레톤 200/20)")
    wn = torch.sqrt(ke / I_eff)
    print(f"  ω_n [rad/s]: min {wn.min():.1f}  max {wn.max():.1f}   ζ: min {(kd / (2 * torch.sqrt(ke * I_eff))).min():.2f}  max {(kd / (2 * torch.sqrt(ke * I_eff))).max():.2f}")

    L = {k: [] for k in ("g", "q", "w", "tau", "ext", "ev", "kps")}
    resets = 0
    term_err, term_vel = 0, 0
    t0 = time.perf_counter()
    for _ in range(args.steps):
        _, _, done, _, _ = env.step(zero)
        c = env._current_context
        L["g"].append(c.assist.theta_g.cpu()); L["q"].append(c.current.dof_pos.cpu()); L["w"].append(c.current.dof_vel.cpu())
        L["tau"].append(c.assist.tau_agent.cpu()); L["ext"].append(comp.external_torque().cpu())
        L["ev"].append(comp.in_stance_event().cpu()); L["kps"].append(comp.gain_schedule()[0].cpu())
        if done.any():
            ids = done.nonzero(as_tuple=False).flatten(); resets += len(ids)
            term_err += int(((c.current.dof_pos - c.assist.theta_g).abs().max(-1).values[ids] > 1.0).sum())
            term_vel += int((c.current.dof_vel.abs().max(-1).values[ids] > 25.0).sum())
            env.reset(ids)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    A = {k: torch.stack(v).numpy() for k, v in L.items()}  # (T, n, d)
    dt = env.dt
    tau, w, q, g, ext = A["tau"], A["w"], A["q"], A["g"], A["ext"]
    acc = np.gradient(w, dt, axis=0)
    rms = lambda x: float(np.sqrt(np.mean(x ** 2)))  # noqa: E731
    print(f"\n=== 무보조 롤아웃 {args.steps} step × {n} env ({args.steps*dt:.0f} s), {wall/args.steps*1e3:.2f} ms/step (게인 갱신 포함) ===")
    print(f"  리셋 {resets}회 (10 s 만기 ≈ {n * int(args.steps * dt // 10)}회) — 조기 종료 원인: 추종 오차>1 rad {term_err}회, 각속도>25 rad/s {term_vel}회")
    print(f"  τ_agent: RMS {rms(tau):.1f} N·m, 범위 [{tau.min():.0f}, {tau.max():.0f}], 피크/RMS {np.abs(tau).max()/rms(tau):.2f}   ← 스켈레톤 보행: RMS 78, [−332, +176], 4.28")
    print(f"  τ_agent p99 |τ| {np.percentile(np.abs(tau), 99):.0f} N·m;  |Δτ|/step RMS {rms(np.diff(tau, axis=0))/dt:.0f} N·m/s")
    print(f"  외부 하중 τ_ext: RMS {rms(ext):.1f}, 범위 [{ext.min():.0f}, {ext.max():.0f}] N·m;  입각 사건 중인 시간 {100*A['ev'].mean():.0f} %;  Kp 배율 최대 {A['kps'].max():.1f}")
    print(f"  |θ̈| RMS {rms(acc):.1f} rad/s², 최대 {np.abs(acc).max():.0f}   ← 스켈레톤: 48 / 408 (mocap 19 / 72)")
    print(f"  |θ_g − θ| 평균 {np.degrees(np.abs(g - q).mean()):.2f}°, p95 {np.degrees(np.percentile(np.abs(g - q), 95)):.1f}°  (v3 진자 2.9°)")
    print(f"  |τ·ω| 평균 {np.abs(tau * w).sum(-1).mean():.1f} W (v3 12 W),  τ² 평균 {(tau ** 2).sum(-1).mean():.0f} (N·m)²  → 보상 weight 크기 판단용")

    # 그림: env 0의 6 s
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"; matplotlib.rcParams["axes.unicode_minus"] = False
    INK, TEAL, ORANGE, GREY = "#141a21", "#0e7a74", "#c2410c", "#9aa8b6"
    e = 0; s = slice(200, 800); tt = np.arange(args.steps)[s] * dt
    fig, ax = plt.subplots(3, 1, figsize=(11.5, 7.5), sharex=True)
    ax[0].plot(tt, np.degrees(g[s, e, 0]), color=GREY, lw=2, label="의도 θ_g"); ax[0].plot(tt, np.degrees(q[s, e, 0]), color=INK, lw=1.3, label="실제 각도")
    ax[1].plot(tt, ext[s, e, 0], color=TEAL, lw=1.6, label="외부 하중 τ_ext (환경)"); ax[1].plot(tt, 40 * A["kps"][s, e, 0], color=GREY, lw=1, ls="--", label="착용자 Kp 배율 ×40")
    ax[2].plot(tt, tau[s, e, 0], color=ORANGE, lw=1.6, label="착용자 토크 τ_agent")
    for a in ax: a.grid(alpha=.25); a.legend(fontsize=9, loc="upper right")
    ax[0].set_ylabel("[deg]"); ax[1].set_ylabel("[N·m]"); ax[2].set_ylabel("[N·m]"); ax[2].set_xlabel("t [s]")
    fig.suptitle(f"v4 진자 env 무보조 — env 0 (hip 관성 {I_eff[0,0]:.2f} kg·m², Kp {ke[0,0]:.0f}, ω_n {wn[0,0]:.1f} rad/s)", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(args.out, dpi=135); print("saved", args.out)


if __name__ == "__main__":
    main()
