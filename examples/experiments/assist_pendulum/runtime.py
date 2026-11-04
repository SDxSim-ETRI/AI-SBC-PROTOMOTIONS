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
"""assist_pendulum 런타임 — 체크포인트/실험 파일에서 env+agent를 만들고 돌리는 도구 한 벌.

`PendulumRuntime`
  * `from_checkpoint(ckpt, num_envs, ...)`: `resolved_configs_inference.pt` 로드 → 컴포넌트 빌드 → env/agent
    생성 → 체크포인트 로드 (inference_agent.py의 구성 경로를 최소한으로 복제).
  * `from_experiment(mlp.py, [--recipe ...], num_envs)`: 실험 파일로 학습 env를 조립 (agent 없음).
  * `rollout(num_steps, seed)`: deterministic(mean_action) rollout 시계열 기록.
  * `play(zero_assist)`: Newton viewer 재생 (θ_g 고스트 다리 · 사람/assist 토크 화살표 오버레이).
  * `compare(...)`: assist 정책 vs 어시스트-제로의 paired rollout 비교 (같은 seed → 같은 θ_g·chunk 노이즈).
  * `check_env(...)`: 학습 전 "스파이크 있는 세상" 점검 — 무보조로 돌려 착용자 토크 모양·랜덤화 적용을 본다.
`AbcEvaluation`: A(assist off) / B(에뮬 chunk) / C(실 ManiFlow chunk) paired 평가 — 에뮬레이션 chunk로
  학습된 LLP가 실 HLP chunk에서도 성능을 유지하면 노이즈·지연 에뮬레이션 설계가 옳았다는 뜻이다.
  조건마다 별도 프로세스로 돈다 (chunk 소스는 env 생성 시점에 정해지고 Newton/Fabric을 한 프로세스에서
  재생성하지 않기 위함).

    python examples/experiments/assist_pendulum/runtime.py play --checkpoint checkpoints/L15_mlp_ppo_pendulum/epoch_6000.ckpt
    python examples/experiments/assist_pendulum/runtime.py compare --checkpoint <ckpt> --num-envs 8 --num-steps 1000
    python examples/experiments/assist_pendulum/runtime.py check-env --num-envs 64 --steps 1500 -- --recipe R0
    python examples/experiments/assist_pendulum/runtime.py abc --checkpoint <ckpt> --zarr <zarr> --split val --out results/eval_abc/seed_val

play 카메라: f 추적 on/off (off면 마우스 자유 시점) · [ ] 궤도 회전 · b/n 후면/정면 · z/x 줌 · r 리셋 · q 종료.
"""

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, Optional, Tuple

from protomotions.utils.simulator_imports import import_simulator_before_torch

import_simulator_before_torch("newton")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from protomotions.utils.hydra_replacement import get_class  # noqa: E402

HERE = Path(__file__).resolve().parent
ROLLOUT_KEYS = ("dof_pos", "dof_vel", "theta_g", "theta_d", "tau_agent", "tau_assist")


def _fabric():
    from lightning.fabric import Fabric

    fabric = Fabric(accelerator="gpu", devices=1, num_nodes=1, loggers=[], callbacks=[])
    fabric.launch()
    return fabric


def _build_env(cfgs: Dict[str, object], device):
    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator
    from protomotions.utils.component_builder import build_all_components

    terrain_config, simulator_config = convert_friction_for_simulator(cfgs["terrain"], cfgs["simulator"])
    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=cfgs["scene_lib"],
        motion_lib_config=cfgs["motion_lib"],
        simulator_config=simulator_config,
        robot_config=cfgs["robot"],
        device=device,
        save_dir=getattr(cfgs["env"], "save_dir", None),
    )
    env_cls = get_class(cfgs["env"]._target_)
    return env_cls(
        config=cfgs["env"],
        robot_config=cfgs["robot"],
        device=device,
        terrain=components["terrain"],
        scene_lib=components["scene_lib"],
        motion_lib=components["motion_lib"],
        simulator=components["simulator"],
    )


def load_experiment_module(path):
    spec = importlib.util.spec_from_file_location("experiment_module", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PendulumRuntime:
    """env(+agent)와 그 위에서 도는 도구들."""

    # ---- 오버레이 스케일/색 (play) ----
    ARROW_SCALE_HUMAN = 0.010  # m per N·m
    ARROW_SCALE_ASSIST = 0.020
    ARROW_MAX_LEN = 0.9  # m
    LEG_LEN = 0.42
    COLOR_HUMAN = (1.0, 0.55, 0.10)
    COLOR_ASSIST = (0.15, 0.90, 0.35)
    COLOR_TARGET = (0.45, 0.75, 0.55)
    # 몸체 색 (Newton MJCF 임포터가 primitive geom의 rgba를 무시하므로 빌드 후 model.shape_color를 geom 이름 기준으로 덮어쓴다)
    BODY_PALETTE = {
        "pelvis_geom": (0.87, 0.84, 0.75),
        "iliac": (0.87, 0.84, 0.75),
        "spine": (0.80, 0.77, 0.68),
        "shoulders": (0.80, 0.77, 0.68),
        "head": (0.87, 0.84, 0.75),
        "right_hip_ball": (0.82, 0.38, 0.26),
        "right_thigh_geom": (0.78, 0.47, 0.37),
        "right_knee": (0.82, 0.38, 0.26),
        "left_hip_ball": (0.24, 0.44, 0.70),
        "left_thigh_geom": (0.36, 0.53, 0.76),
        "left_knee": (0.24, 0.44, 0.70),
    }

    def __init__(self, env, agent=None, configs: Optional[Dict[str, object]] = None, checkpoint: Optional[Path] = None):
        self.env = env
        self.agent = agent
        self.configs = configs
        self.checkpoint = checkpoint

    # ------------------------------------------------------------------
    # 생성
    # ------------------------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        num_envs: int,
        headless: bool = True,
        overrides: Optional[Dict[str, object]] = None,
        config_transform: Optional[Callable[[Dict[str, object]], None]] = None,
    ) -> "PendulumRuntime":
        """체크포인트에서 env + agent 재구성.

        Args:
            checkpoint: last.ckpt 경로 (같은 폴더의 resolved_configs_inference.pt를 읽는다).
            num_envs: 병렬 env 수.
            headless: False면 Newton viewer 창 표시.
            overrides: {"env.assist_torque_limit": 0.0} 형식의 config override.
            config_transform: 스칼라 override 뒤, 컴포넌트 생성 전의 in-memory 변환 (typed 평가 config용).
        """
        ckpt = Path(checkpoint)
        resolved_path = ckpt.parent / "resolved_configs_inference.pt"
        assert resolved_path.exists(), f"resolved configs not found: {resolved_path}"
        resolved = torch.load(resolved_path, map_location="cpu", weights_only=False)

        from protomotions.utils.inference_utils import apply_backward_compatibility_fixes

        apply_backward_compatibility_fixes(resolved["robot"], resolved["simulator"], resolved["env"])
        resolved["simulator"].num_envs = num_envs
        resolved["simulator"].headless = headless
        if overrides:
            from protomotions.utils.config_utils import apply_config_overrides

            apply_config_overrides(
                overrides,
                resolved["env"],
                resolved["simulator"],
                resolved["robot"],
                resolved["agent"],
                resolved.get("terrain"),
                resolved["motion_lib"],
                resolved["scene_lib"],
            )
        if config_transform is not None:
            config_transform(resolved)

        fabric = _fabric()
        cfgs = dict(resolved, terrain=resolved.get("terrain"))
        env = _build_env(cfgs, fabric.device)
        agent_cls = get_class(resolved["agent"]._target_)
        agent = agent_cls(config=resolved["agent"], env=env, fabric=fabric, root_dir=ckpt.parent)
        agent.setup()
        agent.load(str(ckpt), load_env=False)
        agent.model.eval()
        return cls(env, agent, cfgs, ckpt)

    @classmethod
    def from_experiment(
        cls, experiment, experiment_args, num_envs: int, headless: bool = True, experiment_name: str = "check"
    ) -> "PendulumRuntime":
        """실험 파일(경로 또는 모듈)과 그 CLI 인자 목록으로 학습 env를 조립한다 (agent 없음)."""
        from protomotions.utils.config_builder import build_standard_configs

        module = experiment if hasattr(experiment, "env_config") else load_experiment_module(experiment)
        parser = argparse.ArgumentParser()
        module.additional_experiment_arguments(parser)
        exp_args = parser.parse_args(list(experiment_args))
        ns = SimpleNamespace(
            robot_name="hip_pendulum", simulator="newton", headless=headless, num_envs=num_envs,
            experiment_name=experiment_name, motion_file="none", scenes_file=None, batch_size=1, training_max_steps=1,
            **vars(exp_args),
        )
        cfgs = build_standard_configs(
            args=ns, terrain_config_fn=module.terrain_config, scene_lib_config_fn=module.scene_lib_config,
            motion_lib_config_fn=module.motion_lib_config, env_config_fn=module.env_config,
            configure_robot_and_simulator_fn=module.configure_robot_and_simulator, agent_config_fn=module.agent_config,
        )
        fabric = _fabric()
        env = _build_env(cfgs, fabric.device)
        return cls(env, None, cfgs)

    # ------------------------------------------------------------------
    # rollout
    # ------------------------------------------------------------------
    def _policy_action(self, obs_td):
        with torch.no_grad():
            outs = self.agent.model(obs_td)
        return outs.get("mean_action", outs.get("action"))

    def _obs_td(self, obs):
        return self.agent.obs_dict_to_tensordict(self.agent.add_agent_info_to_obs(obs))

    def rollout(self, num_steps: int, seed: int, extra: Optional[Callable[[object], Dict[str, torch.Tensor]]] = None
                ) -> Tuple[Dict[str, torch.Tensor], int]:
        """deterministic(mean_action) rollout을 돌며 상태/토크 시계열 기록.

        Args:
            extra: env를 받아 step마다 추가로 기록할 {키: 텐서}를 돌려주는 콜백.

        Returns:
            (logs, num_resets) — logs[key]: (num_steps, num_envs, ...) CPU 텐서.
            키: dof_pos, dof_vel, theta_g, theta_d, tau_agent, tau_assist (+ extra 키).
        """
        env, agent = self.env, self.agent
        # lazy init(게인 랜덤화 등)이 첫 step에서 RNG를 소비하면 시딩 이후의 난수 스트림이 rollout 간에
        # 어긋난다 — 시딩 전에 미리 초기화해서 같은 seed의 rollout들이 완전히 동일한 궤적을 받도록 한다.
        if hasattr(env, "_lazy_init_assist"):
            env._lazy_init_assist()
        torch.manual_seed(seed)
        obs, _ = env.reset()
        obs_td = self._obs_td(obs)
        logs: Dict[str, list] = {k: [] for k in ROLLOUT_KEYS}
        num_resets = 0
        for _ in range(num_steps):
            actions = self._policy_action(obs_td)
            obs, _rew, done, _term, _extras = env.step(actions)
            ctx = env._current_context
            logs["dof_pos"].append(ctx.current.dof_pos.detach().cpu().clone())
            logs["dof_vel"].append(ctx.current.dof_vel.detach().cpu().clone())
            logs["theta_g"].append(ctx.assist.theta_g.detach().cpu().clone())
            logs["theta_d"].append(ctx.assist.theta_d.detach().cpu().clone())
            logs["tau_agent"].append(ctx.assist.tau_agent.detach().cpu().clone())
            logs["tau_assist"].append(ctx.assist.tau_assist.detach().cpu().clone())
            if extra is not None:
                for k, v in extra(env).items():
                    logs.setdefault(k, []).append(v.detach().cpu().clone())
            done_ids = done.nonzero(as_tuple=False).flatten()
            if len(done_ids) > 0:
                num_resets += len(done_ids)
                env.reset(done_ids)
                obs = env.get_obs()
            obs_td = self._obs_td(obs)
        return {k: torch.stack(v) for k, v in logs.items()}, num_resets

    @staticmethod
    def summarize_rollout(logs: Dict[str, torch.Tensor], dt: float) -> Dict[str, float]:
        power = (logs["tau_agent"] * logs["dof_vel"]).abs().sum(-1)  # (T, N)
        err = (logs["dof_pos"] - logs["theta_g"]).abs().mean(-1)  # (T, N)
        return {
            "human_power_mean": power.mean().item(),
            "human_work_per_episode": (power.mean(0) * dt * power.shape[0]).mean().item(),
            "tracking_err_mean": err.mean().item(),
            "assist_power_mean": (logs["tau_assist"] * logs["dof_vel"]).abs().sum(-1).mean().item(),
            "assist_torque_rms": logs["tau_assist"].square().mean().sqrt().item(),
        }

    def compare(self, num_steps: int, seed: int, out: str) -> Dict[str, Dict[str, float]]:
        """assist 정책 vs 어시스트-제로 베이스라인 paired rollout (mean_action은 RNG를 소비하지 않음)."""
        dt = self.env.dt
        print(f"[1/2] assist policy rollout ({num_steps} steps x {self.env.num_envs} envs)")
        logs_policy, resets_p = self.rollout(num_steps, seed)
        print("[2/2] zero-assist baseline rollout (same seed -> same trajectories)")
        self.env.config.assist_torque_limit = 0.0
        logs_base, resets_b = self.rollout(num_steps, seed)
        sp, sb = self.summarize_rollout(logs_policy, dt), self.summarize_rollout(logs_base, dt)
        reduction = 100.0 * (1.0 - sp["human_power_mean"] / sb["human_power_mean"])
        print("\n===== paired comparison (identical trajectories) =====")
        print(f"{'metric':28s} {'policy':>10s} {'baseline':>10s}")
        for k in sp:
            print(f"{k:28s} {sp[k]:10.3f} {sb[k]:10.3f}")
        print(f"\nhuman power reduction: {reduction:.1f}%")
        print(f"resets during rollout: policy={resets_p}, baseline={resets_b}")
        np.savez(
            out,
            dt=dt,
            **{f"policy_{k}": v.numpy() for k, v in logs_policy.items()},
            **{f"base_{k}": v.numpy() for k, v in logs_base.items()},
        )
        print(f"saved: {out}")
        return {"policy": sp, "baseline": sb}

    # ------------------------------------------------------------------
    # viewer 재생
    # ------------------------------------------------------------------
    def _recolor_shapes(self):
        """geom 이름 기반으로 렌더 색 적용 + 오버레이 선 두께 상향."""
        import warp as wp

        sim = self.env.simulator
        model = sim.model
        if getattr(model, "shape_color", None) is None or not getattr(model, "shape_label", None):
            return
        colors = wp.to_torch(model.shape_color)  # (num_shapes, 3), 모델 메모리 공유
        for i, label in enumerate(model.shape_label):
            if label is None:
                continue
            name = str(label).split("/")[-1]
            for key, c in self.BODY_PALETTE.items():
                if key in name:
                    colors[i, 0], colors[i, 1], colors[i, 2] = c
                    break
        renderer = getattr(sim.viewer, "renderer", None)
        if renderer is not None and hasattr(renderer, "line_width"):
            renderer.line_width = 3.0
        if renderer is not None and hasattr(renderer, "arrow_scale"):
            renderer.arrow_scale = 1.5

    def _setup_camera_controls(self):
        """추적 카메라 토글('f') + 줌 키('z'/'x').

        repo의 Newton render()는 매 프레임 _update_camera()로 카메라를 재설정해 마우스 시점 조작이
        덮어써진다. 추적을 끄면 ViewerGL 기본 마우스 내비게이션이 살아난다.
        """
        sim = self.env.simulator
        sim.set_orbit_camera(azimuth=270.0, distance=3.0, height=1.0)  # 측면 뷰 — 시상면 스윙이 가장 잘 보임
        sim._assist_follow_cam = True
        orig_update = sim._update_camera

        def patched_update():
            if sim._assist_follow_cam:
                orig_update()

        sim._update_camera = patched_update

        def toggle_follow():
            sim._assist_follow_cam = not sim._assist_follow_cam
            if sim._assist_follow_cam:
                print("[Camera] 추적 ON")
            else:
                print("[Camera] 추적 OFF — 마우스 자유 시점 (좌드래그 회전 / 우드래그 팬 / 스크롤 줌)")

        def zoom(delta):
            distance = max(0.8, min(12.0, sim.orbit_camera["distance"] + delta))
            sim.set_orbit_camera(distance=distance)
            print(f"[Camera] distance = {distance:.1f} m")

        sim._custom_key_handlers["f"] = toggle_follow
        sim._custom_key_handlers["z"] = lambda: zoom(-0.5)
        sim._custom_key_handlers["x"] = lambda: zoom(+0.5)

    def _setup_torque_overlay(self):
        """viewer render hook: 목표 고스트 다리 + 토크 화살표."""
        import warp as wp

        env = self.env
        sim = env.simulator
        n, d, device = env.num_envs, env.robot_config.number_of_actions, env.device
        # 다리별 바깥쪽 y 오프셋 (몸과 겹치지 않게): [right, left, ...]
        lateral = torch.tensor([(-0.11 if i % 2 == 0 else 0.11) for i in range(d)], device=device)
        # persistent 버퍼 (매 프레임 내용만 갱신)
        buf = {name: torch.zeros(n * d, 3, device=device, dtype=torch.float32)
               for name in ("tgt_start", "tgt_end", "hum_start", "hum_end", "ast_start", "ast_end")}
        wp_buf = {k: wp.from_torch(v, dtype=wp.vec3) for k, v in buf.items()}
        comp = env.assist_component
        leg_len = self.LEG_LEN

        def leg_dir(th):  # 다리 방향 (아래 기준)
            return torch.stack([-torch.sin(th), torch.zeros_like(th), -torch.cos(th)], dim=-1)

        def tangent(th):  # +토크가 다리를 미는 접선 방향
            return torch.stack([-torch.cos(th), torch.zeros_like(th), torch.sin(th)], dim=-1)

        def hook():
            state = sim.get_robot_state()
            theta = state.dof_pos  # (n, d)
            pivots = state.rigid_body_pos[:, 1: 1 + d, :].clone()  # (n, d, 3)
            pivots[..., 1] += lateral.unsqueeze(0)
            theta_g = comp.get_pd_targets()
            tau_h, tau_a = comp._tau_agent, comp._tau_assist
            # 1) 목표 고스트 다리: pivot -> pivot + L*dir(θ_g)
            buf["tgt_start"][:] = pivots.reshape(-1, 3)
            buf["tgt_end"][:] = (pivots + leg_len * leg_dir(theta_g)).reshape(-1, 3)
            # 2) 사람 토크 화살표 (다리 중간 30% 지점)
            mag_h = (tau_h * self.ARROW_SCALE_HUMAN).clamp(-self.ARROW_MAX_LEN, self.ARROW_MAX_LEN)
            anchor_h = pivots + 0.30 * leg_len * leg_dir(theta)
            buf["hum_start"][:] = anchor_h.reshape(-1, 3)
            buf["hum_end"][:] = (anchor_h + tangent(theta) * mag_h.unsqueeze(-1)).reshape(-1, 3)
            # 3) assist 토크 화살표 (무릎 지점)
            mag_a = (tau_a * self.ARROW_SCALE_ASSIST).clamp(-self.ARROW_MAX_LEN, self.ARROW_MAX_LEN)
            anchor_a = pivots + leg_len * leg_dir(theta)
            buf["ast_start"][:] = anchor_a.reshape(-1, 3)
            buf["ast_end"][:] = (anchor_a + tangent(theta) * mag_a.unsqueeze(-1)).reshape(-1, 3)
            v = sim.viewer
            v.log_lines("assist/target_legs", wp_buf["tgt_start"], wp_buf["tgt_end"], self.COLOR_TARGET)
            v.log_arrows("assist/tau_human", wp_buf["hum_start"], wp_buf["hum_end"], self.COLOR_HUMAN)
            v.log_arrows("assist/tau_assist", wp_buf["ast_start"], wp_buf["ast_end"], self.COLOR_ASSIST)

        sim._render_hook = hook

    def play(self, zero_assist: bool = False) -> None:
        """viewer 창으로 재생 (from_checkpoint(..., headless=False)로 만든 런타임에서)."""
        env = self.env
        self._setup_camera_controls()
        self._setup_torque_overlay()
        self._recolor_shapes()
        obs, _ = env.reset()
        obs_td = self._obs_td(obs)
        mode = "ZERO-ASSIST (사람 단독)" if zero_assist else "ASSIST POLICY"
        print(f"\n=== playing: {mode} ===")
        print("오버레이: 녹색선 = θ_g 고스트 다리 · 주황 화살표 = 사람 τ · 녹색 화살표 = assist τ")
        print("키: f 자유시점 토글 · [ ] 궤도회전 · z/x 줌 · b/n 후면/정면 · r 리셋 · q 종료\n")
        log_every = max(1, int(round(1.0 / env.dt)))
        step = 0
        try:
            while True:
                actions = self._policy_action(obs_td)
                obs, _rew, done, _term, _extras = env.step(actions)
                ctx = env._current_context
                if step % log_every == 0:
                    err = (ctx.current.dof_pos - ctx.assist.theta_g).abs().mean().item()
                    power = (ctx.assist.tau_agent * ctx.current.dof_vel).abs().sum(-1).mean().item()
                    tau_a = ctx.assist.tau_assist.abs().mean().item()
                    print(f"t={step * env.dt:7.1f}s  |err|={err:6.3f} rad  "
                          f"human power={power:7.2f} W  |tau_assist|={tau_a:6.2f} N·m")
                done_ids = done.nonzero(as_tuple=False).flatten()
                if len(done_ids) > 0:
                    env.reset(done_ids)
                    obs = env.get_obs()
                obs_td = self._obs_td(obs)
                step += 1
        except KeyboardInterrupt:
            print("\nstopped.")

    # ------------------------------------------------------------------
    # 학습 전 환경 점검
    # ------------------------------------------------------------------
    def check_env(self, steps: int, seed: int, out: str, diagnostics_json: Optional[str] = None) -> None:
        """무보조로 돌려 착용자 토크 τ_agent의 모양을 스켈레톤 보행 착용자와 같은 잣대로 재고,
        관성·게인 랜덤화가 실제로 적용됐는지 Newton 배열에서 직접 읽는다 (from_experiment 런타임용)."""
        env, cfgs = self.env, self.configs
        torch.manual_seed(seed)
        env.reset()
        comp = env.assist_component
        n, d = env.num_envs, comp.n_dofs
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
        zeta = kd / (2 * torch.sqrt(ke * I_eff))
        print(f"  ω_n [rad/s]: min {wn.min():.1f}  max {wn.max():.1f}   ζ: min {zeta.min():.2f}  max {zeta.max():.2f}")

        L = {k: [] for k in ("g", "q", "w", "tau", "ext", "ev", "kps")}
        resets = term_err = term_vel = 0
        t0 = time.perf_counter()
        for _ in range(steps):
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
        print(f"\n=== 무보조 롤아웃 {steps} step × {n} env ({steps*dt:.0f} s), {wall/steps*1e3:.2f} ms/step (게인 갱신 포함) ===")
        print(f"  리셋 {resets}회 (10 s 만기 ≈ {n * int(steps * dt // 10)}회) — 조기 종료 원인: 추종 오차>1 rad {term_err}회, 각속도>25 rad/s {term_vel}회")
        print(f"  τ_agent: RMS {rms(tau):.1f} N·m, 범위 [{tau.min():.0f}, {tau.max():.0f}], 피크/RMS {np.abs(tau).max()/rms(tau):.2f}   ← 스켈레톤 보행: RMS 78, [−332, +176], 4.28")
        print(f"  τ_agent p99 |τ| {np.percentile(np.abs(tau), 99):.0f} N·m;  |Δτ|/step RMS {rms(np.diff(tau, axis=0))/dt:.0f} N·m/s")
        print(f"  외부 하중 τ_ext: RMS {rms(ext):.1f}, 범위 [{ext.min():.0f}, {ext.max():.0f}] N·m;  입각 사건 중인 시간 {100*A['ev'].mean():.0f} %;  Kp 배율 최대 {A['kps'].max():.1f}")
        print(f"  |θ̈| RMS {rms(acc):.1f} rad/s², 최대 {np.abs(acc).max():.0f}   ← 스켈레톤: 48 / 408 (mocap 19 / 72)")
        print(f"  |θ_g − θ| 평균 {np.degrees(np.abs(g - q).mean()):.2f}°, p95 {np.degrees(np.percentile(np.abs(g - q), 95)):.1f}°  (v3 진자 2.9°)")
        print(f"  |τ·ω| 평균 {np.abs(tau * w).sum(-1).mean():.1f} W (v3 12 W),  τ² 평균 {(tau ** 2).sum(-1).mean():.0f} (N·m)²  → 보상 weight 크기 판단용")

        if diagnostics_json:
            # 요청한 config가 아니라 실제 적용된 시뮬레이터 게인을 읽어 기록한다.
            floor = float(getattr(cfgs["env"], "human_stiffness_floor", 0.0))
            heavy = I_eff >= 3.0
            soft_heavy = heavy & (ke >= 150) & (ke <= 250) & (kd >= 15) & (kd <= 25)
            diag = {"seed": seed, "num_envs": n, "steps": steps,
                    "omega_n_range": cfgs["env"].omega_n_range, "stiffness_floor": floor,
                    "applied_kp_min": float(ke.min()), "applied_kp_max": float(ke.max()),
                    "applied_wn_min": float(wn.min()), "applied_wn_max": float(wn.max()),
                    "applied_zeta_min": float(zeta.min()), "applied_zeta_max": float(zeta.max()),
                    "heavy_hip_count": int(heavy.sum()), "soft_heavy_hip_count": int(soft_heavy.sum()),
                    "resets": resets, "tracking_failures": term_err, "velocity_failures": term_vel,
                    "tracking_error_mean_deg": float(np.degrees(np.abs(g - q).mean())),
                    "wearer_rms_Nm": rms(tau)}
            if floor and float(ke.min()) < floor - 1e-4:
                raise RuntimeError("Stiffness floor did not reach simulator gains")
            target = Path(diagnostics_json)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(diag, indent=2, allow_nan=False) + "\n")

        # 그림: env 0의 6 s
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"; matplotlib.rcParams["axes.unicode_minus"] = False
        INK, TEAL, ORANGE, GREY = "#141a21", "#0e7a74", "#c2410c", "#9aa8b6"
        e = 0; s = slice(200, 800); tt = np.arange(steps)[s] * dt
        fig, ax = plt.subplots(3, 1, figsize=(11.5, 7.5), sharex=True)
        ax[0].plot(tt, np.degrees(g[s, e, 0]), color=GREY, lw=2, label="의도 θ_g"); ax[0].plot(tt, np.degrees(q[s, e, 0]), color=INK, lw=1.3, label="실제 각도")
        ax[1].plot(tt, ext[s, e, 0], color=TEAL, lw=1.6, label="외부 하중 τ_ext (환경)"); ax[1].plot(tt, 40 * A["kps"][s, e, 0], color=GREY, lw=1, ls="--", label="착용자 Kp 배율 ×40")
        ax[2].plot(tt, tau[s, e, 0], color=ORANGE, lw=1.6, label="착용자 토크 τ_agent")
        for a in ax: a.grid(alpha=.25); a.legend(fontsize=9, loc="upper right")
        ax[0].set_ylabel("[deg]"); ax[1].set_ylabel("[N·m]"); ax[2].set_ylabel("[N·m]"); ax[2].set_xlabel("t [s]")
        fig.suptitle(f"진자 env 무보조 — env 0 (hip 관성 {I_eff[0,0]:.2f} kg·m², Kp {ke[0,0]:.0f}, ω_n {wn[0,0]:.1f} rad/s)", fontsize=12, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out, dpi=135); print("saved", out)


def build_env_agent(checkpoint, num_envs, headless=True, overrides=None, config_transform=None):
    """(env, agent) — `PendulumRuntime.from_checkpoint`의 함수형 별칭 (verify_controller.py, evaluate.py)."""
    rt = PendulumRuntime.from_checkpoint(checkpoint, num_envs, headless, overrides, config_transform)
    return rt.env, rt.agent


# ---------------------------------------------------------------------------
# A/B/C paired 평가
# ---------------------------------------------------------------------------
class AbcEvaluation:
    """A(assist off) / B(에뮬 chunk) / C(실 ManiFlow chunk) — 같은 mocap 클립·같은 사람 게인."""

    CONDITIONS = {"A": "assist off (사람 단독)", "B": "에뮬 chunk", "C": "실 ManiFlow chunk"}

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.out_dir = Path(args.out)

    @staticmethod
    def condition_overrides(cond: str, args, channel_offset):
        a = "env.control_components.assist."
        ov = {
            a + "intent_source": "motion_data",
            a + "motion_zarr_path": args.zarr,
            a + "motion_split": args.split,
            a + "motion_seed": args.motion_seed,
            a + "chunk_source": "maniflow" if cond == "C" else "emulated",
        }
        if args.metadata:
            ov[a + "motion_metadata_parquet"] = args.metadata
        if args.type_filter:
            ov[a + "motion_type_filter"] = args.type_filter
        if channel_offset is not None:
            ov[a + "motion_channel_offset"] = tuple(channel_offset)
        if args.refresh_steps:
            ov[a + "chunk_refresh_steps"] = args.refresh_steps
        if cond == "A":
            ov["env.assist_torque_limit"] = 0.0
        return ov

    def run_condition(self, cond: str, channel_offset) -> None:
        """한 조건을 rollout하고 시계열을 npz로 저장 (별도 프로세스에서 호출)."""
        args = self.args
        rt = PendulumRuntime.from_checkpoint(args.checkpoint, args.num_envs, headless=True,
                                             overrides=self.condition_overrides(cond, args, channel_offset))
        env = rt.env
        comp = env.control_manager.components["assist"]
        cfg = comp.config
        print(f"[{cond}] control={type(comp).__name__} intent={type(comp._intent).__name__} "
              f"chunk={type(comp._chunk_src).__name__ if comp._chunk_src else 'emulated'} "
              f"hlp_dt={cfg.hlp_dt} knots={cfg.chunk_knots} refresh={cfg.chunk_refresh_steps} "
              f"assist_limit={env.config.assist_torque_limit}")
        knot_offsets = torch.arange(cfg.chunk_knots, device=env.device, dtype=torch.float32) * cfg.hlp_dt

        def extra(env):
            # chunk knot과 그 시각의 "실제" 의도 — 온라인 HLP 오차 계산용
            t_now = comp._episode_time()
            knot_t = comp._chunk_t0.unsqueeze(1) + knot_offsets.unsqueeze(0)
            tr = comp.in_transition()
            return {"chunk": comp._chunk, "chunk_t0": comp._chunk_t0, "chunk_gt": comp._target_at(knot_t),
                    "in_transition": torch.zeros_like(t_now, dtype=torch.bool) if tr is None else tr,
                    "episode_time": t_now}

        logs, num_resets = rt.rollout(args.num_steps, args.seed, extra=extra)
        out = {k: v.numpy() for k, v in logs.items()}
        out["dt"] = np.float32(env.dt)
        out["hlp_dt"] = np.float32(cfg.hlp_dt)
        out["num_resets"] = np.int64(num_resets)
        path = self.out_dir / f"{cond}.npz"
        np.savez(path, **out)
        print(f"[{cond}] resets={num_resets} → {path}")

    @staticmethod
    def summarize(d, mask):
        """조건 요약. mask (T, N) = 유효 스텝(전환·초기 블렌드 제외)."""
        dt = float(d["dt"])
        m = mask[..., None]
        power = np.abs(d["tau_agent"] * d["dof_vel"]).sum(-1)
        err = np.abs(d["dof_pos"] - d["theta_g"]).mean(-1)
        sel = mask
        return {
            "human_power_mean": float(power[sel].mean()),
            "human_work_per_episode": float(power[sel].mean() * dt * power.shape[0]),
            "tracking_err_mean_deg": float(math.degrees(err[sel].mean())),
            "tracking_err_p95_deg": float(math.degrees(np.percentile(err[sel], 95))),
            "assist_torque_rms": float(np.sqrt((d["tau_assist"] ** 2 * m)[sel].mean())),
            "assist_power_mean": float(np.abs(d["tau_assist"] * d["dof_vel"]).sum(-1)[sel].mean()),
            "num_resets": int(d["num_resets"]),
            "valid_frac": float(sel.mean()),
        }

    @staticmethod
    def hlp_error_per_lead(d, mask):
        """chunk knot vs 그 시각의 실제 의도 — lead별 MAE [deg]."""
        chunk, gt = d["chunk"], d["chunk_gt"]  # (T, N, K, 2)
        hlp_dt = float(d["hlp_dt"])
        return [{"lead_ms": round(k * hlp_dt * 1000),
                 "mae_deg": float(math.degrees(np.abs(chunk[:, :, k] - gt[:, :, k]).mean(-1)[mask].mean()))}
                for k in range(chunk.shape[2])]

    @staticmethod
    def valid_mask(d, warmup_s: float):
        """초기 블렌드·클립 전환 구간과 리셋 직후를 제외한 유효 스텝."""
        return (d["episode_time"] >= warmup_s) & (~d["in_transition"].astype(bool))

    def report(self, conds) -> None:
        args, out_dir = self.args, self.out_dir
        data = {c: dict(np.load(out_dir / f"{c}.npz")) for c in conds}
        masks = {c: self.valid_mask(d, args.warmup) for c, d in data.items()}
        summaries = {c: self.summarize(data[c], masks[c]) for c in conds}
        pairing = {}
        if len(conds) > 1:  # 페어링 확인: θ_g가 조건 간 동일해야 한다
            base = data[conds[0]]["theta_g"]
            for c in conds[1:]:
                pairing[c] = float(np.abs(data[c]["theta_g"] - base).max())
        lines = ["# A/B/C paired 평가", "", f"- checkpoint: `{args.checkpoint}`", f"- zarr: `{args.zarr}` (split={args.split})",
                 f"- envs={args.num_envs} steps={args.num_steps} seed={args.seed} motion_seed={args.motion_seed} warmup={args.warmup}s"]
        if pairing:
            worst = max(pairing.values())
            lines.append(f"- θ_g 페어링: 최대 차 {worst:.2e} rad ({'동일' if worst < 1e-6 else '⚠️ 어긋남 — 리셋 횟수 확인'})")
        lines += ["", "## 지표", "", "| 지표 | " + " | ".join(f"{c} ({self.CONDITIONS[c]})" for c in conds) + " |", "|---|" + "---|" * len(conds)]
        for key in summaries[conds[0]]:
            lines.append(f"| {key} | " + " | ".join(f"{summaries[c][key]:.4g}" for c in conds) + " |")
        if "A" in conds:
            lines += ["", "## 사람 파워 절감 (A 대비)", ""]
            pa = summaries["A"]["human_power_mean"]
            for c in conds:
                if c != "A":
                    lines.append(f"- **{c}** ({self.CONDITIONS[c]}): {100.0 * (1.0 - summaries[c]['human_power_mean'] / pa):.1f} %")
        lines += ["", "## 온라인 HLP 오차 (chunk knot vs 실제 의도)", ""]
        leads = {c: self.hlp_error_per_lead(data[c], masks[c]) for c in conds if c != "A"}
        if leads:
            cols = [c for c in conds if c != "A"]
            lines.append("| lead | " + " | ".join(cols) + " |")
            lines.append("|---|" + "---|" * len(cols))
            for i in range(len(leads[cols[0]])):
                lines.append(f"| {leads[cols[0]][i]['lead_ms']} ms | " + " | ".join(f"{leads[c][i]['mae_deg']:.2f}°" for c in cols) + " |")
        text = "\n".join(lines) + "\n"
        (out_dir / "report.md").write_text(text)
        (out_dir / "report.json").write_text(json.dumps({
            "checkpoint": args.checkpoint, "zarr": args.zarr, "split": args.split, "summaries": summaries,
            "pairing_max_theta_g_diff": pairing, "hlp_error_per_lead": leads}, indent=2))
        print("\n" + text)
        print(f"saved: {out_dir / 'report.md'}")

    def run(self) -> int:
        args, out_dir = self.args, self.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        channel_offset = None
        if args.align_ref_zarr:
            from protomotions.envs.control.motion_intent import channel_mean_offset

            cache = out_dir / "channel_offset.json"
            if cache.exists():
                channel_offset = json.loads(cache.read_text())
            else:
                channel_offset = list(channel_mean_offset(args.align_ref_zarr, args.zarr))
                cache.write_text(json.dumps(channel_offset))
            print("channel offset (deg): " + ", ".join(f"{math.degrees(v):+.2f}" for v in channel_offset))
        if args._condition:
            self.run_condition(args._condition, channel_offset)
            return 0
        conds = [c.strip().upper() for c in args.conditions.split(",") if c.strip()]
        for c in conds:
            assert c in self.CONDITIONS, f"unknown condition {c}"
            print(f"\n===== 조건 {c}: {self.CONDITIONS[c]} =====")
            cmd = [sys.executable, os.path.abspath(__file__), "abc", "--_condition", c]
            for k, v in vars(args).items():
                if k in ("_condition", "command") or v is None or v is False:
                    continue
                cmd += [f"--{k.replace('_', '-')}", str(v)]
            rc = subprocess.call(cmd, env=os.environ.copy())
            if rc != 0:
                raise SystemExit(f"조건 {c} 실패 (rc={rc})")
        self.report(conds)
        return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("play", help="Newton viewer 재생 (토크 오버레이)")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--num-envs", type=int, default=1)
    s.add_argument("--zero-assist", action="store_true", help="assist 토크 0 (사람 단독)")

    s = sub.add_parser("compare", help="assist 정책 vs 어시스트-제로 paired rollout")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--num-envs", type=int, default=8)
    s.add_argument("--num-steps", type=int, default=1000)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", default="/tmp/assist_compare.npz")

    s = sub.add_parser("check-env", help="학습 전 환경 점검 (뒤에 '--'와 실험 파일 인자, 예: -- --recipe R0)")
    s.add_argument("--experiment", default=str(HERE / "mlp.py"))
    s.add_argument("--num-envs", type=int, default=64)
    s.add_argument("--steps", type=int, default=1500, help="10 ms step 수 (1500 = 15 s)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="docs/ai_sbc/pendulum_v4_env_check.png")
    s.add_argument("--diagnostics-json", default=None)

    s = sub.add_parser("abc", help="A/B/C paired 평가 (mocap 의도, 에뮬 vs 실 ManiFlow chunk)")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--zarr", required=True, help="의도 궤적 zarr (flexion_trunk)")
    s.add_argument("--split", default="val", choices=["all", "train", "val"])
    s.add_argument("--metadata", default=None, help="SEED metadata parquet")
    s.add_argument("--type-filter", default=None, help="예: walking,jogging")
    s.add_argument("--align-ref-zarr", default=None, help="이 zarr(=SEED) 규약으로 영점 정렬 (suit14 평가에 필요)")
    s.add_argument("--conditions", default="A,B,C")
    s.add_argument("--num-envs", type=int, default=32)
    s.add_argument("--num-steps", type=int, default=3000)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--motion-seed", type=int, default=0)
    s.add_argument("--refresh-steps", type=int, default=0, help="0=config 값 유지")
    s.add_argument("--warmup", type=float, default=1.0, help="에피소드 앞부분 제외 [s]")
    s.add_argument("--out", required=True)
    s.add_argument("--_condition", default=None, help=argparse.SUPPRESS)

    args, rest = p.parse_known_args(argv)
    if args.command != "check-env" and rest:
        p.error(f"unrecognized arguments: {' '.join(rest)}")

    if args.command == "play":
        overrides = {"env.assist_torque_limit": 0.0} if args.zero_assist else None
        rt = PendulumRuntime.from_checkpoint(args.checkpoint, args.num_envs, headless=False, overrides=overrides)
        rt.play(zero_assist=args.zero_assist)
    elif args.command == "compare":
        rt = PendulumRuntime.from_checkpoint(args.checkpoint, args.num_envs, headless=True)
        rt.compare(args.num_steps, args.seed, args.out)
    elif args.command == "check-env":
        exp_args = rest[1:] if rest[:1] == ["--"] else rest
        rt = PendulumRuntime.from_experiment(args.experiment, exp_args, args.num_envs, experiment_name="check_env")
        rt.check_env(args.steps, args.seed, args.out, args.diagnostics_json)
    else:
        return AbcEvaluation(args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
