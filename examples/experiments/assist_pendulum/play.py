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
"""assist_pendulum 정책을 Newton viewer로 재생 (토크 벡터 오버레이 포함).

오버레이:
  * 회색-녹색 라인 = 사람 의도 궤적 θ_g의 "고스트 다리" (추종 목표 방향)
  * 주황 화살표  = 사람 토크 τ_agent  (다리 중간, 길이 ∝ 크기)
  * 녹색 화살표  = assist 토크 τ_assist (무릎 위치, 길이 ∝ 크기)
  화살표 방향은 해당 토크가 다리를 미는 접선 방향.

카메라 조작:
  f      : 추적 카메라 on/off — off 상태에서 마우스로 자유 시점
           (좌드래그 회전, 우드래그 팬, 스크롤 줌)
  [ / ]  : 좌우 30° 궤도 회전 (추적 모드)
  b / n  : 후면/정면 뷰,  z / x : 줌 인/아웃 (추적 모드)
  r      : 리셋,  q : 종료

    # 학습된 assist 정책
    python examples/experiments/assist_pendulum/play.py \
        --checkpoint results/assist_pendulum_v2/last.ckpt

    # 어시스트 없는 베이스라인 (사람 단독)
    python examples/experiments/assist_pendulum/play.py \
        --checkpoint results/assist_pendulum_v2/last.ckpt --zero-assist
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

import_simulator_before_torch("newton")

import torch  # noqa: E402

from runtime_utils import build_env_agent  # noqa: E402

# 오버레이 스케일/색
_ARROW_SCALE_HUMAN = 0.010  # m per N·m
_ARROW_SCALE_ASSIST = 0.020
_ARROW_MAX_LEN = 0.9  # m
_LEG_LEN = 0.42
_COLOR_HUMAN = (1.0, 0.55, 0.10)
_COLOR_ASSIST = (0.15, 0.90, 0.35)
_COLOR_TARGET = (0.45, 0.75, 0.55)

# 몸체 색 (Newton MJCF 임포터가 primitive geom의 rgba를 무시하므로
# 빌드 후 model.shape_color를 geom 이름 기준으로 직접 덮어쓴다)
_BODY_PALETTE = {
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


def _recolor_shapes(sim):
    """geom 이름 기반으로 렌더 색 적용 + 오버레이 선 두께 상향."""
    import warp as wp

    model = sim.model
    if getattr(model, "shape_color", None) is None or not getattr(
        model, "shape_label", None
    ):
        return
    colors = wp.to_torch(model.shape_color)  # (num_shapes, 3), 모델 메모리 공유
    for i, label in enumerate(model.shape_label):
        if label is None:
            continue
        name = str(label).split("/")[-1]
        for key, c in _BODY_PALETTE.items():
            if key in name:
                colors[i, 0], colors[i, 1], colors[i, 2] = c
                break

    renderer = getattr(sim.viewer, "renderer", None)
    if renderer is not None and hasattr(renderer, "line_width"):
        renderer.line_width = 3.0
    if renderer is not None and hasattr(renderer, "arrow_scale"):
        renderer.arrow_scale = 1.5


def _setup_camera_controls(sim):
    """추적 카메라 토글('f') + 줌 키('z'/'x') 설치.

    repo의 Newton render()는 매 프레임 _update_camera()로 카메라를 재설정해
    마우스 시점 조작이 덮어써진다. 추적을 끄면 ViewerGL 기본 마우스
    내비게이션이 살아난다.
    """
    sim._camera_distance = 3.0
    sim._camera_height = 1.0
    sim._camera_azimuth = 270.0  # 측면 뷰 — 시상면 스윙이 가장 잘 보임

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
        sim._camera_distance = max(0.8, min(12.0, sim._camera_distance + delta))
        print(f"[Camera] distance = {sim._camera_distance:.1f} m")

    sim._custom_key_handlers["f"] = toggle_follow
    sim._custom_key_handlers["z"] = lambda: zoom(-0.5)
    sim._custom_key_handlers["x"] = lambda: zoom(+0.5)


def _setup_torque_overlay(env):
    """viewer render hook: 목표 고스트 다리 + 토크 화살표."""
    import warp as wp

    sim = env.simulator
    n = env.num_envs
    d = env.robot_config.number_of_actions
    device = env.device

    # 다리별 바깥쪽 y 오프셋 (몸과 겹치지 않게): [right, left, ...]
    lateral = torch.tensor(
        [(-0.11 if i % 2 == 0 else 0.11) for i in range(d)], device=device
    )

    # persistent 버퍼 (매 프레임 내용만 갱신)
    buf = {
        name: torch.zeros(n * d, 3, device=device, dtype=torch.float32)
        for name in (
            "tgt_start", "tgt_end",
            "hum_start", "hum_end",
            "ast_start", "ast_end",
        )
    }
    wp_buf = {k: wp.from_torch(v, dtype=wp.vec3) for k, v in buf.items()}

    comp = env.assist_component

    def hook():
        state = sim.get_robot_state()
        dof_pos = state.dof_pos  # (n, d)
        pivots = state.rigid_body_pos[:, 1 : 1 + d, :].clone()  # (n, d, 3)
        pivots[..., 1] += lateral.unsqueeze(0)

        theta = dof_pos
        theta_g = comp.get_pd_targets()
        tau_h = comp._tau_agent
        tau_a = comp._tau_assist

        def leg_dir(th):  # 다리 방향 (아래 기준)
            return torch.stack(
                [-torch.sin(th), torch.zeros_like(th), -torch.cos(th)], dim=-1
            )

        def tangent(th):  # +토크가 다리를 미는 접선 방향
            return torch.stack(
                [-torch.cos(th), torch.zeros_like(th), torch.sin(th)], dim=-1
            )

        # 1) 목표 고스트 다리: pivot -> pivot + L*dir(θ_g)
        buf["tgt_start"][:] = pivots.reshape(-1, 3)
        buf["tgt_end"][:] = (pivots + _LEG_LEN * leg_dir(theta_g)).reshape(-1, 3)

        # 2) 사람 토크 화살표 (다리 중간 30% 지점)
        mag_h = (tau_h * _ARROW_SCALE_HUMAN).clamp(-_ARROW_MAX_LEN, _ARROW_MAX_LEN)
        anchor_h = pivots + 0.30 * _LEG_LEN * leg_dir(theta)
        buf["hum_start"][:] = anchor_h.reshape(-1, 3)
        buf["hum_end"][:] = (anchor_h + tangent(theta) * mag_h.unsqueeze(-1)).reshape(-1, 3)

        # 3) assist 토크 화살표 (무릎 지점)
        mag_a = (tau_a * _ARROW_SCALE_ASSIST).clamp(-_ARROW_MAX_LEN, _ARROW_MAX_LEN)
        anchor_a = pivots + _LEG_LEN * leg_dir(theta)
        buf["ast_start"][:] = anchor_a.reshape(-1, 3)
        buf["ast_end"][:] = (anchor_a + tangent(theta) * mag_a.unsqueeze(-1)).reshape(-1, 3)

        v = sim.viewer
        v.log_lines("assist/target_legs", wp_buf["tgt_start"], wp_buf["tgt_end"], _COLOR_TARGET)
        v.log_arrows("assist/tau_human", wp_buf["hum_start"], wp_buf["hum_end"], _COLOR_HUMAN)
        v.log_arrows("assist/tau_assist", wp_buf["ast_start"], wp_buf["ast_end"], _COLOR_ASSIST)

    sim._render_hook = hook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--zero-assist", action="store_true", help="assist 토크 0 (사람 단독)"
    )
    args = parser.parse_args()

    overrides = {"env.assist_torque_limit": 0.0} if args.zero_assist else None
    env, agent = build_env_agent(
        args.checkpoint, args.num_envs, headless=False, overrides=overrides
    )

    _setup_camera_controls(env.simulator)
    _setup_torque_overlay(env)
    _recolor_shapes(env.simulator)

    obs, _ = env.reset()
    obs_td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))

    mode = "ZERO-ASSIST (사람 단독)" if args.zero_assist else "ASSIST POLICY"
    print(f"\n=== playing: {mode} ===")
    print("오버레이: 녹색선 = θ_g 고스트 다리 · 주황 화살표 = 사람 τ · 녹색 화살표 = assist τ")
    print("키: f 자유시점 토글 · [ ] 궤도회전 · z/x 줌 · b/n 후면/정면 · r 리셋 · q 종료\n")

    log_every = max(1, int(round(1.0 / env.dt)))
    step = 0
    try:
        while True:
            with torch.no_grad():
                outs = agent.model(obs_td)
            actions = outs.get("mean_action", outs.get("action"))
            obs, _rew, done, _term, _extras = env.step(actions)

            ctx = env._current_context
            if step % log_every == 0:
                err = (ctx.current.dof_pos - ctx.assist.theta_g).abs().mean().item()
                power = (
                    (ctx.assist.tau_agent * ctx.current.dof_vel).abs().sum(-1).mean().item()
                )
                tau_a = ctx.assist.tau_assist.abs().mean().item()
                print(
                    f"t={step * env.dt:7.1f}s  |err|={err:6.3f} rad  "
                    f"human power={power:7.2f} W  |tau_assist|={tau_a:6.2f} N·m"
                )

            done_ids = done.nonzero(as_tuple=False).flatten()
            if len(done_ids) > 0:
                env.reset(done_ids)
                obs = env.get_obs()
            obs_td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))
            step += 1
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
