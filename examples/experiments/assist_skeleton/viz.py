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
"""Newton 뷰어 오버레이 — 보조 없는 env를 반투명 고스트 라인으로, assist 토크를 화살표로.

`compare_maniflow_control_newton.py`(β 가산 실험)의 GhostSkeletonRenderer를 env
인덱스만 파라미터화해 옮겼다. Newton 마커 API는 no-op이므로 viewer의
`log_lines/log_arrows`를 `NewtonSimulator._render_hook`에서 직접 호출한다.
"""

import logging

import torch

log = logging.getLogger(__name__)


def _chain_render_hook(simulator, fn):
    prev = getattr(simulator, "_render_hook", None)

    def _hook():
        if prev is not None:
            prev()
        fn()

    simulator._render_hook = _hook


class GhostSkeletonRenderer:
    """`ghost_env`를 라인 스켈레톤으로 그리고 `visible_env`만 메시로 남긴다."""

    def __init__(self, simulator, ghost_env: int, visible_env: int,
                 color=(0.45, 0.80, 1.0), alpha: float = 0.5, line_width: float = 3.5):
        import warp as wp

        self.sim = simulator
        self.env_id = ghost_env
        self.viewer = simulator.viewer
        kin = simulator.robot_config.kinematic_info
        parents = kin.parent_indices
        dev = simulator.device
        self._children = torch.tensor([i for i, p in enumerate(parents) if p >= 0], dtype=torch.long, device=dev)
        self._parents = torch.tensor([p for p in parents if p >= 0], dtype=torch.long, device=dev)
        n = len(self._children)
        self._starts_t = torch.zeros(n, 3, device=dev)
        self._ends_t = torch.zeros(n, 3, device=dev)
        self._starts_wp = wp.from_torch(self._starts_t, dtype=wp.vec3)
        self._ends_wp = wp.from_torch(self._ends_t, dtype=wp.vec3)
        self._color = color
        self._patch_line_alpha(alpha)
        try:
            self.viewer.renderer.line_width = line_width
        except AttributeError:
            pass
        self.viewer.set_visible_worlds([visible_env])
        self.viewer.set_world_offsets((0.0, 0.0, 0.0))
        _chain_render_hook(simulator, self._draw)

    def _patch_line_alpha(self, alpha: float) -> None:
        try:
            shader = self.viewer.renderer._wireframe_shader
            orig = shader.update_frame

            def _update(*a, **kw):
                kw["alpha"] = alpha
                return orig(*a, **kw)

            shader.update_frame = _update
        except AttributeError as e:
            log.warning("고스트 라인 alpha 패치 실패(%s) — 불투명 라인", e)

    def _draw(self) -> None:
        bodies = self.sim._get_simulator_bodies_state()
        pos = bodies.rigid_body_pos[self.env_id][self.sim.data_conversion.body_convert_to_common]
        self._starts_t.copy_(pos[self._parents])
        self._ends_t.copy_(pos[self._children])
        self.viewer.log_lines("/ghost/agent_a", self._starts_wp, self._ends_wp, self._color)


class HipTorqueArrows:
    """B env 좌/우 고관절 옆에 assist 토크를 수평 화살표로 표시 (+x = 굴곡 방향, 길이 ∝ N·m).

    몸체 캡슐 안에 가려지지 않도록 고관절에서 좌우로 `lateral` 만큼 띄운 지점에서
    앞뒤로 그린다.
    """

    def __init__(self, simulator, hook, env_id: int, scale: float = 0.04, lateral: float = 0.25):
        import warp as wp

        self.sim = simulator
        self.hook = hook
        self.env_id = env_id
        self.scale = scale
        self.lateral = lateral
        names = simulator.robot_config.kinematic_info.body_names
        self._femurs = torch.tensor(
            [names.index("femur_r"), names.index("femur_l")], dtype=torch.long, device=simulator.device
        )
        self._starts_t = torch.zeros(2, 3, device=simulator.device)
        self._ends_t = torch.zeros(2, 3, device=simulator.device)
        self._starts_wp = wp.from_torch(self._starts_t, dtype=wp.vec3)
        self._ends_wp = wp.from_torch(self._ends_t, dtype=wp.vec3)
        _chain_render_hook(simulator, self._draw)

    def _draw(self) -> None:
        i = self.hook.i - 1
        if i < 0:
            return
        tau = self.hook.log["tau_assist"][i, self.env_id]  # (2,) 관절 규약 (+ = 굴곡)
        bodies = self.sim._get_simulator_bodies_state()
        pos = bodies.rigid_body_pos[self.env_id][self.sim.data_conversion.body_convert_to_common]
        p = pos[self._femurs].clone()
        p[0, 1] -= self.lateral  # 오른쪽 (−y)
        p[1, 1] += self.lateral  # 왼쪽 (+y)
        self._starts_t.copy_(p)
        ends = p.clone()
        ends[:, 0] += tau * self.scale  # 굴곡(+) = 앞(+x)
        self._ends_t.copy_(ends)
        self.sim.viewer.log_arrows("/assist/tau", self._starts_wp, self._ends_wp, (0.2, 0.8, 0.3), width=0.02)
