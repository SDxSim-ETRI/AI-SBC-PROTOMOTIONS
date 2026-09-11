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
"""Newton 물리 서브스텝 루프 안에서 외부 제어기를 돌리기 위한 시뮬레이터 확장.

왜 필요한가: 착용자 대역(mimic tracker)은 env 주기 50 ms(120 Hz/6)로 학습됐고
관측이 env dt에 묶여 있어 그 주기를 바꿀 수 없다. 반면 assist 제어기(LLP)는
10 ms 주기로 학습됐다. 그래서 env step 하나(decimation개 서브스텝) 안에서
``hook_every`` 서브스텝마다 파이썬 콜백을 불러 LLP를 돌리고 qfrc를 갱신한다.
물리 400 Hz / decimation 20 / hook_every 4 → env 20 Hz, LLP 100 Hz.

콜백 계약::

    def hook(sim: SubstepHookMixin, window: int) -> None

- ``window`` = 0, 1, … (env step 안의 몇 번째 LLP 창인지)
- 호출 시점: 해당 창의 첫 서브스텝 **직전** (state_0 = 최신 상태)
- 콜백 안에서 ``sim.robot_view``/``sim._get_simulator_dof_state()`` 등으로
  상태를 읽고, ``JointTorqueOverride.set_torques``(qfrc)나
  ``sim._apply_simulator_pd_targets``(PD 목표)로 제어를 쓴다
- ``sim.window_tau_pd_common`` = **직전 창**의 서브스텝 평균 qfrc_actuator
  (COMMON DOF ordering, built-in PD 몫). 첫 창(window=0)에서는 이전 env step의
  마지막 창 값이다. LLP 학습 env의 ``get_substep_mean_dof_forces()``와 같은
  정의를 10 ms 창에 적용한 것

CUDA graph는 끈다 — 서브스텝 사이에 파이썬/torch 코드가 끼므로 캡처할 수 없다.
소수 env(평가·시연)용이며 학습용이 아니다.

config 확장 필드 (SimulatorConfig 인스턴스에 setattr로 붙인다; 없으면 기본값):
    substep_hook_every: int = 1      — 몇 서브스텝마다 콜백을 부를지
    fixed_base_z_offset: float = 0.0 — fix_base_link 로봇의 루트를 z로 올릴 양 [m]
                                      (골반 고정 대역에서 발이 지면에 닿지 않게)

사용법: ``simulator_config._target_``을 아래 클래스 중 하나로 바꾸고, env 생성 후
``env.simulator.substep_hook = fn``.
"""

from typing import Callable, Optional

import torch
import warp as wp

from protomotions.robot_configs.base import ControlType
from protomotions.simulator.newton.fixed_base import FixedBaseNewtonSimulator
from protomotions.simulator.newton.simulator import NewtonSimulator, accumulate_qfrc_kernel

SubstepHook = Callable[["SubstepHookMixin", int], None]


class SubstepHookMixin:
    """`_simulate` 서브스텝 루프에 주기적 콜백 + 창 단위 PD 토크 평균을 추가한다."""

    substep_hook: Optional[SubstepHook] = None

    # ------------------------------------------------------------------
    # 빌드 시점 확장
    # ------------------------------------------------------------------
    @property
    def hook_every(self) -> int:
        return int(getattr(self.config, "substep_hook_every", 1))

    def _configure_builder_joint_properties(self) -> None:
        super()._configure_builder_joint_properties()
        z_off = float(getattr(self.config, "fixed_base_z_offset", 0.0))
        if z_off == 0.0 or not self.robot_config.asset.fix_base_link:
            return
        # 루트(부모 = world)의 고정 joint 프레임을 z로 올린다 — finalize 전 빌더 수정
        moved = False
        for j in range(self.robot.joint_count):
            if self.robot.joint_parent[j] != -1:
                continue
            xf = self.robot.joint_X_p[j]
            p = wp.transform_get_translation(xf)
            q = wp.transform_get_rotation(xf)
            self.robot.joint_X_p[j] = wp.transform(wp.vec3(p[0], p[1], p[2] + z_off), q)
            moved = True
            break
        assert moved, "fixed_base_z_offset: root joint (parent == -1) not found"

    def _create_simulation(self) -> None:
        # 콜백이 서브스텝 사이에 끼므로 CUDA graph를 쓸 수 없다. 부모는
        # `wp.is_mempool_enabled()`가 참이면 _simulate를 캡처하려 드는데, 다른
        # 프로세스가 GPU를 쓰는 중이면 캡처가 간헐적으로 프로세스째 죽였다
        # (2026-09-12, 학습과 동시 실행 시 2/14회). 캡처 자체를 건너뛴다.
        self._hook_active = False
        orig = wp.is_mempool_enabled
        wp.is_mempool_enabled = lambda *args, **kwargs: False
        try:
            super()._create_simulation()
        finally:
            wp.is_mempool_enabled = orig
        self.use_cuda_graph = False
        self.graph = None
        self._ensure_window_buffers()
        self._hook_active = True
        print("[INFO] SubstepHook: eager substep loop (CUDA graph off)")

    def _ensure_window_buffers(self) -> None:
        # 부모 _create_simulation의 CUDA graph 캡처가 _simulate를 먼저 부르므로 lazy 할당
        if hasattr(self, "_window_acc_start"):
            return
        n_dofs = self.robot_config.number_of_actions
        self.window_tau_pd_common = torch.zeros(self.num_envs, n_dofs, device=self.device)
        self._window_acc_start = torch.zeros(self.num_envs * self._pd_num_dofs, device=self.device)

    # ------------------------------------------------------------------
    # 서브스텝 루프
    # ------------------------------------------------------------------
    def _finish_window(self, num_substeps: int) -> None:
        """직전 창의 qfrc_actuator 평균 → window_tau_pd_common."""
        if self._qfrc_accum_wp is None or num_substeps <= 0:
            return
        acc = wp.to_torch(self._qfrc_accum_wp)
        mean_sim = ((acc - self._window_acc_start) / float(num_substeps)).view(
            self.num_envs, self._pd_num_dofs
        )
        self.window_tau_pd_common = mean_sim[:, self.data_conversion.dof_convert_to_common]
        self._window_acc_start = acc.clone()

    def _simulate(self) -> None:
        """NewtonSimulator._simulate + hook_every 서브스텝마다 콜백."""
        if not getattr(self, "_hook_active", False):
            return super()._simulate()
        accumulate_qfrc = self._qfrc_accum_wp is not None
        if accumulate_qfrc:
            self._qfrc_accum_wp.zero_()
            self._window_acc_start.zero_()
        every = self.hook_every
        hook = self.substep_hook
        window = 0
        since_hook = 0
        for i in range(self.decimation):
            if hook is not None and i % every == 0:
                if i > 0:
                    self._finish_window(since_hook)
                    since_hook = 0
                hook(self, window)
                window += 1
            self.state_0.clear_forces()
            if self.control_type == ControlType.PROPORTIONAL:
                self._apply_pd_kernel(self.state_0)
            elif self.control_type == ControlType.TORQUE:
                self._apply_torques_kernel_method()
            if self.viewer:
                self.viewer.apply_forces(self.state_0)
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
            )
            self.state_0, self.state_1 = self.state_1, self.state_0
            if accumulate_qfrc:
                wp.launch(
                    kernel=accumulate_qfrc_kernel,
                    dim=self.num_envs * self._pd_num_dofs,
                    inputs=[
                        self.state_0.mujoco.qfrc_actuator,
                        self._qfrc_accum_wp,
                        self._pd_qd_stride,
                        self._pd_qd_dof_start,
                        self._pd_num_dofs,
                    ],
                )
            since_hook += 1
        if hook is not None:
            self._finish_window(since_hook)

        if self.decimation % 2 != 0:
            self.state_0.assign(self.state_1)

    # ------------------------------------------------------------------
    # 콜백용 상태 읽기 (COMMON ordering)
    # ------------------------------------------------------------------
    def hook_dof_state(self):
        """현재 (dof_pos, dof_vel) — (num_envs, num_dofs) COMMON ordering."""
        st = self._get_simulator_dof_state()
        conv = self.data_conversion.dof_convert_to_common
        return st.dof_pos[:, conv], st.dof_vel[:, conv]

    def hook_body_rot(self, common_body_index: int) -> torch.Tensor:
        """특정 body의 월드 회전 (num_envs, 4) xyzw."""
        st = self._get_simulator_bodies_state()
        sim_idx = int(self.data_conversion.body_convert_to_common[common_body_index])
        return st.rigid_body_rot[:, sim_idx]


class SubstepHookNewtonSimulator(SubstepHookMixin, NewtonSimulator):
    """floating-base 로봇용 (mimic tracker 착용자)."""


class SubstepHookFixedBaseNewtonSimulator(SubstepHookMixin, FixedBaseNewtonSimulator):
    """fix_base_link 로봇용 (골반 고정 대역 착용자)."""
