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
"""SkeletonAssistEnv: 보행 착용자(mimic tracker, 20 Hz) + LLP assist torque(100 Hz) 학습 env.

진자 `AssistEnv`의 전신 보행 판. 착용자는 사람 PD가 아니라 **동결된 mimic
tracker 정책**이 만든다 — 관절 27개의 PD 목표를 50 ms마다 갱신하고 built-in
PD(로봇 config 게인)가 그 목표를 추종한다. LLP는 env 주기 10 ms로 hip flexion
2관절에 additive qfrc를 넣는다. 통합 시스템(`examples/experiments/assist_skeleton/
run_system.py --human tracker`)과 시간 구조가 같다.

    τ_hip_total = τ_agent (tracker 목표를 향한 built-in PD) + τ_assist (LLP, qfrc)

매 step:
    1. LLP action(2) → tanh × scale → τ_assist, assist_torque_limit clamp
    2. progress_buf % tracker_period == 0 인 env: tracker가 현재 관측
       (`max_coords_obs`, `mimic_target_poses` + 자체 이전 action)으로 새 PD 목표 산출
    3. JointTorqueOverride.set_torques(τ_assist) — hip 채널만, 게인 무변경
    4. simulator.step(PD 목표 27)   ← 착용자 몫
    5. τ_agent = hip 채널의 substep 평균 qfrc_actuator (built-in PD 몫만)
    6. post_physics: HLP 링버퍼/ chunk 갱신, 관측·보상·종료

policy action 차원은 로봇 DOF(27)가 아니라 보조 관절 수(2)다. BaseEnv가 로봇
DOF 크기로 만든 action 버퍼·state history의 action 축을 __init__ 뒤에 2로 다시
만든다 (`previous_actions` 관측 = LLP action 히스토리).

Newton + BUILT_IN_PD 전용. control_components에 `SkeletonAssistControl`
("assist" 키)이 있어야 한다.
"""

from dataclasses import dataclass

import math

import torch
from torch import Tensor

from protomotions.envs.base_env.config import EnvConfig
from protomotions.envs.base_env.env import BaseEnv
from protomotions.envs.obs.state_history_buffer import StateHistoryBuffer


@dataclass
class SkeletonAssistEnvConfig(EnvConfig):
    """SkeletonAssistEnv 설정.

    Attributes:
        assist_torque_limit: assist 토크 한계 [N·m] (qfrc 경로는 effort limit
            클램프를 받지 않으므로 여기서 직접 제한).
        assist_torque_lpf_hz: 주입 전에 거는 1차 저역통과의 차단주파수 [Hz].
            0이면 끔(기본). **벌점이 아니라 플랜트의 일부**로 다루는 쪽이다 —
            구동기가 못 내는 대역을 학습 때부터 막아 두면 정책이 그 한계 안에서
            위상을 맞추는 법을 배운다. 변화율 벌점을 키우는 방식(action_rate)은
            "아무것도 안 하는" 해로 몰리기 쉽다. 실행 시에는 같은 필터를
            `AssistControllerConfig.torque_lpf_hz`로 켠다.
        assist_component_name: control_components에서 `SkeletonAssistControl`의 키.
    """

    _target_: str = "protomotions.envs.base_env.skeleton_assist_env.SkeletonAssistEnv"

    assist_torque_limit: float = 25.0
    assist_torque_lpf_hz: float = 0.0
    assist_component_name: str = "assist"


class SkeletonAssistEnv(BaseEnv):
    """동결 tracker 착용자 위에 LLP assist torque를 additive 주입하는 env."""

    config: SkeletonAssistEnvConfig
    _torque_override = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        comp = self.assist_component
        n_act = comp.num_assist_dofs
        # policy action = 보조 관절 수. BaseEnv가 만든 로봇 DOF 크기 버퍼를 교체한다.
        self._current_raw_action = torch.zeros(self.num_envs, n_act, device=self.device)
        self._current_processed_action = torch.zeros(self.num_envs, n_act, device=self.device)
        # 구동기 대역 모사 (0이면 통과). α = dt / (dt + 1/(2πf))
        lpf_hz = float(getattr(self.config, "assist_torque_lpf_hz", 0.0) or 0.0)
        self._assist_lpf_alpha = (
            self.dt / (self.dt + 1.0 / (2.0 * math.pi * lpf_hz)) if lpf_hz > 0.0 else 1.0
        )
        self._tau_assist_prev = torch.zeros(self.num_envs, n_act, device=self.device)
        if self.state_history is not None:
            old = self.state_history
            self.state_history = StateHistoryBuffer(
                num_envs=self.num_envs,
                num_history_steps=old.num_history_steps,
                num_bodies=self.robot_config.kinematic_info.num_bodies,
                num_dofs=self.robot_config.kinematic_info.num_dofs,
                action_dim=n_act,
                num_contact_bodies=len(self.contact_body_ids),
                anchor_body_index=self.robot_config.anchor_body_index,
                device=self.device,
                store_noisy=old.store_noisy,
            )
            # 상태 버퍼는 이미 리셋 상태로 채워졌을 수 있으므로 복사 (action 축 제외)
            for name in (
                "rigid_body_pos", "rigid_body_rot", "rigid_body_vel", "rigid_body_ang_vel",
                "dof_pos", "dof_vel", "ground_heights", "body_contacts",
            ):
                getattr(self.state_history, name).copy_(getattr(old, name))
        # 관측 버퍼는 첫 계산 때 로봇 DOF 크기 action 히스토리로 잡혔다 → 새 크기로 재계산
        self._observation_buffer.pop("previous_actions", None)
        self._current_context = None
        all_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._process_observations(self.context, all_ids)

    def reset(self, env_ids=None, **kwargs):
        out = super().reset(env_ids, **kwargs)
        if self._assist_lpf_alpha < 1.0:
            if env_ids is None:
                self._tau_assist_prev.zero_()
            else:
                self._tau_assist_prev[env_ids] = 0.0
        return out

    @property
    def assist_component(self):
        return self.control_manager.components[self.config.assist_component_name]

    def _lazy_init_assist(self):
        """첫 step에서 hip 채널 torque override 생성 (완전히 초기화된 robot_view 필요)."""
        if self._torque_override is not None:
            return
        from protomotions.maniflow.hybrid_control import JointTorqueOverride

        comp = self.assist_component
        self._torque_override = JointTorqueOverride(
            self.simulator,
            env_ids=range(self.num_envs),
            common_dof_indices=comp.assist_dof_indices.tolist(),
        )
        # gain_scale=1.0: 착용자 PD는 그대로, additive 주입만 활성화
        self._torque_override.engage(gain_scale=1.0)

    def step(self, action: Tensor):
        self._lazy_init_assist()
        self.extras = {}
        self._current_context = None
        self._current_noisy_obs = None
        self._current_raw_action[:] = action

        # 1) policy action → assist torque [N·m]
        action_dict = self._process_action(action, self.context)
        tau_assist = torch.clamp(
            action_dict["processed_action"],
            -self.config.assist_torque_limit,
            self.config.assist_torque_limit,
        )
        if self._assist_lpf_alpha < 1.0:
            # 구동기 대역 모사: τ ← τ_prev + α(τ_cmd − τ_prev) (에피소드 리셋 시 0)
            tau_assist = self._tau_assist_prev + self._assist_lpf_alpha * (
                tau_assist - self._tau_assist_prev
            )
            self._tau_assist_prev = tau_assist
        self._current_processed_action[:] = tau_assist

        comp = self.assist_component
        # 2) 착용자: 주기가 된 env는 현재 관측으로 tracker 행동 갱신 (pre-physics)
        comp.tracker_act(self.get_obs())
        pd_targets = comp.get_pd_targets()

        # 3) assist 주입 (hip 채널)
        self._torque_override.set_torques(tau_assist)
        comp._tau_assist[:] = tau_assist

        # 4) 물리
        self.simulator.step(pd_targets, markers_callback=self.get_markers_state)

        # 5) 착용자 토크 계측: hip 채널의 substep 평균 qfrc_actuator (built-in PD 몫)
        tau_agent_all = self.simulator.get_substep_mean_dof_forces()
        comp._tau_agent[:] = tau_agent_all[:, comp.assist_dof_indices]

        self.post_physics_step()

        if self.simulator.user_requested_reset:
            self.user_reset()

        obs = self.get_obs()
        return obs, self.rew_buf, self.reset_buf, self.terminate_buf, self.extras
