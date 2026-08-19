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
"""AssistEnv: 사람(BUILT_IN_PD) + RL assist torque(additive qfrc) 환경.

AI-SBC LLP 구조 (모델 설명 자료 10p 진자 모델):

    τ_total = τ_agent + τ_assist
    τ_agent  = 사람 근사 — 시뮬레이터 BUILT_IN_PD가 의도 궤적 θ_g를 추종
               (Kp(θ_g - θ) - Kd·θ̇, 매 물리 substep implicit 평가)
    τ_assist = RL policy 출력 — JointTorqueOverride로 qfrc_applied에
               additive 주입 (control step 동안 ZOH)

매 step 흐름:
    1. policy action -> tanh -> τ_assist [N·m], assist_torque_limit clamp
    2. JointTorqueOverride.set_torques(τ_assist)  (qfrc, 매 substep 가산)
    3. simulator.step(θ_g)                         (사람 PD 목표)
    4. get_substep_mean_dof_forces()로 τ_agent 계측 (qfrc_actuator는 사람
       PD 몫만 포함 — assist는 qfrc_applied 경로라 분리됨)
    5. 계측값을 assist 컴포넌트 버퍼에 기록 -> 보상/critic 관측에서 사용

Newton + BUILT_IN_PD 전용. control_components에 AssistTargetControl
("assist" 키)이 있어야 한다.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from protomotions.envs.base_env.config import EnvConfig
from protomotions.envs.base_env.env import BaseEnv


@dataclass
class AssistEnvConfig(EnvConfig):
    """AssistEnv 설정.

    Attributes:
        assist_torque_limit: assist 토크 한계 [N·m]. qfrc 경로는 시뮬레이터
            effort limit의 클램프를 받지 않으므로 여기서 직접 제한한다.
        human_gain_scale_min/max: env별 사람 PD 게인(ke/kd) 랜덤 배율 범위.
            "사용자마다 다른 근력/임피던스"의 도메인 랜덤화. 시뮬레이터 초기화
            시 1회 샘플링되어 에피소드 간 고정된다. min==max==1.0이면 비활성.
        assist_component_name: control_components에서 AssistTargetControl의 키.
    """

    _target_: str = "protomotions.envs.base_env.assist_env.AssistEnv"

    assist_torque_limit: float = 25.0
    human_gain_scale_min: float = 0.7
    human_gain_scale_max: float = 1.3
    assist_component_name: str = "assist"
    # 사람 모델에 속도 피드포워드 Kd·θ̇_g 추가:
    #   τ_agent = Kp(θ_g−θ) + Kd(θ̇_g−θ̇)
    # built-in PD(Kp(θ_g−θ) − Kd·θ̇) 위에 τ_ff = Kd·θ̇_g를 qfrc로 가산해 구현.
    # False면 순수 위치 PD (v1 동작).
    human_velocity_feedforward: bool = False


class AssistEnv(BaseEnv):
    """사람 PD(BUILT_IN_PD) 위에 RL assist torque를 additive 주입하는 환경."""

    config: AssistEnvConfig
    _torque_override = None

    @property
    def assist_component(self):
        return self.control_manager.components[self.config.assist_component_name]

    def _lazy_init_assist(self):
        """첫 step에서 torque override 생성 + 사람 게인 랜덤화.

        JointTorqueOverride는 완전히 초기화된 Newton robot_view가 필요하므로
        __init__이 아니라 첫 step에서 생성한다.
        """
        if self._torque_override is not None:
            return
        from protomotions.maniflow.hybrid_control import JointTorqueOverride

        n_dofs = self.robot_config.number_of_actions
        self._torque_override = JointTorqueOverride(
            self.simulator,
            env_ids=range(self.num_envs),
            common_dof_indices=range(n_dofs),
        )
        # gain_scale=1.0: 사람 PD는 그대로 두고 additive 주입만 활성화
        self._torque_override.engage(gain_scale=1.0)
        self._randomize_human_gains()

        # 속도 피드포워드용 env별 Kd (게인 랜덤화 배율 반영)
        kd_base = torch.tensor(
            [
                self.robot_config.control.control_info[name].damping
                for name in self.robot_config.kinematic_info.dof_names
            ],
            device=self.device,
            dtype=torch.float32,
        )
        self._human_kd = kd_base.unsqueeze(0) * self.assist_component._gain_scale

    def _randomize_human_gains(self):
        """env별 사람 PD 게인(ke/kd) 배율 랜덤화 (초기화 시 1회)."""
        cfg = self.config
        comp = self.assist_component
        if cfg.human_gain_scale_max <= cfg.human_gain_scale_min:
            comp._gain_scale[:] = 1.0
            return

        import warp as wp
        from newton.solvers import SolverNotifyFlags

        sim = self.simulator
        n_dofs = self.robot_config.number_of_actions
        scale = cfg.human_gain_scale_min + (
            cfg.human_gain_scale_max - cfg.human_gain_scale_min
        ) * torch.rand(self.num_envs, n_dofs, device=self.device)

        # COMMON dof i가 SIM-ordered 배열에서 차지하는 위치
        sim_idx = sim.data_conversion.dof_convert_to_common[
            torch.arange(n_dofs, device=self.device, dtype=torch.long)
        ]

        ke_wp = sim.robot_view.get_attribute("joint_target_ke", sim.model)
        kd_wp = sim.robot_view.get_attribute("joint_target_kd", sim.model)
        ke = wp.to_torch(ke_wp).view(self.num_envs, -1)
        kd = wp.to_torch(kd_wp).view(self.num_envs, -1)
        ke[:, sim_idx] = ke[:, sim_idx] * scale
        kd[:, sim_idx] = kd[:, sim_idx] * scale
        sim.robot_view.set_attribute("joint_target_ke", sim.model, ke_wp)
        sim.robot_view.set_attribute("joint_target_kd", sim.model, kd_wp)
        sim.solver.notify_model_changed(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

        comp._gain_scale[:] = scale

    def step(self, action: Tensor):
        self._lazy_init_assist()

        self.extras = {}
        self._current_context = None
        self._current_noisy_obs = None
        self._current_raw_action[:] = action

        # policy action -> assist torque [N·m]
        action_dict = self._process_action(action, self.context)
        tau_assist = torch.clamp(
            action_dict["processed_action"],
            -self.config.assist_torque_limit,
            self.config.assist_torque_limit,
        )
        self._current_processed_action[:] = tau_assist

        comp = self.assist_component

        # 사람 PD 목표 = 깨끗한 의도 궤적 θ_g (사람은 자기 의도를 정확히 앎)
        use_ffwd = getattr(self.config, "human_velocity_feedforward", False)
        if use_ffwd:
            pd_targets, target_vel = comp.get_pd_targets_and_vel()
            tau_ff = self._human_kd * target_vel  # 사람 몫의 피드포워드
        else:
            pd_targets = comp.get_pd_targets()
            tau_ff = None

        # qfrc 주입 = τ_assist (+ 사람 피드포워드 τ_ff)
        inject = tau_assist if tau_ff is None else tau_assist + tau_ff
        self._torque_override.set_torques(inject)
        comp._tau_assist[:] = tau_assist

        self.simulator.step(pd_targets, markers_callback=self.get_markers_state)

        # 사람 토크 계측: qfrc_actuator substep 평균 (COMMON ordering) — built-in
        # PD 몫. assist는 qfrc_applied 경로라 미포함이고, 피드포워드는 사람
        # 몫이므로 다시 더한다.
        tau_agent = self.simulator.get_substep_mean_dof_forces()
        if tau_ff is not None:
            tau_agent = tau_agent + tau_ff
        comp._tau_agent[:] = tau_agent

        self.post_physics_step()

        if self.simulator.user_requested_reset:
            self.user_reset()

        obs = self.get_obs()
        return obs, self.rew_buf, self.reset_buf, self.terminate_buf, self.extras
