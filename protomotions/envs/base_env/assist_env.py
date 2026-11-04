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
from typing import Optional

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
    # Legacy runs clamp the sum of independent qfrc sources to the wearer limit.
    # New experiments limit assist and wearer FF separately before adding external loads.
    clamp_combined_injection: bool = True
    human_gain_scale_min: float = 0.7
    human_gain_scale_max: float = 1.3
    assist_component_name: str = "assist"
    # 사람 모델에 속도 피드포워드 Kd·θ̇_g 추가:
    #   τ_agent = Kp(θ_g−θ) + Kd(θ̇_g−θ̇)
    # built-in PD(Kp(θ_g−θ) − Kd·θ̇) 위에 τ_ff = Kd·θ̇_g를 qfrc로 가산해 구현.
    # False면 순수 위치 PD (v1 동작).
    human_velocity_feedforward: bool = False
    # Route FF through the built-in actuator so its effort limit applies to the
    # complete human PD+FF torque. False preserves historical external-FF runs.
    human_feedforward_inside_pd: bool = False
    # Fixed per-environment mixture of position-PD and velocity-FF wearers.
    # Zero preserves the old FF path without consuming an extra random draw.
    human_feedforward_disable_prob: float = 0.0

    # ---- v4 플랜트 랜덤화 (기본값 = 꺼짐, 옛 pickle과 호환) ----
    # 대퇴 body의 **질량중심 관성(body_inertia)** 배율 범위 (env·dof별 1회 샘플). (1,1)이면 비활성.
    # 질량은 그대로 둔다 — 질량을 키우면 중력 모멘트(13 → 107 N·m)가 상시 하중이 되어 정책이
    # "중력 받치기"로 비용을 깎는 법을 배운다. 관성만 키우면 τ→각도 응답만 무거워진다
    # (입각기: 몸은 무겁고 hip 중력 모멘트는 작다). hip 축 관성 I = m·d² + r·I_com이라
    # r=25에서 0.29 + 25×0.13 = 3.6 kg·m² (스켈레톤 다리 3.5).
    inertia_scale_range: tuple = (1.0, 1.0)
    inertia_body_keyword: str = "thigh"
    # 착용자 임피던스: "gain_scale" = ke/kd × U[min,max] (v2/v3) |
    # "natural_freq" = ω_n∈omega_n_range, ζ∈zeta_range에서 Kp=ω_n²·I, Kd=2ζ√(Kp·I)
    # (I = hip 축 유효 관성). 진자 26.4 rad/s·0.66, 스켈레톤 착용자 7.5·0.38.
    impedance_mode: str = "gain_scale"
    omega_n_range: tuple = (26.4, 26.4)
    zeta_range: tuple = (0.66, 0.66)
    # Optional floor when broadening natural frequencies to soft/high-inertia wearers.
    # Zero preserves all legacy experiments. Kd uses the resulting Kp to retain zeta.
    human_stiffness_floor: float = 0.0
    hip_inertia_base: float = 0.418  # (하위 호환) 대퇴 hip 축 관성 [kg·m²]
    hip_inertia_com: float = 0.1316  # 대퇴 질량중심 관성 (hip_pendulum.xml, MuJoCo 계산)
    hip_inertia_parallel: float = 0.2863  # m·d² 평행축 항 (질량 배율에 비례)

    # ---- 질량·보조 한계 랜덤화 (2026-09-29) ----
    # 질량을 관성 배율 r과 함께 r^p 로 키운다 (p=0이면 v4처럼 질량 고정). v4는 관성만 ×1~25라 중력/관성 비
    # m·g·d/I 가 실제 다리(≈15 s⁻²)의 1/4까지 떨어졌고, 그래서 진자 다리는 유각 말기에 스스로 감속하지 않아
    # 정책이 "감속 = 브레이크 도움"을 배웠다(사람 범위 착용자 W1에서 유각 말기 오방향의 원인). p=0.43이면
    # r=25에서 질량 ×4, 중력 모멘트 13 → 53 N·m, 비 ≈ 15 s⁻².
    mass_scale_exponent: float = 0.0
    # 보조 토크 한계를 env마다 U[lo, hi]로 뽑아 clamp 한다 (정책은 관측 못 함 — 자기 토크의 효과를 히스토리로
    # 봐야 한다). (0, 0)이면 비활성 = assist_torque_limit 하나. 배포 때는 하드웨어 한계로 clamp (controller).
    assist_limit_range: tuple = (0.0, 0.0)
    # 2026-10-21 다리별 학습: 보조를 넣는 관절 (비면 전 관절). 나머지 관절의 보조는 0이다.
    # 보상·종료의 관절 범위는 각 MdpComponent의 joint_mask로 따로 정한다.
    assist_active_joints: tuple = ()


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

        n_dofs = self.robot_config.number_of_actions
        sim = self.simulator
        self._dof_sim_idx = sim.data_conversion.dof_convert_to_common[
            torch.arange(n_dofs, device=self.device, dtype=torch.long)
        ]
        self._ke_wp = sim.robot_view.get_attribute("joint_target_ke", sim.model)
        self._kd_wp = sim.robot_view.get_attribute("joint_target_kd", sim.model)

        self._inertia_scale = self._randomize_inertia()  # (n, d), 1.0이면 비활성
        lo, hi = (float(v) for v in getattr(self.config, "assist_limit_range", (0.0, 0.0)))
        if hi > 0:
            self._assist_limit = lo + (hi - lo) * torch.rand(self.num_envs, 1, device=self.device)
        else:
            self._assist_limit = None
        if getattr(self.config, "impedance_mode", "gain_scale") == "natural_freq":
            self._randomize_impedance_natural_freq()
        else:
            self._randomize_human_gains()

        # 랜덤화가 끝난 뒤의 게인을 COMMON 순서로 기억 (매 step 스케줄의 기준)
        import warp as wp

        ke = wp.to_torch(self._ke_wp).view(self.num_envs, -1)
        kd = wp.to_torch(self._kd_wp).view(self.num_envs, -1)
        self._ke_base = ke[:, self._dof_sim_idx].clone()
        self._kd_base = kd[:, self._dof_sim_idx].clone()
        self._human_kp = self._ke_base.clone()
        self._human_kd = self._kd_base.clone()
        if self.config.human_feedforward_inside_pd and (self._human_kp <= 0).any():
            raise ValueError("Human FF inside the PD actuator requires positive stiffness")
        probability = float(getattr(self.config, "human_feedforward_disable_prob", 0.0))
        if not 0.0 <= probability <= 1.0:
            raise ValueError("human_feedforward_disable_prob must be in [0, 1]")
        self._human_ff_enabled = None
        if probability > 0.0 and self.config.human_velocity_feedforward:
            # Both hips belong to the same wearer; do not switch their controller
            # type independently or within an episode. No actor input is added.
            self._human_ff_enabled = (
                torch.rand(self.num_envs, 1, device=self.device) >= probability
            )
        self._gain_scheduled = hasattr(self.assist_component, "gain_schedule")
        # 착용자 토크 한계 (COMMON 순서, (n, d)). built-in PD는 솔버가 이 값으로 clamp하지만
        # 피드포워드는 qfrc 경로라 여기서 직접 clamp해야 한다 — 안 그러면 Kd 150 × θ̇_g 7 rad/s
        # = 1000 N·m 같은 비인간적 토크가 "착용자 토크"로 집계된다 (v4 점검에서 실측).
        self._effort_limit = torch.tensor(
            [self.robot_config.control.control_info[n].effort_limit
             for n in self.robot_config.kinematic_info.dof_names],
            device=self.device, dtype=torch.float32,
        ).unsqueeze(0).expand(self.num_envs, -1).clone()
        if hasattr(self.assist_component, "set_wearer_gains"):
            self.assist_component.set_wearer_gains(
                self._ke_base, self._kd_base, self._effort_limit, self.hip_inertia_eff()
            )

    def _active_joint_mask(self) -> Optional[Tensor]:
        """(1, d) 0/1 보조 관절 마스크. 전 관절이면 None."""
        joints = tuple(getattr(self.config, "assist_active_joints", ()) or ())
        if not joints:
            return None
        mask = getattr(self, "_active_mask", None)
        if mask is None:
            n_dofs = self.robot_config.number_of_actions
            if any(not 0 <= int(j) < n_dofs for j in joints):
                raise ValueError(f"assist_active_joints {joints} outside 0..{n_dofs - 1}")
            mask = torch.zeros(1, n_dofs, device=self.device)
            mask[0, [int(j) for j in joints]] = 1.0
            self._active_mask = mask
        return mask

    def _randomize_inertia(self) -> Tensor:
        """대퇴 body 질량·관성을 env·dof별 배율로 바꾼다 (Newton body_* 속성, 1회).

        배율은 dof(hip)마다 뽑고, 같은 쪽("right"/"left")의 대퇴 body에 건다.
        """
        cfg = self.config
        n_dofs = self.robot_config.number_of_actions
        per_env = getattr(self.assist_component, "env_inertia_range", None)
        if per_env is not None:
            # 의도 집단별 관성 범위 (MotionAssistTargetControl.intent_pools)
            lo_t, hi_t = per_env[:, :1], per_env[:, 1:]
            scale = lo_t + (hi_t - lo_t) * torch.rand(self.num_envs, n_dofs, device=self.device)
        else:
            lo, hi = getattr(cfg, "inertia_scale_range", (1.0, 1.0))
            scale = lo + (hi - lo) * torch.rand(self.num_envs, n_dofs, device=self.device)
            if hi <= lo and abs(lo - 1.0) < 1e-9:
                return torch.ones(self.num_envs, n_dofs, device=self.device)

        import warp as wp
        from newton.solvers import SolverNotifyFlags

        sim = self.simulator
        keyword = getattr(cfg, "inertia_body_keyword", "thigh")
        link_names = list(sim.robot_view.body_names)
        dof_names = list(self.robot_config.kinematic_info.dof_names)
        # dof ↔ body 매칭: 같은 쪽 키워드가 들어간 대퇴 body
        body_for_dof = []
        for dn in dof_names:
            side = "right" if "right" in dn or dn.endswith("_r") else ("left" if "left" in dn or dn.endswith("_l") else None)
            cand = [i for i, ln in enumerate(link_names) if keyword in ln and (side is None or side in ln)]
            body_for_dof.append(cand[0] if cand else None)
        assert all(b is not None for b in body_for_dof), f"inertia bodies not found: {link_names}"

        I_wp = sim.robot_view.get_attribute("body_inertia", sim.model)
        iI_wp = sim.robot_view.get_attribute("body_inv_inertia", sim.model)
        I = wp.to_torch(I_wp).view(self.num_envs, -1, 3, 3)
        iI = wp.to_torch(iI_wp).view(self.num_envs, -1, 3, 3)
        p_mass = float(getattr(cfg, "mass_scale_exponent", 0.0))
        mscale = scale ** p_mass if p_mass > 0 else torch.ones_like(scale)
        self._mass_scale = mscale
        if p_mass > 0:
            m_wp = sim.robot_view.get_attribute("body_mass", sim.model)
            im_wp = sim.robot_view.get_attribute("body_inv_mass", sim.model)
            m = wp.to_torch(m_wp).view(self.num_envs, -1)
            im = wp.to_torch(im_wp).view(self.num_envs, -1)
        for d, b in enumerate(body_for_dof):
            r = scale[:, d]
            # 질량중심 관성은 r배; 질량은 r^p배 (질량중심 관성도 질량에 비례하므로 총 r^(1+p)... 가 아니라
            # 관성 배율 r 자체를 "hip 축 관성 목표"로 두고 질량은 별도 배율로 취급한다)
            I[:, b] = I[:, b] * r.view(-1, 1, 1)
            iI[:, b] = iI[:, b] / r.view(-1, 1, 1)
            if p_mass > 0:
                m[:, b] = m[:, b] * mscale[:, d]
                im[:, b] = im[:, b] / mscale[:, d]
        sim.robot_view.set_attribute("body_inertia", sim.model, I_wp)
        sim.robot_view.set_attribute("body_inv_inertia", sim.model, iI_wp)
        if p_mass > 0:
            sim.robot_view.set_attribute("body_mass", sim.model, m_wp)
            sim.robot_view.set_attribute("body_inv_mass", sim.model, im_wp)
        sim.solver.notify_model_changed(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES)
        return scale

    def hip_inertia_eff(self) -> Tensor:
        """env·dof별 hip 축 유효 관성 [kg·m²] = m·d² + 배율 × I_com."""
        cfg = self.config
        com = float(getattr(cfg, "hip_inertia_com", 0.1316))
        par = float(getattr(cfg, "hip_inertia_parallel", 0.2863))
        mscale = getattr(self, "_mass_scale", None)
        if mscale is None:
            mscale = torch.ones_like(self._inertia_scale)
        return par * mscale + com * self._inertia_scale

    def _randomize_impedance_natural_freq(self):
        """Kp = ω_n²·I, Kd = 2ζ√(Kp·I) — 관성이 바뀌어도 동역학 영역(ω_n, ζ)을 직접 뽑는다."""
        import warp as wp
        from newton.solvers import SolverNotifyFlags

        cfg = self.config
        n_dofs = self.robot_config.number_of_actions
        I_eff = self.hip_inertia_eff()
        wlo, whi = cfg.omega_n_range
        zlo, zhi = cfg.zeta_range
        wn = wlo + (whi - wlo) * torch.rand(self.num_envs, n_dofs, device=self.device)
        ze = zlo + (zhi - zlo) * torch.rand(self.num_envs, n_dofs, device=self.device)
        kp = wn * wn * I_eff
        floor = float(getattr(cfg, "human_stiffness_floor", 0.0))
        if not 0.0 <= floor < float("inf"):
            raise ValueError("human_stiffness_floor must be finite and nonnegative")
        if floor > 0:
            kp = kp.clamp_min(floor)
        kd = 2.0 * ze * torch.sqrt(kp * I_eff)
        ke_t = wp.to_torch(self._ke_wp).view(self.num_envs, -1)
        kd_t = wp.to_torch(self._kd_wp).view(self.num_envs, -1)
        ke_t[:, self._dof_sim_idx] = kp
        kd_t[:, self._dof_sim_idx] = kd
        sim = self.simulator
        sim.robot_view.set_attribute("joint_target_ke", sim.model, self._ke_wp)
        sim.robot_view.set_attribute("joint_target_kd", sim.model, self._kd_wp)
        sim.solver.notify_model_changed(SolverNotifyFlags.JOINT_DOF_PROPERTIES)
        # critic privileged 관측이 쓰는 gain_scale에는 진자 기본(300)에 대한 비율을 남긴다
        base_kp = torch.tensor(
            [self.robot_config.control.control_info[n].stiffness for n in self.robot_config.kinematic_info.dof_names],
            device=self.device, dtype=torch.float32,
        )
        self.assist_component._gain_scale[:] = kp / base_kp.unsqueeze(0)

    def _apply_gain_schedule(self):
        """컴포넌트가 주는 (Kp 배율, Kd 배율)로 built-in PD 게인을 이번 step에 맞게 갱신."""
        import warp as wp
        from newton.solvers import SolverNotifyFlags

        s_kp, s_kd = self.assist_component.gain_schedule()
        ke_t = wp.to_torch(self._ke_wp).view(self.num_envs, -1)
        kd_t = wp.to_torch(self._kd_wp).view(self.num_envs, -1)
        self._human_kp = self._ke_base * s_kp
        ke_t[:, self._dof_sim_idx] = self._human_kp
        kd_now = self._kd_base * s_kd
        kd_t[:, self._dof_sim_idx] = kd_now
        self._human_kd = kd_now
        sim = self.simulator
        sim.robot_view.set_attribute("joint_target_ke", sim.model, self._ke_wp)
        sim.robot_view.set_attribute("joint_target_kd", sim.model, self._kd_wp)
        # notify_model_changed(JOINT_DOF_PROPERTIES)는 게인 복사 뒤에 mujoco_warp.set_const_0
        # (qpos0 상수 재계산: DOF마다 밀집 LDL 풀이)를 매번 다시 돌린다 — 게인과 무관한 계산이고,
        # 매 step 호출에서 세그폴트가 났다(2026-09-23, faulthandler 스택). 게인 복사 커널만 돌린다.
        upd = getattr(sim.solver, "_update_joint_dof_properties", None)
        if upd is not None:
            upd()
        else:
            sim.solver.notify_model_changed(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

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
        self.requested_assist_torque = action_dict["processed_action"].clone()
        lim = getattr(self, "_assist_limit", None)
        if lim is not None:  # env마다 다른 한계
            tau_assist = torch.maximum(torch.minimum(action_dict["processed_action"], lim), -lim)
        else:
            tau_assist = torch.clamp(
                action_dict["processed_action"],
                -self.config.assist_torque_limit,
                self.config.assist_torque_limit,
            )
        active = self._active_joint_mask()
        if active is not None:
            tau_assist = tau_assist * active
            self.requested_assist_torque = self.requested_assist_torque * active
        self._current_processed_action[:] = tau_assist

        comp = self.assist_component
        if getattr(self, "_gain_scheduled", False):
            self._apply_gain_schedule()  # 입각 사건의 착용자 굳힘 (컴포넌트 step은 post_physics에서 갱신됨)

        # 사람 PD 목표 = 깨끗한 의도 궤적 θ_g (사람은 자기 의도를 정확히 앎)
        use_ffwd = getattr(self.config, "human_velocity_feedforward", False)
        if use_ffwd:
            pd_targets, target_vel = comp.get_pd_targets_and_vel()
            tau_ff = self._human_kd * target_vel  # 사람 몫의 피드포워드
            if self.config.human_feedforward_inside_pd:
                if self._human_ff_enabled is not None:
                    tau_ff = tau_ff * self._human_ff_enabled
                # kp*(q_goal + FF/kp - q) - kd*qdot is exactly PD+FF.
                # The solver clips this single human actuator at each substep;
                # external assistance and disturbances remain independent.
                pd_targets = pd_targets + tau_ff / self._human_kp
                tau_ff = None  # Already included in measured actuator force.
            else:
                if getattr(self, "_effort_limit", None) is not None:
                    tau_ff = torch.clamp(tau_ff, -self._effort_limit, self._effort_limit)
                if self._human_ff_enabled is not None:
                    tau_ff = tau_ff * self._human_ff_enabled
        else:
            pd_targets = comp.get_pd_targets()
            tau_ff = None

        # qfrc 주입 = τ_assist (+ 사람 피드포워드 τ_ff) (+ 외부 하중 τ_ext — 착용자 토크 아님)
        inject = tau_assist if tau_ff is None else tau_assist + tau_ff
        if hasattr(comp, "external_torque"):
            inject = inject + comp.external_torque()
        written = self._torque_override.set_torques(
            inject, clamp_to_effort_limits=self.config.clamp_combined_injection
        )
        self.extras["assist/requested_abs_max_Nm"] = self.requested_assist_torque.abs().max()
        self.extras["assist/limit_fraction"] = (
            (self.requested_assist_torque - tau_assist).abs() > 1e-5
        ).float().mean()
        self.extras["assist/injection_error_max_Nm"] = (written - inject).abs().max()
        self.extras["assist/injection_clipped_fraction"] = (
            (written - inject).abs() > 1e-5
        ).float().mean()
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
