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
"""진자 LLP 학습용 외란 — "그네 밀어주기"가 아니라 "스파이크 있는 세상"을 만든다 (v4).

진자 v3는 매끈한 sine 의도를 매끈한 PD가 따라가는 상황만 보고 학습했다. 스켈레톤
보행에서 착용자 hip 토크는 착지마다 −300 N·m급으로 튀는데(`docs/ai_sbc/NEXT_SESSION.md`
§2-E), 그 스파이크가 진자 학습 분포에 한 번도 없었다. 이 컴포넌트는
`MotionAssistTargetControl`을 상속해 의도 궤적·chunk는 그대로 두고, hip 관절에 두
종류의 외란을 던진다. **정책은 외란 값을 관측하지 못한다** — 관절 각도·각속도
히스토리와 HLP 예측창으로 그 효과만 본다 (도메인 랜덤화).

1. **입각 사건 (stance event)** — 착지·하중 수용의 골격. 세 가지가 함께 온다:
     * 외부 하중 모멘트 τ_load (qfrc, 착용자 토크에 **미포함**): 체중이 hip에 요구하는
       모멘트. 부호 무작위(보행 착지는 +굴곡 외력), 크기 0~`stance_load_per_kg`×`reference_mass`.
     * 착용자 굳힘: Kp × `stance_kp_mult`, Kd × `stance_kd_mult` (built-in PD 게인을
       env가 매 step 갱신; **착용자 토크에 포함** — 입각기 근육 공수축). 이것이
       "큰 토크, 작은 각도 변화"를 만든다.
     * 하중 크기는 굳은 강성으로 처지는 각도가 `stance_max_deflection`을 넘지 않게 묶는다
       (무거운 하중은 발이 땅에 붙어 굳었을 때만 온다는 물리를 그대로).
   시점은 시간 기반 무작위(간격 `stance_gap_range`)라 보행 주기에 걸리는 것도, 안
   걸리는 것도 다 나온다. (env, dof)마다 독립.
2. **작은 섭동 (perturbation)** — 굳힘 없이 아무 때나 오는 반정현 펄스, 0~`perturb_amp_per_kg`.
3. **반전 브레이크 (reversal brake, 2026-09-29)** — 의도 θ_g가 반전(θ̇_g 부호 바뀜)에 `reversal_brake_lead` 안으로
   다가오면 확률 `reversal_brake_prob`로 외부 점성 토크 −c·θ̇(실제 각속도)를 건다. 실제 보행에서 유각 말기 대퇴는
   무릎 결합·지면으로 **공짜로 감속**하는데(사람 범위 착용자 W1의 그 구간 hip 토크 ≈ 0), 접촉 없는 진자는 감속에
   토크가 필요해서 정책이 "감속 중 = 브레이크 도움"을 배웠다(2026-09 초기 진자 정책들의 유각 말기 오방향). 이 사건이 그 감속을 대신
   해 주어, 브레이크 도움이 평균적으로 손해가 되게 한다. 환경 토크(착용자 토크 미포함).

θ_g 다양화(계단·램프·비주기)는 부모 config의 `blend_range`·`hold_prob`로 실험 파일에서
넓힌다 — 코드 변경 없음. 관성·기본 임피던스 랜덤화는 `AssistEnv`(v4 필드) 몫.

2026-10 진자 연구의 두 외란 변형도 이 클래스의 method로 들어 있다 (기본값은 모두 꺼짐이며, 꺼진
기능은 난수를 소비하지 않아 옛 체크포인트의 RNG·궤적이 그대로다):
4. **구조화 사건** (`structured_probability`) — 입각 하중을 각도 의존 힘장으로 바꾸고 수동 감쇠 사건을
   더한다. 사건 시각은 경과 시간만으로 뽑아 짝 플랜트가 같은 난수를 쓴다. `human_target_hold_steps`는
   착용자 PD/FF 명령의 sample-and-hold 주기.
5. **유각 감속·반전 충격** (`phase_probability`) — 합성 의도의 감속·반전 시각에 실제 각속도에 비례한
   유한 소산 펄스. 접촉·관성 변화 모델이 아니다.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, TYPE_CHECKING

import torch
from torch import Tensor

from protomotions.envs.control.assist_motion_control import (
    MotionAssistTargetControl,
    MotionAssistTargetControlConfig,
)

if TYPE_CHECKING:
    from protomotions.envs.base_env.env import BaseEnv


@dataclass
class DisturbedAssistTargetControlConfig(MotionAssistTargetControlConfig):
    """새 필드는 plain default만 (옛 pickle 호환)."""

    _target_: str = (
        "protomotions.envs.control.assist_disturbance_control.DisturbedAssistTargetControl"
    )

    # 하중 크기의 기준 질량 [kg] — 문헌 보행 hip 모멘트 피크 ≈ 1 N·m/kg (착용자 90.7 kg)
    reference_mass: float = 90.0

    # 입각 사건
    stance_gap_range: Tuple[float, float] = (0.4, 1.6)   # 사건 사이 간격 [s] (보행 주기 안팎)
    stance_dur_range: Tuple[float, float] = (0.08, 0.45)  # 사건 길이 [s]
    stance_ramp: float = 0.03                            # 코사인 오르내림 [s]
    stance_load_per_kg: float = 2.0                      # 하중 상한 [N·m/kg] (0~이 값 균일)
    stance_load_pos_prob: float = 0.5                    # +굴곡 외력일 확률
    stance_kp_mult_range: Tuple[float, float] = (2.0, 8.0)
    stance_kd_mult_range: Tuple[float, float] = (1.5, 4.0)
    stance_max_deflection: float = 0.3                   # 하중/Kp_eff 상한 [rad]
    stance_prob: float = 0.85                            # 간격이 찼을 때 실제로 사건이 일어날 확률

    # 작은 섭동
    perturb_gap_range: Tuple[float, float] = (0.5, 3.0)
    perturb_dur_range: Tuple[float, float] = (0.03, 0.25)
    perturb_amp_per_kg: float = 0.3

    # 반전 브레이크 (0이면 꺼짐 = v4 호환)
    reversal_brake_prob: float = 0.0
    reversal_brake_lead: float = 0.15                  # 반전 전 이 시간 안이면 창 시작 [s]
    reversal_brake_c_range: Tuple[float, float] = (2.0, 30.0)  # 점성 계수 [N·m·s/rad]
    reversal_brake_ramp: float = 0.03
    reversal_brake_min_vel: float = 0.3                # 이보다 느리면 반전으로 안 침 [rad/s]
    # 반전 사건의 종류 (2026-09-29): "viscous" = −c·θ̇ | "feedforward" = κ·I_eff·θ̈_g — 의도가 반전에서
    # 요구하는 관성 토크의 κ배를 환경이 대신 낸다(실제 다리의 무릎 결합·중력이 하는 일). 진자 정책은 "의도의 관성
    # 토크를 대신 내는 피드포워드"를 배우는데, 실제 다리에선 반전 관성이 공짜라 그 브레이크가 손해였다(2026-09 초기 진자 정책 공통).
    reversal_brake_mode: str = "viscous"
    reversal_ff_gain_range: Tuple[float, float] = (0.5, 1.2)   # κ
    reversal_ff_after: float = 0.10                           # 반전 뒤에도 이만큼 유지 [s]

    disturbance_enabled: bool = True
    # 의도 실현 가능성: 필요 각가속도 ≤ 이 비율 × (착용자 토크 한계 / hip 관성). 0이면 안 묶음.
    intent_accel_frac: float = 0.5

    # 구조화 사건 (2026-10-14 연구, 0이면 꺼짐)
    structured_probability: float = 0.0   # 구조화 에피소드일 확률
    damping_probability: float = 0.75     # 감쇠 사건이 실제 계수를 가질 확률
    damping_max: float = 30.0             # 감쇠 계수 상한 [N·m·s/rad]
    damping_torque_limit: float = 80.0    # 감쇠 토크 clamp [N·m]
    human_target_hold_steps: int = 1      # 착용자 명령 sample-and-hold 주기 [policy step], 1 = 매 step

    # 유각 감속·반전 충격 (2026-10-15 일반화 연구, 0이면 꺼짐)
    phase_probability: float = 0.0
    swing_damping_max: float = 20.0
    swing_zero_probability: float = 0.25
    swing_duration_range: Tuple[float, float] = (0.06, 0.18)
    shock_duration_range: Tuple[float, float] = (0.02, 0.05)
    shock_damping_max: float = 35.0
    phase_torque_limit: float = 60.0


class DisturbedAssistTargetControl(MotionAssistTargetControl):
    """의도·chunk는 부모, 여기서는 외란 스케줄만 관리한다."""

    def __init__(self, config: DisturbedAssistTargetControlConfig, env: "BaseEnv"):
        n, d, dev = env.num_envs, env.robot_config.number_of_actions, env.device
        z = lambda *s: torch.zeros(*s, device=dev)  # noqa: E731
        # 입각 사건 상태 (env, dof)
        self._ev_start, self._ev_end = z(n, d), z(n, d)
        self._ev_load, self._ev_kp, self._ev_kd = z(n, d), torch.ones(n, d, device=dev), torch.ones(n, d, device=dev)
        self._ev_next = z(n, d)
        # 섭동 상태
        self._pt_start, self._pt_end, self._pt_amp, self._pt_next = z(n, d), z(n, d), z(n, d), z(n, d)
        # env가 넘겨주는 착용자 기본 Kp (하중 상한 계산용). 모르면 300.
        self._kp_base = torch.full((n, d), 300.0, device=dev)
        self._effort_limit = torch.full((n, d), 150.0, device=dev)
        self._inertia = torch.full((n, d), 0.418, device=dev)
        # 반전 브레이크 상태 (env, dof)
        self._rb_near = torch.zeros(n, d, dtype=torch.bool, device=dev)
        self._rb_c, self._rb_t0 = z(n, d), z(n, d)
        self._rb_end = z(n, d)
        self._tau_brake = z(n, d)
        # 이번 step 결과
        self._tau_ext = z(n, d)
        self._kp_scale = torch.ones(n, d, device=dev)
        self._kd_scale = torch.ones(n, d, device=dev)
        # 구조화 사건·명령 홀드 상태
        self._structured = torch.zeros(n, 1, dtype=torch.bool, device=dev)
        for name in ("_field_center", "_field_slope", "_field_bias", "_damp_start", "_damp_end", "_damp_next",
                     "_damp_c", "_tau_field", "_tau_damping", "_held_target", "_held_velocity"):
            setattr(self, name, z(n, d))
        self._hold_step = torch.full((n,), -1, dtype=torch.long, device=dev)
        # 유각 감속·반전 충격 상태
        self._phase_enabled = torch.zeros(n, d, dtype=torch.bool, device=dev)
        self._decelerating = torch.zeros(n, d, dtype=torch.bool, device=dev)
        for name in ("_previous_intent_velocity", "_swing_start", "_swing_end", "_swing_c",
                     "_shock_start", "_shock_end", "_shock_c", "_tau_swing", "_tau_shock"):
            setattr(self, name, z(n, d))
        self._validate_event_config(config, env)
        super().__init__(config, env)  # 마지막에 self.reset(all) 호출

    @staticmethod
    def _validate_event_config(config, env) -> None:
        g = lambda name, default: getattr(config, name, default)  # noqa: E731  (옛 pickle엔 필드가 없다)
        for name in ("structured_probability", "damping_probability", "phase_probability", "swing_zero_probability"):
            if not 0 <= g(name, 0.0) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("damping_max", "damping_torque_limit", "swing_damping_max", "shock_damping_max", "phase_torque_limit"):
            if not 0 <= g(name, 0.0) < float("inf"):
                raise ValueError(f"Invalid {name}")
        if g("human_target_hold_steps", 1) < 1:
            raise ValueError("human_target_hold_steps must be positive")
        dt = getattr(env, "dt", None)
        for name, default in (("swing_duration_range", (0.06, 0.18)), ("shock_duration_range", (0.02, 0.05))):
            lo, hi = g(name, default)
            if not (dt is None or dt <= lo) or not lo <= hi < float("inf"):
                raise ValueError(f"Invalid {name}")

    # ------------------------------------------------------------------
    def set_wearer_gains(
        self, kp: Tensor, kd: Tensor, effort_limit: Optional[Tensor] = None, inertia: Optional[Tensor] = None
    ) -> None:
        """env가 (랜덤화된) 착용자 기본 게인·토크 한계·hip 관성을 알려준다.

        * 하중 상한 = min(처짐 한계 × Kp_eff, 0.8 × effort_limit): 착용자 PD가 버틸 수 없는
          하중은 다리를 날려 버려 종료만 만들 뿐 배울 게 없다.
        * 의도 진폭 상한: 관성 I 다리로 진폭 a·주파수 f sine을 따르려면 a(2πf)²·I 의 토크가
          필요하다. 3.5 kg·m²에 2 Hz·0.6 rad은 330 N·m — 한계 150을 넘어 추종 실패로만 끝난다.
          사람도 그런 의도는 내지 않으므로 세그먼트 샘플 후 `intent_accel_frac`×한계/I 로 진폭을 묶는다.
        """
        self._kp_base[:] = kp
        if effort_limit is not None:
            self._effort_limit[:] = effort_limit
        if inertia is not None:
            self._inertia[:] = inertia
        # 게인이 정해진 뒤 현재 세그먼트도 실현 가능하게 다시 묶는다
        all_ids = torch.arange(self.env.num_envs, device=self.env.device)
        for slot in (0, 1):
            self._clamp_intent_amplitude(slot, all_ids)
        # 동작 데이터 의도도 같은 원리로: 이 다리가 따라갈 수 없는 클립은 뽑지 않는다
        self.update_motion_feasibility(self._effort_limit, self._inertia)

    def _clamp_intent_amplitude(self, slot: int, env_ids: Tensor) -> None:
        """a·(2πf)² + a₂·(4πf)² ≤ intent_accel_frac × effort / I 가 되게 a, a₂를 비례 축소."""
        cfg: DisturbedAssistTargetControlConfig = self.config
        frac = float(getattr(cfg, "intent_accel_frac", 0.5))
        if frac <= 0:
            return
        f = self._params["freq"][slot, env_ids]
        a = self._params["amp"][slot, env_ids]
        a2 = self._params["amp2"][slot, env_ids]
        w2 = (2.0 * math.pi * f) ** 2
        need = a * w2 + a2 * 4.0 * w2  # 필요한 최대 각가속도 [rad/s²]
        cap = frac * self._effort_limit[env_ids] / self._inertia[env_ids]
        scale = torch.where(need > cap, cap / need.clamp(min=1e-6), torch.ones_like(need))
        self._params["amp"][slot, env_ids] = a * scale
        self._params["amp2"][slot, env_ids] = a2 * scale

    def _sample_segment(self, slot: int, env_ids: Tensor, allow_hold: bool = True):
        super()._sample_segment(slot, env_ids, allow_hold)
        self._clamp_intent_amplitude(slot, env_ids)

    def _u(self, lo: float, hi: float, *shape) -> Tensor:
        return lo + (hi - lo) * torch.rand(*shape, device=self.env.device)

    def _gate(self, t: Tensor, start: Tensor, end: Tensor, ramp: float) -> Tensor:
        """[start, end] 안에서 1, 양끝 ramp 구간은 코사인 (2D 텐서 연산)."""
        d_in = t - start
        d_out = end - t
        active = (d_in >= 0) & (d_out >= 0)
        r = max(ramp, 1e-4)
        g = torch.minimum((d_in / r).clamp(0, 1), (d_out / r).clamp(0, 1))
        g = 0.5 * (1.0 - torch.cos(math.pi * g))
        return torch.where(active, g, torch.zeros_like(g))

    # ------------------------------------------------------------------
    def reset(self, env_ids: Tensor) -> None:
        super().reset(env_ids)
        if len(env_ids) == 0:
            return
        cfg: DisturbedAssistTargetControlConfig = self.config
        d = self.n_dofs
        m = len(env_ids)
        self._ev_start[env_ids] = -1.0
        self._ev_end[env_ids] = -1.0
        self._ev_load[env_ids] = 0.0
        self._ev_kp[env_ids] = 1.0
        self._ev_kd[env_ids] = 1.0
        # 첫 사건은 start_blend 뒤 간격 하나 지난 다음부터
        self._ev_next[env_ids] = cfg.start_blend + self._u(*cfg.stance_gap_range, m, d)
        self._pt_start[env_ids] = -1.0
        self._pt_end[env_ids] = -1.0
        self._pt_amp[env_ids] = 0.0
        self._pt_next[env_ids] = cfg.start_blend + self._u(*cfg.perturb_gap_range, m, d)
        self._tau_ext[env_ids] = 0.0
        self._kp_scale[env_ids] = 1.0
        self._kd_scale[env_ids] = 1.0
        self._rb_near[env_ids] = False
        self._rb_c[env_ids] = 0.0
        self._rb_end[env_ids] = -1.0
        self._tau_brake[env_ids] = 0.0
        self._reset_structured(env_ids)
        self._reset_phase(env_ids)

    def _reset_structured(self, env_ids: Tensor) -> None:
        cfg: DisturbedAssistTargetControlConfig = self.config
        self._hold_step[env_ids] = -1
        self._tau_field[env_ids] = 0
        self._tau_damping[env_ids] = 0
        prob = float(getattr(cfg, "structured_probability", 0.0))
        if not len(env_ids) or prob == 0:
            return
        m, d = len(env_ids), self.n_dofs
        self._structured[env_ids] = self._u(0, 1, m, 1) < prob
        self._field_center[env_ids] = self._u(*cfg.offset_range, m, d)
        # 무차원 계수: 이미 묶인 하중 사건 크기에 비례해 쓴다
        self._field_slope[env_ids] = self._u(-1, 1, m, d) / self._u(0.15, 0.6, m, d)
        self._field_bias[env_ids] = self._u(-0.25, 0.25, m, d)
        self._damp_start[env_ids] = -1
        self._damp_end[env_ids] = -1
        self._damp_c[env_ids] = 0
        self._damp_next[env_ids] = cfg.start_blend + self._u(*cfg.stance_gap_range, m, d)

    def _reset_phase(self, env_ids: Tensor) -> None:
        self._phase_enabled[env_ids] = False
        self._decelerating[env_ids] = False
        for name in ("_previous_intent_velocity", "_swing_c", "_shock_c", "_tau_swing", "_tau_shock"):
            getattr(self, name)[env_ids] = 0
        for name in ("_swing_start", "_swing_end", "_shock_start", "_shock_end"):
            getattr(self, name)[env_ids] = -1
        prob = float(getattr(self.config, "phase_probability", 0.0))
        if prob > 0 and len(env_ids):
            self._phase_enabled[env_ids] = self._u(0, 1, len(env_ids), self.n_dofs) < prob

    def step(self) -> None:
        super().step()
        cfg: DisturbedAssistTargetControlConfig = self.config
        if not getattr(cfg, "disturbance_enabled", True):
            return
        t = self._episode_time().unsqueeze(1).expand(-1, self.n_dofs)  # (n, d)
        dev = self.env.device
        n, d = t.shape

        # --- 입각 사건: 새로 시작할 (env, dof) ---
        due = (t >= self._ev_next) & (t > self._ev_end)
        if due.any():
            k = int(due.sum())
            fire = torch.rand(k, device=dev) < cfg.stance_prob
            dur = self._u(*cfg.stance_dur_range, k)
            kp_m = self._u(*cfg.stance_kp_mult_range, k)
            kd_m = self._u(*cfg.stance_kd_mult_range, k)
            sign = torch.where(torch.rand(k, device=dev) < cfg.stance_load_pos_prob, 1.0, -1.0)
            mag = torch.rand(k, device=dev) * cfg.stance_load_per_kg * cfg.reference_mass
            # 처짐 한계: 굳은 강성으로 stance_max_deflection 이상 처지지 않게
            mag = torch.minimum(mag, cfg.stance_max_deflection * self._kp_base[due] * kp_m)
            mag = torch.minimum(mag, 0.8 * self._effort_limit[due])
            t_due = t[due]
            self._ev_start[due] = torch.where(fire, t_due, torch.full_like(t_due, -1.0))
            self._ev_end[due] = torch.where(fire, t_due + dur, torch.full_like(t_due, -1.0))
            self._ev_load[due] = torch.where(fire, sign * mag, torch.zeros_like(mag))
            self._ev_kp[due] = torch.where(fire, kp_m, torch.ones_like(kp_m))
            self._ev_kd[due] = torch.where(fire, kd_m, torch.ones_like(kd_m))
            gap = self._u(*cfg.stance_gap_range, k)
            self._ev_next[due] = torch.where(fire, t_due + dur, t_due) + gap

        # --- 섭동 ---
        due_p = (t >= self._pt_next) & (t > self._pt_end)
        if due_p.any():
            k = int(due_p.sum())
            dur = self._u(*cfg.perturb_dur_range, k)
            amp = (torch.rand(k, device=dev) * 2 - 1) * cfg.perturb_amp_per_kg * cfg.reference_mass
            t_due = t[due_p]
            self._pt_start[due_p] = t_due
            self._pt_end[due_p] = t_due + dur
            self._pt_amp[due_p] = amp
            self._pt_next[due_p] = t_due + dur + self._u(*cfg.perturb_gap_range, k)

        # --- 이번 step 값 ---
        g = self._gate(t, self._ev_start, self._ev_end, cfg.stance_ramp)
        self._kp_scale = 1.0 + g * (self._ev_kp - 1.0)
        self._kd_scale = 1.0 + g * (self._ev_kd - 1.0)
        tau_load = g * self._ev_load
        span = (self._pt_end - self._pt_start).clamp(min=1e-4)
        in_p = (t >= self._pt_start) & (t <= self._pt_end)
        tau_p = torch.where(in_p, self._pt_amp * torch.sin(math.pi * (t - self._pt_start) / span), torch.zeros_like(t))
        self._tau_ext = tau_load + tau_p + self._reversal_brake(t[:, 0])
        self._structured_events(t)
        self._phase_events()

    def _structured_events(self, t: Tensor) -> None:
        """구조화 에피소드: 입각 하중을 각도 의존 힘장으로 바꾸고 수동 감쇠 사건을 더한다 (꺼지면 난수 소비 없음)."""
        cfg: DisturbedAssistTargetControlConfig = self.config
        if float(getattr(cfg, "structured_probability", 0.0)) == 0:
            return
        due = (t >= self._damp_next) & (t > self._damp_end)
        if due.any():
            count = int(due.sum())
            duration = self._u(*cfg.stance_dur_range, count)
            self._damp_start[due] = t[due]
            self._damp_end[due] = t[due] + duration
            self._damp_next[due] = self._damp_end[due] + self._u(*cfg.stance_gap_range, count)
            self._damp_c[due] = self._u(0, cfg.damping_max, count) * (
                self._u(0, 1, count) < cfg.damping_probability)
        hist = self.env.state_history
        q, velocity = hist.dof_pos[:, 0], hist.dof_vel[:, 0]
        amplitude = self._ev_load.abs()
        slope_limit = 0.5 * self._kp_base * self._kp_scale
        slope = (amplitude * self._field_slope).clamp(-slope_limit, slope_limit)
        field = (slope * (q - self._field_center) + self._field_bias * amplitude).clamp(-amplitude, amplitude)
        load_gate = self._gate(t, self._ev_start, self._ev_end, cfg.stance_ramp)
        self._tau_field = self._structured * load_gate * field
        damping_gate = self._gate(t, self._damp_start, self._damp_end, cfg.stance_ramp)
        self._tau_damping = self._structured * (-damping_gate * self._damp_c * velocity).clamp(
            -cfg.damping_torque_limit, cfg.damping_torque_limit)
        # 옛 하중을 겹치지 않고 바꾼다; 게인 스케줄·섭동은 그대로. 비구조화 에피소드는 원래 힘을 유지.
        self._tau_ext = (self._tau_ext - self._structured * load_gate * self._ev_load
                         + self._tau_field + self._tau_damping)

    def _phase_events(self) -> None:
        """합성 의도의 감속·반전 시각에 실제 각속도 비례 소산 펄스 (꺼지면 난수 소비 없음)."""
        cfg: DisturbedAssistTargetControlConfig = self.config
        if float(getattr(cfg, "phase_probability", 0.0)) == 0:
            return
        time = self._episode_time()
        dt = self.env.dt
        q = self._target_at(torch.stack((time - dt, time, time + dt), dim=1))
        velocity = (q[:, 2] - q[:, 0]) / (2 * dt)
        acceleration = (q[:, 2] - 2 * q[:, 1] + q[:, 0]) / dt ** 2
        t = time[:, None].expand_as(velocity)
        eligible = self._phase_enabled & (t > cfg.start_blend)
        decelerating = (velocity * acceleration < 0) & (velocity.abs() > .3)
        onset = eligible & decelerating & ~self._decelerating & (t > self._swing_end)
        if onset.any():
            count = int(onset.sum())
            self._swing_start[onset] = t[onset]
            self._swing_end[onset] = t[onset] + self._u(*cfg.swing_duration_range, count)
            self._swing_c[onset] = self._u(0, cfg.swing_damping_max, count) * (
                self._u(0, 1, count) >= cfg.swing_zero_probability)
        reversal = eligible & (velocity * self._previous_intent_velocity < 0) & (t > self._shock_end)
        if reversal.any():
            count = int(reversal.sum())
            self._shock_start[reversal] = t[reversal]
            self._shock_end[reversal] = t[reversal] + self._u(*cfg.shock_duration_range, count)
            self._shock_c[reversal] = self._u(0, cfg.shock_damping_max, count)
        actual_velocity = self.env.state_history.dof_vel[:, 0]
        for event, ramp in (("swing", .02), ("shock", .01)):
            gate = self._gate(t, getattr(self, f"_{event}_start"), getattr(self, f"_{event}_end"), ramp)
            torque = (-gate * getattr(self, f"_{event}_c") * actual_velocity).clamp(
                -cfg.phase_torque_limit, cfg.phase_torque_limit)
            setattr(self, f"_tau_{event}", torque)
            self._tau_ext = self._tau_ext + torque
        self._decelerating.copy_(decelerating)
        self._previous_intent_velocity.copy_(velocity)

    def get_pd_targets_and_vel(self):
        """착용자 PD/FF 명령; `human_target_hold_steps` > 1이면 그 주기로 sample-and-hold."""
        hold = int(getattr(self.config, "human_target_hold_steps", 1))
        if hold == 1:
            return super().get_pd_targets_and_vel()
        step = self.env.progress_buf
        due = (self._hold_step < 0) | (step - self._hold_step >= hold)
        if due.any():
            target, velocity = super().get_pd_targets_and_vel()
            self._held_target[due] = target[due]
            self._held_velocity[due] = velocity[due]
            self._hold_step[due] = step[due]
        return self._held_target, self._held_velocity

    def get_pd_targets(self):
        if int(getattr(self.config, "human_target_hold_steps", 1)) == 1:
            return super().get_pd_targets()
        return self.get_pd_targets_and_vel()[0]

    def _reversal_brake(self, t: Tensor) -> Tensor:
        """의도가 반전에 다가올 때 실제 각속도에 비례한 외부 점성 토크 (n, d). 꺼져 있으면 0."""
        cfg: DisturbedAssistTargetControlConfig = self.config
        prob = float(getattr(cfg, "reversal_brake_prob", 0.0))
        if prob <= 0:
            return self._tau_brake.zero_()
        lead = float(getattr(cfg, "reversal_brake_lead", 0.15))
        mode = str(getattr(cfg, "reversal_brake_mode", "viscous"))
        delta = 0.01
        tt = torch.stack([t - delta, t, t + delta, t + lead, t + lead + delta], dim=1)  # (n, 5)
        th = self._target_at(tt)  # (n, 5, d)
        v_now = (th[:, 2] - th[:, 1]) / delta
        a_now = (th[:, 2] - 2 * th[:, 1] + th[:, 0]) / (delta * delta)   # θ̈_g
        v_lead = (th[:, 4] - th[:, 3]) / delta
        near = (v_now * v_lead < 0) & (v_now.abs() > float(getattr(cfg, "reversal_brake_min_vel", 0.3)))
        rising = near & ~self._rb_near
        tcol = t.unsqueeze(1).expand_as(near)
        if rising.any():
            k = int(rising.sum())
            dev = t.device
            on = torch.rand(k, device=dev) < prob
            if mode == "feedforward":
                lo, hi = getattr(cfg, "reversal_ff_gain_range", (0.5, 1.2))
            else:
                lo, hi = cfg.reversal_brake_c_range
            c = lo + (hi - lo) * torch.rand(k, device=dev)
            self._rb_c[rising] = torch.where(on, c, torch.zeros_like(c))
            self._rb_t0[rising] = tcol[rising]
            # 창의 끝: 반전(≈ lead 뒤) + after
            self._rb_end[rising] = tcol[rising] + lead + float(getattr(cfg, "reversal_ff_after", 0.10))
        if mode == "feedforward":
            active = tcol <= self._rb_end
        else:
            active = near
        self._rb_c = torch.where(active, self._rb_c, torch.zeros_like(self._rb_c))
        self._rb_near = near
        ramp = max(float(getattr(cfg, "reversal_brake_ramp", 0.03)), 1e-4)
        g = ((tcol - self._rb_t0) / ramp).clamp(0.0, 1.0)
        if mode == "feedforward":
            # 환경이 의도의 관성 토크 κ·I·θ̈_g 를 대신 낸다 (착용자 토크 아님)
            self._tau_brake = g * self._rb_c * self._inertia * a_now
            return self._tau_brake
        hist = getattr(self.env, "state_history", None)
        dof_vel = hist.dof_vel[:, 0] if hist is not None and hasattr(hist, "dof_vel") else None
        if dof_vel is None:
            return self._tau_brake.zero_()
        self._tau_brake = -g * self._rb_c * dof_vel
        return self._tau_brake

    def in_reversal_brake(self) -> Tensor:
        """진단용: 반전 브레이크가 걸린 (n, d) 마스크."""
        return self._rb_c > 0

    # ------------------------------------------------------------------
    # env가 매 step 읽는 값
    # ------------------------------------------------------------------
    def external_torque(self) -> Tensor:
        """hip에 가할 외부 토크 (n, d) [N·m] — 착용자 토크에 포함하지 않는다."""
        return self._tau_ext

    def gain_schedule(self) -> Tuple[Tensor, Tensor]:
        """이번 step의 착용자 (Kp 배율, Kd 배율) (n, d)."""
        return self._kp_scale, self._kd_scale

    def in_stance_event(self) -> Tensor:
        """진단용: 입각 사건 진행 중인 (n, d) 마스크."""
        t = self._episode_time().unsqueeze(1)
        return (t >= self._ev_start) & (t <= self._ev_end)
