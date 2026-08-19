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
"""AI-SBC LLP용 의도 궤적(θ_g) 생성 + HLP 예측 chunk 에뮬레이션 컴포넌트.

역할 (AssistEnv와 한 쌍):
  1. **의도 궤적 θ_g(t)**: 사람(내부 PD)이 추종하는 관절 목표. 보행 유사
     sine 기반이지만 정형화 방지를 위해 세그먼트 단위로 파라미터(케이던스·
     진폭·오프셋·위상·고조파)를 재샘플링하고 smoothstep으로 블렌딩하며,
     hold(정지) 세그먼트와 좌우 비상관 모드를 섞는다. 사람 PD는 항상
     이 "깨끗한" θ_g를 추종한다 (사람은 자기 의도를 정확히 앎).
  2. **HLP chunk 에뮬레이션**: 저주파(chunk_refresh_steps)마다 θ_g에서
     미래 knot K개를 떠서 노이즈(바이어스+백색)와 지연을 주입한 chunk를
     만들고, policy step마다 knot 사이를 선형 interpolation한 미래 목표
     창(theta_d_future)을 관측으로 제공한다. 학습 시점에 HLP(ManiFlow)
     예측 오차에 대한 강건성을 확보하는 장치다.
  3. **계측 버퍼**: AssistEnv가 기록하는 사람 토크(tau_agent), 인가된
     assist 토크(tau_assist), env별 사람 게인 배율(gain_scale)을 컨텍스트로
     노출한다 (critic 전용 privileged 관측).

시간축은 에피소드 시간 t = progress_buf * dt (리셋 시 0). 세그먼트/chunk
상태는 모두 (num_envs, ...) 텐서로 완전히 벡터화되어 있다.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import torch
from torch import Tensor

from protomotions.envs.control.base import ControlComponent, ControlComponentConfig
from protomotions.envs.context_views import AssistContext

if TYPE_CHECKING:
    from protomotions.envs.base_env.env import BaseEnv
    from protomotions.envs.context_views import EnvContext
    from protomotions.simulator.base_simulator.config import (
        MarkerState,
        VisualizationMarkerConfig,
    )

_SEGMENT_PARAM_NAMES = ("freq", "amp", "phase", "amp2", "phase2", "offset")


def _smoothstep(x: Tensor) -> Tensor:
    x = x.clamp(0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


@dataclass
class AssistTargetControlConfig(ControlComponentConfig):
    """AssistTargetControl 설정.

    Attributes:
        hlp_dt: chunk knot 간격 (s). HLP 저주파 주기 (예: 0.1 = 10 Hz).
        chunk_knots: chunk가 담는 미래 knot 수 K. K*hlp_dt가
            (chunk 최대 age + 관측 미래 창)보다 길어야 interpolation이
            잘리지 않는다.
        chunk_refresh_steps: chunk 재생성 주기 (policy step 단위).
        chunk_noise_std: knot별 백색 노이즈 std (rad).
        chunk_bias_std: chunk 단위 바이어스 노이즈 std (rad).
        chunk_delay_max: HLP 예측 지연 에뮬레이션 상한 (s). chunk가 이만큼
            과거 기준으로 계산된 것처럼 시간축을 밀어 예측 오차를 만든다.
        obs_future_samples: 관측 미래 창 샘플 수.
        obs_future_dt: 관측 미래 창 샘플 간격 (s).
        freq_range: 궤적 기본 주파수 범위 (Hz). 보행 케이던스 수준.
        amp_range: 진폭 범위 (rad).
        offset_range: 중립 오프셋 범위 (rad).
        harmonic_amp_frac_max: 2차 고조파 진폭의 기본 진폭 대비 최대 비율.
        antiphase_jitter: 좌우 위상차 π 주변 지터 (rad).
        uncorrelated_prob: 좌우 다리를 완전히 독립으로 샘플링할 확률.
        hold_prob: 세그먼트가 정지(hold)일 확률.
        segment_gap_range: 세그먼트 유지 시간 범위 (s) — 블렌드 완료 후
            다음 블렌드 시작까지.
        blend_range: 세그먼트 전환 블렌딩 시간 범위 (s).
        start_blend: 에피소드 시작 시 기본 자세 -> 궤적 블렌딩 시간 (s).
        dof_limit_margin: 목표를 관절 한계의 이 비율 안으로 clamp.
    """

    _target_: str = "protomotions.envs.control.assist_target_control.AssistTargetControl"

    # HLP chunk 에뮬레이션
    hlp_dt: float = 0.1
    chunk_knots: int = 8
    chunk_refresh_steps: int = 10
    chunk_noise_std: float = 0.02
    chunk_bias_std: float = 0.03
    chunk_delay_max: float = 0.05

    # 관측 미래 창
    obs_future_samples: int = 5
    obs_future_dt: float = 0.1

    # 의도 궤적 생성기
    freq_range: Tuple[float, float] = (0.4, 2.0)
    amp_range: Tuple[float, float] = (0.1, 0.6)
    offset_range: Tuple[float, float] = (-0.2, 0.3)
    harmonic_amp_frac_max: float = 0.3
    antiphase_jitter: float = 0.6
    uncorrelated_prob: float = 0.25
    hold_prob: float = 0.15
    segment_gap_range: Tuple[float, float] = (1.5, 4.0)
    blend_range: Tuple[float, float] = (0.4, 1.2)
    start_blend: float = 0.6
    dof_limit_margin: float = 0.95

    # 시각화: θ_g 방향 다리 끝점 마커 (viewer 전용). 0이면 비활성.
    marker_leg_length: float = 0.42


class AssistTargetControl(ControlComponent):
    """의도 궤적 θ_g와 HLP chunk를 관리하는 control component."""

    def __init__(self, config: AssistTargetControlConfig, env: "BaseEnv"):
        super().__init__(config, env)
        self.config: AssistTargetControlConfig = config

        n = env.num_envs
        d = env.robot_config.number_of_actions
        device = env.device
        self.n_dofs = d

        # 세그먼트 파라미터 2세트: 슬롯 0 = 현재(cur), 슬롯 1 = 다음(nxt)
        self._params = {
            name: torch.zeros(2, n, d, device=device) for name in _SEGMENT_PARAM_NAMES
        }
        # 블렌드 상태: seg_start에서 blend_dur 동안 cur -> nxt 전환
        self._seg_start = torch.zeros(n, device=device)
        self._blend_dur = torch.ones(n, device=device)

        # HLP chunk 버퍼
        k = config.chunk_knots
        self._chunk = torch.zeros(n, k, d, device=device)
        self._chunk_t0 = torch.zeros(n, device=device)

        # AssistEnv가 기록하는 계측 버퍼
        self._tau_agent = torch.zeros(n, d, device=device)
        self._tau_assist = torch.zeros(n, d, device=device)
        self._gain_scale = torch.ones(n, d, device=device)

        # 목표 clamp 범위 (관절 한계 * margin)
        kin = env.robot_config.kinematic_info
        self._target_low = (
            kin.dof_limits_lower.to(device=device, dtype=torch.float32)
            * config.dof_limit_margin
        )
        self._target_high = (
            kin.dof_limits_upper.to(device=device, dtype=torch.float32)
            * config.dof_limit_margin
        )
        self._default_dof_pos = env.robot_config.default_dof_pos.to(
            device=device, dtype=torch.float32
        )

        # 전체 env 초기 시딩 (env.reset 전에 populate_context가 불려도 동작)
        self.reset(torch.arange(n, device=device, dtype=torch.long))

    # ------------------------------------------------------------------
    # 시간/궤적 유틸
    # ------------------------------------------------------------------
    def _episode_time(self) -> Tensor:
        """에피소드 경과 시간 (num_envs,) [s]."""
        return self.env.progress_buf.float() * self.env.dt

    def _uniform(self, lo: float, hi: float, *shape) -> Tensor:
        return lo + (hi - lo) * torch.rand(*shape, device=self.env.device)

    def _eval_segment(self, slot: int, t: Tensor, env_ids: Optional[Tensor]) -> Tensor:
        """슬롯의 세그먼트 궤적 평가.

        Args:
            slot: 0(cur) 또는 1(nxt).
            t: 에피소드 시간 (n, K).
            env_ids: 대상 env 인덱스 (n,) 또는 None(전체).

        Returns:
            (n, K, d) 관절 목표.
        """
        p = {}
        for name in _SEGMENT_PARAM_NAMES:
            v = self._params[name][slot]
            p[name] = v[env_ids] if env_ids is not None else v
        tt = t.unsqueeze(-1)  # (n, K, 1)
        two_pi = 2.0 * math.pi
        theta = (
            p["offset"].unsqueeze(1)
            + p["amp"].unsqueeze(1)
            * torch.sin(two_pi * p["freq"].unsqueeze(1) * tt + p["phase"].unsqueeze(1))
            + p["amp2"].unsqueeze(1)
            * torch.sin(
                2.0 * two_pi * p["freq"].unsqueeze(1) * tt + p["phase2"].unsqueeze(1)
            )
        )
        return theta

    def _target_at(self, t: Tensor, env_ids: Optional[Tensor] = None) -> Tensor:
        """cur/nxt 세그먼트를 블렌딩한 의도 궤적 θ_g.

        Args:
            t: 에피소드 시간 (n, K).
            env_ids: 대상 env 인덱스 (n,) 또는 None(전체).

        Returns:
            (n, K, d) 관절 목표 (관절 한계 내로 clamp).
        """
        seg_start = self._seg_start[env_ids] if env_ids is not None else self._seg_start
        blend_dur = self._blend_dur[env_ids] if env_ids is not None else self._blend_dur
        s = _smoothstep((t - seg_start.unsqueeze(1)) / blend_dur.unsqueeze(1))
        a = self._eval_segment(0, t, env_ids)
        b = self._eval_segment(1, t, env_ids)
        theta = a + (b - a) * s.unsqueeze(-1)
        return theta.clamp(self._target_low, self._target_high)

    def get_pd_targets(self) -> Tensor:
        """사람 PD가 추종할 현재 목표 θ_g(t) (num_envs, num_dofs), COMMON ordering."""
        t = self._episode_time()
        return self._target_at(t.unsqueeze(1)).squeeze(1)

    def get_pd_targets_and_vel(self) -> Tuple[Tensor, Tensor]:
        """목표 θ_g(t)와 목표 속도 θ̇_g(t) (중앙 차분).

        사람 모델의 속도 피드포워드 항 Kd·θ̇_g 계산용. 궤적이 해석적이므로
        ±dt/2 중앙 차분으로 충분히 정확하다.

        Returns:
            (theta_g, theta_g_dot) — 각각 (num_envs, num_dofs).
        """
        t = self._episode_time()
        delta = 0.5 * self.env.dt
        theta = self._target_at(t.unsqueeze(1)).squeeze(1)
        t_lo = (t - delta).clamp(min=0.0)
        t_hi = t + delta
        th_lo = self._target_at(t_lo.unsqueeze(1)).squeeze(1)
        th_hi = self._target_at(t_hi.unsqueeze(1)).squeeze(1)
        vel = (th_hi - th_lo) / (t_hi - t_lo).unsqueeze(1)
        return theta, vel

    # ------------------------------------------------------------------
    # 세그먼트 샘플링
    # ------------------------------------------------------------------
    def _sample_segment(self, slot: int, env_ids: Tensor, allow_hold: bool = True):
        """세그먼트 파라미터 재샘플링 (보행 유사 + 다양화).

        좌우 다리는 기본적으로 케이던스를 공유하고 위상차 π±jitter의
        antiphase 관계지만, uncorrelated_prob 확률로 완전 독립 샘플링해
        정형화된 좌우 상관 관계에 과적합되는 것을 막는다.
        """
        cfg = self.config
        n = len(env_ids)
        d = self.n_dofs
        dev = self.env.device
        two_pi = 2.0 * math.pi

        # 주파수: 공유 케이던스 vs 다리별 독립
        freq_shared = self._uniform(*cfg.freq_range, n, 1).expand(n, d)
        freq_indep = self._uniform(*cfg.freq_range, n, d)
        uncorr = torch.rand(n, 1, device=dev) < cfg.uncorrelated_prob
        freq = torch.where(uncorr, freq_indep, freq_shared)

        amp = self._uniform(*cfg.amp_range, n, d)
        offset = self._uniform(*cfg.offset_range, n, d)

        # 위상: [기본 위상 + antiphase 패턴 + 지터] vs 완전 랜덤
        base_phase = self._uniform(0.0, two_pi, n, 1)
        antiphase = torch.zeros(n, d, device=dev)
        if d >= 2:
            antiphase[:, 1::2] = math.pi  # DOF ordering: [right, left, ...]
        jitter = self._uniform(-cfg.antiphase_jitter, cfg.antiphase_jitter, n, d)
        phase_corr = base_phase + antiphase + jitter
        phase_rand = self._uniform(0.0, two_pi, n, d)
        phase = torch.where(uncorr, phase_rand, phase_corr)

        amp2 = amp * self._uniform(0.0, cfg.harmonic_amp_frac_max, n, d)
        phase2 = self._uniform(0.0, two_pi, n, d)

        if allow_hold:
            hold = torch.rand(n, 1, device=dev) < cfg.hold_prob
            amp = torch.where(hold, torch.zeros_like(amp), amp)
            amp2 = torch.where(hold, torch.zeros_like(amp2), amp2)

        new_values = {
            "freq": freq,
            "amp": amp,
            "phase": phase,
            "amp2": amp2,
            "phase2": phase2,
            "offset": offset,
        }
        for name, value in new_values.items():
            self._params[name][slot, env_ids] = value

    # ------------------------------------------------------------------
    # HLP chunk
    # ------------------------------------------------------------------
    def _refresh_chunk(self, env_ids: Tensor, t_now: Tensor):
        """HLP 예측 chunk 재생성 (노이즈 + 지연 주입).

        knot k의 값은 θ_g(t_now - delay + k*hlp_dt)로 계산하되 chunk의
        시간축 원점은 t_now로 기록한다 — delay만큼 시간축이 밀린 예측
        오차(HLP 추론 지연)를 에뮬레이션하는 것.
        """
        cfg = self.config
        n = len(env_ids)
        k = cfg.chunk_knots
        dev = self.env.device

        delay = torch.rand(n, device=dev) * cfg.chunk_delay_max
        base = t_now - delay
        knot_offsets = torch.arange(k, device=dev, dtype=torch.float32) * cfg.hlp_dt
        tt = base.unsqueeze(1) + knot_offsets.unsqueeze(0)  # (n, K)

        knots = self._target_at(tt, env_ids)  # (n, K, d)
        bias = torch.randn(n, 1, self.n_dofs, device=dev) * cfg.chunk_bias_std
        white = torch.randn(n, k, self.n_dofs, device=dev) * cfg.chunk_noise_std
        knots = (knots + bias + white).clamp(self._target_low, self._target_high)

        self._chunk[env_ids] = knots
        self._chunk_t0[env_ids] = t_now

    def _interp_chunk(self, tt: Tensor) -> Tensor:
        """chunk knot 선형 interpolation (전체 env).

        Args:
            tt: 조회 시각 (num_envs, S) [s].

        Returns:
            (num_envs, S, d) interpolation된 목표. chunk 범위 밖은 양 끝
            knot 값으로 clamp (ZOH).
        """
        cfg = self.config
        k = cfg.chunk_knots
        rel = (tt - self._chunk_t0.unsqueeze(1)) / cfg.hlp_dt  # knot 단위
        rel = rel.clamp(0.0, float(k - 1))
        idx0 = rel.floor().long().clamp(max=k - 2)
        w = (rel - idx0.float()).clamp(0.0, 1.0).unsqueeze(-1)  # (n, S, 1)

        idx0e = idx0.unsqueeze(-1).expand(-1, -1, self.n_dofs)
        knot0 = torch.gather(self._chunk, 1, idx0e)
        knot1 = torch.gather(self._chunk, 1, idx0e + 1)
        return knot0 * (1.0 - w) + knot1 * w

    # ------------------------------------------------------------------
    # ControlComponent 인터페이스
    # ------------------------------------------------------------------
    def reset(self, env_ids: Tensor):
        """에피소드 시작: 기본 자세 hold -> 새 궤적으로 start_blend 블렌딩."""
        if len(env_ids) == 0:
            return
        cfg = self.config

        # cur 슬롯 = 기본 자세 hold (진폭 0, offset = 기본 관절 각)
        for name in _SEGMENT_PARAM_NAMES:
            self._params[name][0, env_ids] = 0.0
        self._params["offset"][0, env_ids] = self._default_dof_pos.unsqueeze(0)
        # freq=0 나눗셈은 없지만 (sin 인자만 사용) 안전하게 유효값 유지
        self._params["freq"][0, env_ids] = cfg.freq_range[0]

        # nxt 슬롯 = 실제 궤적, t=0부터 start_blend 동안 블렌드 인
        self._sample_segment(1, env_ids, allow_hold=False)
        self._seg_start[env_ids] = 0.0
        self._blend_dur[env_ids] = cfg.start_blend

        # chunk/계측 초기화
        t0 = torch.zeros(len(env_ids), device=self.env.device)
        self._refresh_chunk(env_ids, t0)
        self._tau_agent[env_ids] = 0.0
        self._tau_assist[env_ids] = 0.0

    def step(self):
        """세그먼트 승격(블렌드 완료 시 재샘플링) + 저주파 chunk 갱신."""
        cfg = self.config
        t = self._episode_time()

        # 1) 블렌드가 끝난 env: nxt -> cur 승격, 새 nxt 샘플링
        done = t >= (self._seg_start + self._blend_dur)
        env_ids = done.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            for v in self._params.values():
                v[0, env_ids] = v[1, env_ids]
            self._sample_segment(1, env_ids, allow_hold=True)
            gap = self._uniform(*cfg.segment_gap_range, len(env_ids))
            self._seg_start[env_ids] = t[env_ids] + gap
            self._blend_dur[env_ids] = self._uniform(*cfg.blend_range, len(env_ids))

        # 2) HLP 저주파 chunk 갱신
        refresh = (self.env.progress_buf % cfg.chunk_refresh_steps) == 0
        env_ids = refresh.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self._refresh_chunk(env_ids, t[env_ids])

    def create_visualization_markers(
        self, headless: bool
    ) -> Dict[str, "VisualizationMarkerConfig"]:
        """θ_g 목표 방향 다리 끝점을 녹색 구로 표시 (DOF당 1개)."""
        if headless or self.config.marker_leg_length <= 0:
            return {}
        from protomotions.simulator.base_simulator.config import (
            MarkerConfig,
            VisualizationMarkerConfig,
        )

        return {
            "assist_target_markers": VisualizationMarkerConfig(
                type="sphere",
                color=(0.1, 0.85, 0.2),
                markers=[MarkerConfig(size="small") for _ in range(self.n_dofs)],
            ),
        }

    def get_markers_state(self) -> Dict[str, "MarkerState"]:
        """마커 위치 = 대퇴 피벗 + R_y(θ_g)로 회전시킨 다리 끝점.

        pelvis가 월드에 고정(회전 identity)이고 hinge 축이 +Y라는 가정
        (hip_pendulum 로봇 전용 시각화).
        """
        if self.env.simulator.headless or self.config.marker_leg_length <= 0:
            return {}
        from protomotions.simulator.base_simulator.config import MarkerState

        theta_g = self.get_pd_targets()  # (n, d)
        body_pos = self.env.simulator.get_robot_state().rigid_body_pos
        # DOF i의 대퇴 body는 kinematic 순서상 root 다음 (i+1)
        pivot = body_pos[:, 1 : 1 + self.n_dofs, :]  # (n, d, 3)

        length = self.config.marker_leg_length
        offset = torch.zeros_like(pivot)
        offset[..., 0] = -length * torch.sin(theta_g)
        offset[..., 2] = -length * torch.cos(theta_g)
        translation = pivot + offset

        orientation = torch.zeros(
            self.env.num_envs, self.n_dofs, 4, device=self.env.device
        )
        orientation[..., 3] = 1.0  # identity (xyzw)

        return {
            "assist_target_markers": MarkerState(
                translation=translation,
                orientation=orientation,
            ),
        }

    def populate_context(self, ctx: "EnvContext") -> None:
        cfg = self.config
        t = self._episode_time()

        obs_offsets = (
            torch.arange(cfg.obs_future_samples, device=self.env.device)
            * cfg.obs_future_dt
        )
        tt = t.unsqueeze(1) + obs_offsets.unsqueeze(0)  # (n, S)
        theta_d_future = self._interp_chunk(tt)

        ctx.assist = AssistContext(
            theta_d=theta_d_future[:, 0],
            theta_d_future=theta_d_future,
            chunk_age=(t - self._chunk_t0).unsqueeze(1),
            theta_g=self._target_at(t.unsqueeze(1)).squeeze(1),
            tau_agent=self._tau_agent,
            tau_assist=self._tau_assist,
            gain_scale=self._gain_scale,
        )
