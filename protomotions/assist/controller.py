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
"""HLP(ManiFlow flexion 예측) + LLP(assist torque 정책)를 묶은 독립 제어기.

ProtoMotions env에 의존하지 않는다. 입력은 실기기에서 측정 가능한 값뿐이다:

    hip_pos  (N, 2) 고관절 flexion 각 [rad], 관절 규약 (오른쪽, 왼쪽)
    hip_vel  (N, 2) 각속도 [rad/s]
    trunk    (N, 2) 몸통 pitch/roll [rad] (앞숙임 +, 오른쪽 기울임 +) — 없으면 0

출력은 같은 관절 규약의 assist 토크 (N, 2) [N·m]. 호출 주기는 LLP 학습 주기
(10 ms)여야 한다.

내부 동작 (LLP 학습 env `AssistEnv` + `MotionAssistTargetControl`과 1:1 대응):

1. `observe(s_k)` — 관절 상태 히스토리 push, HLP 링 버퍼 push,
   `chunk_refresh_steps`마다 HLP 예측으로 chunk(knot K개 × 2, 간격 25 ms) 갱신
2. `act()` — chunk를 선형 interpolation한 미래 목표 창(S개 × obs_future_dt),
   관절 히스토리, 과거 raw action으로 관측을 조립 → LLP actor → tanh × scale
   → clamp → 토크
3. `reset(env_ids, s_0)` — 히스토리를 현재 상태로 채우고(edge padding) chunk를
   t=0에서 1회 예측

부호 규약: 데이터/HLP 규약은 +θ = 굴곡. LLP 체크포인트마다 "데이터 flexion →
LLP 관절각" 부호 `flexion_sign`이 config에 있다 (hip_pendulum −1: +θ = 신전,
skeleton 학습 +1). `joint_sign`은 실제 관절각 → 데이터 flexion 부호(skeleton +1,
hip_pendulum −1). 내부 LLP 규약으로는 `θ_llp = flexion_sign·joint_sign·θ_joint`,
`τ_joint = flexion_sign·joint_sign·τ_llp`로 양방향 변환한다.

등가성은 `python -m protomotions.assist.verify_controller`로 진자 env와 step
단위로 대조한다.
"""

import logging
import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
from torch import Tensor

from protomotions.assist.llp_policy import LLPObsLayout, LLPPolicy
from protomotions.maniflow.angle_estimator import ManiFlowAngleEstimator
from protomotions.maniflow.chunk_source import DEFAULT_DENOISE_STEPS, ManiFlowChunkSource

logger = logging.getLogger(__name__)


@dataclass
class AssistControllerConfig:
    """`AssistController` 설정.

    Attributes:
        llp_checkpoint: `assist_pendulum` 체크포인트 (`last.ckpt`).
        hlp_checkpoint: ManiFlow flexion 체크포인트. None이면 flexion40 run에서
            자동 탐색 (`verify_angle_estimator.FLEXION40_RUN_DIR`).
        hlp_denoise_steps: ManiFlow denoise 스텝 (권장 3).
        hlp_obs_offset: HLP 관측 4ch 영점 오프셋 [rad] — 입력 스트림을 HLP 학습
            (SEED) 규약으로 정렬. 관측에 가산되고, 앞 2ch은 예측에서 감산된다.
        joint_sign: 관절각 → 데이터 flexion(+굴곡) 부호. skeleton +1,
            hip_pendulum −1.
        dt: 호출 주기 [s]. None이면 LLP 학습 dt.
        torque_limit: assist 토크 한계 [N·m]. None이면 LLP 학습값.
        torque_lpf_hz: 출력 1차 저역통과 차단주파수 [Hz]. 0이면 끔(기본).
            학습된 정책은 매 10 ms마다 토크를 크게 바꿀 수 있는데, 실제 구동기는
            그 대역을 못 내고 착용자도 진동으로 느낀다. 여기서 걸면 시뮬과 실기기가
            같은 필터를 쓴다.
        torque_slew_limit: 출력 변화율 한계 [N·m/s]. None이면 끔. 구동기의
            토크 상승률 한계를 그대로 모사한다 (저역통과 뒤에 적용).
        measure_timing: True면 HLP 예측·LLP 추론 시간을 기록 (CUDA sync 포함).
    """

    llp_checkpoint: str
    hlp_checkpoint: Optional[str] = None
    hlp_denoise_steps: int = DEFAULT_DENOISE_STEPS
    hlp_obs_offset: Optional[Sequence[float]] = None
    joint_sign: float = 1.0
    dt: Optional[float] = None
    torque_limit: Optional[float] = None
    torque_lpf_hz: float = 0.0
    torque_slew_limit: Optional[float] = None
    measure_timing: bool = False


class AssistController:
    """HLP + LLP 결합 assist torque 제어기 (N개 병렬 인스턴스)."""

    def __init__(
        self,
        config: AssistControllerConfig,
        num_envs: int,
        device: torch.device,
    ):
        self.config = config
        self.num_envs = int(num_envs)
        self.device = torch.device(device)

        # --- LLP ---
        self.llp = LLPPolicy(config.llp_checkpoint, self.device)
        lay: LLPObsLayout = self.llp.layout
        self.layout = lay
        self.dt = float(config.dt) if config.dt is not None else lay.dt
        if abs(self.dt - lay.dt) > 1e-9:
            logger.warning(
                "AssistController dt=%.4f ≠ LLP 학습 dt=%.4f — 히스토리 시간 창이 달라진다",
                self.dt,
                lay.dt,
            )
        self.torque_limit = (
            float(config.torque_limit) if config.torque_limit is not None else lay.torque_limit
        )

        # --- HLP ---
        hlp_ckpt = config.hlp_checkpoint
        if not hlp_ckpt:
            from protomotions.maniflow.loader import discover_best_checkpoint
            from protomotions.maniflow.verify_angle_estimator import FLEXION40_RUN_DIR

            hlp_ckpt = str(discover_best_checkpoint(FLEXION40_RUN_DIR))
        self.hlp_checkpoint = hlp_ckpt
        estimator = ManiFlowAngleEstimator.from_checkpoint(
            hlp_ckpt,
            num_envs=self.num_envs,
            device=self.device,
            denoise_steps=int(config.hlp_denoise_steps),
        )
        if config.hlp_obs_offset is not None:
            off = torch.as_tensor(config.hlp_obs_offset, dtype=torch.float32)
            assert off.shape == (4,), f"hlp_obs_offset must have 4 channels, got {tuple(off.shape)}"
            estimator.set_calibration(obs_offset=off, output_bias=off[:2])
        self.hlp = ManiFlowChunkSource(
            estimator,
            num_envs=self.num_envs,
            dt=self.dt,
            device=self.device,
            flexion_sign=lay.flexion_sign,  # HLP(데이터 규약) knot → LLP 관절각 규약
        )
        # chunk 격자는 모델이 결정한다 (MotionAssistTargetControl과 동일 규칙)
        self.hlp_dt = float(self.hlp.knot_dt)
        self.num_knots = int(self.hlp.num_knots)
        if self.hlp_dt != lay.hlp_dt or self.num_knots != lay.chunk_knots:
            logger.warning(
                "HLP chunk 격자(%.3f s × %d)가 LLP 학습 격자(%.3f s × %d)와 다르다",
                self.hlp_dt,
                self.num_knots,
                lay.hlp_dt,
                lay.chunk_knots,
            )

        # --- 상태 버퍼 (LLP 규약) ---
        n, d = self.num_envs, lay.num_dofs
        # θ_llp = flexion_sign·joint_sign·θ_joint (부호 곱은 자기 역원 → 토크 역변환에도 사용)
        self._joint_to_llp = float(lay.flexion_sign) * float(config.joint_sign)
        self._q = torch.zeros(n, lay.history_steps + 1, d, device=self.device)
        self._qd = torch.zeros(n, lay.history_steps + 1, d, device=self.device)
        self._a = torch.zeros(n, lay.prev_action_steps + 1, d, device=self.device)
        self._chunk = torch.zeros(n, self.num_knots, d, device=self.device)
        self._chunk_t0 = torch.zeros(n, device=self.device)
        self._k = torch.zeros(n, dtype=torch.long, device=self.device)
        # 마지막 act()의 raw action — 다음 observe()에서 히스토리에 push된다
        # (env: step 중 적용된 action을 post-physics에서 push하는 것과 동일)
        self._pending_raw = torch.zeros(n, d, device=self.device)
        # 출력 필터 상태 (LLP 규약). act()는 멱등이어야 하므로 갱신은 observe()에서.
        self._tau_prev = torch.zeros(n, d, device=self.device)
        self._pending_tau = torch.zeros(n, d, device=self.device)
        lpf_hz = float(config.torque_lpf_hz or 0.0)
        # 1차 저역통과의 이산 계수: α = dt / (dt + 1/(2πf))
        self._lpf_alpha = (
            self.dt / (self.dt + 1.0 / (2.0 * math.pi * lpf_hz)) if lpf_hz > 0.0 else 1.0
        )
        self._slew_step = (
            float(config.torque_slew_limit) * self.dt
            if config.torque_slew_limit is not None
            else None
        )
        self._future_offsets = (
            torch.arange(lay.obs_future_samples, device=self.device, dtype=torch.float32)
            * lay.obs_future_dt
        )
        self._all_ids = torch.arange(n, device=self.device, dtype=torch.long)

        # 검증용: True면 내부 HLP 갱신을 건너뛰고 set_chunk()로 받은 chunk를 쓴다
        self.external_chunk = False

        # 진단 (마지막 act 기준, LLP 규약)
        self.theta_d_future: Optional[Tensor] = None
        self.chunk_age: Optional[Tensor] = None
        self.last_raw_action: Optional[Tensor] = None
        self.last_tau_llp: Optional[Tensor] = None
        self.hlp_ms: list = []
        self.llp_ms: list = []

        logger.info(
            "AssistController: LLP %s (epoch %d, dt %.3f s, limit %.1f N·m, flexion_sign %+.0f, "
            "dofs %s) + HLP %s (denoise %d, knots %d × %.3f s)",
            config.llp_checkpoint,
            self.llp.epoch,
            self.dt,
            self.torque_limit,
            lay.flexion_sign,
            lay.assist_dof_names or "all",
            hlp_ckpt,
            config.hlp_denoise_steps,
            self.num_knots,
            self.hlp_dt,
        )

    # ------------------------------------------------------------------
    # 규약 변환
    # ------------------------------------------------------------------
    def _to_llp(self, x: Tensor) -> Tensor:
        return x.to(device=self.device, dtype=torch.float32) * self._joint_to_llp

    def _to_joint(self, x: Tensor) -> Tensor:
        return x * self._joint_to_llp  # 부호 곱은 자기 역원

    @staticmethod
    def _trunk(trunk: Optional[Tensor], n: int, device) -> Optional[Tensor]:
        if trunk is None:
            return None
        return trunk.to(device=device, dtype=torch.float32)[:, :2]

    # ------------------------------------------------------------------
    # chunk
    # ------------------------------------------------------------------
    def _refresh_chunk(self, env_ids: Tensor) -> None:
        t_now = self._k[env_ids].float() * self.dt
        if self.config.measure_timing:
            torch.cuda.synchronize(self.device) if self.device.type == "cuda" else None
            t_start = time.perf_counter()
        knots, t0 = self.hlp.predict_knots(env_ids, t_now)
        if self.config.measure_timing:
            torch.cuda.synchronize(self.device) if self.device.type == "cuda" else None
            self.hlp_ms.append((time.perf_counter() - t_start) * 1e3)
        lay = self.layout
        self._chunk[env_ids] = knots.clamp(lay.target_low, lay.target_high)
        self._chunk_t0[env_ids] = t0

    def set_chunk(self, chunk_llp: Tensor, t0: Tensor, env_ids: Optional[Tensor] = None) -> None:
        """외부 chunk 주입 (LLP 규약, (m, K, d)) — 검증/재생용."""
        ids = self._all_ids if env_ids is None else env_ids
        self._chunk[ids] = chunk_llp.to(self.device)
        self._chunk_t0[ids] = t0.to(self.device)

    def _interp_chunk(self, tt: Tensor) -> Tensor:
        """`AssistTargetControl._interp_chunk`와 동일한 선형 interpolation."""
        k = self.num_knots
        d = self.layout.num_dofs
        rel = (tt - self._chunk_t0.unsqueeze(1)) / self.hlp_dt
        rel = rel.clamp(0.0, float(k - 1))
        idx0 = rel.floor().long().clamp(max=k - 2)
        w = (rel - idx0.float()).clamp(0.0, 1.0).unsqueeze(-1)
        idx0e = idx0.unsqueeze(-1).expand(-1, -1, d)
        knot0 = torch.gather(self._chunk, 1, idx0e)
        knot1 = torch.gather(self._chunk, 1, idx0e + 1)
        return knot0 * (1.0 - w) + knot1 * w

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def reset(
        self,
        env_ids: Optional[Tensor],
        hip_pos: Tensor,
        hip_vel: Tensor,
        trunk: Optional[Tensor] = None,
    ) -> None:
        """에피소드 시작: 히스토리를 현재 상태로 채우고 chunk를 t=0에서 예측.

        Args:
            env_ids: 리셋할 인스턴스 (m,) 또는 None(전체).
            hip_pos / hip_vel: 리셋 직후 관절 상태 (m, 2), 관절 규약.
            trunk: 몸통 pitch/roll (m, 2) [rad] 또는 None.
        """
        ids = self._all_ids if env_ids is None else env_ids.to(self.device, dtype=torch.long)
        if ids.numel() == 0:
            return
        q = self._to_llp(hip_pos)
        qd = self._to_llp(hip_vel)
        self._q[ids] = q.unsqueeze(1).expand(-1, self._q.shape[1], -1)
        self._qd[ids] = qd.unsqueeze(1).expand(-1, self._qd.shape[1], -1)
        self._a[ids] = 0.0
        self._pending_raw[ids] = 0.0
        self._tau_prev[ids] = 0.0
        self._pending_tau[ids] = 0.0
        self._k[ids] = 0
        self.hlp.reset(ids, q, self._trunk(trunk, ids.numel(), self.device))
        if not self.external_chunk:
            self._refresh_chunk(ids)

    @torch.no_grad()
    def observe(self, hip_pos: Tensor, hip_vel: Tensor, trunk: Optional[Tensor] = None) -> None:
        """새 관측 s_k 반영 (매 dt). 전체 인스턴스 (N, 2)."""
        q = self._to_llp(hip_pos)
        qd = self._to_llp(hip_vel)
        self._k += 1
        self._q = self._q.roll(shifts=1, dims=1)
        self._qd = self._qd.roll(shifts=1, dims=1)
        self._q[:, 0] = q
        self._qd[:, 0] = qd
        self._a = self._a.roll(shifts=1, dims=1)
        self._a[:, 0] = self._pending_raw
        self._tau_prev = self._pending_tau
        self.hlp.observe(q, self._trunk(trunk, self.num_envs, self.device))
        if not self.external_chunk:
            refresh = (self._k % self.layout.chunk_refresh_steps) == 0
            ids = refresh.nonzero(as_tuple=False).flatten()
            if ids.numel() > 0:
                self._refresh_chunk(ids)

    @torch.no_grad()
    def act(self) -> Tensor:
        """현재 상태에서 assist 토크 계산 (N, 2), 관절 규약 [N·m].

        같은 상태에서 여러 번 불러도 결과가 같다 (히스토리 갱신은 observe에서).
        """
        lay = self.layout
        n = self.num_envs
        t = self._k.float() * self.dt
        tt = t.unsqueeze(1) + self._future_offsets.unsqueeze(0)  # (N, S)
        theta_d_future = self._interp_chunk(tt)  # (N, S, d)
        q = self._q[:, 0]
        qd = self._qd[:, 0]
        s = lay.dof_vel_scale
        obs: Dict[str, Tensor] = {
            LLPObsLayout.KEY_PROPRIO: torch.cat(
                [
                    q,
                    qd * s,
                    self._q[:, 1:].reshape(n, -1),
                    (self._qd[:, 1:] * s).reshape(n, -1),
                ],
                dim=-1,
            ),
            LLPObsLayout.KEY_TARGET: torch.cat(
                [
                    theta_d_future.reshape(n, -1),
                    (theta_d_future - q.unsqueeze(1)).reshape(n, -1),
                    (t - self._chunk_t0).unsqueeze(1),
                ],
                dim=-1,
            ),
            LLPObsLayout.KEY_PREV_ACTIONS: self._a[:, 1 : 1 + lay.prev_action_steps].reshape(n, -1),
        }
        if self.config.measure_timing:
            torch.cuda.synchronize(self.device) if self.device.type == "cuda" else None
            t_start = time.perf_counter()
        raw = self.llp.act(obs)
        tau_llp = torch.clamp(self.llp.to_torque(raw), -self.torque_limit, self.torque_limit)
        if self.config.measure_timing:
            torch.cuda.synchronize(self.device) if self.device.type == "cuda" else None
            self.llp_ms.append((time.perf_counter() - t_start) * 1e3)

        tau_llp = self._smooth(tau_llp)

        self._pending_raw = raw
        self._pending_tau = tau_llp
        self.theta_d_future = theta_d_future
        self.chunk_age = t - self._chunk_t0
        self.last_raw_action = raw
        self.last_tau_llp = tau_llp
        return self._to_joint(tau_llp)

    def _smooth(self, tau: Tensor) -> Tensor:
        """출력 저역통과 + 변화율 제한 (LLP 규약). 둘 다 꺼져 있으면 그대로 반환."""
        if self._lpf_alpha < 1.0:
            tau = self._tau_prev + self._lpf_alpha * (tau - self._tau_prev)
        if self._slew_step is not None:
            tau = self._tau_prev + (tau - self._tau_prev).clamp(-self._slew_step, self._slew_step)
        return tau

    @torch.no_grad()
    def step(self, hip_pos: Tensor, hip_vel: Tensor, trunk: Optional[Tensor] = None) -> Tensor:
        """`observe` + `act` (일반 제어 루프용)."""
        self.observe(hip_pos, hip_vel, trunk)
        return self.act()

    # ------------------------------------------------------------------
    # 진단 (관절 규약)
    # ------------------------------------------------------------------
    def target_future_joint(self) -> Optional[Tensor]:
        """마지막 act의 미래 목표 창 (N, S, 2), 관절 규약."""
        return None if self.theta_d_future is None else self._to_joint(self.theta_d_future)

    def chunk_joint(self) -> Tensor:
        """현재 chunk knot (N, K, 2), 관절 규약."""
        return self._to_joint(self._chunk)

    @property
    def episode_time(self) -> Tensor:
        return self._k.float() * self.dt
