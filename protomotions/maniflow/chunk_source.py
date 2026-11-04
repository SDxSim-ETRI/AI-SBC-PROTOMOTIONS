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
"""실 HLP(ManiFlow flexion 모델)를 LLP의 chunk 소스로 연결 — AI-SBC Phase C.

`AssistTargetControl._refresh_chunk`의 **에뮬레이션을 실제 예측으로 교체**하는
지점. 두 정책의 유일한 접점인 chunk 버퍼(knot K개 × DOF, 간격 hlp_dt,
기준시각 t0)를 그대로 채우므로 LLP 쪽 interpolation·관측 코드는 무변경이다.

해결해야 하는 시간축 불일치
---------------------------
- LLP policy 주기 = 10 ms (물리 400 Hz / decimation 4)
- HLP 모델 native 프레임 = ``estimator.frame_dt`` (H1: 10 ms; 지운 40 fps 판은 25 ms라 정수배가 아니었다)
- LLP의 관절 히스토리 버퍼와 HLP 관측 창(H1: 10프레임 = 100 ms)의 길이가 다르다

따라서 이 모듈이 **자체 링 버퍼**(policy 주기, 320 ms)를 들고, 예측 시각마다
모델 프레임 격자로 선형 재표집해 모델 창을 만든다. dt가 모델 프레임 간격과 같으면
재표집이 프레임 정확 복원이 된다(검증 하네스가 이용).

부호·채널 규약
--------------
모델 관측은 `[hip_flexion_r, hip_flexion_l, trunk_pitch, trunk_roll]`(rad,
flexion + = 굴곡)이고 hip_pendulum 로봇은 +θ = 신전이므로 `flexion_sign`
(기본 −1)로 양방향 변환한다. trunk 2ch은 로봇 관절이 아니므로 의도 소스가
제공하는 GT를 그대로 쓴다(진자에는 몸통이 없다).

미래 범위: 현 체크포인트는 **현재 + 미래 6프레임 = 150 ms**뿐이다
(`angle_estimator` 모듈 docstring의 horizon 정렬 참고). knot 수·간격을
모델에서 유도하므로 더 긴 horizon으로 재학습해도 코드는 무변경이다.
"""

import logging
from typing import Optional, Tuple, TYPE_CHECKING

import torch
from torch import Tensor

from protomotions.maniflow.angle_estimator import (
    FLEXION_FRAME_DT,
    ManiFlowAngleEstimator,
)

if TYPE_CHECKING:
    from protomotions.envs.base_env.env import BaseEnv
    from protomotions.envs.control.assist_motion_control import (
        MotionAssistTargetControlConfig,
    )

logger = logging.getLogger(__name__)

# 링 버퍼 길이 [s] — 모델 관측 창(H1: 100 ms)보다 넉넉하게
RING_SECONDS = 0.32
# denoise 스텝 (2026-09-14 실측으로 3 → 1). suit14 교차평가에서 정확도는 스텝 수와
# 무관했고(lead 25 ms MAE 0.47/0.45/0.45/0.47° @ 1/2/3/6스텝) 지연만 스텝당 ~3 ms씩
# 늘었다 (batch-1 median 4.94 / 8.13 / 11.04 ms).
DEFAULT_DENOISE_STEPS = 1


class ManiFlowChunkSource:
    """실 HLP 예측으로 LLP chunk knot을 생성하는 소스.

    호출 순서 (`AssistTargetControl`에서):

    1. `reset(env_ids, dof_pos, trunk)` — 리셋된 env의 링 버퍼를 현재
       관측으로 채우고 estimator history를 무효화.
    2. `observe(dof_pos, trunk)` — **매 policy step** 링 버퍼에 push.
    3. `predict_knots(env_ids, t_now)` — chunk 갱신 시각에 호출.
    """

    def __init__(
        self,
        estimator: ManiFlowAngleEstimator,
        num_envs: int,
        dt: float,
        device: torch.device,
        *,
        flexion_sign: float = -1.0,
        ring_seconds: float = RING_SECONDS,
    ):
        self.estimator = estimator
        self.num_envs = int(num_envs)
        self.dt = float(dt)
        self.device = torch.device(device)
        self.flexion_sign = float(flexion_sign)

        self.knot_dt = float(getattr(estimator, "frame_dt", FLEXION_FRAME_DT))
        self.n_obs = estimator.n_obs_steps
        self.num_knots = estimator.num_future_steps
        self.obs_dim = estimator.obs_dim

        # 링 길이: 모델 창 전체(=(n_obs−1)·knot_dt) 보간에 필요한 샘플 + 여유
        needed = int((self.n_obs - 1) * self.knot_dt / self.dt) + 2
        self.ring_len = max(needed, int(round(ring_seconds / self.dt)))
        self._ring = torch.zeros(
            self.num_envs, self.ring_len, self.obs_dim, device=self.device
        )
        self.last_pred: Optional[Tensor] = None

        # 모델 프레임 격자 재표집 계수 (창 끝 = 현재 시각)
        lags = (
            torch.arange(self.n_obs - 1, -1, -1, device=self.device, dtype=torch.float32)
            * self.knot_dt
            / self.dt
        )  # (n_obs,) — 몇 "policy step" 전인지 (실수)
        floor = lags.floor()
        self._src_hi = (self.ring_len - 1 - floor.long()).clamp(min=0)
        self._src_lo = (self.ring_len - 2 - floor.long()).clamp(min=0)
        self._src_w = (lags - floor).view(1, self.n_obs, 1)

        logger.info(
            "ManiFlowChunkSource: knot_dt=%.3f knots=%d (미래 %.0f ms) "
            "n_obs=%d denoise=%d ring=%d steps(dt=%.4f)",
            self.knot_dt,
            self.num_knots,
            (self.num_knots - 1) * self.knot_dt * 1e3,
            self.n_obs,
            estimator.num_inference_steps,
            self.ring_len,
            self.dt,
        )

    # ------------------------------------------------------------------
    # 구성
    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls, config: "MotionAssistTargetControlConfig", env: "BaseEnv"
    ) -> "ManiFlowChunkSource":
        """`MotionAssistTargetControlConfig`에서 소스를 만든다 (체크포인트 생략 시 현역 H1)."""
        from protomotions.maniflow.loader import DEFAULT_H1_CHECKPOINT

        ckpt = config.maniflow_ckpt or str(DEFAULT_H1_CHECKPOINT)
        denoise = config.maniflow_denoise_steps or DEFAULT_DENOISE_STEPS
        estimator = ManiFlowAngleEstimator.from_checkpoint(
            ckpt,
            num_envs=env.num_envs,
            device=env.device,
            denoise_steps=int(denoise),
        )
        logger.info("HLP 체크포인트: %s", ckpt)
        return cls(
            estimator,
            num_envs=env.num_envs,
            dt=env.dt,
            device=env.device,
            flexion_sign=config.flexion_sign,
        )

    # ------------------------------------------------------------------
    # 관측
    # ------------------------------------------------------------------
    def _make_obs(self, dof_pos: Tensor, trunk: Optional[Tensor]) -> Tensor:
        """로봇 상태 → 모델 관측 4ch (데이터 규약). (n, 2) → (n, 4)."""
        flexion = dof_pos[:, :2] * self.flexion_sign
        if trunk is None:
            trunk = torch.zeros_like(flexion)
        return torch.cat([flexion, trunk[:, :2]], dim=-1)

    def reset(
        self, env_ids: Tensor, dof_pos: Tensor, trunk: Optional[Tensor] = None
    ) -> None:
        """리셋된 env: 링 버퍼를 현재 관측으로 채우고(edge padding) history 무효화."""
        if env_ids.numel() == 0:
            return
        obs = self._make_obs(dof_pos, trunk)  # (m, 4)
        self._ring[env_ids] = obs.unsqueeze(1).expand(-1, self.ring_len, -1)
        self.estimator.reset(env_ids)

    def observe(self, dof_pos: Tensor, trunk: Optional[Tensor] = None) -> None:
        """매 policy step 링 버퍼에 push (index −1 = 최신)."""
        obs = self._make_obs(dof_pos, trunk)
        self._ring = torch.cat([self._ring[:, 1:], obs.unsqueeze(1)], dim=1)

    # ------------------------------------------------------------------
    # 예측
    # ------------------------------------------------------------------
    def _resample_history(self, env_ids: Tensor) -> Tensor:
        """링 버퍼 → 모델 프레임 격자 창. (m, n_obs, 4)."""
        ring = self._ring[env_ids]  # (m, R, 4)
        hi = ring[:, self._src_hi]  # (m, n_obs, 4)
        lo = ring[:, self._src_lo]
        return hi + (lo - hi) * self._src_w

    def predict_knots(self, env_ids: Tensor, t_now: Tensor) -> Tuple[Tensor, Tensor]:
        """실 HLP 예측으로 chunk knot 생성.

        Args:
            env_ids: 갱신할 env (m,).
            t_now: 해당 env의 현재 에피소드 시각 (m,) [s].

        Returns:
            (knots, t0) — knots (m, num_knots, 2) **로봇 규약**, t0 = t_now.
            knot k의 시각은 t_now + k·knot_dt (k=0이 현재).
        """
        history = self._resample_history(env_ids)
        self.estimator.set_history(env_ids, history)
        pred = self.estimator.predict_horizon(env_ids)  # (m, horizon, 2)
        self.last_pred = pred
        knots = pred[:, self.n_obs - 1 :] * self.flexion_sign
        return knots, t_now
