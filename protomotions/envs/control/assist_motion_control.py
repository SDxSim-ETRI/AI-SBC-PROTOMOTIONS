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
"""mocap 의도 궤적 + 실 HLP chunk를 쓰는 AssistTargetControl 확장 — Phase C.

`AssistTargetControl`(합성 sine 의도 + chunk 에뮬레이션)을 **상속만 하고
원본은 건드리지 않는다**. 실험 파일에서 `_target_`을 이 클래스로 바꾸면
아래 두 축을 독립적으로 교체할 수 있다:

| 축 | 기본값 | 대안 |
|----|--------|------|
| 의도 궤적 θ_g | 합성 sine (부모 구현) | `intent_source="motion_data"` → mocap zarr |
| HLP chunk | 에뮬레이션 (부모 구현) | `chunk_source="maniflow"` → 실 ManiFlow 예측 |

`AssistTargetControlConfig`를 그대로 쓰면 부모와 100% 동일하게 동작하므로,
v2 재현과 v3 실험이 한 클래스에서 갈린다.

교체 지점은 부모의 훅 4개뿐이다:
  - `_target_at` — 의도 궤적의 유일한 소스 (임의 연속 시각으로 조회됨)
  - `_refresh_chunk` — chunk 생성
  - `reset` / `step` — 소스의 에피소드/스텝 bookkeeping
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, TYPE_CHECKING

import torch
from torch import Tensor

from protomotions.envs.control.assist_target_control import (
    AssistTargetControl,
    AssistTargetControlConfig,
)

if TYPE_CHECKING:
    from protomotions.envs.base_env.env import BaseEnv

logger = logging.getLogger(__name__)


@dataclass
class MotionAssistTargetControlConfig(AssistTargetControlConfig):
    """`AssistTargetControlConfig` + 의도/chunk 소스 선택.

    새 필드는 반드시 plain default만 쓴다 — 기존 체크포인트의
    `resolved_configs*.pt`(pickle)는 `__init__`을 거치지 않고 클래스 속성으로
    해석되므로 `field(default_factory=...)`는 복원되지 않는다.

    Attributes:
        intent_source: "synthetic"(부모 sine) | "motion_data"(mocap zarr).
        chunk_source: "emulated"(부모 노이즈 주입) | "maniflow"(실 HLP 예측).
        flexion_sign: 데이터 flexion(+굴곡) → 로봇 관절각 부호.
            hip_pendulum은 hinge 축 +Y라 +θ = 신전이므로 −1.
        motion_zarr_path: `extract_soma23_hip_imu_zarr.py --features
            flexion_trunk` 형식 zarr 경로.
        motion_split: "all" | "train" | "val" — HLP validation 분할 재현
            (`motion_intent.maniflow_val_mask`). 실 HLP 평가는 "val".
        motion_metadata_parquet: SEED 메타데이터 parquet (동작 종류 필터용).
        motion_type_filter: `content_type_of_movement` CSV (예 "walking,jogging").
        motion_channel_offset: 4ch 영점 오프셋 [rad] — 다른 규약의 데이터를
            모델 학습 규약으로 정렬할 때 (`motion_intent.channel_mean_offset`).
        motion_seed: 클립 샘플링 전용 generator seed (전역 RNG와 분리 —
            ManiFlow denoise가 전역 RNG를 소비하므로 paired 비교에 필수).
        chunk_drift_std: 에뮬레이션에 추가하는 lead 비례 상관 오차 [rad/s].
            실 HLP 오차는 knot마다 독립이 아니라 선행 시간에 따라 매끄럽게
            커진다 (lead별 MAE 0.36°@25 ms → 3.11°@150 ms).
        maniflow_ckpt: HLP 체크포인트 (None이면 flexion40 run에서 자동 탐색).
        maniflow_denoise_steps: denoise 스텝 (권장 3).
    """

    _target_: str = (
        "protomotions.envs.control.assist_motion_control.MotionAssistTargetControl"
    )

    intent_source: str = "synthetic"
    chunk_source: str = "emulated"
    flexion_sign: float = -1.0

    motion_zarr_path: Optional[str] = None
    motion_split: str = "all"
    motion_metadata_parquet: Optional[str] = None
    motion_type_filter: Optional[str] = None
    motion_channel_offset: Optional[Tuple[float, float, float, float]] = None
    motion_seed: int = 0

    chunk_drift_std: float = 0.0

    maniflow_ckpt: Optional[str] = None
    maniflow_denoise_steps: int = 3


class MotionAssistTargetControl(AssistTargetControl):
    """의도/chunk 소스를 교체할 수 있는 `AssistTargetControl`.

    두 소스가 모두 기본값이면 부모와 동작이 완전히 같다 (v2 재현 경로).
    """

    def __init__(self, config: MotionAssistTargetControlConfig, env: "BaseEnv"):
        # 부모 __init__이 chunk 버퍼를 config.chunk_knots로 할당하고 마지막에
        # self.reset()을 부르므로, 소스는 그 전에 준비돼 있어야 한다.
        self._intent = None
        if getattr(config, "intent_source", "synthetic") == "motion_data":
            from protomotions.envs.control.motion_intent import MotionIntentSource

            self._intent = MotionIntentSource.from_config(config, env)

        self._chunk_src = None
        if getattr(config, "chunk_source", "emulated") == "maniflow":
            from protomotions.maniflow.chunk_source import ManiFlowChunkSource

            self._chunk_src = ManiFlowChunkSource.from_config(config, env)
            # chunk 격자는 모델이 결정한다 (25 ms 간격, horizon − n_obs + 1 knot)
            if (
                config.hlp_dt != self._chunk_src.knot_dt
                or config.chunk_knots != self._chunk_src.num_knots
            ):
                logger.warning(
                    "chunk_source=maniflow: hlp_dt %.3f→%.3f, chunk_knots %d→%d "
                    "(모델 기준으로 덮어씀)",
                    config.hlp_dt,
                    self._chunk_src.knot_dt,
                    config.chunk_knots,
                    self._chunk_src.num_knots,
                )
            config.hlp_dt = self._chunk_src.knot_dt
            config.chunk_knots = self._chunk_src.num_knots

        super().__init__(config, env)

    # ------------------------------------------------------------------
    # 훅 1: 의도 궤적
    # ------------------------------------------------------------------
    def _target_at(self, t: Tensor, env_ids: Optional[Tensor] = None) -> Tensor:
        """mocap 의도 궤적 (없으면 부모의 합성 sine)."""
        if self._intent is None:
            return super()._target_at(t, env_ids)
        theta = self._intent.theta_at(t, env_ids)
        return theta.clamp(self._target_low, self._target_high)

    # ------------------------------------------------------------------
    # 훅 2: chunk 생성
    # ------------------------------------------------------------------
    def _refresh_chunk(self, env_ids: Tensor, t_now: Tensor) -> None:
        """실 HLP 예측 또는 (부모 에뮬레이션 + lead 비례 drift)."""
        if self._chunk_src is not None:
            knots, t0 = self._chunk_src.predict_knots(env_ids, t_now)
            self._chunk[env_ids] = knots.clamp(self._target_low, self._target_high)
            self._chunk_t0[env_ids] = t0
            return

        super()._refresh_chunk(env_ids, t_now)

        drift_std = getattr(self.config, "chunk_drift_std", 0.0)
        if drift_std > 0.0:
            k = self.config.chunk_knots
            dev = self.env.device
            drift = torch.randn(len(env_ids), 1, self.n_dofs, device=dev) * drift_std
            offsets = (
                torch.arange(k, device=dev, dtype=torch.float32) * self.config.hlp_dt
            ).view(1, k, 1)
            self._chunk[env_ids] = (self._chunk[env_ids] + drift * offsets).clamp(
                self._target_low, self._target_high
            )

    # ------------------------------------------------------------------
    # 훅 3/4: 에피소드·스텝 bookkeeping
    # ------------------------------------------------------------------
    def _hlp_observation(self, t: Tensor, env_ids: Optional[Tensor] = None):
        """실 HLP 관측용 (관절각, trunk) 쌍. trunk는 의도 소스의 GT."""
        dof_pos = self.env.state_history.dof_pos[:, 0]
        if env_ids is not None:
            dof_pos = self.env.state_history.dof_pos[env_ids, 0]
        trunk = None if self._intent is None else self._intent.trunk_at(t, env_ids)
        return dof_pos, trunk

    def reset(self, env_ids: Tensor) -> None:
        if len(env_ids) == 0:
            return
        t0 = torch.zeros(len(env_ids), device=self.env.device)

        if self._intent is not None:
            self._intent.reset(env_ids)
        # 실 HLP: 링 버퍼를 리셋 직후 상태로 채운다 (edge padding)
        if self._chunk_src is not None:
            dof_pos, trunk = self._hlp_observation(t0, env_ids)
            self._chunk_src.reset(env_ids, dof_pos, trunk)

        if self._intent is None:
            # 부모가 합성 세그먼트 초기화 + _refresh_chunk + 계측 버퍼 리셋까지 한다
            super().reset(env_ids)
            return

        self._refresh_chunk(env_ids, t0)
        self._tau_agent[env_ids] = 0.0
        self._tau_assist[env_ids] = 0.0

    def step(self) -> None:
        t = self._episode_time()

        # 실 HLP 관측은 매 policy step push (링 버퍼가 25 ms 격자로 재표집)
        if self._chunk_src is not None:
            dof_pos, trunk = self._hlp_observation(t)
            self._chunk_src.observe(dof_pos, trunk)

        if self._intent is None:
            super().step()
            return

        self._intent.step(t)
        refresh = (self.env.progress_buf % self.config.chunk_refresh_steps) == 0
        env_ids = refresh.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self._refresh_chunk(env_ids, t[env_ids])

    # ------------------------------------------------------------------
    # 진단
    # ------------------------------------------------------------------
    def in_transition(self) -> Optional[Tensor]:
        """mocap 클립 전환(crossfade) 중인 env — 평가 마스킹용 (관측 아님)."""
        if self._intent is None:
            return None
        return self._intent.in_transition(self._episode_time())
