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
        intent_pools: env 단위 의도 집단 (2026-10-21). 비어 있으면 기존 동작.
            항목 = (이름, env 비율, 소스 "synthetic"|"motion", 관성 배율 lo, hi,
            ((동작 그룹, 시간 비율), ...)). env는 한 집단에 고정 배정되고, AssistEnv가
            그 집단의 관성 배율 범위로 대퇴 관성을 뽑는다 (관성은 env 수명 동안 고정).
        motion_groups: (그룹 이름, `content_type_of_movement` 정규식) — 처음 맞는 그룹.
        motion_accel_frac: 실현 가능성 검사. 클립의 |θ̈| 99 백분위 × hip 관성이
            frac × 착용자 토크 한계를 넘는 클립은 그 env에서 뽑지 않는다 (0 = 검사 안 함).
            합성 궤적의 진폭 상한(`intent_accel_frac`)과 같은 원리다.
        motion_accel_check_dofs: 검사할 관절 (비면 전 관절).
        intent_pool_seed: 집단 배정 전용 seed.
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

    intent_pools: tuple = ()
    motion_groups: tuple = ()
    motion_accel_frac: float = 0.0
    motion_accel_check_dofs: tuple = ()
    intent_pool_seed: int = 0


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

        # env 단위 의도 집단 (합성/동작 혼합). 부모 __init__의 reset 전에 배정한다.
        self._pools = tuple(getattr(config, "intent_pools", ()) or ())
        self._mixed = bool(self._pools)
        self._motion_env: Optional[Tensor] = None
        self.env_pool: Optional[Tensor] = None
        self.env_inertia_range: Optional[Tensor] = None
        self.pool_summary: list = []
        if self._mixed:
            self._setup_pools(config, env)

        super().__init__(config, env)

    def _setup_pools(self, config: MotionAssistTargetControlConfig, env: "BaseEnv") -> None:
        """env를 집단에 정확한 비율로 고정 배정하고, 동작 집단의 클립 확률을 정한다."""
        import numpy as np

        from protomotions.envs.control.motion_intent import group_clip_probabilities

        n, dev = env.num_envs, env.device
        fractions = [float(p[1]) for p in self._pools]
        if any(f < 0 for f in fractions) or abs(sum(fractions) - 1.0) > 1e-6:
            raise ValueError(f"intent_pools fractions must be nonnegative and sum to 1: {fractions}")
        sources = [str(p[2]) for p in self._pools]
        if any(src not in ("synthetic", "motion") for src in sources):
            raise ValueError(f"intent_pools sources must be synthetic|motion: {sources}")
        counts = [int(round(f * n)) for f in fractions]
        counts[-1] = n - sum(counts[:-1])
        gen = torch.Generator().manual_seed(int(getattr(config, "intent_pool_seed", 0)))
        perm = torch.randperm(n, generator=gen)
        pool = torch.empty(n, dtype=torch.long)
        start = 0
        for i, c in enumerate(counts):
            pool[perm[start : start + c]] = i
            start += c
        self.env_pool = pool.to(dev)
        is_motion = torch.tensor([src == "motion" for src in sources], dtype=torch.bool)
        self._motion_env = is_motion.to(dev)[self.env_pool]
        ranges = torch.tensor([[float(p[3]), float(p[4])] for p in self._pools], dtype=torch.float32)
        if bool((ranges[:, 0] <= 0).any()) or bool((ranges[:, 1] < ranges[:, 0]).any()):
            raise ValueError("intent_pools inertia ranges must satisfy 0 < lo <= hi")
        self.env_inertia_range = ranges.to(dev)[self.env_pool]
        self.pool_summary = [
            dict(name=str(p[0]), fraction=float(p[1]), source=str(p[2]), envs=int(c),
                 inertia_scale=[float(p[3]), float(p[4])], weights=[list(w) for w in p[5]])
            for p, c in zip(self._pools, counts)
        ]
        if not bool(is_motion.any()):
            return
        if self._intent is None:
            raise ValueError("motion intent pools require intent_source='motion_data'")
        if self._chunk_src is not None:
            raise ValueError("mixed intent pools support emulated chunks only")
        table = self._intent.table
        if table.clip_type is None:
            raise ValueError("motion intent pools require motion_metadata_parquet")
        lengths = table.clip_len.cpu().numpy()
        rows = []
        for summary, p in zip(self.pool_summary, self._pools):
            if str(p[2]) == "motion":
                probs, group_index = group_clip_probabilities(
                    table.clip_type, lengths, config.motion_groups, p[5], self._intent.min_frames
                )
                names = [str(g[0]) for g in config.motion_groups]
                summary["group_clips"] = {
                    name: int((group_index == j).sum()) for j, name in enumerate(names)
                }
            else:
                probs = np.full(table.num_clips, 1.0 / table.num_clips)
            rows.append(probs)
        self._intent.configure_sampling(torch.tensor(np.stack(rows)), self.env_pool)
        logger.info("intent pools: %s", self.pool_summary)

    def update_motion_feasibility(self, effort_limit: Tensor, inertia: Tensor) -> int:
        """착용자 토크 한계·hip 관성이 정해진 뒤 클립 실현 가능성 상한을 건다.

        Returns:
            다시 뽑은 슬롯 수 (첫 step에서만 0이 아닐 수 있다).
        """
        frac = float(getattr(self.config, "motion_accel_frac", 0.0))
        if self._intent is None or frac <= 0.0:
            return 0
        cap = frac * effort_limit / inertia
        if self._motion_env is not None:
            cap = torch.where(self._motion_env.unsqueeze(1), cap, torch.full_like(cap, float("inf")))
        dofs = tuple(getattr(self.config, "motion_accel_check_dofs", ()) or ()) or tuple(range(self.n_dofs))
        fixed = self._intent.set_accel_cap(cap, dofs)
        if fixed:
            all_ids = torch.arange(self.env.num_envs, device=self.env.device)
            self._refresh_chunk(all_ids, self._episode_time())
        return fixed

    # ------------------------------------------------------------------
    # 훅 1: 의도 궤적
    # ------------------------------------------------------------------
    def _target_at(self, t: Tensor, env_ids: Optional[Tensor] = None) -> Tensor:
        """mocap 의도 궤적 (없으면 부모의 합성 sine, 집단 혼합이면 env별 선택)."""
        if self._intent is None:
            return super()._target_at(t, env_ids)
        theta = self._intent.theta_at(t, env_ids).clamp(self._target_low, self._target_high)
        if not self._mixed:
            return theta
        synthetic = super()._target_at(t, env_ids)
        motion = self._motion_env if env_ids is None else self._motion_env[env_ids]
        return torch.where(motion.view(-1, 1, 1), theta, synthetic)

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
        if self._mixed:
            # 합성 세그먼트도 함께 초기화 (chunk는 env별로 고른 의도에서 뜬다)
            super().reset(env_ids)
            return
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
        if self._mixed:
            super().step()  # 합성 세그먼트 승격 + chunk 갱신
            return
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
        moving = self._intent.in_transition(self._episode_time())
        return moving if self._motion_env is None else moving & self._motion_env
