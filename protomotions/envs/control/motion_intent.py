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
"""mocap 기반 의도 궤적(θ_g) 소스 — AI-SBC Phase C.

`AssistTargetControl`의 합성 sine 궤적을 실제 mocap flexion 궤적으로
대체한다. HLP(ManiFlow flexion 모델)는 locomotion mocap으로 학습됐으므로
합성 sine 위에서는 예측이 성립하지 않는다 — 실 HLP를 붙이려면 사람의
의도 궤적 자체가 같은 분포(mocap)에서 와야 한다.

데이터 규약
-----------
입력 zarr는 `data/scripts/extract_soma23_hip_imu_zarr.py --features
flexion_trunk`가 만든 것과 동일한 레이아웃이다:

    data/state       (F, 4) float32  [hip_flexion_r, hip_flexion_l,
                                      trunk_pitch, trunk_roll], rad
    meta/episode_ends (C,) int64     클립 경계 (누적 끝 인덱스)
    attrs: features="flexion_trunk", obs_spec, action_dof_names, motion_names

부호 규약: 데이터는 **skeleton hinge flexion(+ = 굴곡)**, hip_pendulum 로봇은
hinge 축 +Y라 **+θ = 신전(extension)**이다. 따라서 로봇 관절각으로 넘길 때
`flexion_sign`(기본 −1)을 곱한다. trunk 2ch은 로봇 관절이 아니라 HLP 관측용
이므로 부호 변환 없이 데이터 규약 그대로 전달한다.

시간축은 `AssistTargetControl`과 동일한 에피소드 시간 t = progress_buf·dt.
클립이 에피소드보다 짧으면 새 클립으로 이어 붙이며(smoothstep crossfade),
그 구간은 `in_transition`으로 표시해 평가에서 제외할 수 있게 한다.

클립 샘플링은 **전역 torch RNG가 아닌 전용 generator**를 사용한다 — ManiFlow
denoise가 전역 RNG를 소비하므로(§8 함정) 그렇지 않으면 emulated/maniflow
조건의 paired 비교에서 의도 궤적이 어긋난다.
"""

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    from protomotions.envs.base_env.env import BaseEnv
    from protomotions.envs.control.assist_motion_control import (
        MotionAssistTargetControlConfig,
    )

logger = logging.getLogger(__name__)

# 모델·데이터 공통 프레임 레이트 (2026-08-13 통일, 프레임워크 문서 §8).
# zarr에 `fps` 속성이 있으면 그 값을 쓴다 (100 fps 데이터 = H1 학습 데이터, 2026-10-21).
MOTION_FPS = 40.0
# 클립 전환 crossfade 길이 [s]
MOTION_BLEND_S = 0.5
# 클립에서 실제로 쓸 최소 길이 (1.5 s; 40 fps에서 60프레임). 전환 예약이 항상 조회
# 지평선(chunk span + 관측 창 ≈ 0.55 s)보다 앞서도록 보장한다.
MOTION_MIN_CLIP_FRAMES = 60
MOTION_MIN_CLIP_S = MOTION_MIN_CLIP_FRAMES / MOTION_FPS
# 클립별 각가속도 대표값: 5프레임 이동평균 뒤 2차 차분 |θ̈|의 99 백분위 [rad/s²]
MOTION_ACCEL_SMOOTH_FRAMES = 5
MOTION_ACCEL_PERCENTILE = 99.0

_STATE_KEY = "data/state"
_EPISODE_ENDS_KEY = "meta/episode_ends"
_EXPECTED_FEATURES = "flexion_trunk"
_EXPECTED_CHANNELS = ("hip_flexion_r", "hip_flexion_l", "trunk_pitch", "trunk_roll")


def maniflow_val_mask(
    n_episodes: int, val_ratio: float = 0.05, seed: int = 42
) -> np.ndarray:
    """ManiFlow 학습 시의 validation 분할을 그대로 재현한다.

    `ManiFlow/maniflow/common/sampler.py:get_val_mask`와 동일한 규칙
    (n_episodes, val_ratio, seed)만으로 결정적이다. flexion40 체크포인트는
    68,960 클립 zarr에 val_ratio 0.05 / seed 42로 학습됐으므로 val = 3,448개.

    Returns:
        (n_episodes,) bool — True인 클립이 HLP의 **미학습분**.
    """
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    mask = np.zeros(n_episodes, dtype=bool)
    mask[val_idxs] = True
    return mask


def _open_zarr(path: str):
    import zarr

    z = zarr.open(os.path.expanduser(path), mode="r")
    features = z.attrs.get("features")
    if features != _EXPECTED_FEATURES:
        raise ValueError(f"{path}: features={features!r}, expected 'flexion_trunk'")
    spec = z.attrs.get("obs_spec")
    if spec is not None:
        channels = tuple(json.loads(spec)["channels"])
        if channels != _EXPECTED_CHANNELS:
            raise ValueError(f"{path}: obs channels {channels} != {_EXPECTED_CHANNELS}")
    return z


def _streaming_channel_mean(z, chunk: int = 1 << 20) -> np.ndarray:
    """`data/state`의 채널별 평균 (전체 프레임을 한 번에 올리지 않음)."""
    arr = z[_STATE_KEY]
    total = arr.shape[0]
    acc = np.zeros(arr.shape[1], dtype=np.float64)
    for start in range(0, total, chunk):
        acc += arr[start : start + chunk].astype(np.float64).sum(axis=0)
    return acc / float(total)


def channel_mean_offset(ref_zarr_path: str, zarr_path: str) -> Tuple[float, ...]:
    """두 데이터셋의 채널 평균 차 (= 중립자세 영점 오프셋) [rad].

    SEED와 ETRI suit는 중립자세 각도 규약이 채널별 상수만큼 다르다
    (프레임워크 문서 §3.1). 반환값을 `motion_channel_offset`으로 넘기면
    해당 zarr이 ref(=SEED) 규약으로 정렬되어, HLP를 무보정으로 돌릴 수 있다.

    Returns:
        mean(ref) − mean(target), 채널 순서는 `_EXPECTED_CHANNELS`.
    """
    ref = _streaming_channel_mean(_open_zarr(ref_zarr_path))
    tgt = _streaming_channel_mean(_open_zarr(zarr_path))
    offset = tuple(float(v) for v in (ref - tgt))
    logger.info(
        "channel_mean_offset(%s vs %s) = %s deg",
        os.path.basename(ref_zarr_path),
        os.path.basename(zarr_path),
        [round(math.degrees(v), 2) for v in offset],
    )
    return offset


@dataclass
class MotionClipTable:
    """필터를 통과한 클립들을 하나로 이어 붙인 프레임 테이블.

    Attributes:
        frames: (F, 4) float32 device 텐서. `channel_offset` 적용 후,
            **데이터 규약**(flexion +) 그대로.
        clip_start: (C,) int64 — 각 클립의 첫 프레임 인덱스.
        clip_len: (C,) int64 — 각 클립의 프레임 수.
        names: 클립 이름 (진단용).
        fps: 프레임 레이트 (zarr `fps` 속성, 없으면 `MOTION_FPS`).
        clip_type: 클립별 `content_type_of_movement` (metadata를 줬을 때만).
        clip_accel99: (C, 2) 고관절 2ch의 |θ̈| 99 백분위 [rad/s²]
            (`with_accel99=True`일 때만). 실현 가능성 검사용.
    """

    frames: Tensor
    clip_start: Tensor
    clip_len: Tensor
    names: List[str]
    fps: float = MOTION_FPS
    clip_type: Optional[List[str]] = None
    clip_accel99: Optional[Tensor] = None

    @property
    def num_clips(self) -> int:
        return int(self.clip_start.numel())


def _load_clip_metadata(
    z, metadata_parquet: str, type_filter: Optional[str], n_clips: int
) -> Tuple[np.ndarray, List[str]]:
    """metadata parquet의 `content_type_of_movement`로 클립 마스크 생성.

    `motion_names`(basename, 보통 `.motion` 확장자 포함)를 parquet의
    `filename` 열과 1:1 조인한다. 매칭 실패분은 제외하고 개수를 로깅한다.

    Returns:
        (mask, kinds) — kinds는 클립별 동작 종류 (미매칭은 "").
    """
    import pandas as pd

    names = list(z.attrs.get("motion_names", []))
    if len(names) != n_clips:
        raise ValueError(
            f"motion_names {len(names)} != episode count {n_clips}; "
            "metadata filter를 쓸 수 없다"
        )
    stems = [os.path.splitext(n)[0] for n in names]

    df = pd.read_parquet(
        os.path.expanduser(metadata_parquet),
        columns=["filename", "content_type_of_movement"],
    )
    df["filename"] = df["filename"].map(lambda s: os.path.splitext(str(s))[0])
    lookup = dict(zip(df["filename"], df["content_type_of_movement"]))

    wanted = None
    if type_filter:
        wanted = {s.strip() for s in type_filter.split(",") if s.strip()}

    mask = np.zeros(n_clips, dtype=bool)
    kinds = [""] * n_clips
    unmatched = 0
    for i, stem in enumerate(stems):
        kind = lookup.get(stem)
        if kind is None:
            unmatched += 1
            continue
        kinds[i] = str(kind)
        mask[i] = True if wanted is None else (str(kind) in wanted)
    if unmatched:
        logger.warning(
            "metadata 미매칭 클립 %d/%d개 제외 (%s)",
            unmatched,
            n_clips,
            os.path.basename(metadata_parquet),
        )
    return mask, kinds


def clip_accel_percentile(
    x: np.ndarray,
    fps: float,
    smooth_frames: int = MOTION_ACCEL_SMOOTH_FRAMES,
    percentile: float = MOTION_ACCEL_PERCENTILE,
) -> np.ndarray:
    """한 클립의 채널별 |θ̈| 백분위 [rad/s²]. x (T, k) → (k,).

    mocap 차분 잡음이 최댓값을 지배하지 않도록 이동평균 뒤 2차 차분의 백분위를 쓴다.
    """
    if x.shape[0] < smooth_frames + 2:
        return np.zeros(x.shape[1], dtype=np.float32)
    kernel = np.ones(smooth_frames, dtype=np.float64) / smooth_frames
    smooth = np.stack(
        [np.convolve(x[:, j].astype(np.float64), kernel, mode="valid") for j in range(x.shape[1])],
        axis=1,
    )
    accel = np.abs(np.diff(smooth, n=2, axis=0)) * fps * fps
    return np.percentile(accel, percentile, axis=0).astype(np.float32)


def load_motion_clips(
    zarr_path: str,
    *,
    split: str = "all",
    metadata_parquet: Optional[str] = None,
    type_filter: Optional[str] = None,
    channel_offset: Optional[Sequence[float]] = None,
    flexion_sign: float = -1.0,
    target_low: Optional[Tensor] = None,
    target_high: Optional[Tensor] = None,
    min_frames: Optional[int] = None,
    with_accel99: bool = False,
    device: torch.device = torch.device("cpu"),
) -> MotionClipTable:
    """zarr에서 클립을 필터링해 `MotionClipTable`로 올린다.

    필터 순서: split(HLP 학습/미학습) → metadata(동작 종류) → 길이
    (`min_frames`, 기본 1.5 s 분량) → 관절 범위(로봇 한계를 넘는 클립 제외).

    Args:
        split: "all" | "train" | "val" — HLP의 validation 분할 기준
            (`maniflow_val_mask`). 실 HLP 평가는 "val"을 써야 미학습분 검증이
            된다.
        channel_offset: 4ch에 가산할 영점 오프셋 [rad] (`channel_mean_offset`).
        flexion_sign: 데이터 flexion → 로봇 관절각 부호.
        target_low/target_high: 로봇 관절 목표 한계 (d=2). 주면 이 범위를
            벗어나는 프레임이 있는 클립을 **통째로 제외**한다 (clamp로 궤적을
            왜곡하지 않기 위함).
    """
    z = _open_zarr(zarr_path)
    fps = float(z.attrs.get("fps", MOTION_FPS))
    if min_frames is None:
        min_frames = int(round(MOTION_MIN_CLIP_S * fps))
    ends = np.asarray(z[_EPISODE_ENDS_KEY][:], dtype=np.int64)
    n_clips = int(ends.shape[0])
    starts = np.concatenate([[0], ends[:-1]]).astype(np.int64)
    lengths = (ends - starts).astype(np.int64)

    keep = np.ones(n_clips, dtype=bool)
    if split not in ("all", "train", "val"):
        raise ValueError(f"split must be all|train|val, got {split!r}")
    if split != "all":
        val_mask = maniflow_val_mask(n_clips)
        keep &= val_mask if split == "val" else ~val_mask
    n_after_split = int(keep.sum())

    kinds_all = None
    if metadata_parquet:
        meta_mask, kinds_all = _load_clip_metadata(z, metadata_parquet, type_filter, n_clips)
        keep &= meta_mask
    elif type_filter:
        raise ValueError("type_filter를 쓰려면 metadata_parquet도 지정해야 한다")
    n_after_meta = int(keep.sum())

    keep &= lengths >= int(min_frames)
    n_after_len = int(keep.sum())

    state = np.asarray(z[_STATE_KEY][:], dtype=np.float32)
    if channel_offset is not None:
        state = state + np.asarray(channel_offset, dtype=np.float32)[None, :]

    if target_low is not None and target_high is not None:
        lo = (target_low.detach().cpu().numpy().astype(np.float64))[:2]
        hi = (target_high.detach().cpu().numpy().astype(np.float64))[:2]
        theta = state[:, :2].astype(np.float64) * float(flexion_sign)
        # 클립별 min/max (reduceat은 빈 구간이 없어야 하므로 starts를 그대로 사용)
        clip_min = np.minimum.reduceat(theta, starts, axis=0)
        clip_max = np.maximum.reduceat(theta, starts, axis=0)
        in_range = np.all(clip_min >= lo[None, :], axis=1) & np.all(
            clip_max <= hi[None, :], axis=1
        )
        keep &= in_range
    n_after_range = int(keep.sum())

    if n_after_range == 0:
        raise ValueError(
            f"{zarr_path}: 필터를 통과한 클립이 없다 "
            f"(split={split} type_filter={type_filter})"
        )

    kept = np.nonzero(keep)[0]
    kept_lengths = lengths[kept]
    frames_np = np.concatenate(
        [state[starts[i] : ends[i]] for i in kept], axis=0
    )  # (F, 4)
    new_starts = np.concatenate([[0], np.cumsum(kept_lengths)[:-1]]).astype(np.int64)
    names_all = list(z.attrs.get("motion_names", []))
    names = [str(names_all[i]) if i < len(names_all) else f"clip{i}" for i in kept]

    accel99 = None
    if with_accel99:
        accel99 = torch.from_numpy(np.stack(
            [clip_accel_percentile(state[starts[i] : ends[i], :2], fps) for i in kept]
        )).to(device=device)

    frames = torch.from_numpy(frames_np).to(device=device, dtype=torch.float32)
    logger.info(
        "MotionIntentSource: %s split=%s → clips %d→%d (split %d, meta %d, len %d, "
        "range %d) / frames %d (%.1f s) / %.0f MB",
        os.path.basename(str(zarr_path).rstrip("/")),
        split,
        n_clips,
        len(kept),
        n_after_split,
        n_after_meta,
        n_after_len,
        n_after_range,
        frames.shape[0],
        frames.shape[0] / fps,
        frames.numel() * 4 / 1e6,
    )
    return MotionClipTable(
        frames=frames,
        clip_start=torch.from_numpy(new_starts).to(device=device),
        clip_len=torch.from_numpy(kept_lengths).to(device=device),
        names=names,
        fps=fps,
        clip_type=None if kinds_all is None else [kinds_all[i] for i in kept],
        clip_accel99=accel99,
    )


def group_clip_probabilities(
    clip_type: Sequence[str],
    clip_len: np.ndarray,
    groups: Sequence[Tuple[str, str]],
    weights: Sequence[Tuple[str, float]],
    min_frames: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """동작 그룹 가중치를 **시간 비율**로 맞추는 클립 확률.

    그룹은 (이름, 정규식) 순서대로 `content_type_of_movement`에 처음 맞는 것이다.
    클립 c를 뽑으면 평균 (len_c + min_frames)/2 프레임 동안 쓰이므로(시작 위치 균등),
    p_c = w_g / Σ_{c'∈g} r_{c'} 로 두면 그룹 g가 차지하는 시간 비율이 w_g가 된다.

    Returns:
        (probs (C,), group_index (C,)) — 그룹에 속하지 않으면 −1.
    """
    import re

    patterns = [(str(name), re.compile(str(rx))) for name, rx in groups]
    names = [name for name, _ in patterns]
    group_index = np.full(len(clip_type), -1, dtype=np.int64)
    for i, kind in enumerate(clip_type):
        for j, (_, rx) in enumerate(patterns):
            if rx.search(str(kind)):
                group_index[i] = j
                break
    remaining = (np.asarray(clip_len, dtype=np.float64) + float(min_frames)) / 2.0
    probs = np.zeros(len(clip_type), dtype=np.float64)
    for name, w in weights:
        if name not in names:
            raise ValueError(f"unknown motion group {name!r}; known {names}")
        if float(w) <= 0.0:
            continue
        member = group_index == names.index(name)
        if not member.any():
            raise ValueError(f"motion group {name!r} has no clips")
        probs[member] = float(w) / remaining[member].sum()
    total = probs.sum()
    if total <= 0.0:
        raise ValueError("motion group weights select no clips")
    return probs / total, group_index


def _smoothstep(x: Tensor) -> Tensor:
    x = x.clamp(0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


class MotionIntentSource:
    """mocap 클립을 이어 붙여 연속 의도 궤적 θ_g(t)를 제공한다.

    `AssistTargetControl`의 세그먼트 구조와 같은 2슬롯(cur/nxt) + smoothstep
    블렌드 방식이라 `_target_at`이 임의의 연속 시각으로 조회해도(중앙차분
    속도, chunk knot) 동작한다.

    슬롯 상태는 모두 (2, num_envs) 텐서다. 슬롯 0 = 현재, 1 = 다음.
    """

    def __init__(
        self,
        table: MotionClipTable,
        num_envs: int,
        device: torch.device,
        default_dof_pos: Tensor,
        *,
        fps: Optional[float] = None,
        flexion_sign: float = -1.0,
        start_blend: float = 0.6,
        blend_s: float = MOTION_BLEND_S,
        min_frames: Optional[int] = None,
        seed: int = 0,
    ):
        self.table = table
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.fps = float(table.fps if fps is None else fps)
        self.flexion_sign = float(flexion_sign)
        self.start_blend = float(start_blend)
        self.blend_s = float(blend_s)
        self.min_frames = int(
            round(MOTION_MIN_CLIP_S * self.fps) if min_frames is None else min_frames
        )
        # 집단별 가중 샘플링·실현 가능성 (configure_sampling / set_accel_cap). 없으면 균등.
        self._pool_probs: Optional[Tensor] = None
        self._env_pool: Optional[Tensor] = None
        self._fallback_clip: Optional[Tensor] = None
        self._accel_cap: Optional[Tensor] = None
        self._cap_channels: List[int] = [0, 1]
        self.max_tries = 16
        self.fallback_count = 0

        n = num_envs
        dev = self.device
        z2 = (2, n)
        self._clip = torch.zeros(z2, dtype=torch.long, device=dev)
        self._base = torch.zeros(z2, dtype=torch.long, device=dev)
        self._frame_lo = torch.zeros(z2, dtype=torch.long, device=dev)
        self._frame_hi = torch.zeros(z2, dtype=torch.long, device=dev)
        self._t0 = torch.zeros(z2, device=dev)
        self._hold = torch.zeros(z2, dtype=torch.bool, device=dev)
        self._hold_value = torch.zeros(2, n, 4, device=dev)
        self._blend_start = torch.zeros(n, device=dev)
        self._blend_dur = torch.full((n,), self.start_blend, device=dev)

        # 기본 자세 hold 값 (데이터 규약으로 되돌려 저장; trunk = 0)
        dpos = default_dof_pos.to(device=dev, dtype=torch.float32)
        self._default_obs4 = torch.zeros(4, device=dev)
        self._default_obs4[:2] = dpos[:2] / self.flexion_sign

        # 클립 샘플링 전용 generator (전역 RNG와 분리 — §8 함정)
        self._gen = torch.Generator(device=dev)
        self.seed(seed)

        self.reset(torch.arange(n, device=dev, dtype=torch.long))

    # ------------------------------------------------------------------
    # 구성
    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls, config: "MotionAssistTargetControlConfig", env: "BaseEnv"
    ) -> "MotionIntentSource":
        """`MotionAssistTargetControlConfig`에서 소스를 만든다."""
        if not config.motion_zarr_path:
            raise ValueError("intent_source='motion_data'에는 motion_zarr_path 필요")
        d = env.robot_config.number_of_actions
        if d != 2:
            raise ValueError(
                f"MotionIntentSource는 hip flexion 2 DOF 전용 (로봇 DOF={d})"
            )
        kin = env.robot_config.kinematic_info
        margin = config.dof_limit_margin
        low = kin.dof_limits_lower.to(device=env.device, dtype=torch.float32) * margin
        high = kin.dof_limits_upper.to(device=env.device, dtype=torch.float32) * margin

        table = load_motion_clips(
            config.motion_zarr_path,
            split=config.motion_split,
            metadata_parquet=config.motion_metadata_parquet,
            type_filter=config.motion_type_filter,
            channel_offset=config.motion_channel_offset,
            flexion_sign=config.flexion_sign,
            target_low=low,
            target_high=high,
            with_accel99=float(getattr(config, "motion_accel_frac", 0.0)) > 0.0,
            device=env.device,
        )
        return cls(
            table,
            num_envs=env.num_envs,
            device=env.device,
            default_dof_pos=env.robot_config.default_dof_pos,
            flexion_sign=config.flexion_sign,
            start_blend=config.start_blend,
            seed=config.motion_seed,
        )

    def seed(self, seed: int) -> None:
        """클립 샘플링 generator 재시딩 (paired 평가에서 조건마다 호출)."""
        self._gen.manual_seed(int(seed))

    # ------------------------------------------------------------------
    # 클립 배치
    # ------------------------------------------------------------------
    def _assign_clip(self, slot: int, env_ids: Tensor, t0: Tensor) -> None:
        """슬롯에 랜덤 클립 + 랜덤 시작 프레임을 배정한다.

        시작 오프셋은 `[0, len − min_frames]`에서 뽑으므로 배정 직후
        최소 `min_frames/fps`(1.5 s) 분량이 남는다.
        """
        m = int(env_ids.numel())
        if m == 0:
            return
        c = self.table.num_clips
        if self._pool_probs is None:
            clip = torch.randint(0, c, (m,), generator=self._gen, device=self.device)
        else:
            clip = self._sample_pool_clips(env_ids)
        start = self.table.clip_start[clip]
        length = self.table.clip_len[clip]
        span = (length - self.min_frames).clamp(min=0)
        frac = torch.rand(m, generator=self._gen, device=self.device)
        offset = (frac * (span + 1).float()).long().clamp(max=span)

        self._clip[slot, env_ids] = clip
        self._base[slot, env_ids] = start + offset
        self._frame_lo[slot, env_ids] = start + offset
        self._frame_hi[slot, env_ids] = start + length - 1
        self._t0[slot, env_ids] = t0
        self._hold[slot, env_ids] = False

    # ------------------------------------------------------------------
    # 집단별 가중 샘플링 + 실현 가능성 (2026-10-21)
    # ------------------------------------------------------------------
    def configure_sampling(self, pool_probs: Tensor, env_pool: Tensor) -> None:
        """env 집단별 클립 확률을 정한다 (이후 배정부터 적용).

        Args:
            pool_probs: (P, C) 집단별 클립 확률, 각 행의 합 1.
            env_pool: (n,) env별 집단 번호 0..P−1.
        """
        probs = pool_probs.to(device=self.device, dtype=torch.float32)
        if probs.shape[1] != self.table.num_clips:
            raise ValueError(f"pool_probs columns {probs.shape[1]} != clips {self.table.num_clips}")
        if not torch.allclose(probs.sum(dim=1), torch.ones(probs.shape[0], device=self.device)):
            raise ValueError("each pool_probs row must sum to 1")
        pools = env_pool.to(device=self.device, dtype=torch.long)
        if pools.shape != (self.num_envs,) or int(pools.min()) < 0 or int(pools.max()) >= probs.shape[0]:
            raise ValueError("env_pool must assign every env to a valid pool")
        self._pool_probs = probs
        self._env_pool = pools
        # 모든 시도가 실현 불가능할 때 쓰는 대체: 집단 안에서 가장 느린 클립
        if self.table.clip_accel99 is not None:
            need = self.table.clip_accel99.max(dim=1).values
            masked = torch.where(probs > 0, need.unsqueeze(0), torch.full_like(probs, float("inf")))
            self._fallback_clip = masked.argmin(dim=1)
        else:
            self._fallback_clip = probs.argmax(dim=1)

    def set_accel_cap(self, cap: Tensor, channels: Sequence[int] = (0, 1)) -> int:
        """env별 허용 |θ̈| [rad/s²] (n, 2)를 정하고, 이미 배정된 클립 중 불가능한 것을 다시 뽑는다.

        에피소드 시작 직후(관성·게인이 정해지는 첫 step)에만 부른다 — 슬롯 0은 그때 기본 자세
        hold라 다시 뽑아도 궤적이 끊기지 않는다.

        Returns:
            다시 뽑은 슬롯 수.
        """
        if self.table.clip_accel99 is None:
            raise ValueError("set_accel_cap requires a table loaded with with_accel99=True")
        self._accel_cap = cap.to(device=self.device, dtype=torch.float32)
        self._cap_channels = [int(c) for c in channels]
        all_ids = torch.arange(self.num_envs, device=self.device)
        fixed = 0
        for slot in (0, 1):
            bad = (~self._hold[slot]) & ~self._feasible(self._clip[slot], all_ids)
            ids = bad.nonzero(as_tuple=False).flatten()
            if ids.numel():
                self._assign_clip(slot, ids, self._t0[slot, ids].clone())
                fixed += int(ids.numel())
        return fixed

    def _feasible(self, clip: Tensor, env_ids: Tensor) -> Tensor:
        """클립의 |θ̈| 99 %가 env 허용치 이하인지 (검사 채널 전부)."""
        if self._accel_cap is None or self.table.clip_accel99 is None:
            return torch.ones(clip.shape[0], dtype=torch.bool, device=self.device)
        ch = self._cap_channels
        need = self.table.clip_accel99[clip][:, ch]
        cap = self._accel_cap[env_ids][:, ch]
        return (need <= cap).all(dim=1)

    def _sample_pool_clips(self, env_ids: Tensor) -> Tensor:
        """집단 확률로 뽑고, 실현 불가능한 것은 최대 `max_tries`번 다시 뽑는다.

        거절 표본추출이므로 결과는 "그 env에서 가능한 클립만으로 다시 정규화한 집단 확률"과 같다.
        """
        m = int(env_ids.numel())
        pools = self._env_pool[env_ids]
        clip = torch.zeros(m, dtype=torch.long, device=self.device)
        todo = torch.ones(m, dtype=torch.bool, device=self.device)
        for _ in range(self.max_tries):
            for p in range(self._pool_probs.shape[0]):
                sel = todo & (pools == p)
                k = int(sel.sum())
                if k:
                    clip[sel] = torch.multinomial(
                        self._pool_probs[p], k, replacement=True, generator=self._gen
                    )
            todo = todo & ~self._feasible(clip, env_ids)
            if not bool(todo.any()):
                break
        if bool(todo.any()):
            clip[todo] = self._fallback_clip[pools[todo]]
            self.fallback_count += int(todo.sum())
        return clip

    def _slot_end_time(self, slot: int, env_ids: Optional[Tensor] = None) -> Tensor:
        """슬롯 클립이 소진되는 에피소드 시각 [s]."""
        idx = slice(None) if env_ids is None else env_ids
        remaining = (self._frame_hi[slot][idx] - self._base[slot][idx]).float()
        return self._t0[slot][idx] + remaining / self.fps

    # ------------------------------------------------------------------
    # 평가
    # ------------------------------------------------------------------
    def _eval_slot(self, slot: int, t: Tensor, env_ids: Optional[Tensor]) -> Tensor:
        """슬롯의 4ch 데이터 규약 값. t (n, K) → (n, K, 4)."""
        idx = slice(None) if env_ids is None else env_ids
        t0 = self._t0[slot][idx].unsqueeze(1)
        base = self._base[slot][idx].unsqueeze(1)
        hi = self._frame_hi[slot][idx].unsqueeze(1)

        max_rel = (hi - base).float()
        rel = ((t - t0) * self.fps).clamp(min=0.0)
        rel = torch.minimum(rel, max_rel)
        step = rel.floor()
        frac = (rel - step).unsqueeze(-1)

        idx0 = base + step.long()
        idx1 = torch.minimum(idx0 + 1, hi)
        f0 = self.table.frames[idx0]  # (n, K, 4)
        f1 = self.table.frames[idx1]
        out = f0 + (f1 - f0) * frac

        hold = self._hold[slot][idx].unsqueeze(1).unsqueeze(2)
        hold_value = self._hold_value[slot][idx].unsqueeze(1)
        return torch.where(hold, hold_value, out)

    def obs4_at(self, t: Tensor, env_ids: Optional[Tensor] = None) -> Tensor:
        """블렌딩된 4ch 값 (데이터 규약). t (n, K) → (n, K, 4)."""
        idx = slice(None) if env_ids is None else env_ids
        blend_start = self._blend_start[idx].unsqueeze(1)
        blend_dur = self._blend_dur[idx].unsqueeze(1)
        s = _smoothstep((t - blend_start) / blend_dur.clamp(min=1e-6))
        a = self._eval_slot(0, t, env_ids)
        b = self._eval_slot(1, t, env_ids)
        return a + (b - a) * s.unsqueeze(-1)

    def theta_at(self, t: Tensor, env_ids: Optional[Tensor] = None) -> Tensor:
        """의도 관절각 θ_g. t (n, K) → (n, K, 2) **로봇 규약** (clamp 안 함)."""
        return self.obs4_at(t, env_ids)[..., :2] * self.flexion_sign

    def trunk_at(self, t: Tensor, env_ids: Optional[Tensor] = None) -> Tensor:
        """HLP 관측용 trunk pitch/roll. t (n,) → (n, 2) 데이터 규약."""
        return self.obs4_at(t.unsqueeze(1), env_ids)[:, 0, 2:]

    def in_transition(self, t: Tensor) -> Tensor:
        """클립 전환(crossfade) 진행 중 여부. t (n,) → (n,) bool."""
        return (t >= self._blend_start) & (t <= self._blend_start + self._blend_dur)

    # ------------------------------------------------------------------
    # ControlComponent 훅
    # ------------------------------------------------------------------
    def reset(self, env_ids: Tensor) -> None:
        """에피소드 시작: 기본 자세 hold → 새 클립으로 start_blend 블렌딩."""
        if env_ids.numel() == 0:
            return
        zeros = torch.zeros(env_ids.numel(), device=self.device)
        self._hold[0, env_ids] = True
        self._hold_value[0, env_ids] = self._default_obs4
        self._t0[0, env_ids] = 0.0
        self._base[0, env_ids] = 0
        self._frame_lo[0, env_ids] = 0
        self._frame_hi[0, env_ids] = 0

        self._assign_clip(1, env_ids, zeros)
        self._blend_start[env_ids] = 0.0
        self._blend_dur[env_ids] = self.start_blend

    def step(self, t: Tensor) -> None:
        """블렌드 완료 env를 승격하고 다음 클립 전환을 예약한다."""
        done = t >= (self._blend_start + self._blend_dur)
        env_ids = done.nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return

        for buf in (self._clip, self._base, self._frame_lo, self._frame_hi, self._t0):
            buf[0, env_ids] = buf[1, env_ids]
        self._hold[0, env_ids] = self._hold[1, env_ids]
        self._hold_value[0, env_ids] = self._hold_value[1, env_ids]

        # cur 클립이 소진되기 blend_s 전에 crossfade 시작 (항상 t보다 미래)
        end_t = self._slot_end_time(0, env_ids)
        blend_start = torch.maximum(end_t - self.blend_s, t[env_ids] + 1e-3)
        self._blend_start[env_ids] = blend_start
        self._blend_dur[env_ids] = self.blend_s
        self._assign_clip(1, env_ids, blend_start)
