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
"""`motion_intent` 단위 테스트 — mocap 의도 궤적 소스 (AI-SBC Phase C).

CPU에서 합성 zarr로 도는 테스트가 기본이고, 실제 SEED zarr가 있으면
클립 필터 개수까지 확인한다 (없으면 skip).

    pytest protomotions/tests/test_motion_intent.py
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from protomotions.envs.control.motion_intent import (
    MOTION_FPS,
    MOTION_MIN_CLIP_FRAMES,
    MotionIntentSource,
    load_motion_clips,
    maniflow_val_mask,
)

SEED_ZARR = Path(
    "/home/bak/Projects/ManiFlow_Policy/ManiFlow/data/hip-flexion-seed-locomotion-40fps.zarr"
)
LIMIT = 1.6 * 0.95  # hip_pendulum 관절 한계 × dof_limit_margin


def _write_zarr(path: Path, clips, *, embodiment="soma23"):
    """flexion_trunk 규약 zarr을 합성해 기록 (clips = (T, 4) 배열 리스트)."""
    import zarr

    zarr.config.set({"default_zarr_format": 2})
    root = zarr.open(str(path), mode="w")
    state = np.concatenate(clips, axis=0).astype(np.float32)
    ends = np.cumsum([len(c) for c in clips]).astype(np.int64)
    data = root.create_group("data")
    data.create_array("state", shape=state.shape, dtype="float32")[:] = state
    data.create_array("action", shape=(state.shape[0], 2), dtype="float32")[:] = state[
        :, :2
    ]
    meta = root.create_group("meta")
    meta.create_array("episode_ends", shape=ends.shape, dtype="int64")[:] = ends
    root.attrs["features"] = "flexion_trunk"
    root.attrs["embodiment"] = embodiment
    root.attrs["convention"] = "skeleton"
    root.attrs["action_dof_names"] = ["hip_flexion_r", "hip_flexion_l"]
    root.attrs["obs_spec"] = json.dumps(
        {
            "channels": [
                "hip_flexion_r",
                "hip_flexion_l",
                "trunk_pitch",
                "trunk_roll",
            ]
        }
    )
    root.attrs["motion_names"] = [f"clip{i}.motion" for i in range(len(clips))]
    return path


def _ramp_clip(n, *, start=0.0, slope=0.01, trunk=0.1):
    """프레임 인덱스에 선형인 클립 — 보간 정확도 검증용."""
    t = np.arange(n, dtype=np.float32)
    return np.stack(
        [start + slope * t, start - slope * t, np.full(n, trunk), np.zeros(n)], axis=1
    )


# ---------------------------------------------------------------------------
# validation 분할 재현
# ---------------------------------------------------------------------------
def test_val_mask_reproduces_maniflow_split():
    """flexion40 체크포인트가 학습된 68,960 클립 → val 3,448개."""
    mask = maniflow_val_mask(68960)
    assert mask.sum() == 3448
    # 결정적이며 train/val이 서로소
    assert np.array_equal(mask, maniflow_val_mask(68960))
    assert (mask & ~mask).sum() == 0


def test_val_mask_edge_cases():
    assert maniflow_val_mask(2).sum() == 1  # 최소 1개, n−1 이하
    assert maniflow_val_mask(10, val_ratio=0.5, seed=1).sum() == 5


# ---------------------------------------------------------------------------
# 클립 로딩·필터
# ---------------------------------------------------------------------------
def test_load_applies_offset_and_split(tmp_path):
    path = _write_zarr(tmp_path / "a.zarr", [_ramp_clip(100) for _ in range(4)])
    table = load_motion_clips(str(path), min_frames=10)
    assert table.num_clips == 4
    assert table.frames.shape == (400, 4)
    assert table.names[0] == "clip0.motion"

    # channel_offset은 로드 시 가산된다
    off = (0.5, -0.5, 0.25, 0.0)
    shifted = load_motion_clips(str(path), channel_offset=off, min_frames=10)
    delta = (shifted.frames - table.frames).mean(0)
    assert torch.allclose(delta, torch.tensor(off), atol=1e-6)

    # split은 val + train = 전체
    n_val = load_motion_clips(str(path), split="val", min_frames=10).num_clips
    n_train = load_motion_clips(str(path), split="train", min_frames=10).num_clips
    assert n_val + n_train == 4


def test_load_filters_short_clips_and_out_of_range(tmp_path):
    clips = [
        _ramp_clip(100),  # 정상
        _ramp_clip(5),  # 너무 짧음
        _ramp_clip(100, start=2.0, slope=0.0),  # 로봇 관절 한계 초과 (2.0 rad)
    ]
    path = _write_zarr(tmp_path / "b.zarr", clips)
    low = torch.tensor([-LIMIT, -LIMIT])
    high = torch.tensor([LIMIT, LIMIT])
    table = load_motion_clips(
        str(path), target_low=low, target_high=high, min_frames=10
    )
    assert table.num_clips == 1
    assert table.frames.shape[0] == 100

    with pytest.raises(ValueError):
        load_motion_clips(str(path), min_frames=10_000)  # 통과 클립 0


def test_load_rejects_wrong_features(tmp_path):
    import zarr

    zarr.config.set({"default_zarr_format": 2})
    root = zarr.open(str(tmp_path / "bad.zarr"), mode="w")
    root.attrs["features"] = "hip_imu"
    with pytest.raises(ValueError, match="flexion_trunk"):
        load_motion_clips(str(tmp_path / "bad.zarr"))


def test_type_filter_requires_metadata(tmp_path):
    path = _write_zarr(tmp_path / "c.zarr", [_ramp_clip(100)])
    with pytest.raises(ValueError, match="metadata_parquet"):
        load_motion_clips(str(path), type_filter="walking", min_frames=10)


# ---------------------------------------------------------------------------
# 궤적 평가
# ---------------------------------------------------------------------------
def _source(tmp_path, clips, **kwargs):
    path = _write_zarr(tmp_path / "src.zarr", clips)
    table = load_motion_clips(str(path), min_frames=10)
    kwargs.setdefault("min_frames", 10)
    return MotionIntentSource(
        table,
        num_envs=kwargs.pop("num_envs", 4),
        device=torch.device("cpu"),
        default_dof_pos=torch.zeros(2),
        **kwargs,
    )


def test_theta_at_is_frame_exact_and_sign_flipped(tmp_path):
    """프레임 시각에서는 보간 오차 0, 부호는 flexion_sign만큼 뒤집힌다."""
    src = _source(tmp_path, [_ramp_clip(200)], num_envs=3, seed=0)
    frames = src._base[1]  # 슬롯 1(실제 클립)의 시작 프레임
    for k in (0, 1, 37, 100):
        t = torch.full((3, 1), k / MOTION_FPS)
        got = src._eval_slot(1, t, None)[:, 0]
        # 클립 끝을 넘어가면 마지막 프레임으로 고정(ZOH)되므로 기대값도 clamp
        want = src.table.frames[torch.minimum(frames + k, src._frame_hi[1])]
        assert torch.allclose(got, want, atol=1e-6)
        theta = got[:, :2] * src.flexion_sign
        assert torch.allclose(theta, -want[:, :2], atol=1e-6)


def test_theta_at_midpoint_is_linear(tmp_path):
    src = _source(tmp_path, [_ramp_clip(200, slope=0.02)], num_envs=2, seed=1)
    t_mid = torch.full((2, 1), 0.5 / MOTION_FPS)
    mid = src._eval_slot(1, t_mid, None)[:, 0]
    a = src._eval_slot(1, torch.zeros(2, 1), None)[:, 0]
    b = src._eval_slot(1, torch.full((2, 1), 1 / MOTION_FPS), None)[:, 0]
    assert torch.allclose(mid, 0.5 * (a + b), atol=1e-6)


def test_holds_past_clip_end(tmp_path):
    """클립 끝을 넘어선 조회는 마지막 프레임으로 고정(ZOH)."""
    src = _source(tmp_path, [_ramp_clip(60)], num_envs=2, seed=2)
    last_t = (60 - 1) / MOTION_FPS
    end = src._eval_slot(1, torch.full((2, 1), last_t), None)
    beyond = src._eval_slot(1, torch.full((2, 1), last_t + 5.0), None)
    assert torch.allclose(end, beyond, atol=1e-6)
    # 음수 시각도 첫 프레임으로 고정
    before = src._eval_slot(1, torch.full((2, 1), -1.0), None)
    first = src._eval_slot(1, torch.zeros(2, 1), None)
    assert torch.allclose(before, first, atol=1e-6)


def test_trunk_channels_keep_data_convention(tmp_path):
    src = _source(tmp_path, [_ramp_clip(100, trunk=0.3)], num_envs=2, seed=3)
    trunk = src.trunk_at(torch.full((2,), 1.0))
    assert trunk.shape == (2, 2)
    assert torch.allclose(trunk[:, 0], torch.full((2,), 0.3), atol=1e-6)


def test_reset_starts_from_default_pose(tmp_path):
    """t=0에서는 기본 자세(hold)에서 시작해 start_blend 동안 블렌드 인."""
    src = _source(tmp_path, [_ramp_clip(200, start=0.5)], num_envs=2, seed=4)
    theta0 = src.theta_at(torch.zeros(2, 1))[:, 0]
    assert torch.allclose(theta0, torch.zeros(2, 2), atol=1e-6)
    assert bool(src.in_transition(torch.zeros(2)).all())


def test_clip_chaining_is_continuous(tmp_path):
    """클립이 소진되며 이어 붙을 때 관절각이 튀지 않는다."""
    clips = [_ramp_clip(MOTION_MIN_CLIP_FRAMES + 5, slope=0.005) for _ in range(6)]
    src = _source(tmp_path, clips, num_envs=4, seed=7)
    dt = 0.01
    prev = None
    max_jump = 0.0
    saw_transition = False
    for i in range(1200):  # 12 s — 클립(1.6 s)이 여러 번 교체된다
        t = torch.full((4,), i * dt)
        src.step(t)
        cur = src.theta_at(t.unsqueeze(1))[:, 0]
        if prev is not None:
            max_jump = max(max_jump, float((cur - prev).abs().max()))
        saw_transition |= bool(src.in_transition(t).any())
        prev = cur
    assert saw_transition, "클립 전환이 한 번도 일어나지 않았다"
    assert max_jump < math.radians(10.0), f"전환에서 점프 {math.degrees(max_jump):.1f}°"


def test_lookahead_never_outruns_reserved_clip(tmp_path):
    """0.55 s 앞을 조회해도 예약된 다음 클립 안에서 해결된다 (NaN/점프 없음)."""
    clips = [_ramp_clip(MOTION_MIN_CLIP_FRAMES + 2) for _ in range(4)]
    src = _source(tmp_path, clips, num_envs=4, seed=8)
    dt = 0.01
    leads = torch.tensor([0.0, 0.15, 0.4, 0.55])
    for i in range(800):
        t = torch.full((4,), i * dt)
        src.step(t)
        q = t.unsqueeze(1) + leads.unsqueeze(0)
        out = src.theta_at(q)
        assert torch.isfinite(out).all()
        assert float(out.abs().max()) < 10.0


def test_dedicated_generator_is_isolated_from_global_rng(tmp_path):
    """전역 RNG를 소비해도 클립 배정이 바뀌지 않는다 (paired 비교 전제)."""
    path = _write_zarr(tmp_path / "g.zarr", [_ramp_clip(120) for _ in range(20)])
    table = load_motion_clips(str(path), min_frames=10)

    def make():
        return MotionIntentSource(
            table,
            num_envs=8,
            device=torch.device("cpu"),
            default_dof_pos=torch.zeros(2),
            seed=11,
        )

    a = make()
    torch.manual_seed(999)
    torch.randn(10_000)
    b = make()
    assert torch.equal(a._clip[1], b._clip[1])
    assert torch.equal(a._base[1], b._base[1])

    # seed()로 재시딩하면 같은 스트림이 다시 나온다
    c = make()
    c.seed(11)
    c.reset(torch.arange(8))
    assert torch.equal(a._clip[1], c._clip[1])


# ---------------------------------------------------------------------------
# 실제 SEED zarr (있을 때만)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not SEED_ZARR.exists(), reason="SEED zarr 없음 (로컬 전용)")
def test_real_seed_zarr_filter_counts():
    low = torch.tensor([-LIMIT, -LIMIT])
    high = torch.tensor([LIMIT, LIMIT])
    val = load_motion_clips(
        str(SEED_ZARR), split="val", target_low=low, target_high=high
    )
    train = load_motion_clips(
        str(SEED_ZARR), split="train", target_low=low, target_high=high
    )
    # 2026-09-08 실측 (관절 범위 필터 후)
    assert val.num_clips == 3130
    assert train.num_clips == 59840
    assert val.frames.shape[1] == 4
    assert float(val.frames[:, :2].abs().max()) <= LIMIT + 1e-6
