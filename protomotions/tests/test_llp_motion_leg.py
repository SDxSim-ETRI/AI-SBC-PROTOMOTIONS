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
"""LLP 20 N·m 다리별·동작 의도 학습(2026-10-21) 구성 요소 단위 테스트.

    pytest protomotions/tests/test_llp_motion_leg.py
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from tensordict import TensorDict

from protomotions.agents.common.config import MLPLayerConfig
from protomotions.agents.common.shared_joint_mlp import SingleJointMLP, SingleJointMLPConfig
from protomotions.envs.control.assist_disturbance_control import (
    DisturbedAssistTargetControl,
    DisturbedAssistTargetControlConfig,
)
from protomotions.envs.control.motion_intent import (
    MotionIntentSource,
    group_clip_probabilities,
    load_motion_clips,
)
from protomotions.envs.rewards.assist import (
    compute_assist_effort_rew,
    compute_human_power_rew,
    compute_target_tracking_rew,
)
from protomotions.envs.terminations.assist import (
    compute_dof_vel_failure_term,
    compute_tracking_failure_term,
)
from protomotions.robot_configs.hip_pendulum import HipPendulumRobotConfig
from protomotions.tests.test_motion_intent import _write_zarr

KINDS = ["walking", "running", "running, turning", "sitting", "dancing", "walking"]
FREQS = [1.0, 3.0, 2.5, 0.3, 0.6, 1.2]  # Hz → 달리기 클립이 가장 빠름
GROUPS = (("running", r"^running"), ("deep", r"sitting"), ("walking", r"^walking$"), ("other", r".*"))


def _clips(fps=100.0, n=400):
    t = np.arange(n) / fps
    out = []
    for f in FREQS:
        theta = 0.4 * np.sin(2 * np.pi * f * t)
        out.append(np.stack([theta, -theta, np.zeros(n), np.zeros(n)], 1).astype(np.float32))
    return out


def _dataset(tmp_path):
    path = _write_zarr(tmp_path / "m.zarr", _clips())
    import zarr

    zarr.open(str(path), mode="r+").attrs["fps"] = 100.0
    parquet = tmp_path / "meta.parquet"
    pd.DataFrame({"filename": [f"clip{i}.motion" for i in range(len(KINDS))],
                  "content_type_of_movement": KINDS}).to_parquet(parquet)
    return str(path), str(parquet)


def test_group_probabilities_match_time_shares():
    kinds = ["running", "walking", "walking", "sitting", "dancing"]
    lengths = np.array([300, 200, 600, 400, 1000])
    weights = (("running", 0.5), ("walking", 0.3), ("deep", 0.2))
    probs, group = group_clip_probabilities(kinds, lengths, GROUPS, weights, min_frames=150)
    assert list(group) == [0, 2, 2, 1, 3]
    remaining = (lengths + 150) / 2
    share = {name: float((probs * remaining)[group == j].sum()) for j, (name, _) in enumerate(GROUPS)}
    total = sum(share.values())
    assert share["other"] == 0.0
    for name, w in weights:
        assert share[name] / total == pytest.approx(w)
    with pytest.raises(ValueError):
        group_clip_probabilities(kinds, lengths, GROUPS, (("missing", 1.0),), 150)


def test_loader_reads_fps_kinds_and_acceleration(tmp_path):
    path, parquet = _dataset(tmp_path)
    table = load_motion_clips(path, metadata_parquet=parquet, with_accel99=True)
    assert table.fps == 100.0 and table.clip_type == KINDS
    expected = 0.4 * (2 * np.pi * np.asarray(FREQS)) ** 2
    accel = table.clip_accel99[:, 0].numpy()
    assert np.all(np.abs(accel / expected - 1) < 0.1)  # 5프레임 평활의 감쇠 몇 % 이내
    src = MotionIntentSource(table, num_envs=2, device=torch.device("cpu"),
                             default_dof_pos=torch.zeros(2))
    assert src.fps == 100.0 and src.min_frames == 150


def test_pool_sampling_only_draws_feasible_clips(tmp_path):
    path, parquet = _dataset(tmp_path)
    table = load_motion_clips(path, metadata_parquet=parquet, with_accel99=True)
    n = 64
    src = MotionIntentSource(table, num_envs=n, device=torch.device("cpu"),
                             default_dof_pos=torch.zeros(2), seed=3)
    lengths = table.clip_len.numpy()
    fast, _ = group_clip_probabilities(KINDS, lengths, GROUPS, (("running", 1.0),), src.min_frames)
    calm, _ = group_clip_probabilities(KINDS, lengths, GROUPS, (("walking", .5), ("deep", .5)), src.min_frames)
    pools = torch.arange(n) % 2
    src.configure_sampling(torch.tensor(np.stack([fast, calm])), pools)
    accel = table.clip_accel99.max(dim=1).values
    cap = torch.full((n, 2), float("inf"))
    cap[0::4] = float(accel[2]) + 1.0  # 집단 0 일부: 달리기 클립 1(3 Hz) 불가능, 2(2.5 Hz)만 가능
    src.set_accel_cap(cap)
    ids = torch.arange(n)
    for _ in range(30):
        src._assign_clip(1, ids, torch.zeros(n))
        clip = src._clip[1]
        assert torch.all(torch.isin(clip[pools == 0], torch.tensor([1, 2])))
        assert torch.all(torch.isin(clip[pools == 1], torch.tensor([0, 3, 5])))
        assert torch.all(clip[0::4] == 2)
    assert src.fallback_count == 0
    cap[0::2] = 1.0  # 어떤 달리기 클립도 불가능 → 집단에서 가장 느린 클립으로 대체
    src.set_accel_cap(cap)
    src._assign_clip(1, ids, torch.zeros(n))
    assert torch.all(src._clip[1][0::2] == 2) and src.fallback_count > 0


def _fake_env(n):
    return SimpleNamespace(num_envs=n, dt=.01, device=torch.device("cpu"),
                           robot_config=HipPendulumRobotConfig(),
                           progress_buf=torch.zeros(n, dtype=torch.long),
                           state_history=SimpleNamespace(dof_pos=torch.zeros(n, 1, 2),
                                                         dof_vel=torch.zeros(n, 1, 2)))


def test_mixed_pools_assign_sources_inertia_and_feasibility(tmp_path):
    path, parquet = _dataset(tmp_path)
    pools = (("synthetic", .25, "synthetic", 1.0, 25.0, ()),
             ("fast", .5, "motion", 1.0, 3.9, (("running", 1.0),)),
             ("calm", .25, "motion", 1.0, 25.0, (("walking", .5), ("deep", .5))))
    cfg = DisturbedAssistTargetControlConfig(
        intent_source="motion_data", motion_zarr_path=path, motion_metadata_parquet=parquet,
        intent_pools=pools, motion_groups=GROUPS, motion_accel_frac=0.8,
        motion_accel_check_dofs=(0,), hlp_dt=.01, chunk_knots=6, chunk_refresh_steps=1,
        obs_future_samples=3, obs_future_dt=.01)
    n = 16
    torch.manual_seed(0)
    comp = DisturbedAssistTargetControl(cfg, _fake_env(n))
    counts = torch.bincount(comp.env_pool, minlength=3).tolist()
    assert counts == [4, 8, 4]
    assert torch.equal(comp.env_inertia_range[comp.env_pool == 1], torch.tensor([[1.0, 3.9]]).expand(8, 2))
    t = torch.full((n, 1), 0.9)
    target = comp._target_at(t)[:, 0]
    motion = comp._intent.theta_at(t)[:, 0].clamp(comp._target_low, comp._target_high)
    synthetic = super(type(comp).__mro__[1], comp)._target_at(t)[:, 0]
    m = comp._motion_env
    torch.testing.assert_close(target[m], motion[m])
    torch.testing.assert_close(target[~m], synthetic[~m])
    # 관성이 정해지면 실현 가능성 상한이 걸리고, 합성 env는 검사하지 않는다
    inertia = torch.full((n, 2), 0.5)
    comp.set_wearer_gains(torch.full((n, 2), 300.), torch.full((n, 2), 15.),
                          torch.full((n, 2), 150.), inertia)
    cap = comp._intent._accel_cap
    assert torch.isinf(cap[~m]).all()
    torch.testing.assert_close(cap[m], torch.full_like(cap[m], 0.8 * 150 / 0.5))
    for slot in (0, 1):
        live = ~comp._intent._hold[slot] & m
        need = comp._intent.table.clip_accel99[comp._intent._clip[slot]][:, 0]
        assert torch.all(need[live] <= cap[live, 0])


def test_joint_masks_restrict_rewards_and_terminations():
    tau = torch.tensor([[3.0, 100.0]])
    vel = torch.tensor([[2.0, 50.0]])
    mask = (1.0, 0.0)
    assert compute_human_power_rew(tau, vel, False, mask).item() == 6.0
    assert compute_human_power_rew(tau, vel, True, mask).item() == 9.0
    assert compute_assist_effort_rew(tau, mask).item() == 9.0
    assert compute_human_power_rew(tau, vel, True).item() == 9.0 + 1e4  # 비면 기존 동작
    q, goal = torch.tensor([[0.1, 2.0]]), torch.tensor([[0.0, 0.0]])
    assert compute_target_tracking_rew(q, goal, 40.0, mask).item() == pytest.approx(np.exp(-40 * 0.01))
    assert not compute_tracking_failure_term(q, goal, 1.0, mask).item()
    assert compute_tracking_failure_term(q, goal, 1.0).item()
    assert not compute_dof_vel_failure_term(vel, 25.0, mask).item()
    assert compute_dof_vel_failure_term(vel, 25.0).item()


@pytest.mark.parametrize("joint", [0, 1])
def test_single_joint_actor_reads_only_its_joint(joint):
    cfg = SingleJointMLPConfig(num_out=2, normalize_obs=True, joint_index=joint,
                               in_keys=["assist_proprio", "assist_target", "previous_actions"],
                               out_keys=["mu"], layers=[MLPLayerConfig(units=16)])
    model = SingleJointMLP(cfg).eval()
    obs = TensorDict({"assist_proprio": torch.randn(4, 104),
                      "assist_target": torch.randn(4, 13),
                      "previous_actions": torch.randn(4, 20)}, batch_size=[4])
    mu = model(obs.clone())["mu"]
    assert torch.all(mu[:, 1 - joint] == 0)
    changed = obs.clone()
    for key in cfg.in_keys:
        stop = -1 if key == "assist_target" else None
        changed[key][..., :stop][..., 1 - joint::2] += 5.0
    torch.testing.assert_close(model(changed)["mu"], mu, rtol=0, atol=0)
    model(obs.clone())["mu"].sum().backward()
    assert all(p.grad is not None for p in model.parameters())
