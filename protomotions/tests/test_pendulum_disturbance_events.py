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
"""진자 외란 컴포넌트의 구조화 사건·명령 홀드·유각 감속 기능 (2026-10-26 병합).

2026-10-14/15 연구의 `PendulumResearchControl`·`PendulumGeneralControl`을 `DisturbedAssistTargetControl`의
method로 합쳤다. 아래 해시는 병합 직전(스냅샷 커밋 8058f3c)의 옛 클래스 세 개로 같은 시나리오를 돌려
기록한 값이다: 기능이 꺼진 기본 config는 옛 `DisturbedAssistTargetControl`과, 켠 config는 옛
연구 클래스와 외부 토크·PD 명령·게인 스케줄·RNG 상태까지 같아야 한다.

    pytest protomotions/tests/test_pendulum_disturbance_events.py
"""

import hashlib
from types import SimpleNamespace

import torch

from protomotions.envs.control.assist_disturbance_control import (
    DisturbedAssistTargetControl,
    DisturbedAssistTargetControlConfig,
)
from protomotions.robot_configs.hip_pendulum import HipPendulumRobotConfig

# (config 필드, seed, step 수, 옛 클래스로 기록한 sha256)
LEGACY_DIGESTS = {
    "base_default": ({}, 21, 300, "9e07b5e24db1de7f8eb6cc3da4ee95774036a146c158ff0560b55a855ec8936d"),
    "research_structured": (dict(structured_probability=1.0, human_target_hold_steps=5), 7, 400,
                            "55888770601499117ebc8364f373c26800b8a0c5344e5f1bbb845ada98778d99"),
    "general_phase": (dict(structured_probability=.5, phase_probability=.5, human_target_hold_steps=5), 11, 400,
                      "d8989a3a12d322dc46558170cdbbad205d0343bda2e8553195d0d4b652606006"),
    "general_phase_only": (dict(phase_probability=1.0), 3, 400,
                           "2affc1595fde5a275246ee4368d682395ca9a3d81efc22d6d292569bceaf9eb9"),
}


def control(config, n=8):
    env = SimpleNamespace(num_envs=n, dt=.01, device=torch.device("cpu"), robot_config=HipPendulumRobotConfig(),
                          progress_buf=torch.zeros(n, dtype=torch.long),
                          state_history=SimpleNamespace(dof_pos=torch.zeros(n, 1, 2), dof_vel=torch.zeros(n, 1, 2)))
    return DisturbedAssistTargetControl(config, env)


def _digest(fields, seed, steps):
    torch.manual_seed(seed)
    obj = control(DisturbedAssistTargetControlConfig(**fields))
    h = hashlib.sha256()
    h.update(torch.get_rng_state().numpy().tobytes())
    for step in range(steps):
        obj.env.progress_buf[:] = step
        obj.env.state_history.dof_vel[:] = torch.sin(torch.arange(8, dtype=torch.float32)[:, None, None] + step * 0.03) * 4
        obj.env.state_history.dof_pos[:] = torch.cos(torch.arange(8, dtype=torch.float32)[:, None, None] * 0.5 + step * 0.02) * 0.4
        if step % 97 == 50:
            obj.reset(torch.tensor([1, 4]))
        obj.step()
        tgt, vel = obj.get_pd_targets_and_vel()
        for t in (obj.external_torque(), tgt, vel, *obj.gain_schedule()):
            h.update(t.detach().numpy().tobytes())
    h.update(torch.get_rng_state().numpy().tobytes())
    return h.hexdigest()


def test_merged_control_reproduces_the_pre_merge_classes():
    for name, (fields, seed, steps, expected) in LEGACY_DIGESTS.items():
        assert _digest(fields, seed, steps) == expected, name


def test_disabled_features_consume_no_random_numbers():
    torch.manual_seed(5)
    obj = control(DisturbedAssistTargetControlConfig())
    for step in range(50):
        obj.env.progress_buf[:] = step
        before = torch.get_rng_state()
        obj._structured_events(obj._episode_time().unsqueeze(1).expand(-1, obj.n_dofs))
        obj._phase_events()
        assert torch.equal(before, torch.get_rng_state())
        assert not obj._tau_field.any() and not obj._tau_damping.any()
        assert not obj._tau_swing.any() and not obj._tau_shock.any()


def test_target_hold_keeps_command_but_not_intent_frozen():
    obj = control(DisturbedAssistTargetControlConfig(human_target_hold_steps=5))
    obj.env.progress_buf[:] = 20
    first = tuple(x.clone() for x in obj.get_pd_targets_and_vel())
    obj.env.progress_buf[:] = 24
    for a, b in zip(first, obj.get_pd_targets_and_vel()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not torch.equal(first[0], obj._target_at(obj._episode_time().unsqueeze(1)).squeeze(1))
    obj.env.progress_buf[:] = 25
    assert not torch.equal(first[0], obj.get_pd_targets())
    obj.reset(torch.tensor([0, 3]))
    obj.env.progress_buf[[0, 3]] = 0
    obj.get_pd_targets()
    assert obj._hold_step.tolist() == [0, 25, 25, 0, 25, 25, 25, 25]


def test_damping_is_passive_bounded_and_state_independent_rng():
    cfg = DisturbedAssistTargetControlConfig(structured_probability=1.0)
    torch.manual_seed(7)
    a = control(cfg)
    torch.manual_seed(7)
    b = control(cfg)
    a.env.state_history.dof_vel[:] = 4
    b.env.state_history.dof_vel[:] = -4
    b.env.state_history.dof_pos[:] = .4
    active = 0
    for step in range(400):
        a.env.progress_buf[:] = b.env.progress_buf[:] = step
        before = torch.get_rng_state()
        a.step()
        after = torch.get_rng_state()
        torch.set_rng_state(before)
        b.step()
        assert torch.equal(after, torch.get_rng_state())
        for obj in (a, b):
            assert (obj._tau_damping * obj.env.state_history.dof_vel[:, 0] <= 0).all()
            assert obj._tau_damping.abs().max() <= cfg.damping_torque_limit
            assert (obj._tau_field.abs() <= obj._ev_load.abs() + 1e-5).all()
            assert torch.isfinite(obj.external_torque()).all()
        active += int(a._tau_damping.count_nonzero())
    assert active > 0


def test_phase_events_are_paired_passive_bounded_and_reset_locally():
    cfg = DisturbedAssistTargetControlConfig(phase_probability=1., structured_probability=.5)
    torch.manual_seed(7)
    a = control(cfg)
    torch.manual_seed(7)
    b = control(cfg)
    a.env.state_history.dof_vel[:] = 4.
    b.env.state_history.dof_vel[:] = -3.
    counts = {"swing": 0, "shock": 0}
    for step in range(1000):
        a.env.progress_buf[:] = b.env.progress_buf[:] = step
        before = torch.get_rng_state()
        a.step()
        after = torch.get_rng_state()
        torch.set_rng_state(before)
        b.step()
        assert torch.equal(after, torch.get_rng_state())
        for event in counts:
            torch.testing.assert_close(getattr(a, f"_{event}_start"), getattr(b, f"_{event}_start"), rtol=0, atol=0)
            counts[event] += int(getattr(a, f"_tau_{event}").count_nonzero())
            for obj in (a, b):
                torque = getattr(obj, f"_tau_{event}")
                assert (torque * obj.env.state_history.dof_vel[:, 0] <= 0).all()
                assert torque.abs().max() <= cfg.phase_torque_limit
    assert all(counts.values()), counts
    other = a._swing_start[1:].clone()
    a.reset(torch.tensor([0]))
    assert (a._swing_start[0] == -1).all()
    torch.testing.assert_close(other, a._swing_start[1:], rtol=0, atol=0)
    assert not a._tau_swing[0].any()
