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
"""FPO++ flow actor · Temporal Transformer · FPO agent 불변량 (시뮬레이터 없이).

    pytest protomotions/tests/test_fpo_agent.py
"""

import argparse
import importlib.util
import types
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from protomotions.agents.common.config import MLPLayerConfig
from protomotions.agents.fpo.agent import FPO, fpo_surrogate
from protomotions.agents.fpo.config import FlowActorConfig, TemporalTransformerConfig
from protomotions.agents.fpo.model import FlowActor, TemporalTransformer

ROOT = Path(__file__).resolve().parents[2]
KEYS = ["assist_proprio", "assist_target", "previous_actions"]
SMALL = dict(num_dofs=2, history_steps=4, prev_action_steps=2, future_samples=2)  # proprio 20, target 9, prev 4


def small_obs(batch, seed=0):
    g = torch.Generator().manual_seed(seed)
    d, H, P, S = SMALL["num_dofs"], SMALL["history_steps"], SMALL["prev_action_steps"], SMALL["future_samples"]
    return {
        "assist_proprio": torch.randn(batch, d * (1 + H) * 2, generator=g),
        "assist_target": torch.randn(batch, S * d * 2 + 1, generator=g),
        "previous_actions": torch.randn(batch, P * d, generator=g),
    }


def make_actor(transformer: bool, steps=4, samples=8, scale=0.25, layers=(32, 32)):
    ctx = TemporalTransformerConfig(d_model=16, num_heads=2, num_layers=1, ff_size=32, **SMALL) if transformer else None
    cfg = FlowActorConfig(
        in_keys=KEYS, num_out=2, normalize_obs=True, norm_clamp_value=5, context=ctx,
        layers=[MLPLayerConfig(units=u, activation="relu") for u in layers],
        sampling_steps=steps, n_samples_per_action=samples, actor_scale=scale,
    )
    actor = FlowActor(cfg)
    with torch.no_grad():
        actor.eval()(TensorDict(small_obs(3), batch_size=3))  # lazy 모듈 굳히기
    return actor


# ----------------------------------------------------------------------
# 토큰화: 평탄 관측의 각 조각이 맞는 시간 슬롯으로 간다
# ----------------------------------------------------------------------
def test_tokenizer_maps_history_and_actions_to_time_slots():
    d, H, P, S = 2, 4, 2, 2
    tfm = TemporalTransformer(TemporalTransformerConfig(num_dofs=d, history_steps=H, prev_action_steps=P, future_samples=S))
    q = torch.tensor([[1.0, -1.0]]); qd = torch.tensor([[2.0, -2.0]])
    hist_q = torch.stack([torch.tensor([10.0 + k, -(10.0 + k)]) for k in range(H)]).unsqueeze(0)  # lag 1..H
    hist_qd = torch.stack([torch.tensor([20.0 + k, -(20.0 + k)]) for k in range(H)]).unsqueeze(0)
    theta = torch.tensor([[[30.0, -30.0], [31.0, -31.0]]]); rel = theta - q.unsqueeze(1); age = torch.tensor([[0.7]])
    prev = torch.tensor([[[40.0, -40.0], [41.0, -41.0]]])  # j = 0, 1 → lag 2, 3
    flat = torch.cat([q, qd, hist_q.reshape(1, -1), hist_qd.reshape(1, -1),
                      theta.reshape(1, -1), rel.reshape(1, -1), age, prev.reshape(1, -1)], dim=-1)
    past, future, current = tfm.tokenize(flat)
    assert past.shape == (1, H + 1, 3 * d + 1) and future.shape == (1, S, 2 * d + 1)
    torch.testing.assert_close(past[0, 0, :2], q[0]); torch.testing.assert_close(past[0, 0, 2:4], qd[0])
    for k in range(1, H + 1):
        torch.testing.assert_close(past[0, k, :2], hist_q[0, k - 1]); torch.testing.assert_close(past[0, k, 2:4], hist_qd[0, k - 1])
    torch.testing.assert_close(past[0, 2, 4:6], prev[0, 0]); torch.testing.assert_close(past[0, 3, 4:6], prev[0, 1])
    assert past[0, :, 6].tolist() == [0.0, 0.0, 1.0, 1.0, 0.0]  # action 유효 플래그
    torch.testing.assert_close(future[0, :, :2], theta[0]); torch.testing.assert_close(future[0, :, 2:4], rel[0])
    assert torch.all(future[0, :, 4] == 0.7)
    torch.testing.assert_close(current[0, :4], torch.cat([q[0], qd[0]]))
    assert tfm(flat).shape == (1, tfm.out_dim)


# ----------------------------------------------------------------------
# context는 action당 한 번, 속도장만 flow step 수만큼
# ----------------------------------------------------------------------
def test_transformer_runs_once_per_action_not_per_flow_step():
    actor = make_actor(transformer=True, steps=5, samples=8)
    calls = {"ctx": 0, "vel": 0}
    actor.context_net.register_forward_hook(lambda *_: calls.__setitem__("ctx", calls["ctx"] + 1))
    actor.velocity_net.register_forward_hook(lambda *_: calls.__setitem__("vel", calls["vel"] + 1))
    td = TensorDict(small_obs(6), batch_size=6)
    with torch.no_grad():
        actor(td.clone())
    assert calls["ctx"] == 1 and calls["vel"] == 2 * 5 + 1  # 표본 적분 5 + 결정적 적분 5 + CFM 표본 1
    calls.update(ctx=0, vel=0)
    with torch.no_grad():
        actor.act_deterministic(td.clone())
    assert calls["ctx"] == 1 and calls["vel"] == 5
    calls.update(ctx=0, vel=0)
    out = actor(td.clone())
    actor.cfm_loss(td.clone(), out["action"], out["cfm_eps"], out["cfm_t"])
    assert calls["ctx"] == 2


def test_deterministic_action_is_repeatable_and_outputs_have_shapes():
    actor = make_actor(transformer=False, steps=3, samples=5)
    td = TensorDict(small_obs(4), batch_size=4)
    with torch.no_grad():
        a1 = actor.act_deterministic(td.clone())["mean_action"]
        a2 = actor.act_deterministic(td.clone())["mean_action"]
        out = actor(td.clone())
    torch.testing.assert_close(a1, a2, rtol=0, atol=0)
    assert out["action"].shape == (4, 2) and out["cfm_eps"].shape == (4, 5, 2)
    assert out["cfm_t"].shape == (4, 5, 1) and out["cfm_loss_old"].shape == (4, 5)
    assert torch.all((out["cfm_t"] > 0.0) & (out["cfm_t"] < 1.0))


# ----------------------------------------------------------------------
# FPO 비율: 같은 가중치면 1, gradient는 transformer까지 흐른다
# ----------------------------------------------------------------------
def test_ratio_is_one_for_unchanged_weights_and_gradients_reach_transformer():
    actor = make_actor(transformer=True, steps=3, samples=6)
    td = TensorDict(small_obs(5, seed=3), batch_size=5)
    with torch.no_grad():
        out = actor(td.clone())
    new_loss, x0 = actor.cfm_loss(td.clone(), out["action"], out["cfm_eps"], out["cfm_t"])
    torch.testing.assert_close(new_loss, out["cfm_loss_old"], rtol=1e-5, atol=1e-6)
    adv = torch.randn(5, 1)
    loss, info = fpo_surrogate(out["cfm_loss_old"], new_loss, adv, clip=0.05)
    assert abs(info["actor/ratio_mean"].item() - 1.0) < 1e-5 and info["actor/clip_frac"].item() == 0.0
    loss.backward()
    grads = [p.grad for p in actor.context_net.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_surrogate_pushes_ratio_up_for_positive_advantage_and_clips():
    old = torch.zeros(4, 3)
    adv = torch.tensor([[1.0], [1.0], [-1.0], [-1.0]])
    new = torch.tensor([[-0.01], [-0.2], [-0.01], [-0.2]]).expand(4, 3)  # ratio 1.01, 1.22, 1.01, 1.22
    loss, info = fpo_surrogate(old, new, adv, clip=0.05, mode="aspo")
    # 양의 advantage: 비율 1.22는 1.05로 잘려 1.01보다 손실이 작아지지 않는다 (PPO clip)
    ppo_loss, _ = fpo_surrogate(old[:2], new[:2], adv[:2], clip=0.05, mode="ppo")
    spo_loss, _ = fpo_surrogate(old[2:], new[2:], adv[2:], clip=0.05, mode="spo")
    torch.testing.assert_close(loss, 0.5 * (ppo_loss + spo_loss))
    assert ppo_loss.item() == pytest.approx(-0.5 * (torch.exp(torch.tensor(0.01)).item() + 1.05), abs=1e-6)


# ----------------------------------------------------------------------
# 합성 bandit: FPO++ 갱신이 flow 정책을 보상 최대점으로 옮긴다 (부호·방향 검증)
# ----------------------------------------------------------------------
def test_fpo_updates_move_flow_policy_toward_reward_optimum():
    torch.manual_seed(0)
    actor = make_actor(transformer=False, steps=4, samples=8, scale=0.25, layers=(64, 64))
    actor.config.action_perturb_std = 0.0
    actor.norm.running_obs_norm.fabric = types.SimpleNamespace(  # train 모드의 통계 기록에 필요 (단일 프로세스 흉내)
        world_size=1, global_rank=0, broadcast=lambda x, src=0: x)
    opt = torch.optim.Adam(actor.parameters(), lr=3e-3)
    target = torch.tensor([0.4, -0.3])
    B = 256

    def mean_error():
        actor.eval()
        with torch.no_grad():
            a = actor.act_deterministic(TensorDict(small_obs(B, seed=99), batch_size=B))["mean_action"]
        return (a - target).norm(dim=-1).mean().item()

    err0 = mean_error()
    for it in range(150):
        td = TensorDict(small_obs(B, seed=it), batch_size=B)
        actor.eval()
        with torch.no_grad():
            out = actor(td.clone())
        reward = -(out["action"] - target).square().sum(-1)
        adv = ((reward - reward.mean()) / (reward.std() + 1e-8)).unsqueeze(-1)
        actor.train()
        for _ in range(4):
            new_loss, _ = actor.cfm_loss(td.clone(), out["action"], out["cfm_eps"], out["cfm_t"])
            loss, _ = fpo_surrogate(out["cfm_loss_old"], new_loss, adv, clip=0.05)
            opt.zero_grad(); loss.backward(); opt.step()
    err1 = mean_error()
    assert err0 > 0.3, err0
    assert err1 < 0.1, (err0, err1)


# ----------------------------------------------------------------------
# checkpoint: 저장본 actor = EMA 가중치, resume은 원 가중치
# ----------------------------------------------------------------------
class _Model(torch.nn.Module):
    def __init__(self, actor):
        super().__init__()
        self._actor = actor
        self._critic = torch.nn.Linear(2, 1)


def _bare_agent(actor, ema_active):
    agent = FPO.__new__(FPO)
    agent.model = _Model(actor)
    agent.device = torch.device("cpu")
    agent.current_epoch, agent.step_count, agent.fit_start_time, agent.best_evaluated_score = 7, 100, 0.0, None
    agent.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    agent.critic_optimizer = torch.optim.Adam(agent.model._critic.parameters(), lr=1e-3)
    agent.adv_mean_ema, agent.adv_std_ema = torch.zeros(1), torch.ones(1)
    agent._ema = {n: torch.full_like(p, 0.5) for n, p in actor.named_parameters() if p.requires_grad}
    agent._ema_active = ema_active
    from protomotions.agents.fpo.config import FPOAgentConfig
    agent.config = FPOAgentConfig(batch_size=1, training_max_steps=1)
    agent.config.normalize_rewards = False
    return agent


def test_checkpoint_bakes_ema_into_actor_and_resume_restores_raw_weights():
    actor = make_actor(transformer=True)
    raw = {k: v.clone() for k, v in actor.state_dict().items()}
    agent = _bare_agent(actor, ema_active=True)
    state = agent.get_state_dict({})
    name = next(n for n, p in actor.named_parameters() if p.requires_grad)
    assert torch.all(state["model"][f"_actor.{name}"] == 0.5)  # 평가·배포용 actor = EMA
    torch.testing.assert_close(state["fpo_actor_raw"][name], raw[name], rtol=0, atol=0)
    torch.testing.assert_close(actor.state_dict()[name], raw[name], rtol=0, atol=0)  # 살아 있는 모델은 그대로
    fresh = _bare_agent(make_actor(transformer=True), ema_active=False)
    fresh.load_parameters(state)
    torch.testing.assert_close(fresh.model._actor.state_dict()[name], raw[name], rtol=0, atol=0)
    assert fresh._ema_active and torch.all(fresh._ema[name] == 0.5)
    before = _bare_agent(make_actor(transformer=True), ema_active=False)
    assert "fpo_actor_raw" not in before.get_state_dict({})  # warmup 전 저장본은 원 가중치 그대로


# ----------------------------------------------------------------------
# 레시피: llp_walk.py --policy tfm_fpo 가 출발 checkpoint의 관측 구성을 transformer 토큰 설정으로 옮긴다
# ----------------------------------------------------------------------
SOURCE = ROOT / "checkpoints/L15_mlp_ppo_pendulum/epoch_6000.ckpt"


@pytest.mark.skipif(not (SOURCE.parent / "resolved_configs.pt").exists(), reason="source checkpoint not available")
@pytest.mark.parametrize("policy", ["mlp_fpo", "tfm_fpo"])
def test_llp_walk_recipe_builds_fpo_agent_config(policy):
    spec = importlib.util.spec_from_file_location("llp_walk", ROOT / "examples/experiments/assist_skeleton/llp_walk.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    parser = argparse.ArgumentParser(); module.additional_experiment_arguments(parser)
    args = parser.parse_args(["--initial-actor-checkpoint", str(SOURCE), "--actor-init", "random", "--policy", policy,
                              "--finetune-actor-lr", "1e-4"])
    args.batch_size, args.training_max_steps = 2048, 8192 * 20
    cfg = module._agent_finetune(args)
    from protomotions.agents.fpo.config import FPOAgentConfig
    assert isinstance(cfg, FPOAgentConfig) and cfg.model.actor.in_keys == KEYS and cfg.model.actor.num_out == 2
    assert cfg.actor_clip_frac_threshold is None and not cfg.clip_critic_loss and cfg.num_mini_epochs == 8
    assert cfg.model.actor_optimizer._target_ == "torch.optim.AdamW" and cfg.model.actor_optimizer.lr == 1e-4
    assert [layer.units for layer in cfg.model.actor.layers] == [512, 256, 128]
    if policy == "tfm_fpo":
        ctx = cfg.model.actor.context
        assert (ctx.num_dofs, ctx.history_steps, ctx.prev_action_steps, ctx.future_samples) == (2, 25, 10, 3)
        tfm = TemporalTransformer(ctx)
        assert tfm.flat_dim == 2 * 26 * 2 + 3 * 2 * 2 + 1 + 10 * 2 == 137 and tfm.n_tokens == 29
    else:
        assert cfg.model.actor.context is None
    with pytest.raises(AssertionError):
        args.actor_init = "checkpoint"; module._agent_finetune(args)
