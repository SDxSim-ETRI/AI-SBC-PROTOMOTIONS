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
"""Flow actor(FPO++)와 Temporal Transformer 컨텍스트 인코더.

데이터 흐름 (rollout 한 step):
    관측 dict ──cat·정규화──▶ flat ──┬─(MLP 변형) context = flat
                                   └─(transformer 변형) 토큰화 → encoder → 현재 토큰 읽기 (+ 현재 특징) = context
    context ──▶ [x_1 ~ N(0,I)] ─ N번 Euler: x ← x + v(context, t, x)·dt ─▶ x_0 ──×actor_scale──▶ action
    context ──▶ (ε, t) K개 표본 ──▶ CFM 손실 L_old (학습 때 비율 exp(L_old − L_new)의 분자)

context는 step마다 **한 번** 계산되고 flow 적분의 N번·CFM의 K개 표본 모두에 재사용된다 — transformer는
flow step마다 다시 돌지 않는다 (`TemporalTransformer.forward` 호출 수 = action당 1).

Key Classes:
    - TemporalTransformer: 평탄 관측 → 시간별 토큰 → self-attention → context
    - FlowActor: TensorDictModuleBase; forward(rollout 표본), act_deterministic(평가), cfm_loss(학습)
"""

from typing import Optional, Tuple

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from torch import Tensor, nn

from protomotions.agents.common.common import NormObsBase
from protomotions.agents.fpo.config import FlowActorConfig, TemporalTransformerConfig
from protomotions.agents.utils.training import get_activation_func

TRANSFORMER_IN_KEYS = ("assist_proprio", "assist_target", "previous_actions")


class TemporalTransformer(nn.Module):
    """평탄 LLP 관측을 시간별 토큰으로 되돌려 요약한다 (설계는 `TemporalTransformerConfig` 참조).

    입력 flat = [assist_proprio | assist_target | previous_actions] (정규화 후, 이 순서).
        assist_proprio   = [q(d), q̇·s(d), hist_q(H·d, index 0 = 1 step 전), hist_q̇·s(H·d)]
        assist_target    = [θ_d(S·d), θ_d − q(S·d), chunk_age(1)]
        previous_actions = [a(P·d), index j = (j + action_lag_offset) step 전에 계산된 raw action]
    """

    def __init__(self, config: TemporalTransformerConfig):
        super().__init__()
        self.config = config
        d, H, P, S = config.num_dofs, config.history_steps, config.prev_action_steps, config.future_samples
        self.d, self.H, self.P, self.S = d, H, P, S
        self.proprio_dim = d * (1 + H) * 2
        self.target_dim = S * d * 2 + 1
        self.prev_dim = P * d
        self.flat_dim = self.proprio_dim + self.target_dim + self.prev_dim
        self.n_past = H + 1
        self.n_future = S
        self.n_tokens = self.n_past + self.n_future
        self.past_dim = 3 * d + 1  # q, q̇, a, a_valid
        self.future_dim = 2 * d + 1  # θ_d, θ_d − q, chunk_age
        self.current_dim = 2 * d + self.target_dim  # q, q̇, θ_d 창, 상대값, chunk_age
        self.embed_past = nn.Linear(self.past_dim, config.d_model)
        self.embed_future = nn.Linear(self.future_dim, config.d_model)
        self.slot_embed = nn.Parameter(torch.randn(1, self.n_tokens, config.d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.ff_size,
            dropout=config.dropout,
            activation=get_activation_func(config.activation, return_type="functional"),
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers, enable_nested_tensor=False)
        self.readout_norm = nn.LayerNorm(config.d_model)
        self.out_dim = config.d_model + (self.current_dim if config.concat_current_features else 0)
        # action이 들어가는 lag 슬롯: previous_actions[j] → lag j + offset (H 이내만)
        lags = [j + config.action_lag_offset for j in range(P)]
        self._action_slots = [(j, lag) for j, lag in enumerate(lags) if lag <= H]

    # ------------------------------------------------------------------
    def tokenize(self, flat: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """flat (B, flat_dim) → (past (B, H+1, past_dim), future (B, S, future_dim), current (B, current_dim))."""
        assert flat.shape[-1] == self.flat_dim, f"flat obs {flat.shape[-1]} ≠ expected {self.flat_dim}"
        B, d, H, S = flat.shape[0], self.d, self.H, self.S
        proprio = flat[:, : self.proprio_dim]
        target = flat[:, self.proprio_dim : self.proprio_dim + self.target_dim]
        prev = flat[:, self.proprio_dim + self.target_dim :]
        q = proprio[:, :d]
        qd = proprio[:, d : 2 * d]
        hist_q = proprio[:, 2 * d : 2 * d + H * d].reshape(B, H, d)
        hist_qd = proprio[:, 2 * d + H * d :].reshape(B, H, d)
        pos = torch.cat([q.unsqueeze(1), hist_q], dim=1)  # lag 0..H
        vel = torch.cat([qd.unsqueeze(1), hist_qd], dim=1)
        act = flat.new_zeros(B, H + 1, d)
        valid = flat.new_zeros(B, H + 1, 1)
        prev_a = prev.reshape(B, self.P, d)
        for j, lag in self._action_slots:
            act[:, lag] = prev_a[:, j]
            valid[:, lag] = 1.0
        past = torch.cat([pos, vel, act, valid], dim=-1)
        theta_d = target[:, : S * d].reshape(B, S, d)
        rel = target[:, S * d : 2 * S * d].reshape(B, S, d)
        age = target[:, 2 * S * d :].unsqueeze(1).expand(B, S, 1)
        future = torch.cat([theta_d, rel, age], dim=-1)
        current = torch.cat([q, qd, target], dim=-1)
        return past, future, current

    def forward(self, flat: Tensor) -> Tensor:
        past, future, current = self.tokenize(flat)
        tokens = torch.cat([self.embed_past(past), self.embed_future(future)], dim=1) + self.slot_embed
        out = self.encoder(tokens)  # (B, n_tokens, d_model), 마스크 없음 (양방향)
        ctx = self.readout_norm(out[:, 0])  # 현재 상태 토큰 (lag 0)
        if self.config.concat_current_features:
            ctx = torch.cat([ctx, current], dim=-1)
        return ctx


class FlowActor(TensorDictModuleBase):
    """FPO++ flow 정책 (설계·수식은 모듈 docstring과 `FlowActorConfig` 참조)."""

    config: FlowActorConfig

    def __init__(self, config: FlowActorConfig):
        TensorDictModuleBase.__init__(self)
        self.config = config
        assert config.in_keys, "FlowActor requires in_keys"
        assert config.num_out is not None and config.num_out > 0, "FlowActor requires num_out"
        assert config.timestep_embed_dim % 2 == 0, "timestep_embed_dim must be even"
        self.norm = NormObsBase(config)
        self.context_net: Optional[TemporalTransformer] = None
        if config.context is not None:
            assert tuple(config.in_keys) == TRANSFORMER_IN_KEYS, (
                f"TemporalTransformer expects in_keys {TRANSFORMER_IN_KEYS}, got {config.in_keys}"
            )
            self.context_net = TemporalTransformer(config.context)
        act = config.velocity_activation
        layers = []
        units = [layer.units for layer in config.layers]
        assert units, "FlowActor requires at least one hidden layer"
        layers.append(nn.LazyLinear(units[0]))
        layers.append(get_activation_func(act))
        for i in range(1, len(units)):
            layers.append(nn.Linear(units[i - 1], units[i]))
            layers.append(get_activation_func(act))
        final = nn.Linear(units[-1], config.num_out)
        if config.final_layer_weight_scale is not None and config.final_layer_weight_scale != 1.0:
            with torch.no_grad():
                final.weight.mul_(config.final_layer_weight_scale)
                final.bias.mul_(config.final_layer_weight_scale)
        layers.append(final)
        self.velocity_net = nn.Sequential(*layers)
        self.register_buffer(
            "_freqs", 2.0 ** torch.arange(config.timestep_embed_dim // 2, dtype=torch.float32), persistent=False
        )
        self.in_keys = list(config.in_keys)
        self.out_keys = list(config.out_keys)
        for key in ["action", "mean_action", "cfm_eps", "cfm_t", "cfm_loss_old"]:
            assert key in self.out_keys, f"FlowActor output key {key} not in out_keys {self.out_keys}"

    # ------------------------------------------------------------------
    # context (step당 한 번)
    # ------------------------------------------------------------------
    def flat_obs(self, tensordict: TensorDict) -> Tensor:
        return torch.cat([tensordict[key] for key in self.in_keys], dim=-1)

    def context(self, tensordict: TensorDict) -> Tensor:
        norm_obs = self.norm(self.flat_obs(tensordict))
        if self.context_net is None:
            return norm_obs
        return self.context_net(norm_obs)

    # ------------------------------------------------------------------
    # 속도장과 적분
    # ------------------------------------------------------------------
    def embed_timestep(self, t: Tensor) -> Tensor:
        """(..., 1) → (..., timestep_embed_dim): [cos(t·2^k), sin(t·2^k)]."""
        scaled = t * self._freqs
        return torch.cat([torch.cos(scaled), torch.sin(scaled)], dim=-1)

    def velocity(self, ctx: Tensor, t: Tensor, x: Tensor) -> Tensor:
        """v(context, t, x_t); ctx·t·x는 같은 선행 차원 (B, ·) 또는 (B, K, ·)."""
        h = torch.cat([ctx, self.embed_timestep(t), x], dim=-1)
        return self.velocity_net(h) * self.config.mlp_output_scale

    def integrate(self, ctx: Tensor, x1: Tensor) -> Tensor:
        """x_1 → x_0: t = 1 → 0을 N등분한 Euler 적분. context는 고정(재계산 없음)."""
        n = int(self.config.sampling_steps)
        x = x1
        dt = -1.0 / n
        for i in range(n):
            t = x.new_full((*x.shape[:-1], 1), 1.0 - i / n)
            x = x + self.velocity(ctx, t, x) * dt
        return x

    # ------------------------------------------------------------------
    # CFM 손실
    # ------------------------------------------------------------------
    def sample_cfm_times(self, batch: int, device) -> Tuple[Tensor, Tensor]:
        """(ε (B, K, A), t (B, K, 1)); t ~ Beta(1, β)를 [0.005, 0.995]로."""
        K, A = int(self.config.n_samples_per_action), int(self.config.num_out)
        eps = torch.randn(batch, K, A, device=device)
        u = torch.rand(batch, K, 1, device=device)
        t = 0.005 + 0.99 * (1.0 - (1.0 - u) ** (1.0 / float(self.config.t_inverse_cdf_beta)))
        return eps, t

    def cfm_from_context(self, ctx: Tensor, actions: Tensor, eps: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """(손실 (B, K), x_0 예측 (B, K, A)). 손실 = ‖v − (ε − a/scale)‖² / √A (sqrt reduction)."""
        B, K, A = eps.shape
        scaled = actions / float(self.config.actor_scale)
        x_t = t * eps + (1.0 - t) * scaled.unsqueeze(1)
        ctx_k = ctx.unsqueeze(1).expand(B, K, ctx.shape[-1])
        v = self.velocity(ctx_k, t, x_t)
        target = eps - scaled.unsqueeze(1)
        sq = (v - target).square()
        red = self.config.cfm_loss_reduction
        if red == "mean":
            loss = sq.mean(dim=-1)
        elif red == "sum":
            loss = sq.sum(dim=-1)
        elif red == "sqrt":
            loss = sq.sum(dim=-1) / (A**0.5)
        else:
            raise ValueError(f"unknown cfm_loss_reduction {red}")
        x0_pred = x_t - t * v
        return loss, x0_pred

    def cfm_loss(self, tensordict: TensorDict, actions: Tensor, eps: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """학습용: 저장된 action과 (ε, t)에서 현재 가중치의 CFM 손실 (context 한 번 계산)."""
        return self.cfm_from_context(self.context(tensordict), actions, eps, t)

    # ------------------------------------------------------------------
    # TensorDict 인터페이스
    # ------------------------------------------------------------------
    def forward(self, tensordict: TensorDict) -> TensorDict:
        """rollout: 표본 action(+perturb), 결정적 action, CFM 표본과 수집 시점 손실."""
        ctx = self.context(tensordict)
        B, A = ctx.shape[0], int(self.config.num_out)
        scale = float(self.config.actor_scale)
        action = scale * self.integrate(ctx, torch.randn(B, A, device=ctx.device, dtype=ctx.dtype))
        if self.config.action_perturb_std > 0:
            action = action + self.config.action_perturb_std * torch.randn_like(action)
        mean_action = scale * self.integrate(ctx, torch.zeros(B, A, device=ctx.device, dtype=ctx.dtype))
        eps, t = self.sample_cfm_times(B, ctx.device)
        loss_old, _ = self.cfm_from_context(ctx, action, eps, t)
        tensordict["action"] = action
        tensordict["mean_action"] = mean_action
        tensordict["cfm_eps"] = eps
        tensordict["cfm_t"] = t
        tensordict["cfm_loss_old"] = loss_old
        return tensordict

    def act_deterministic(self, tensordict: TensorDict) -> TensorDict:
        """평가: x_1 = 0에서 적분한 action을 mu_key에 넣는다 (RNG 소비 없음)."""
        ctx = self.context(tensordict)
        B, A = ctx.shape[0], int(self.config.num_out)
        zeros = torch.zeros(B, A, device=ctx.device, dtype=ctx.dtype)
        tensordict[self.config.mu_key] = float(self.config.actor_scale) * self.integrate(ctx, zeros)
        return tensordict
