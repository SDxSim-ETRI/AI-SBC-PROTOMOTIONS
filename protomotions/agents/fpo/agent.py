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
"""FPO++ agent — PPO의 rollout·GAE·critic은 그대로 두고 actor 손실만 CFM 비율로 바꾼다.

PPO와 다른 점 (fpo-control `algorithms/fpo.py`를 ProtoMotions 틀에 맞춘 것):
    * actor 출력은 Gaussian이 아니라 flow 표본 (`FlowActor`). neglogp 대신 rollout 때의 CFM 손실 L_old와
      그 표본 (ε, t)를 experience buffer에 저장한다 (모델 출력 키라 자동 등록).
    * actor_step: 같은 (ε, t)에서 현재 가중치의 L_new를 다시 재고 ratio = exp(clamp(L_old − L_new)).
      신뢰영역은 ASPO(advantage > 0: PPO clip, < 0: SPO 2차 벌점), ε = 0.05.
    * actor 가중치 EMA (epoch마다, warmup 뒤). checkpoint의 `model._actor.*`는 EMA 가중치(평가·배포용),
      원 가중치는 `fpo_actor_raw`에 따로 저장해 resume은 원 가중치로 이어진다.
    * PPO의 `actor_clip_frac_threshold`(clip 비율이 크면 actor 갱신 건너뜀)는 좁은 ε에서 거의 항상
      걸리므로 FPO 레시피에서는 None으로 둔다. critic clip(`clip_critic_loss`)도 FPO++처럼 끈다.

Key Classes:
    - FPO: PPO 서브클래스
Key Functions:
    - fpo_surrogate: 비율·신뢰영역 손실 (단위 시험용으로 Fabric 없이 호출 가능)
    - knn_entropy: Kozachenko–Leonenko k-NN 엔트로피 추정 (선택)
"""

import logging
import math
from pathlib import Path
from typing import Dict, Tuple

import torch
from tensordict import TensorDict
from torch import Tensor

from protomotions.agents.base_agent.agent import BaseAgent
from protomotions.agents.fpo.config import FPOAgentConfig
from protomotions.agents.ppo.agent import PPO

log = logging.getLogger(__name__)


def clamp_ste(x: Tensor, max: float) -> Tensor:
    """forward는 clamp, backward는 항등 (straight-through)."""
    return x + (x.clamp(max=max) - x).detach()


def fpo_surrogate(
    old_loss: Tensor,
    new_loss: Tensor,
    advantages: Tensor,
    *,
    clip: float,
    mode: str = "aspo",
    cfm_loss_clamp: float = 20.0,
    diff_clamp_max: float = 10.0,
    clamp_negative: bool = True,
    clamp_negative_max: float = 20.0,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """FPO++ actor 손실. old_loss·new_loss (B, K), advantages (B, 1) → (스칼라 손실, 로그 dict)."""
    adv = advantages
    if cfm_loss_clamp > 0:
        old_loss = old_loss.clamp(max=cfm_loss_clamp)
        new_loss = new_loss.clamp(max=cfm_loss_clamp)
    if clamp_negative:
        new_loss = torch.where(adv < 0, new_loss.clamp(max=clamp_negative_max), new_loss)
    log_ratio = clamp_ste(old_loss - new_loss, max=diff_clamp_max)
    ratio = torch.exp(log_ratio)
    ppo_loss = torch.max(-adv * ratio, -adv * torch.clamp(ratio, 1.0 - clip, 1.0 + clip))
    spo_loss = -(ratio * adv - adv.abs() / (2.0 * clip) * (ratio - 1.0).square())
    if mode == "ppo":
        surrogate = ppo_loss
    elif mode == "spo":
        surrogate = spo_loss
    elif mode == "aspo":
        surrogate = torch.where(adv > 0, ppo_loss, spo_loss)
    else:
        raise ValueError(f"unknown trust_region_mode {mode}")
    loss = surrogate.mean()
    info = {
        "actor/ratio_mean": ratio.detach().mean(),
        "actor/log_ratio_abs_mean": log_ratio.detach().abs().mean(),
        "actor/clip_frac": ((ratio.detach() - 1.0).abs() > clip).float().mean(),
        "actor/cfm_loss_old": old_loss.detach().mean(),
        "actor/cfm_loss_new": new_loss.detach().mean(),
    }
    return loss, info


def knn_entropy(x0_pred: Tensor, k: int = 1) -> Tensor:
    """x0_pred (B, K, A) → Kozachenko–Leonenko 엔트로피 추정 (배치 평균, nats)."""
    B, K, A = x0_pred.shape
    assert 1 <= k < K, f"k must be in [1, K), got k={k}, K={K}"
    dists = torch.cdist(x0_pred, x0_pred, p=2)
    dists = dists + torch.eye(K, device=x0_pred.device).unsqueeze(0) * 1e10
    kth = torch.topk(dists, k=k, dim=2, largest=False, sorted=True)[0][:, :, -1].clamp(min=1e-6, max=1e9)
    psi_n = torch.digamma(torch.tensor(float(K), device=x0_pred.device))
    psi_k = torch.digamma(torch.tensor(float(k), device=x0_pred.device))
    log_cd = (A / 2) * math.log(math.pi) - float(torch.lgamma(torch.tensor(A / 2 + 1)))
    return (psi_n - psi_k + log_cd + A * torch.log(kth).mean(dim=1)).mean()


class FPO(PPO):
    """FPO++ agent (모듈 docstring 참조)."""

    config: FPOAgentConfig

    # -----------------------------
    # Setup / checkpoint
    # -----------------------------
    def setup(self):
        super().setup()
        assert not self.config.initial_actor_checkpoint, (
            "FPO flow actor cannot warm-start from a Gaussian MLP checkpoint; use --actor-init random"
        )
        self._ema = None
        self._ema_active = False
        if self.config.ema_decay > 0:
            self._ema = {
                name: p.detach().clone() for name, p in self.model._actor.named_parameters() if p.requires_grad
            }
            log.info(
                f"FPO actor EMA enabled: decay={self.config.ema_decay}, warmup_epochs={self.config.ema_warmup_epochs}"
            )

    def load(self, checkpoint: Path, load_env: bool = True):
        super().load(checkpoint, load_env=load_env)
        if self.config.evaluator.eval_metrics_every is None:
            # 이 레시피는 run_system A/B로 따로 평가한다 (AssistFinetunePPO와 같은 이유).
            self.just_loaded_checkpoint_should_evaluate = False

    def load_parameters(self, state_dict):
        BaseAgent.load_parameters(self, state_dict)  # PPO의 logstd 처리는 flow actor에 없다
        if "fpo_actor_raw" in state_dict:
            self.model._actor.load_state_dict(state_dict["fpo_actor_raw"], strict=True)
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
        if self.config.adaptive_lr.enabled and "adaptive_lr" in state_dict:
            self.actor_lr = state_dict["adaptive_lr"]["actor_lr"]
            self.critic_lr = state_dict["adaptive_lr"]["critic_lr"]
            for group in self.actor_optimizer.param_groups:
                group["lr"] = self.actor_lr
            for group in self.critic_optimizer.param_groups:
                group["lr"] = self.critic_lr
        if self.config.advantage_normalization.enabled and self.config.advantage_normalization.use_ema:
            if "adv_mean_ema" in state_dict:
                self.adv_mean_ema.copy_(state_dict["adv_mean_ema"])
            if "adv_std_ema" in state_dict:
                self.adv_std_ema.copy_(state_dict["adv_std_ema"])
        if self._ema is not None and "fpo_ema" in state_dict:
            self._ema = {k: v.to(self.device).clone() for k, v in state_dict["fpo_ema"].items()}
            self._ema_active = bool(state_dict.get("fpo_ema_active", False))

    def get_state_dict(self, state_dict):
        state_dict = super().get_state_dict(state_dict)
        if self._ema is not None:
            state_dict["fpo_ema"] = {k: v.clone() for k, v in self._ema.items()}
            state_dict["fpo_ema_active"] = self._ema_active
            if self._ema_active:
                # 저장본의 actor = EMA 가중치 (평가·배포), 원 가중치는 fpo_actor_raw (resume)
                state_dict["fpo_actor_raw"] = {k: v.clone() for k, v in self.model._actor.state_dict().items()}
                baked = dict(state_dict["model"])
                for name, shadow in self._ema.items():
                    baked[f"_actor.{name}"] = shadow.clone()
                state_dict["model"] = baked
        return state_dict

    # -----------------------------
    # Optimization
    # -----------------------------
    def optimize_model(self) -> Dict:
        out = super().optimize_model()
        self._update_ema()
        return out

    @torch.no_grad()
    def _update_ema(self):
        if self._ema is None:
            return
        done = self.current_epoch + 1  # 이번 갱신까지 끝난 epoch 수
        warmup, decay = int(self.config.ema_warmup_epochs), float(self.config.ema_decay)
        params = dict(self.model._actor.named_parameters())
        if done == warmup:
            for name, shadow in self._ema.items():
                shadow.copy_(params[name].detach())
            self._ema_active = True
        elif done > warmup:
            for name, shadow in self._ema.items():
                shadow.mul_(decay).add_(params[name].detach(), alpha=1.0 - decay)
            self._ema_active = True

    def actor_step(self, batch_dict) -> Tuple[Tensor, Dict]:
        batch_td = TensorDict(batch_dict, batch_size=batch_dict["action"].shape[0])
        new_loss, x0_pred = self.model._actor.cfm_loss(
            batch_td, batch_dict["action"], batch_dict["cfm_eps"], batch_dict["cfm_t"]
        )
        advantages = batch_dict["advantages"].unsqueeze(-1)
        fpo_loss, info = fpo_surrogate(
            batch_dict["cfm_loss_old"],
            new_loss,
            advantages,
            clip=float(self.config.fpo_clip),
            mode=self.config.trust_region_mode,
            cfm_loss_clamp=float(self.config.cfm_loss_clamp),
            diff_clamp_max=float(self.config.cfm_diff_clamp_max),
            clamp_negative=bool(self.config.cfm_loss_clamp_negative_advantages),
            clamp_negative_max=float(self.config.cfm_loss_clamp_negative_advantages_max),
        )
        actor_loss = fpo_loss
        entropy = torch.tensor(0.0, device=self.device)
        if self.config.knn_entropy_coef > 0:
            entropy = knn_entropy(x0_pred, k=int(self.config.knn_entropy_k))
            actor_loss = actor_loss - self.config.knn_entropy_coef * entropy
        extra_loss, extra_log = self.calculate_extra_actor_loss(batch_td)
        actor_loss = actor_loss + extra_loss
        log_dict = {
            "actor/fpo_loss": fpo_loss.detach(),
            "actor/extra_loss": extra_loss.detach(),
            "actor/entropy_loss": entropy.detach(),
            "actor/exploration_rms": (batch_dict["action"] - batch_dict["mean_action"]).pow(2).mean().sqrt(),
            "losses/actor_loss": actor_loss.detach(),
        }
        log_dict.update(info)
        log_dict.update(extra_log)
        return actor_loss, log_dict
