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
"""FPO++ 레시피 도우미 — 실험 파일이 공유하는 CLI 인자와 PPO → FPO++ agent config 변환 (2026-11-03).

진자(`examples/experiments/assist_pendulum/mlp.py`)와 전신(`examples/experiments/assist_skeleton/llp_walk.py`) 실험 파일이
같은 `--policy mlp_ppo|mlp_fpo|tfm_fpo`·`--fpo-*`·`--tfm-*` 인자를 받고, 이미 조립한 PPO 설정(critic·보상 정규화·
advantage 정규화·γ·λ·rollout 길이·checkpoint 주기)을 그대로 둔 채 actor를 flow 정책으로, actor 손실을 FPO++의
CFM 비율로 바꾼다. 토큰 시간축(히스토리·이전 action·예측 창 수)은 각 실험의 env 관측 구성에서 넘겨받는다.

Key Functions:
    - add_policy_arguments: --policy와 FPO++·Temporal Transformer 인자
    - fpo_agent_config: PPOAgentConfig → FPOAgentConfig
"""

import argparse
import copy
from dataclasses import fields

from protomotions.agents.base_agent.config import OptimizerConfig
from protomotions.agents.fpo.config import (
    FlowActorConfig,
    FPOAgentConfig,
    FPOModelConfig,
    TemporalTransformerConfig,
)
from protomotions.agents.ppo.config import PPOAgentConfig

POLICIES = ("mlp_ppo", "mlp_fpo", "tfm_fpo")


def add_policy_arguments(parser: argparse.ArgumentParser, policy_note: str = "") -> None:
    """`--policy`와 FPO++ / Temporal Transformer 인자. env·관측·보상·토크 한계는 실험 파일의 다른 인자가 정한다."""
    parser.add_argument("--policy", choices=list(POLICIES), default="mlp_ppo",
                        help="mlp_ppo: 기존 Gaussian MLP + PPO (기준선). mlp_fpo: 같은 은닉층 크기의 flow actor + FPO++. "
                             "tfm_fpo: Temporal Transformer context + flow MLP + FPO++. " + policy_note)
    parser.add_argument("--fpo-samples", type=int, default=32, help="action당 CFM 표본 수 K (FPO++ 휴머노이드 32)")
    parser.add_argument("--fpo-sampling-steps", type=int, default=16, help="flow Euler 적분 step 수 N (rollout·평가 공통)")
    parser.add_argument("--fpo-clip", type=float, default=0.05, help="비율 clip ε (FPO++ 보행 0.05)")
    parser.add_argument("--fpo-trust", choices=["ppo", "spo", "aspo"], default="aspo", help="신뢰영역 (FPO++ 기본 aspo)")
    parser.add_argument("--fpo-mini-epochs", type=int, default=8,
                        help="rollout 한 번의 데이터를 몇 번 재사용하는가 (minibatch 수 = num_envs×32 / --batch-size). "
                             "기준선 PPO는 1 epoch")
    parser.add_argument("--fpo-actor-scale", type=float, default=0.25,
                        help="flow 출력 x_0 → raw action 배율. 초기 탐색 폭 = 배율·N(0,1) raw (tanh·한계 N·m 전)")
    parser.add_argument("--fpo-ema-decay", type=float, default=0.95, help="actor 가중치 EMA 감쇠 (epoch마다; 0 = 끔)")
    parser.add_argument("--fpo-ema-warmup", type=int, default=200, help="EMA 시작 epoch")
    parser.add_argument("--fpo-weight-decay", type=float, default=1e-4, help="actor AdamW weight decay (FPO++ 1e-4)")
    parser.add_argument("--fpo-grad-clip", type=float, default=1.0, help="gradient norm clip (FPO++ 1.0; 기준선 PPO 50)")
    parser.add_argument("--tfm-d-model", type=int, default=64, help="Temporal Transformer 토큰 차원")
    parser.add_argument("--tfm-layers", type=int, default=2, help="Temporal Transformer 블록 수")
    parser.add_argument("--tfm-heads", type=int, default=4, help="attention head 수")
    parser.add_argument("--tfm-ff", type=int, default=128, help="feed-forward 은닉 차원")


def fpo_agent_config(cfg: PPOAgentConfig, args: argparse.Namespace, *, env_cfg, actor_cfg, actor_lr: float) -> FPOAgentConfig:
    """PPO 설정 `cfg`를 FPO++ 설정으로 바꾼다.

    actor_cfg: 기준선 Gaussian MLP actor 설정 — 관측 키·action 차원·은닉층(`mu_model.layers`)을 flow 속도 MLP에 옮긴다.
    env_cfg: 학습 env 설정 — 토큰 시간축(상태 히스토리 길이, 관측의 이전 action 수, HLP 예측 창 표본 수)을 읽는다.
             진자 `AssistEnvConfig`와 전신 `SkeletonAssistEnvConfig`가 같은 자리에 둔다.
    mlp_fpo: 속도 MLP = 기준선 actor와 같은 은닉층; 입력 [정규화 평탄 관측, t 임베딩(8), x_t].
    tfm_fpo: 평탄 관측을 시간별 토큰(현재+과거 상태 토큰 + HLP 예측 토큰)으로 되돌려 소형 transformer가 요약한
             context [현재 토큰 출력 d_model + 현재 특징]을 같은 속도 MLP에 조건으로 준다.
    critic은 두 변형 모두 `cfg`의 MLP critic. FPO++와 PPO 틀의 차이: actor_clip_frac_threshold(좁은 ε에서 항상 걸림)와
    critic clip을 끄고, gradient clip·AdamW·mini-epoch 수는 `--fpo-*` 값.
    """
    assert args.policy in POLICIES[1:], f"fpo_agent_config needs --policy mlp_fpo|tfm_fpo, got {args.policy}"
    num_out = int(actor_cfg.num_out)
    context = None
    if args.policy == "tfm_fpo":
        context = TemporalTransformerConfig(
            num_dofs=num_out, history_steps=int(env_cfg.num_state_history_steps),
            prev_action_steps=int(env_cfg.observation_components["previous_actions"].static_params["history_steps"]),
            future_samples=int(env_cfg.control_components["assist"].obs_future_samples),
            d_model=int(args.tfm_d_model), num_layers=int(args.tfm_layers),
            num_heads=int(args.tfm_heads), ff_size=int(args.tfm_ff),
        )
    actor = FlowActorConfig(
        in_keys=list(actor_cfg.in_keys), num_out=num_out,
        normalize_obs=True, norm_clamp_value=5, context=context,
        layers=copy.deepcopy(list(actor_cfg.mu_model.layers)),
        sampling_steps=int(args.fpo_sampling_steps), n_samples_per_action=int(args.fpo_samples),
        actor_scale=float(args.fpo_actor_scale),
    )
    model = FPOModelConfig(
        in_keys=list(cfg.model.in_keys), actor=actor, critic=copy.deepcopy(cfg.model.critic),
        actor_optimizer=OptimizerConfig(_target_="torch.optim.AdamW", lr=float(actor_lr),
                                        weight_decay=float(args.fpo_weight_decay)),
        critic_optimizer=copy.deepcopy(cfg.model.critic_optimizer),
    )
    base = {f.name: copy.deepcopy(getattr(cfg, f.name)) for f in fields(cfg)
            if f.name not in ("_target_", "model", "initial_actor_checkpoint")}
    out = FPOAgentConfig(
        **base, model=model,
        fpo_clip=float(args.fpo_clip), trust_region_mode=str(args.fpo_trust),
        ema_decay=float(args.fpo_ema_decay), ema_warmup_epochs=int(args.fpo_ema_warmup),
    )
    out.num_mini_epochs = int(args.fpo_mini_epochs)
    out.actor_clip_frac_threshold = None
    out.clip_critic_loss = False
    out.gradient_clip_val = float(args.fpo_grad_clip)
    return out
