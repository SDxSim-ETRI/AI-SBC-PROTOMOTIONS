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
"""FPO++(Flow Policy Optimization) 설정 — Temporal Transformer 컨텍스트 + Flow MLP actor.

FPO++(Yi·Choi et al. 2026, `fpo-control`)의 핵심만 ProtoMotions의 PPO 틀 위에 얹는다:
Gaussian actor 대신 noise → action ODE(flow)가 정책이고, 확률비 대신 저장해 둔 (ε, t) 표본에서 잰
conditional flow matching(CFM) 손실의 차이 exp(L_old − L_new)를 PPO 비율 자리에 넣는다.
critic·GAE·보상 정규화·advantage 정규화·experience buffer는 PPO와 같다.

Key Classes:
    - TemporalTransformerConfig: 관측 이력·HLP 예측을 시간별 토큰으로 읽는 소형 transformer
    - FlowActorConfig: context(MLP용 평탄 관측 또는 transformer 요약) + 시각 임베딩 + x_t → 속도 MLP
    - FPOModelConfig: FlowActor + MLP critic
    - FPOAgentConfig: FPO++ 학습 설정 (ASPO 신뢰영역, CFM 손실 clamp, actor EMA)
"""

from dataclasses import dataclass, field
from typing import List, Optional

from protomotions.agents.common.config import (
    MLPLayerConfig,
    ModuleContainerConfig,
    NormObsBaseConfig,
)
from protomotions.agents.ppo.config import PPOAgentConfig, PPOModelConfig


@dataclass
class TemporalTransformerConfig:
    """평탄 관측(assist_proprio·assist_target·previous_actions)을 시간별 토큰으로 되돌려 self-attention으로 요약한다.

    토큰 = [현재 + 과거 history_steps개 상태 토큰] + [HLP 예측 창 future_samples개 토큰].
    상태 토큰(lag k): [q, q̇·scale, 그 시점에 계산된 raw action, action 유효 플래그]
    예측 토큰(offset s): [θ_d, θ_d − q, chunk 경과시간]
    슬롯마다 학습형 위치 임베딩(RPL의 학습형 state_pos_encoding과 같은 방식), pre-LN 블록, GELU, 마스크 없음.
    읽기 = 현재 상태 토큰의 출력(RPL `aggregation_method="last"`와 같은 뜻: 가장 새 프레임).
    """

    num_dofs: int = field(default=2, metadata={"help": "보조 관절 수 (hip 좌우)."})
    history_steps: int = field(default=25, metadata={"help": "상태 히스토리 길이 H (현재 제외)."})
    prev_action_steps: int = field(default=10, metadata={"help": "관측에 들어오는 과거 raw action 수 P."})
    action_lag_offset: int = field(
        default=2,
        metadata={"help": "previous_actions[j]가 몇 step 전에 계산된 action인가 (env 히스토리 버퍼 규약: j + 2)."},
    )
    future_samples: int = field(default=3, metadata={"help": "HLP 예측 창 표본 수 S (offset 0부터)."})
    d_model: int = field(default=64, metadata={"help": "토큰 임베딩 차원."})
    num_heads: int = field(default=4, metadata={"help": "attention head 수 (d_model을 나눠야 한다)."})
    num_layers: int = field(default=2, metadata={"help": "encoder 블록 수."})
    ff_size: int = field(default=128, metadata={"help": "feed-forward 은닉 차원."})
    dropout: float = field(default=0.0, metadata={"help": "dropout (RL이라 0)."})
    activation: str = field(default="gelu", metadata={"help": "feed-forward 활성 함수."})
    concat_current_features: bool = field(
        default=True,
        metadata={"help": "context에 현재 시점 원 특징(q, q̇, 예측 창, chunk 경과시간)을 덧붙인다 (RPL head의 raw state concat)."},
    )


@dataclass
class FlowActorConfig(NormObsBaseConfig):
    """Flow actor: context c(s) 한 번 → N번의 Euler step 동안 속도장 v(c, t, x_t)만 평가.

    정책 표본: x_1 ~ N(0, I) → x_0 = ∫ v dt (t: 1 → 0) → action = actor_scale · x_0 (+ 탐색용 perturb).
    결정적(평가) 행동: x_1 = 0에서 같은 적분. CFM 손실: (ε, t) 표본에서 x_t = t·ε + (1−t)·a/actor_scale,
    목표 속도 ε − a/actor_scale, ‖v − 목표‖² / √A (`sqrt` reduction).
    """

    _target_: str = "protomotions.agents.fpo.model.FlowActor"
    in_keys: List[str] = field(default_factory=list, metadata={"help": "actor 관측 키 (순서대로 이어 붙여 정규화)."})
    out_keys: List[str] = field(
        default_factory=lambda: ["action", "mean_action", "cfm_eps", "cfm_t", "cfm_loss_old"],
        metadata={"help": "rollout 출력: 표본 action, 결정적 action, CFM 표본(ε, t)과 수집 시점 CFM 손실."},
    )
    num_out: int = field(default=None, metadata={"help": "action 차원."})
    mu_key: str = field(default="mean_action", metadata={"help": "결정적 action 키 (LLPPolicy가 읽는다)."})
    context: Optional[TemporalTransformerConfig] = field(
        default=None, metadata={"help": "None = 정규화한 평탄 관측을 그대로 context로 (MLP+FPO++)."}
    )
    layers: List[MLPLayerConfig] = field(default_factory=list, metadata={"help": "속도 MLP 은닉층."})
    timestep_embed_dim: int = field(default=8, metadata={"help": "flow 시각 t의 sin/cos 임베딩 차원 (짝수)."})
    sampling_steps: int = field(default=16, metadata={"help": "Euler 적분 step 수 N (rollout·평가 공통)."})
    n_samples_per_action: int = field(default=32, metadata={"help": "action당 CFM 표본 수 K."})
    t_inverse_cdf_beta: float = field(default=1.0, metadata={"help": "t ~ Beta(1, β) (1 = 균일), [0.005, 0.995]로 축소."})
    actor_scale: float = field(default=0.25, metadata={"help": "x_0 → raw action 배율 (초기 탐색 폭 = 배율·N(0,1))."})
    mlp_output_scale: float = field(default=1.0, metadata={"help": "속도 MLP 출력 배율."})
    final_layer_weight_scale: Optional[float] = field(
        default=0.01, metadata={"help": "속도 MLP 마지막 층 초기 가중치 배율 (작게 = 초기 flow ≈ 항등)."}
    )
    action_perturb_std: float = field(default=0.02, metadata={"help": "rollout action에 더하는 가우시안 잡음 (엔트로피 역할)."})
    cfm_loss_reduction: str = field(default="sqrt", metadata={"help": "mean | sum | sqrt (FPO++ 기본 sqrt)."})
    velocity_activation: str = field(default="relu", metadata={"help": "속도 MLP 활성 함수 (기준선 PPO MLP와 같은 relu)."})


@dataclass
class FPOModelConfig(PPOModelConfig):
    """FlowActor + PPO critic. forward는 PPOModel 그대로 (actor → critic)."""

    _target_: str = "protomotions.agents.ppo.model.PPOModel"
    out_keys: List[str] = field(
        default_factory=lambda: ["action", "mean_action", "cfm_eps", "cfm_t", "cfm_loss_old", "value"],
    )
    actor: FlowActorConfig = field(default_factory=FlowActorConfig)
    critic: ModuleContainerConfig = field(default_factory=ModuleContainerConfig)


@dataclass
class FPOAgentConfig(PPOAgentConfig):
    """FPO++ 학습 설정. PPO 필드 중 e_clip은 critic clip에만 남고 actor 신뢰영역은 fpo_clip을 쓴다."""

    _target_: str = "protomotions.agents.fpo.agent.FPO"
    model: FPOModelConfig = field(default_factory=FPOModelConfig)
    fpo_clip: float = field(default=0.05, metadata={"help": "비율 clip ε (FPO++ 보행 0.05; PPO 0.2보다 훨씬 좁다)."})
    trust_region_mode: str = field(default="aspo", metadata={"help": "ppo | spo | aspo (양의 advantage는 PPO clip, 음은 SPO 2차 벌점)."})
    cfm_loss_clamp: float = field(default=20.0, metadata={"help": "old·new CFM 손실 상한 (0 이하 = 끔)."})
    cfm_diff_clamp_max: float = field(default=10.0, metadata={"help": "log 비율 (L_old − L_new) 상한, straight-through."})
    cfm_loss_clamp_negative_advantages: bool = field(default=True, metadata={"help": "음의 advantage에서 new CFM 손실 상한 적용."})
    cfm_loss_clamp_negative_advantages_max: float = field(default=20.0)
    knn_entropy_coef: float = field(default=0.0, metadata={"help": "k-NN 엔트로피 보너스 계수 (0 = 끔)."})
    knn_entropy_k: int = field(default=1)
    ema_decay: float = field(default=0.95, metadata={"help": "actor 가중치 EMA 감쇠 (epoch마다; 0 = 끔). checkpoint의 actor는 EMA 가중치."})
    ema_warmup_epochs: int = field(default=200, metadata={"help": "EMA를 시작하는 epoch (그 전 checkpoint는 원 가중치)."})
    initial_actor_checkpoint: str = field(
        default="", metadata={"help": "레시피 호환용. flow actor는 Gaussian MLP 가중치를 이어받을 수 없어 항상 비어 있어야 한다."}
    )
