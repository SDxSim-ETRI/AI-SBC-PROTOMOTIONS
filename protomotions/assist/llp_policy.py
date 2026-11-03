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
"""LLP(assist torque PPO 정책) actor를 env·Fabric 없이 복원하는 로더.

`assist_pendulum` 학습 결과 디렉터리(`last.ckpt` + `resolved_configs_inference.pt`)
만으로 actor를 만들고, 관측 레이아웃(히스토리 길이, 미래 창, 스케일, 토크
한계)을 같은 config에서 읽어 `LLPObsLayout`으로 노출한다. 실기기/시뮬 하네스는
이 레이아웃대로 관측을 조립하기만 하면 된다 (`AssistController` 참조).

복원 경로는 `inference_agent.py`와 같은 pickle을 쓰되 시뮬레이터·env를 만들지
않는다: `PPOModel(config)` → 학습 레이아웃 크기의 0 관측으로 actor를 한 번
통과시켜 lazy 모듈(`nn.LazyLinear`, `RunningMeanStd`)의 형상을 굳힌 뒤
`ckpt["model"]`의 `_actor.*`만 로드한다.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tensordict import TensorDict
from torch import Tensor

from protomotions.utils.hydra_replacement import get_class


@dataclass(frozen=True)
class LLPObsLayout:
    """LLP actor가 학습된 관측·행동 계약 (env config에서 추출).

    Attributes:
        num_dofs: 보조 관절 수 (hip_pendulum = 2: [right, left]).
        history_steps: 관절 상태 히스토리 길이 H (현재 제외).
        dof_vel_scale: 관절 각속도 스케일.
        prev_action_steps: 관측에 들어가는 과거 raw action 수 P.
        obs_future_samples / obs_future_dt: 미래 목표 창 S개, 간격 [s].
        chunk_refresh_steps: chunk 갱신 주기 (policy step 단위).
        hlp_dt / chunk_knots: chunk knot 간격 [s] / 개수.
        dt: policy 주기 [s] (물리 dt × decimation).
        torque_scale / action_transform: raw action → 토크 변환.
        torque_limit: 최종 assist 토크 clamp [N·m].
        target_low / target_high: chunk knot clamp 범위 (관절 한계 × margin).
        in_keys: actor 입력 키 순서.
    """

    num_dofs: int
    history_steps: int
    dof_vel_scale: float
    prev_action_steps: int
    obs_future_samples: int
    obs_future_dt: float
    chunk_refresh_steps: int
    hlp_dt: float
    chunk_knots: int
    dt: float
    torque_scale: float
    action_transform: Optional[str]
    torque_limit: float
    target_low: Tensor
    target_high: Tensor
    in_keys: List[str]
    # 데이터 flexion(+굴곡) → LLP 관절각 부호 (hip_pendulum −1, skeleton +1)
    flexion_sign: float = -1.0
    # 보조 관절 이름 (전신 로봇에서 LLP가 담당하는 DOF; 진자는 전 관절)
    assist_dof_names: tuple = ()

    # actor in_keys 이름 (assist_pendulum/mlp.py의 observation_components 키)
    KEY_PROPRIO = "assist_proprio"
    KEY_TARGET = "assist_target"
    KEY_PREV_ACTIONS = "previous_actions"


def _static_param(component, name: str, default):
    params = getattr(component, "static_params", None) or {}
    return params.get(name, default)


class LLPPolicy:
    """`assist_pendulum` 체크포인트의 actor (deterministic, mean action).

    FPO++ flow actor(`protomotions.agents.fpo.model.FlowActor`) 체크포인트도 같은 경로로 복원한다:
    `mu` 모델 대신 `act_deterministic`(초기 잡음 0에서 flow 적분)을 부른다.

    `share_legs=True`: 단일 관절 진자로 학습한 정책 하나를 양쪽 다리에 같은 가중치로 적용한다
    (2026-10-21 가중치 공유). `SingleJointMLP`(한쪽 관절로 학습) 또는 `SharedJointMLP` 체크포인트의
    관절 모델을 다리마다 그 다리의 관측(각도·속도 히스토리, 미래 목표, 이전 명령)으로 한 번씩 돌린다.
    두 다리는 서로의 관측을 보지 않는다.
    """

    def __init__(self, checkpoint: str, device: torch.device, share_legs: bool = False):
        self.checkpoint = Path(checkpoint)
        self.checkpoints = (self.checkpoint,)
        self.device = torch.device(device)
        self.share_legs = bool(share_legs)
        resolved_path = self.checkpoint.parent / "resolved_configs_inference.pt"
        assert resolved_path.exists(), f"resolved configs not found: {resolved_path}"
        resolved = torch.load(resolved_path, map_location="cpu", weights_only=False)
        self.resolved = resolved
        env_cfg = resolved["env"]
        agent_cfg = resolved["agent"]
        robot_cfg = resolved["robot"]
        sim_cfg = resolved["simulator"]

        # --- 관측 레이아웃 (actor의 lazy 모듈을 굳히는 데도 필요) ---
        comp = env_cfg.control_components["assist"]
        obs_components = env_cfg.observation_components
        action_cfg = env_cfg.action_config
        kin = robot_cfg.kinematic_info
        margin = float(getattr(comp, "dof_limit_margin", 1.0))
        # 보조 관절: 전신 로봇이면 config의 assist_dof_names, 진자면 전 관절
        assist_names = tuple(getattr(comp, "assist_dof_names", ()) or ())
        if assist_names:
            dof_idx = [list(kin.dof_names).index(n) for n in assist_names]
        else:
            dof_idx = list(range(int(robot_cfg.number_of_actions)))
        num_dofs = int(getattr(agent_cfg.model.actor, "num_out", None) or len(dof_idx))
        assert num_dofs == len(dof_idx), (
            f"actor num_out {num_dofs} ≠ assist dofs {len(dof_idx)} ({assist_names or 'all'})"
        )
        idx_t = torch.tensor(dof_idx, dtype=torch.long)
        self.layout = LLPObsLayout(
            num_dofs=num_dofs,
            history_steps=int(env_cfg.num_state_history_steps),
            dof_vel_scale=float(
                _static_param(obs_components[LLPObsLayout.KEY_PROPRIO], "dof_vel_scale", 1.0)
            ),
            prev_action_steps=int(
                _static_param(obs_components[LLPObsLayout.KEY_PREV_ACTIONS], "history_steps", 1)
            ),
            obs_future_samples=int(comp.obs_future_samples),
            obs_future_dt=float(comp.obs_future_dt),
            chunk_refresh_steps=int(comp.chunk_refresh_steps),
            hlp_dt=float(comp.hlp_dt),
            chunk_knots=int(comp.chunk_knots),
            dt=float(sim_cfg.sim.decimation) / float(sim_cfg.sim.fps),
            torque_scale=float(action_cfg["torque_scale"]),
            action_transform=action_cfg.get("action_transform", "tanh"),
            torque_limit=float(env_cfg.assist_torque_limit),
            target_low=(torch.tensor(comp.target_low_override).to(self.device)
                        if getattr(comp, "target_low_override", ()) else
                        (kin.dof_limits_lower.float()[idx_t] * margin).to(self.device)),
            target_high=(torch.tensor(comp.target_high_override).to(self.device)
                         if getattr(comp, "target_high_override", ()) else
                         (kin.dof_limits_upper.float()[idx_t] * margin).to(self.device)),
            in_keys=list(agent_cfg.model.actor.in_keys),
            flexion_sign=float(getattr(comp, "flexion_sign", -1.0)),
            assist_dof_names=assist_names,
        )
        expected = {
            LLPObsLayout.KEY_PROPRIO,
            LLPObsLayout.KEY_TARGET,
            LLPObsLayout.KEY_PREV_ACTIONS,
        }
        assert set(self.layout.in_keys) == expected, (
            f"unsupported actor in_keys {self.layout.in_keys}; "
            f"AssistController assembles {sorted(expected)} only"
        )

        # --- actor 복원 ---
        # 모델은 LazyLinear + lazy RunningMeanStd라 state dict를 받기 전에 형상을
        # 굳혀야 한다 (BaseAgent.setup의 dummy forward와 같은 역할). 학습 레이아웃
        # 크기의 0 관측으로 mu 모델만 한 번 통과시킨 뒤 actor 부분만 로드한다.
        model_cls = get_class(agent_cfg.model._target_)
        model = model_cls(config=agent_cfg.model)
        actor = model._actor.eval()
        with torch.no_grad():
            td0 = TensorDict(self.zero_obs(batch=1), batch_size=1)
            if hasattr(actor, "mu"):
                actor.mu(td0)
            else:
                actor(td0)  # FlowActor(FPO++): context·속도 MLP의 lazy 모듈을 굳힌다
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        actor_state = {
            k[len("_actor.") :]: v for k, v in state["model"].items() if k.startswith("_actor.")
        }
        actor.load_state_dict(actor_state, strict=True)
        self.actor = actor.to(self.device).eval()
        self.mu_key = agent_cfg.model.actor.mu_key
        self.epoch = int(state.get("epoch", -1))
        if self.share_legs:
            from protomotions.agents.common.shared_joint_mlp import SharedJointMLP

            if not isinstance(self.actor.mu, SharedJointMLP):
                raise ValueError(f"weight sharing needs a per-joint actor, got {type(self.actor.mu).__name__}")

    # ------------------------------------------------------------------
    def obs_dims(self) -> Dict[str, int]:
        """actor 입력 키별 차원 (학습 env의 관측 커널과 동일한 계산)."""
        lay = self.layout
        d = lay.num_dofs
        return {
            LLPObsLayout.KEY_PROPRIO: d * (1 + lay.history_steps) * 2,
            LLPObsLayout.KEY_TARGET: lay.obs_future_samples * d * 2 + 1,
            LLPObsLayout.KEY_PREV_ACTIONS: lay.prev_action_steps * d,
        }

    def zero_obs(self, batch: int = 1) -> Dict[str, Tensor]:
        return {k: torch.zeros(batch, n) for k, n in self.obs_dims().items()}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs: Dict[str, Tensor]) -> Tensor:
        """관측 dict → raw mean action (num_envs, num_dofs).

        `PPOActor.forward`가 아니라 mu 모델만 호출한다 — forward는 탐색용
        action을 샘플링하며 전역 RNG를 소비하므로, paired 비교/실기기에서는
        평균만 계산하는 편이 맞다.
        """
        batch = obs[self.layout.in_keys[0]].shape[0]
        td = TensorDict(
            {k: obs[k].to(self.device) for k in self.layout.in_keys},
            batch_size=batch,
            device=self.device,
        )
        if self.share_legs:
            from protomotions.agents.common.shared_joint_mlp import SharedJointMLP

            # SingleJointMLP.forward는 학습 관절 하나만 내므로, 같은 관절 모델을 두 관절에 돌리는
            # SharedJointMLP.forward를 쓴다.
            return SharedJointMLP.forward(self.actor.mu, td)[self.mu_key]
        if hasattr(self.actor, "act_deterministic"):
            # FlowActor(FPO++): x_1 = 0에서 flow를 적분한 결정적 action (RNG 소비 없음)
            return self.actor.act_deterministic(td)[self.mu_key]
        td = self.actor.mu(td)
        return td[self.mu_key]

    def requested_torque(self, raw_action: Tensor) -> Tensor:
        """raw action → requested N·m, before the final assist limit."""
        lay = self.layout
        a = raw_action
        if lay.action_transform == "tanh":
            a = torch.tanh(a)
        elif lay.action_transform == "clamp":
            a = torch.clamp(a, -1.0, 1.0)
        return a * lay.torque_scale

    def to_torque(self, raw_action: Tensor) -> Tensor:
        """raw action → assist 토크 [N·m] (env의 action 처리 + limit clamp 재현)."""
        lay = self.layout
        tau = self.requested_torque(raw_action)
        return torch.clamp(tau, -lay.torque_limit, lay.torque_limit)
