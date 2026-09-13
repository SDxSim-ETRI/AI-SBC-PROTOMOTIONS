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
"""스켈레톤 LLP 학습용 control component — 내장 tracker 착용자 + 실 HLP chunk.

`SkeletonAssistEnv`(env 10 ms)에서 두 가지를 담당한다.

1. **착용자 = 동결 mimic tracker** (`TrackerPolicy`): tracker는 env 주기 50 ms로
   학습됐으므로 `tracker_period_steps`(5) env step마다 한 번 행동한다. 관측은
   tracker가 학습 때 쓰던 것과 같은 함수·정적 파라미터로 env가 매 step 계산해
   두는 `max_coords_obs`·`mimic_target_poses`(미래 레퍼런스는 +50 ms —
   `TrackerMimicControl`)에, tracker 자신의 이전 action(자체 버퍼)을 더한 것.
   출력 raw action → `normalized_pd_fixed_gains_action` → PD 목표 27개를 다음
   행동 전까지 유지한다. 착용자의 "의도" θ_g = **mocap 레퍼런스의 hip flexion**
   (`ctx.mimic.ref_state.dof_pos`; 평가·critic용. tracker의 PD 목표는 의도가 아니라
   토크를 내기 위한 명령이라 실제 각도와 15~20° 차이가 난다).
2. **HLP = 실 ManiFlow** (`ManiFlowChunkSource`): hip flexion + trunk(pelvis 자세
   × lumbar FK, 학습 데이터 추출기와 동일 규약)를 매 step 링버퍼에 넣고
   `chunk_refresh_steps`마다 예측해 chunk(knot 7 × 2, 25 ms)를 갱신. 영점 보정은
   재생 mocap의 채널 평균을 SEED 평균에 맞추는 오프셋(런타임과 동일 함수).

LLP 관측(hip 2ch 상태·히스토리, 미래 목표 창, 이전 action)은 `AssistContext`의
부분집합 필드로 노출되어 진자 학습과 같은 커널(`assist_obs.py`)을 그대로 쓴다.
"""

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import torch
from tensordict import TensorDict
from torch import Tensor

from protomotions.assist.skeleton_features import (
    calibration_offset_from_motion,
    hip_flexion_indices,
    hlp_channels,
    lumbar_indices,
)
from protomotions.envs.context_views import AssistContext
from protomotions.envs.control.assist_motion_control import MotionAssistTargetControlConfig
from protomotions.envs.control.base import ControlComponent
from protomotions.envs.control.mimic_control import MimicControl, MimicControlConfig
from protomotions.utils.hydra_replacement import get_class

if TYPE_CHECKING:
    from protomotions.envs.base_env.env import BaseEnv
    from protomotions.envs.context_views import EnvContext

logger = logging.getLogger(__name__)

PELVIS_BODY = 0


# ---------------------------------------------------------------------------
# tracker용 mimic 레퍼런스: 미래 프레임 간격을 env dt가 아니라 tracker 주기로
# ---------------------------------------------------------------------------
class _DtProxy:
    """`MimicControl.populate_context`가 읽는 `env.dt`만 바꿔치기한 env 프록시."""

    def __init__(self, env, dt: float):
        self._env = env
        self.dt = float(dt)

    def __getattr__(self, name):
        return getattr(self._env, name)


@dataclass
class TrackerMimicControlConfig(MimicControlConfig):
    """`MimicControlConfig` + 미래 레퍼런스 간격 고정 (tracker 학습 주기)."""

    _target_: str = "protomotions.envs.control.skeleton_assist_control.TrackerMimicControl"
    future_dt: float = 0.05


class TrackerMimicControl(MimicControl):
    """미래 레퍼런스를 `future_dt` 간격으로 만드는 MimicControl (나머지 동일)."""

    def __init__(self, config: TrackerMimicControlConfig, env: "BaseEnv"):
        super().__init__(config, env)
        self._real_env = env
        self.env = _DtProxy(env, config.future_dt)


# ---------------------------------------------------------------------------
# 동결 tracker 정책
# ---------------------------------------------------------------------------
class TrackerPolicy:
    """mimic tracker 체크포인트의 actor + action→PD 목표 변환 (env·Fabric 없이).

    actor는 lazy 모듈이라 첫 실제 관측으로 형상을 굳힌 뒤 state dict를 싣는다.
    """

    def __init__(self, checkpoint: str, device: torch.device):
        self.checkpoint = Path(checkpoint)
        self.device = torch.device(device)
        resolved_path = self.checkpoint.parent / "resolved_configs_inference.pt"
        assert resolved_path.exists(), f"tracker resolved configs not found: {resolved_path}"
        resolved = torch.load(resolved_path, map_location="cpu", weights_only=False)
        self.env_cfg = resolved["env"]
        agent_cfg = resolved["agent"]
        self.in_keys = list(agent_cfg.model.actor.in_keys)
        self.mu_key = agent_cfg.model.actor.mu_key
        self.num_actions = int(agent_cfg.model.actor.num_out)
        model_cls = get_class(agent_cfg.model._target_)
        self._model = model_cls(config=agent_cfg.model)
        # lazy 모듈은 형상을 굳히기 전에 장치로 옮겨야 한다 (BaseAgent.setup과 동일 순서)
        self._actor = self._model._actor.to(self.device).eval()
        self._loaded = False
        # action → PD 목표 (tracker env의 action_config 그대로)
        self._action_fn = self.env_cfg.action_config["fn"]
        self._action_params = {
            k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
            for k, v in self.env_cfg.action_config.items()
            if k != "fn"
        }
        prev_cfg = self.env_cfg.observation_components.get("previous_actions")
        self.prev_action_steps = int((getattr(prev_cfg, "static_params", None) or {}).get("history_steps", 1))
        assert self.prev_action_steps == 1, "embedded tracker supports previous_actions history_steps=1"

    def _lazy_load(self, td: TensorDict) -> None:
        with torch.no_grad():
            self._actor.mu(td)
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        actor_state = {k[len("_actor.") :]: v for k, v in state["model"].items() if k.startswith("_actor.")}
        self._actor.load_state_dict(actor_state, strict=True)
        self._actor = self._actor.to(self.device).eval()
        self._loaded = True
        logger.info("TrackerPolicy loaded: %s (epoch %s)", self.checkpoint, state.get("epoch"))

    @torch.no_grad()
    def act(self, obs: Dict[str, Tensor]) -> Tensor:
        """관측 dict(in_keys 포함) → raw mean action (N, num_actions)."""
        n = obs[self.in_keys[0]].shape[0]
        td = TensorDict({k: obs[k].to(self.device) for k in self.in_keys}, batch_size=n, device=self.device)
        if not self._loaded:
            self._lazy_load(td)
        td = self._actor.mu(td)
        return td[self.mu_key]

    def pd_targets(self, raw_action: Tensor) -> Tensor:
        return self._action_fn(action=raw_action, **self._action_params)["processed_action"]


# ---------------------------------------------------------------------------
# assist control component
# ---------------------------------------------------------------------------
@dataclass
class SkeletonAssistControlConfig(MotionAssistTargetControlConfig):
    """스켈레톤 LLP 학습용 설정.

    새 필드는 plain default만 쓴다 (pickle 복원 규칙).

    Attributes:
        assist_dof_names: LLP가 담당하는 관절 (COMMON dof 이름).
        tracker_checkpoint: 착용자 대역 mimic tracker `last.ckpt`. 쉼표로 여러 개를 주면
            **착용자 앙상블** — env마다 리셋 때 무작위로 하나를 배정한다 (걸음 스타일이
            다른 착용자들 = 사용자 다양성; walk LLP v3가 다른 tracker에서 −31→−2 %로
            떨어진 착용자 특이성 대책).
        tracker_period_steps: tracker 행동 주기 [env step] (50 ms / 10 ms = 5).
        hlp_calibration: "motion"(재생 mocap 평균 → SEED 평균 오프셋) | "none" |
            "p,q,r,s"(직접 지정, deg). 착용자(tracker)가 실제로 걷는 자세는 재생 mocap과
            10° 가까이 다를 수 있어, 착용자 스트림 평균으로 잰 값을 직접 넣는 쪽이
            예측 오차가 작다 (`run_system.py --calibration wearer`가 그 값을 측정해 찍어 준다).
        motion_file_for_calibration: 오프셋 계산에 쓸 motion .pt (None이면 env의 motion_lib 파일).
    """

    _target_: str = "protomotions.envs.control.skeleton_assist_control.SkeletonAssistControl"
    assist_dof_names: Tuple[str, ...] = ("hip_flexion_r", "hip_flexion_l")
    tracker_checkpoint: str = ""
    tracker_period_steps: int = 5
    hlp_calibration: str = "motion"
    motion_file_for_calibration: Optional[str] = None
    # 부모 기본값 덮어쓰기: 실 HLP 격자, 착용자 의도 = tracker, skeleton 부호 규약
    intent_source: str = "tracker"
    chunk_source: str = "maniflow"
    flexion_sign: float = 1.0
    hlp_dt: float = 0.025
    chunk_knots: int = 7
    chunk_refresh_steps: int = 2
    obs_future_samples: int = 4
    obs_future_dt: float = 0.05


class SkeletonAssistControl(ControlComponent):
    """내장 tracker 착용자 + 실 HLP chunk + LLP 컨텍스트."""

    def __init__(self, config: SkeletonAssistControlConfig, env: "BaseEnv"):
        super().__init__(config, env)
        self.config: SkeletonAssistControlConfig = config
        n = env.num_envs
        dev = env.device
        robot = env.robot_config
        names = list(robot.kinematic_info.dof_names)
        self.assist_dof_indices = torch.tensor(
            [names.index(nm) for nm in config.assist_dof_names], device=dev, dtype=torch.long
        )
        self.num_assist_dofs = int(self.assist_dof_indices.numel())
        self._hip_idx = list(hip_flexion_indices(names))
        self._lumbar_idx = list(lumbar_indices(names))
        d = self.num_assist_dofs

        # --- 착용자: 동결 tracker (여러 개면 env별 배정) ---
        assert config.tracker_checkpoint, "SkeletonAssistControlConfig.tracker_checkpoint is required"
        paths = [p.strip() for p in str(config.tracker_checkpoint).split(",") if p.strip()]
        self.trackers = [TrackerPolicy(p, dev) for p in paths]
        self.tracker = self.trackers[0]
        for t in self.trackers[1:]:
            assert t.in_keys == self.tracker.in_keys and t.num_actions == self.tracker.num_actions, (
                "tracker ensemble members must share observation keys and action size"
            )
        self._tracker_id = torch.zeros(n, dtype=torch.long, device=dev)
        n_dofs = robot.number_of_actions
        self._pd_targets = robot.default_dof_pos.to(dev, torch.float32).unsqueeze(0).repeat(n, 1)
        # tracker 관측의 previous_actions는 두 결정 전 action (state history 규약)
        self._tracker_prev1 = torch.zeros(n, n_dofs, device=dev)
        self._tracker_prev2 = torch.zeros(n, n_dofs, device=dev)

        # --- HLP: 실 ManiFlow ---
        from protomotions.maniflow.angle_estimator import ManiFlowAngleEstimator
        from protomotions.maniflow.chunk_source import DEFAULT_DENOISE_STEPS, ManiFlowChunkSource
        from protomotions.maniflow.loader import discover_best_checkpoint
        from protomotions.maniflow.verify_angle_estimator import FLEXION40_RUN_DIR

        ckpt = config.maniflow_ckpt or str(discover_best_checkpoint(FLEXION40_RUN_DIR))
        estimator = ManiFlowAngleEstimator.from_checkpoint(
            ckpt, num_envs=n, device=dev,
            denoise_steps=int(config.maniflow_denoise_steps or DEFAULT_DENOISE_STEPS),
        )
        offset = self._calibration_offset(names)
        if offset is not None:
            estimator.set_calibration(obs_offset=offset, output_bias=offset[:2])
            logger.info("HLP obs offset (deg): %s", [round(math.degrees(v), 2) for v in offset.tolist()])
        self._chunk_src = ManiFlowChunkSource(
            estimator, num_envs=n, dt=env.dt, device=dev, flexion_sign=config.flexion_sign
        )
        if config.hlp_dt != self._chunk_src.knot_dt or config.chunk_knots != self._chunk_src.num_knots:
            logger.warning(
                "chunk 격자 %.3f×%d → 모델 기준 %.3f×%d로 덮어씀",
                config.hlp_dt, config.chunk_knots, self._chunk_src.knot_dt, self._chunk_src.num_knots,
            )
        config.hlp_dt = float(self._chunk_src.knot_dt)
        config.chunk_knots = int(self._chunk_src.num_knots)
        k = config.chunk_knots
        self._chunk = torch.zeros(n, k, d, device=dev)
        self._chunk_t0 = torch.zeros(n, device=dev)
        kin = robot.kinematic_info
        self._target_low = (kin.dof_limits_lower.float() * config.dof_limit_margin).to(dev)[self.assist_dof_indices]
        self._target_high = (kin.dof_limits_upper.float() * config.dof_limit_margin).to(dev)[self.assist_dof_indices]

        # --- 계측 버퍼 (env가 기록) ---
        self._tau_agent = torch.zeros(n, d, device=dev)
        self._tau_assist = torch.zeros(n, d, device=dev)
        self._gain_scale = torch.ones(n, d, device=dev)
        self._future_offsets = torch.arange(config.obs_future_samples, device=dev, dtype=torch.float32) * config.obs_future_dt

        logger.info(
            "SkeletonAssistControl: assist dofs %s, trackers %s (period %d steps), HLP %s",
            config.assist_dof_names, paths, config.tracker_period_steps, ckpt,
        )

    # ------------------------------------------------------------------
    def _calibration_offset(self, dof_names) -> Optional[Tensor]:
        cfg = self.config
        if cfg.hlp_calibration == "none":
            return None
        if "," in cfg.hlp_calibration:
            vals = [float(v) for v in cfg.hlp_calibration.split(",")]
            assert len(vals) == 4, "hlp_calibration 'p,q,r,s'는 4채널(deg)"
            return torch.deg2rad(torch.tensor(vals, dtype=torch.float32))
        motion_file = cfg.motion_file_for_calibration or getattr(self.env.motion_lib, "motion_file", None)
        if motion_file is None:
            mcfg = getattr(getattr(self.env, "motion_lib", None), "config", None)
            motion_file = getattr(mcfg, "motion_file", None)
        assert motion_file, "hlp_calibration=motion needs a motion file"
        # 보행 착용자: pelvis 자세는 시뮬레이션의 실제 값 → fixed_root=False
        return calibration_offset_from_motion(str(motion_file), dof_names, fixed_root=False)

    def _episode_time(self) -> Tensor:
        return self.env.progress_buf.float() * self.env.dt

    def _hlp_observation(self, env_ids: Optional[Tensor] = None):
        """(hip_pos, trunk) — 데이터 규약 입력 (부호 변환은 chunk source가 함)."""
        sh = self.env.state_history
        if env_ids is None:
            dof_pos = sh.dof_pos[:, 0]
            pelvis_q = sh.rigid_body_rot[:, 0, PELVIS_BODY]
        else:
            dof_pos = sh.dof_pos[env_ids, 0]
            pelvis_q = sh.rigid_body_rot[env_ids, 0, PELVIS_BODY]
        ch = hlp_channels(dof_pos, pelvis_q, self._hip_idx, self._lumbar_idx)
        # chunk source는 관절각(로봇 규약)에 flexion_sign을 곱해 데이터 규약을 만든다.
        # skeleton은 flexion_sign=+1이라 관절각 = 데이터 flexion.
        return dof_pos[:, self.assist_dof_indices], ch[:, 2:]

    def _refresh_chunk(self, env_ids: Tensor, t_now: Tensor) -> None:
        knots, t0 = self._chunk_src.predict_knots(env_ids, t_now)
        self._chunk[env_ids] = knots.clamp(self._target_low, self._target_high)
        self._chunk_t0[env_ids] = t0

    def _interp_chunk(self, tt: Tensor) -> Tensor:
        """`AssistTargetControl._interp_chunk`와 동일 (knot 선형 보간, 범위 밖 ZOH)."""
        cfg = self.config
        k = cfg.chunk_knots
        rel = (tt - self._chunk_t0.unsqueeze(1)) / cfg.hlp_dt
        rel = rel.clamp(0.0, float(k - 1))
        idx0 = rel.floor().long().clamp(max=k - 2)
        w = (rel - idx0.float()).clamp(0.0, 1.0).unsqueeze(-1)
        idx0e = idx0.unsqueeze(-1).expand(-1, -1, self.num_assist_dofs)
        knot0 = torch.gather(self._chunk, 1, idx0e)
        knot1 = torch.gather(self._chunk, 1, idx0e + 1)
        return knot0 * (1.0 - w) + knot1 * w

    # ------------------------------------------------------------------
    # 착용자 (env.step에서 pre-physics 호출)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def tracker_act(self, obs: Dict[str, Tensor]) -> None:
        """주기가 된 env의 PD 목표를 현재 관측으로 갱신."""
        due = (self.env.progress_buf % self.config.tracker_period_steps) == 0
        if not bool(due.any()):
            return
        tracker_obs = dict(obs)
        tracker_obs["previous_actions"] = self._tracker_prev2
        if len(self.trackers) == 1:
            raw = self.tracker.act(tracker_obs)
            targets = self.tracker.pd_targets(raw)
        else:
            # 앙상블: 정책마다 전 배치를 계산하고(1024×6 MLP, 비용 미미) env별 배정으로 고른다
            raw = torch.zeros(self.env.num_envs, self.tracker.num_actions, device=self._pd_targets.device)
            targets = torch.zeros_like(self._pd_targets)
            for k, tracker in enumerate(self.trackers):
                sel = self._tracker_id == k
                if not bool(sel.any()):
                    continue
                r = tracker.act(tracker_obs)
                raw[sel] = r[sel]
                targets[sel] = tracker.pd_targets(r)[sel]
        self._pd_targets[due] = targets[due]
        self._tracker_prev2[due] = self._tracker_prev1[due]
        self._tracker_prev1[due] = raw[due]

    def get_pd_targets(self) -> Tensor:
        return self._pd_targets

    # ------------------------------------------------------------------
    # ControlComponent 인터페이스
    # ------------------------------------------------------------------
    def reset(self, env_ids: Tensor) -> None:
        if len(env_ids) == 0:
            return
        self._tracker_prev1[env_ids] = 0.0
        self._tracker_prev2[env_ids] = 0.0
        self._pd_targets[env_ids] = self.env.robot_config.default_dof_pos.to(self._pd_targets)
        if len(self.trackers) > 1:
            self._tracker_id[env_ids] = torch.randint(
                len(self.trackers), (len(env_ids),), device=self._tracker_id.device
            )
        dof_pos, trunk = self._hlp_observation(env_ids)
        self._chunk_src.reset(env_ids, dof_pos, trunk)
        self._refresh_chunk(env_ids, torch.zeros(len(env_ids), device=self.env.device))
        self._tau_agent[env_ids] = 0.0
        self._tau_assist[env_ids] = 0.0

    def step(self) -> None:
        dof_pos, trunk = self._hlp_observation()
        self._chunk_src.observe(dof_pos, trunk)
        refresh = (self.env.progress_buf % self.config.chunk_refresh_steps) == 0
        ids = refresh.nonzero(as_tuple=False).flatten()
        if ids.numel() > 0:
            self._refresh_chunk(ids, self._episode_time()[ids])

    def populate_context(self, ctx: "EnvContext") -> None:
        t = self._episode_time()
        tt = t.unsqueeze(1) + self._future_offsets.unsqueeze(0)
        theta_d_future = self._interp_chunk(tt)
        idx = self.assist_dof_indices
        mimic = getattr(ctx, "mimic", None)
        if mimic is not None and getattr(mimic, "ref_state", None) is not None:
            theta_g = mimic.ref_state.dof_pos[:, idx]  # mocap 의도 (control_components 순서: mimic → assist)
        else:
            theta_g = self._pd_targets[:, idx]
        ctx.assist = AssistContext(
            theta_d=theta_d_future[:, 0],
            theta_d_future=theta_d_future,
            chunk_age=(t - self._chunk_t0).unsqueeze(1),
            theta_g=theta_g,
            tau_agent=self._tau_agent,
            tau_assist=self._tau_assist,
            gain_scale=self._gain_scale,
            dof_pos=ctx.current.dof_pos[:, idx],
            dof_vel=ctx.current.dof_vel[:, idx],
            historical_dof_pos=None if ctx.historical is None else ctx.historical.dof_pos[:, :, idx],
            historical_dof_vel=None if ctx.historical is None else ctx.historical.dof_vel[:, :, idx],
        )
