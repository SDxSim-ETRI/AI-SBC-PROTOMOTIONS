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
"""AI-SBC HLP — hip flexion 각도 예측 ManiFlow 온라인 래퍼.

BONES-SEED locomotion으로 학습된 flexion 예측 모델(ManiFlow lowdim,
run: ``...locomotion-flexion40-run01_seed42``)을 시뮬레이션/실기기 루프 안에서
receding-horizon으로 돌리기 위한 래퍼. 관측은 ``RobotState``가 아니라 **각도 4채널 벡터**다
(진자 env·skeleton env·실기기 IMU가 각자 방식으로 이 4ch를 구성).

관측/출력 계약 (flexion40 모델, ``docs/ai_sbc/AI_SBC_FRAMEWORK.md`` §3.1):

    obs(4)    = [hip_flexion_r, hip_flexion_l, trunk_pitch, trunk_roll]
                (rad, skeleton hinge 규약; pitch=앞숙임+, roll=오른쪽+)
    action(2) = [hip_flexion_r, hip_flexion_l] 미래 각도 (rad)
                × n_action_steps(=4) 프레임 chunk

시간 규약: 모델의 native 프레임은 **40 fps(25 ms)** — :meth:`observe`는 이
주기로 호출해야 한다 (진자 env policy 100 Hz와는 다름 — 결합 시
`AssistTargetControl`의 ``hlp_dt=0.025``로 흡수). :meth:`predict`가 반환하는
chunk의 첫 프레임(lead 0)은 **마지막 observe와 같은 시각**이고, 나머지가
+25/+50/+75 ms 미래다.

⚠️ **horizon 정렬 — 미래는 400 ms가 아니라 150 ms다** (2026-09-08 확인):
모델의 `horizon`(=16)은 **관측 창과 정렬**돼 있다. `action_pred` index k는
상대 프레임 `k − (n_obs_steps − 1)`, 즉 k=0..8은 과거(−225…−25 ms),
k=9가 현재, k=10..15가 미래(+25…**+150 ms**)다. :meth:`predict`가 주는
`action`은 그 중 k=9..12(현재+75 ms)일 뿐이고, 실제로 쓸 수 있는 미래
전체는 :meth:`predict_horizon`으로 꺼내야 한다(현재+미래 7프레임).
lead별 MAE(suit14 walk, denoise 3): +25 ms 0.36° / +50 0.90° / +75 1.44° /
+100 2.00° / +125 2.58° / +150 3.11°. 더 긴 선행 시간이 필요하면 horizon을
`n_obs_steps + 미래프레임수`로 늘려 **재학습**해야 한다.

캘리브레이션 (영점 오프셋, §3.1): SEED와 실측(suit) 데이터는 중립자세 각도
규약이 채널별 상수만큼 다르다. ``obs_offset``(관측에 가산)과
``output_bias``(예측에서 감산)를 상수로 주입한다 — 오프라인 검증은 데이터
평균 정렬로 산출했고(suit14 기준 obs +13.9°/+12.0°/−10.2°/−0.1°,
출력 +13.8°/+12.1°), 실기기에서는 시작 시 중립자세 영점조정으로 대체 예정.

denoise 스텝: 권장 3 (suit14 MAE 0.95°, batch-1 지연 5.95 ms). 기본값은
체크포인트 설정(10)이므로 명시 권장.

등가성 검증: ``python -m protomotions.maniflow.verify_angle_estimator`` —
suit14 zarr를 스트리밍으로 흘려 오프라인 기준 스크립트
(ManiFlow_Policy/scripts/compare_hip_imu_reference.py)의 수치를 재현한다.
"""

from typing import Optional, Sequence, Union

import logging

import torch

# flexion40 모델의 채널 계약 — zarr attrs(obs_spec/action_dof_names)와 대조용.
FLEXION_OBS_CHANNELS = (
    "hip_flexion_r",
    "hip_flexion_l",
    "trunk_pitch",
    "trunk_roll",
)
FLEXION_ACTION_CHANNELS = ("hip_flexion_r", "hip_flexion_l")

logger = logging.getLogger(__name__)

# 모델 native 프레임 간격 [s] — 40 fps 모델의 기본값.
# 100 fps 등 다른 간격으로 재학습한 모델은 학습 zarr의 `frame_dt` attr에서 읽는다
# (`_frame_dt_from_cfg`), 없으면 이 값으로 둔다.
FLEXION_FRAME_DT = 0.025

_TensorLike = Union[torch.Tensor, Sequence[float]]


def _frame_dt_from_cfg(cfg) -> Optional[float]:
    """학습에 쓴 zarr의 `frame_dt` attr을 읽어 모델 프레임 간격을 알아낸다.

    zarr가 그 자리에 없거나(배포 환경) attr이 없으면 None — 호출부가 기본값을 쓴다.
    """
    try:
        path = cfg.walking_task.dataset.zarr_path
        import zarr

        attrs = dict(zarr.open_group(str(path), mode="r").attrs)
        dt = attrs.get("frame_dt")
        if dt:
            logger.info("HLP 프레임 간격을 학습 데이터에서 읽음: %.4f s (%s)", float(dt), path)
            return float(dt)
    except Exception as exc:  # noqa: BLE001 — 진단용, 실패해도 기본값으로 진행
        logger.debug("frame_dt 자동 인식 실패 (%s) — 기본값 사용", exc)
    return None


class ManiFlowAngleEstimator:
    """rolling 관측 history 기반 receding-horizon flexion 각도 예측기.

    사용 패턴:

    - :meth:`observe` — 모델 native 프레임 주기(40 fps)마다 현재 각도 4ch를
      history에 push.
    - :meth:`predict` — 새 chunk가 필요할 때 호출 (receding-horizon replay면
      ``n_action_steps`` 프레임마다 한 번).
    - :meth:`reset` — 리셋된 env는 다음 observe에서 history가 그 관측으로
      전체 채워진다 (학습 시 ``pad_before`` edge-padding과 동일한 프라이밍).
    """

    def __init__(
        self,
        policy,
        num_envs: int,
        device: torch.device,
        obs_offset: Optional[_TensorLike] = None,
        output_bias: Optional[_TensorLike] = None,
        denoise_steps: Optional[int] = None,
        frame_dt: Optional[float] = None,
    ):
        self.policy = policy
        self.num_envs = num_envs
        self.device = torch.device(device)
        # 모델이 내는 시퀀스의 프레임 간격. chunk 격자·관측 재표집이 모두 이 값을 쓴다.
        self.frame_dt = float(frame_dt) if frame_dt else FLEXION_FRAME_DT

        self.n_obs_steps = int(policy.n_obs_steps)
        self.n_action_steps = int(policy.n_action_steps)
        self.action_dim = int(policy.action_dim)
        self.obs_key = policy.obs_encoder.state_key  # 'agent_pos'
        self.obs_dim = int(policy.obs_encoder.state_shape[0])

        if denoise_steps is not None:
            self.policy.num_inference_steps = int(denoise_steps)

        self._obs_offset = torch.zeros(self.obs_dim, device=self.device)
        self._output_bias = torch.zeros(self.action_dim, device=self.device)
        self.set_calibration(obs_offset=obs_offset, output_bias=output_bias)

        self._history = torch.zeros(
            num_envs, self.n_obs_steps, self.obs_dim, device=self.device
        )
        self._primed = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str,
        num_envs: int,
        device: torch.device,
        obs_offset: Optional[_TensorLike] = None,
        output_bias: Optional[_TensorLike] = None,
        denoise_steps: Optional[int] = None,
        use_ema: Optional[bool] = None,
        maniflow_root: Optional[str] = None,
        frame_dt: Optional[float] = None,
    ) -> "ManiFlowAngleEstimator":
        from protomotions.maniflow.loader import load_maniflow_policy

        policy, cfg, _info = load_maniflow_policy(
            ckpt_path, device=str(device), use_ema=use_ema, maniflow_root=maniflow_root
        )
        return cls(
            policy,
            num_envs=num_envs,
            device=device,
            obs_offset=obs_offset,
            output_bias=output_bias,
            denoise_steps=denoise_steps,
            frame_dt=frame_dt or _frame_dt_from_cfg(cfg),
        )

    @property
    def num_inference_steps(self) -> int:
        return int(self.policy.num_inference_steps)

    @property
    def horizon(self) -> int:
        """모델 예측 창 길이 [프레임]. 관측 창과 정렬돼 있다(모듈 docstring)."""
        return int(self.policy.horizon)

    @property
    def num_future_steps(self) -> int:
        """현재 프레임을 포함한 미래 프레임 수 = ``horizon − n_obs_steps + 1``."""
        return self.horizon - self.n_obs_steps + 1

    def set_calibration(
        self,
        obs_offset: Optional[_TensorLike] = None,
        output_bias: Optional[_TensorLike] = None,
    ) -> None:
        """영점 오프셋 갱신 — None인 인자는 기존 값을 유지한다. 전 env 공통 벡터이며,
        :meth:`set_calibration_envs`로 만든 env별 표가 있으면 그 표를 버리고 공통 벡터로 돌아간다."""
        if obs_offset is not None:
            off = torch.as_tensor(obs_offset, dtype=torch.float32, device=self.device)
            assert off.shape == (self.obs_dim,), (
                f"obs_offset must be ({self.obs_dim},), got {tuple(off.shape)}"
            )
            self._obs_offset = off
        if output_bias is not None:
            bias = torch.as_tensor(output_bias, dtype=torch.float32, device=self.device)
            assert bias.shape == (self.action_dim,), (
                f"output_bias must be ({self.action_dim},), got {tuple(bias.shape)}"
            )
            self._output_bias = bias

    def set_calibration_envs(
        self,
        env_ids: torch.Tensor,
        obs_offset: _TensorLike,
        output_bias: Optional[_TensorLike] = None,
    ) -> None:
        """env별 영점 오프셋 — 한 배치 안에서 동작(걷기·달리기)마다 영점이 다를 때 쓴다.

        처음 호출되면 공통 벡터를 (num_envs, dim) 표로 펼치고, 이후 호출은 ``env_ids`` 행만 바꾼다.
        :meth:`observe`/:meth:`set_history`/:meth:`predict`/:meth:`predict_horizon`은 표가 있으면 행별로 적용한다.
        """
        if self._obs_offset.dim() == 1:
            self._obs_offset = self._obs_offset.unsqueeze(0).repeat(self.num_envs, 1)
        if self._output_bias.dim() == 1:
            self._output_bias = self._output_bias.unsqueeze(0).repeat(self.num_envs, 1)
        m = int(env_ids.numel())
        off = torch.as_tensor(obs_offset, dtype=torch.float32, device=self.device)
        assert off.shape == (m, self.obs_dim), f"obs_offset must be ({m}, {self.obs_dim}), got {tuple(off.shape)}"
        self._obs_offset[env_ids] = off
        if output_bias is not None:
            bias = torch.as_tensor(output_bias, dtype=torch.float32, device=self.device)
            assert bias.shape == (m, self.action_dim), f"output_bias must be ({m}, {self.action_dim}), got {tuple(bias.shape)}"
            self._output_bias[env_ids] = bias

    def _obs_offset_for(self, env_ids: Optional[torch.Tensor], seq: bool) -> torch.Tensor:
        off = self._obs_offset
        if off.dim() == 2:
            if env_ids is not None:
                off = off[env_ids]
            if seq:
                off = off.unsqueeze(1)
        return off

    def _output_bias_for(self, env_ids: Optional[torch.Tensor], seq: bool) -> torch.Tensor:
        bias = self._output_bias
        if bias.dim() == 2:
            if env_ids is not None:
                bias = bias[env_ids]
            if seq:
                bias = bias.unsqueeze(1)
        return bias

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        """리셋된 env를 표시 — 다음 observe에서 history가 재프라이밍된다."""
        if env_ids is None:
            self._primed[:] = False
        else:
            self._primed[env_ids] = False

    def observe(self, obs: torch.Tensor) -> torch.Tensor:
        """현재 각도 관측을 history에 push.

        Args:
            obs: (num_envs, 4) rad — ``FLEXION_OBS_CHANNELS`` 순서.
                ``obs_offset`` 캘리브레이션은 내부에서 가산된다.

        Returns:
            캘리브레이션 적용 후의 관측 (history에 들어간 값).
        """
        obs = obs.to(device=self.device, dtype=torch.float32)
        if obs.shape != (self.num_envs, self.obs_dim):
            raise ValueError(
                f"obs must be ({self.num_envs}, {self.obs_dim}), "
                f"got {tuple(obs.shape)}"
            )
        obs = obs + self._obs_offset
        rolled = torch.cat([self._history[:, 1:], obs[:, None]], dim=1)
        primed_full = obs[:, None].expand(-1, self.n_obs_steps, -1)
        self._history = torch.where(self._primed[:, None, None], rolled, primed_full)
        self._primed[:] = True
        return obs

    @torch.no_grad()
    def predict(self) -> torch.Tensor:
        """현재 history에서 flexion 각도 chunk를 예측.

        Returns:
            (num_envs, n_action_steps, action_dim) rad, ``output_bias`` 감산
            완료. 첫 프레임(lead 0)은 마지막 :meth:`observe` 시각과 동일,
            이후 프레임이 +25/+50/... ms 미래다.
        """
        if not bool(self._primed.all()):
            raise RuntimeError("predict() called before observe() primed all envs")
        result = self.policy.predict_action({self.obs_key: self._history})
        return result["action"].detach() - self._output_bias_for(None, seq=True)

    def set_history(self, env_ids: Optional[torch.Tensor], history: torch.Tensor) -> None:
        """history를 외부에서 직접 써넣는다 (25 ms 격자 재표집 등).

        ``observe``와 달리 rolling push가 아니라 창 전체를 덮어쓴다. 시뮬레이션
        policy 주기(10 ms)가 모델 프레임(25 ms)의 정수배가 아닐 때, 호출자가
        자체 링 버퍼를 25 ms 격자로 보간해 넣는 용도다.

        Args:
            env_ids: 대상 env (None이면 전체).
            history: (m, n_obs_steps, obs_dim) rad — ``obs_offset``은 내부에서
                가산된다 (``observe``와 동일 규약).
        """
        history = history.to(device=self.device, dtype=torch.float32)
        m = self.num_envs if env_ids is None else int(env_ids.numel())
        expected = (m, self.n_obs_steps, self.obs_dim)
        if tuple(history.shape) != expected:
            raise ValueError(f"history must be {expected}, got {tuple(history.shape)}")
        history = history + self._obs_offset_for(env_ids, seq=True)
        if env_ids is None:
            self._history = history
            self._primed[:] = True
        else:
            self._history[env_ids] = history
            self._primed[env_ids] = True

    @torch.no_grad()
    def predict_horizon(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """예측 창 **전체**를 반환 (미래 150 ms까지 접근하는 유일한 경로).

        Args:
            env_ids: 대상 env (None이면 전체).

        Returns:
            (m, horizon, action_dim) rad, ``output_bias`` 감산 완료.
            index k = 상대 프레임 ``k − (n_obs_steps − 1)`` — 즉
            ``k = n_obs_steps − 1``이 현재 시각이고 그 이후가 미래다.
            미래 부분만 쓰려면 ``[:, n_obs_steps - 1:]`` (7프레임).
        """
        history = self._history if env_ids is None else self._history[env_ids]
        primed = self._primed if env_ids is None else self._primed[env_ids]
        if not bool(primed.all()):
            raise RuntimeError("predict_horizon() called before observe/set_history")
        result = self.policy.predict_action({self.obs_key: history})
        return result["action_pred"].detach() - self._output_bias_for(env_ids, seq=True)
