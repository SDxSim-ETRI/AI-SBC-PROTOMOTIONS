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
receding-horizon으로 돌리기 위한 래퍼. ``ManiFlowTorqueEstimator``와 같은
history 관리 구조를 쓰되, 관측이 ``RobotState``가 아니라 **각도 4채널 벡터**라는
점이 다르다 (진자 env·skeleton env·실기기 IMU가 각자 방식으로 이 4ch를 구성).

관측/출력 계약 (flexion40 모델, ``docs/AI_SBC_FRAMEWORK.md`` §3.1):

    obs(4)    = [hip_flexion_r, hip_flexion_l, trunk_pitch, trunk_roll]
                (rad, skeleton hinge 규약; pitch=앞숙임+, roll=오른쪽+)
    action(2) = [hip_flexion_r, hip_flexion_l] 미래 각도 (rad)
                × n_action_steps(=4) 프레임 chunk

시간 규약: 모델의 native 프레임은 **40 fps(25 ms)** — :meth:`observe`는 이
주기로 호출해야 한다 (진자 env policy 100 Hz와는 다름 — 결합 시
`AssistTargetControl`의 ``hlp_dt=0.025``로 흡수). :meth:`predict`가 반환하는
chunk의 첫 프레임(lead 0)은 **마지막 observe와 같은 시각**이고, 나머지가
+25/+50/+75 ms 미래다.

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

import torch

# flexion40 모델의 채널 계약 — zarr attrs(obs_spec/action_dof_names)와 대조용.
FLEXION_OBS_CHANNELS = (
    "hip_flexion_r",
    "hip_flexion_l",
    "trunk_pitch",
    "trunk_roll",
)
FLEXION_ACTION_CHANNELS = ("hip_flexion_r", "hip_flexion_l")

_TensorLike = Union[torch.Tensor, Sequence[float]]


class ManiFlowAngleEstimator:
    """rolling 관측 history 기반 receding-horizon flexion 각도 예측기.

    사용 패턴 (``ManiFlowTorqueEstimator``와 동일):

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
    ):
        self.policy = policy
        self.num_envs = num_envs
        self.device = torch.device(device)

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
    ) -> "ManiFlowAngleEstimator":
        from protomotions.maniflow.loader import load_maniflow_policy

        policy, _cfg, _info = load_maniflow_policy(
            ckpt_path, device=str(device), use_ema=use_ema, maniflow_root=maniflow_root
        )
        return cls(
            policy,
            num_envs=num_envs,
            device=device,
            obs_offset=obs_offset,
            output_bias=output_bias,
            denoise_steps=denoise_steps,
        )

    @property
    def num_inference_steps(self) -> int:
        return int(self.policy.num_inference_steps)

    def set_calibration(
        self,
        obs_offset: Optional[_TensorLike] = None,
        output_bias: Optional[_TensorLike] = None,
    ) -> None:
        """영점 오프셋 갱신 — None인 인자는 기존 값을 유지한다."""
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
        return result["action"].detach() - self._output_bias
