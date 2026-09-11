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
"""LLP v3 — 실 HLP(150 ms)와 형식이 일치하는 chunk로 재학습.

v2와 달라지는 것은 **chunk 형식과 관측 미래 창**뿐이다. 의도 궤적 θ_g는
v2와 같은 **합성 sine을 유지**한다 (2026-09-08 결정): 합성 궤적이 mocap보다
케이던스·진폭·좌우 상관을 넓게 흔들어 주므로 "다양한 상황에서 목표각 도달을
돕는다"는 LLP 설계 목표에 더 맞고, mocap 궤적에서의 성능은 **일반화 시험**으로
평가 단계에서 확인한다.

v2 → v3 변경점

| 항목 | v2 | v3 | 이유 |
|------|----|----|------|
| `hlp_dt` / `chunk_knots` | 0.1 / 8 (0.7 s span) | 0.025 / 7 (0~150 ms) | 실 HLP의 예측 격자·범위와 일치 |
| `chunk_refresh_steps` | 10 (10 Hz) | 2 (50 Hz) | 실 HLP는 매 프레임 갱신 가능; age가 0/10 ms로 유지 |
| 관측 미래 창 | 5×0.1 s (0~0.4 s) | 4×0.05 s (0~0.15 s) | 150 ms를 넘는 창은 ZOH 허구값 |
| chunk 노이즈 | 백색 0.02 + 바이어스 0.03 | 백색 0.004 + 바이어스 0.005 + **drift 0.45 rad/s** | 실 HLP 오차는 lead에 비례해 매끄럽게 커진다 (0.36°@25 ms → 3.11°@150 ms) |
| control 클래스 | `AssistTargetControl` | `MotionAssistTargetControl` | 평가 때 `--overrides`로 mocap 의도·실 HLP chunk로 전환 가능 |

drift 계수 근거: 반정규분포에서 σ = MAE/0.798이므로 측정 MAE로부터
σ(25 ms)≈0.008 rad, σ(150 ms)≈0.068 rad → 기울기 ≈0.48 rad/s. 아래 값은
예측 MAE 0.59/1.07/2.08/3.10° (@25/50/100/150 ms)로 측정값에 근사한다.

평가 시 전환 (학습 후, `--overrides`만으로):
    env.control_components.assist.intent_source=motion_data
    env.control_components.assist.motion_zarr_path=<SEED zarr>
    env.control_components.assist.motion_split=val
    env.control_components.assist.chunk_source=maniflow
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mlp import (  # noqa: E402, F401
    ASSIST_TORQUE_LIMIT,
    agent_config,
    apply_inference_overrides,
    configure_robot_and_simulator,
    motion_lib_config,
    scene_lib_config,
    terrain_config,
)
from mlp import env_config as _v2_env_config  # noqa: E402

from protomotions.robot_configs.base import RobotConfig  # noqa: E402

# 실 HLP(flexion40) 격자: 25 ms × (현재 + 미래 6) = 0~150 ms
HLP_KNOT_DT = 0.025
HLP_CHUNK_KNOTS = 7


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace):
    """v2 설정에서 chunk 형식·관측 창·control 클래스만 교체."""
    from protomotions.envs.control.assist_motion_control import (
        MotionAssistTargetControlConfig,
    )

    cfg = _v2_env_config(robot_cfg, args)
    v2 = cfg.control_components["assist"]

    cfg.control_components["assist"] = MotionAssistTargetControlConfig(
        # 실 HLP와 동일한 chunk 격자 (0~150 ms)
        hlp_dt=HLP_KNOT_DT,
        chunk_knots=HLP_CHUNK_KNOTS,
        chunk_refresh_steps=2,
        # 실 HLP 오차 통계에 맞춘 에뮬레이션
        chunk_noise_std=0.004,
        chunk_bias_std=0.005,
        chunk_drift_std=0.45,
        chunk_delay_max=0.01,
        # 관측 미래 창: 0.0~0.15 s
        obs_future_samples=4,
        obs_future_dt=0.05,
        # 의도 궤적은 v2와 동일한 합성 sine (기본값) — 파라미터도 v2 그대로
        freq_range=v2.freq_range,
        amp_range=v2.amp_range,
        offset_range=v2.offset_range,
        harmonic_amp_frac_max=v2.harmonic_amp_frac_max,
        antiphase_jitter=v2.antiphase_jitter,
        uncorrelated_prob=v2.uncorrelated_prob,
        hold_prob=v2.hold_prob,
        segment_gap_range=v2.segment_gap_range,
        blend_range=v2.blend_range,
        start_blend=v2.start_blend,
        dof_limit_margin=v2.dof_limit_margin,
        marker_leg_length=v2.marker_leg_length,
    )
    return cfg
