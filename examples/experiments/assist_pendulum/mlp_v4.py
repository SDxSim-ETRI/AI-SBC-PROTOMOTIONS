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
"""LLP v4 — 랜덤화된 진자: 어떤 입력이 와도 예측 목표 쪽으로 옳게 미는 정책.

v3(합성 sine + 매끈한 PD)는 "그네 밀어주기"만 배워서 보행 착지의 −300 N·m 스파이크에
대응하지 못했다 (`docs/NEXT_SESSION.md` §2-D·§2-E). v4는 의도·관측 계약·보상 뼈대는
v3 그대로 두고 **플랜트와 외란의 분포**를 넓힌다 — 정책은 그 값을 관측하지 못하고
각도·각속도 히스토리와 HLP 예측창으로 효과만 본다.

| 축 | v3 | v4 |
|----|----|----|
| 하중 | 없음 | 입각 사건(외부 모멘트 0~2 N·m/kg + 착용자 굳힘 Kp×2~8) + 작은 섭동, 시점·부호 무작위 |
| 관성 | 0.42 고정 | 질량중심 관성 ×1~25 → hip 축 0.42~3.6 kg·m² (스켈레톤 다리 3.5). 질량·중력은 고정 |
| 착용자 임피던스 | 게인 ±30 % | ω_n 12~24 rad/s, ζ 0.3~0.8 (진자 26·0.66 근방; 12 미만은 중력을 못 버팀) |
| θ_g | sine + hold, 블렌드 0.4~1.2 s | 블렌드 0.25~2.5 s(빠른 재배치·램프), hold 30 % |
| 히스토리 | 100 ms / 이전 action 50 ms | 250 ms / 100 ms (관성 10배 범위의 암묵적 식별용) |
| 비용 | \\|τ·ω\\| | \\|τ·ω\\| + τ² (속도로 못 내리는 항) |

HLP chunk는 v3와 같은 에뮬레이션(θ_g 미래 + 실측 오차 통계) — chunk는 의도이고 하중은
외란이라, 정책이 배우는 원리는 "예측에서 밀려나는 만큼 예측 쪽으로 밀어라".

학습 (sbc env):
    python protomotions/train_agent.py --robot-name hip_pendulum --simulator newton \\
        --experiment-path examples/experiments/assist_pendulum/mlp_v4.py \\
        --experiment-name assist_pendulum_v4 --motion-file none --num-envs 4096 --batch-size 16384
점검 (학습 전, 무보조 롤아웃 파형): examples/experiments/assist_pendulum/check_env_v4.py
평가: 진자 A/B(`eval_abc.py`) + **스켈레톤(안 본 플랜트)** `assist_skeleton/run_system.py --llp-checkpoint ...`
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mlp import (  # noqa: E402, F401
    agent_config,
    apply_inference_overrides,
    configure_robot_and_simulator,
    motion_lib_config,
    scene_lib_config,
    terrain_config,
)
from mlp_v3 import HLP_CHUNK_KNOTS, HLP_KNOT_DT  # noqa: E402
from mlp_v3 import env_config as _v3_env_config  # noqa: E402

from protomotions.robot_configs.base import RobotConfig  # noqa: E402


def _pair(s: str):
    lo, hi = (float(v) for v in s.split(","))
    return (lo, hi)


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    g = parser.add_argument_group("v4 plant/disturbance randomization")
    g.add_argument("--assist-limit", type=float, default=40.0, help="assist 토크 한계 [N·m] (tanh 스케일 = clamp)")
    g.add_argument("--inertia-range", type=_pair, default=(1.0, 25.0), help="대퇴 질량중심 관성 배율 lo,hi (질량·중력은 고정; hip 축 관성 0.42 → ×25 = 3.6 kg·m²)")
    g.add_argument("--omega-n-range", type=_pair, default=(12.0, 24.0), help="착용자 PD 고유진동수 [rad/s] lo,hi (12 미만은 중력도 못 버텨 조기 종료)")
    g.add_argument("--zeta-range", type=_pair, default=(0.3, 0.8), help="착용자 PD 감쇠비 lo,hi")
    g.add_argument("--stance-load-per-kg", type=float, default=2.0, help="입각 하중 상한 [N·m/kg] (문헌 보행 피크 ≈ 1)")
    g.add_argument("--stance-kp-mult", type=_pair, default=(2.0, 8.0), help="입각 굳힘 Kp 배율 lo,hi")
    g.add_argument("--stance-gap", type=_pair, default=(0.4, 1.6), help="입각 사건 간격 [s] lo,hi")
    g.add_argument("--stance-dur", type=_pair, default=(0.08, 0.45), help="입각 사건 길이 [s] lo,hi")
    g.add_argument("--perturb-amp-per-kg", type=float, default=0.3, help="작은 섭동 상한 [N·m/kg]")
    g.add_argument("--no-disturbance", action="store_true", help="외란 끔 (절제: 관성·임피던스 랜덤화만)")
    g.add_argument("--blend-range", type=_pair, default=(0.25, 2.5), help="θ_g 세그먼트 전환 시간 [s] (0.25 = 빠른 재배치, 2.5 = 완만한 램프; 0.05는 초당 640° 의도가 돼 비인간적)")
    g.add_argument("--hold-prob", type=float, default=0.3, help="θ_g 정지 세그먼트 확률")
    g.add_argument("--history-steps", type=int, default=25, help="관절 히스토리 길이 (10 ms 단위; 25 = 250 ms)")
    g.add_argument("--prev-actions", type=int, default=10, help="이전 action 수 (10 = 100 ms)")
    g.add_argument("--human-cost", choices=["work", "torque_sq", "both"], default="both")
    g.add_argument("--work-weight", type=float, default=-0.0016, help="|τ·ω| 항 weight (무보조 ≈126 W → −0.2/step; v3는 12 W에 −0.01)")
    g.add_argument("--torque-sq-weight", type=float, default=-3.0e-5, help="τ² 항 weight (무보조 ≈6,200 → −0.19/step)")
    g.add_argument("--tracking-weight", type=float, default=1.0, help="θ_g 추종 exp 보상 weight (퇴화 해 차단)")


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace):
    from protomotions.envs.component_factories import previous_actions_factory
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.control.assist_disturbance_control import (
        DisturbedAssistTargetControlConfig,
    )
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.rewards.assist import compute_human_power_rew
    from protomotions.envs.action import make_torque_action_config

    cfg = _v3_env_config(robot_cfg, args)
    v3 = cfg.control_components["assist"]

    # --- 의도·chunk는 v3 그대로, 외란 컴포넌트로 교체 + θ_g 다양화 ---
    cfg.control_components["assist"] = DisturbedAssistTargetControlConfig(
        hlp_dt=HLP_KNOT_DT,
        chunk_knots=HLP_CHUNK_KNOTS,
        chunk_refresh_steps=v3.chunk_refresh_steps,
        chunk_noise_std=v3.chunk_noise_std,
        chunk_bias_std=v3.chunk_bias_std,
        chunk_drift_std=v3.chunk_drift_std,
        chunk_delay_max=v3.chunk_delay_max,
        obs_future_samples=v3.obs_future_samples,
        obs_future_dt=v3.obs_future_dt,
        freq_range=v3.freq_range,
        amp_range=v3.amp_range,
        offset_range=v3.offset_range,
        harmonic_amp_frac_max=v3.harmonic_amp_frac_max,
        antiphase_jitter=v3.antiphase_jitter,
        uncorrelated_prob=v3.uncorrelated_prob,
        hold_prob=float(args.hold_prob),
        segment_gap_range=v3.segment_gap_range,
        blend_range=tuple(args.blend_range),
        start_blend=v3.start_blend,
        dof_limit_margin=v3.dof_limit_margin,
        marker_leg_length=v3.marker_leg_length,
        # 외란
        disturbance_enabled=not args.no_disturbance,
        stance_load_per_kg=float(args.stance_load_per_kg),
        stance_kp_mult_range=tuple(args.stance_kp_mult),
        stance_gap_range=tuple(args.stance_gap),
        stance_dur_range=tuple(args.stance_dur),
        perturb_amp_per_kg=float(args.perturb_amp_per_kg),
    )

    # --- 플랜트 랜덤화 (AssistEnv v4 필드) ---
    cfg.inertia_scale_range = tuple(args.inertia_range)
    cfg.impedance_mode = "natural_freq"
    cfg.omega_n_range = tuple(args.omega_n_range)
    cfg.zeta_range = tuple(args.zeta_range)

    # --- 관측: 긴 히스토리 (암묵적 시스템 식별) ---
    cfg.num_state_history_steps = int(args.history_steps)
    cfg.observation_components["previous_actions"] = previous_actions_factory(
        history_steps=int(args.prev_actions)
    )

    # --- 비용: |τ·ω| + τ² ---
    def cost(weight: float, torque_squared: bool):
        return MdpComponent(
            compute_func=compute_human_power_rew,
            dynamic_vars={
                "tau_agent": EnvContext.assist.tau_agent,
                "dof_vel": EnvContext.current.dof_vel,
            },
            static_params={
                "use_torque_squared": torque_squared,
                "weight": float(weight),
                "min_value": -3.0,
                "zero_during_grace_period": True,
            },
        )

    cfg.reward_components.pop("human_power", None)
    if args.human_cost in ("work", "both"):
        cfg.reward_components["human_power"] = cost(args.work_weight, False)
    if args.human_cost in ("torque_sq", "both"):
        cfg.reward_components["human_torque_sq"] = cost(args.torque_sq_weight, True)
    cfg.reward_components["tracking"].static_params["weight"] = float(args.tracking_weight)

    # --- 토크 한계 ---
    cfg.action_config = make_torque_action_config(torque_scale=float(args.assist_limit), action_transform="tanh")
    cfg.assist_torque_limit = float(args.assist_limit)
    return cfg
