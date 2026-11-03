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
"""AI-SBC LLP 진자 학습 레시피 — L1 → R0 → 20 N·m 다리별(motion_leg) 계보를 프리셋 하나로.

사람(BUILT_IN_PD)이 의도 궤적 θ_g를 추종하는 동안 policy가 additive assist torque(τ_assist)를
출력해 사람 토크를 줄이는 태스크. HLP(ManiFlow) 예측은 노이즈/지연이 주입된 chunk로 에뮬레이션한다.
정책은 플랜트·외란 값을 관측하지 못하고 각도·각속도 히스토리와 HLP 예측창으로 효과만 본다.

프리셋 (`--recipe`, 기본 L1):

| 이름 | 내용 | 보관 모델 |
|------|------|-----------|
| `L1` | 랜덤화 진자 (v4): 입각 하중·섭동, 관성 ×1–25, 착용자 ω_n 12–24 rad/s·ζ 0.3–0.8, 250 ms 히스토리, \\|τ·ω\\|+τ² 비용, 40·tanh | `checkpoints/L1_llp_pendulum` |
| `R0` | L1 + 올바른 토크 경로: 착용자 PD+FF를 구동기 하나로 한 번만 제한하고 보조·하중은 독립 주입 | `results/llp_relative_R0_s0_20261020` |
| `motion_leg` | R0 + 20·tanh(한계 20 N·m, 배율 1) + SEED 실제 동작 의도(가벼운 다리 집단 포함) + 다리 하나만 보조·보상·관측(`SingleJointMLP`); 배포는 같은 가중치를 양다리에 (`LLPPolicy(share_legs=True)`) | `results/llp20_motion_*_right_s0_20261021` |

계보의 단계는 아래 `_env_*`/`_agent_*` 함수가 차례로 적용한다 — v2 기본(`_env_base`) → v3 chunk 형식
(`_env_hlp_chunk_format`) → v4 외란·플랜트 랜덤화(`_env_randomized_plant`, L1) → R0 토크 경로
(`_env_corrected_torque_path`) → motion_leg(`_env_motion_intent_per_leg`, `_agent_single_joint_actor`).
2026-10-26 이전의 mlp_v3.py / mlp_v4.py / mlp_relative.py / mlp_motion_leg.py가 각각 그 함수다.

v4 뒤의 실패한 변형(L2–L5, 2026-09-29)은 모두 이 파일의 CLI 인자로 남아 있다: L2 `--mass-exponent 0.43
--assist-limit 50 --assist-limit-range 10,50`, L3 `--stance-load-per-kg 1.0 --stance-prob 0.6
--reversal-brake-prob 0.7`, L4 `--chunk-anchor actual`, L5 `--reversal-mode feedforward`. v4 정점(epoch 6,000)
뒤 두 절약 항이 "덜 밀기"를 이끌어 물러났고(v5 `--assist-effort-weight=-5e-6 --action-rate-weight=-0.002`도
정점 유지 실패), 보관본 v4@6000 = L1이다.

학습 (sbc env):
    python protomotions/train_agent.py --robot-name hip_pendulum --simulator newton \\
        --experiment-path examples/experiments/assist_pendulum/mlp.py --recipe motion_leg --leg right \\
        --experiment-name llp20_motion_A0D0_right_s0 --motion-file none \\
        --num-envs 4096 --batch-size 16384 --seed 0 --training-max-steps 786432000 \\
        --assist-effort-weight=0 --action-rate-weight=0
`--motion-file`은 CLI상 필수지만 이 실험은 모션 라이브러리를 쓰지 않는다. 무보조 평가는
`--overrides env.assist_torque_limit=0`. 학습 전 환경 점검은 `runtime.py check-env`, 짝 평가는 `evaluate.py`.

`--policy mlp_fpo|tfm_fpo`(2026-11-03): 같은 env·관측·보상·토크 한계에서 actor만 flow 정책(FPO++) 또는 Temporal Transformer
context + flow 정책으로 바꾼다 (`protomotions/agents/fpo/recipe.py`, 전신 `llp_walk.py --policy`와 같은 인자). motion_leg 프리셋의
SingleJointMLP와는 함께 쓸 수 없다.
"""

import argparse
from dataclasses import dataclass, fields
from pathlib import Path

from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.envs.base_env.assist_env import AssistEnvConfig
from protomotions.agents.fpo.recipe import add_policy_arguments, fpo_agent_config
from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.components.terrains.config import TerrainConfig
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.motion_lib import MotionLibConfig

ROOT = Path(__file__).resolve().parents[3]

# v2 기본 어시스트 토크 한계 [N·m] (v4부터는 --assist-limit)
ASSIST_TORQUE_LIMIT = 25.0
# v3: 실 HLP(flexion40) 격자 25 ms × (현재 + 미래 6) = 0~150 ms (v4부터는 --hlp-dt/--chunk-knots)
HLP_KNOT_DT = 0.025
HLP_CHUNK_KNOTS = 7

# motion_leg: SEED 고관절 굴곡 100 fps 데이터 (H1 학습분할만 학습에 쓴다; 미학습분은 평가용)
SEED_ZARR = "/home/bak/Projects/ManiFlow_Policy/ManiFlow/data/hip-flexion-seed-locomotion-100fps.zarr"
SEED_METADATA = str(ROOT / "data/seed/metadata/seed_metadata_v004.parquet")
LEGS = {"right": 0, "left": 1}  # hip_pendulum COMMON 순서 = 데이터 채널 순서 [r, l]
# (그룹, content_type_of_movement 정규식) — 처음 맞는 그룹
MOTION_GROUPS = (
    ("running", r"^running"),
    ("jogging", r"^jogging"),
    ("deep", r"sitting|kneel|transition|crouch"),
    ("jumping", r"^jumping"),
    ("walking", r"^walking(, turning)?$"),
    ("other", r".*"),
)
# (집단, env 비율, 의도 소스, 관성 배율 lo, hi, 동작 그룹 시간 비율)
# hip 축 관성 I = 0.2863 + 0.1316 × 배율 → 배율 3.9 = 0.80 kg·m²
INTENT_POOLS = (
    ("synthetic", 0.2, "synthetic", 1.0, 25.0, ()),
    ("light_fast", 0.4, "motion", 1.0, 3.9,
     (("running", 0.6), ("jogging", 0.3), ("jumping", 0.1))),
    ("general", 0.4, "motion", 1.0, 25.0,
     (("walking", 0.35), ("jogging", 0.2), ("jumping", 0.1), ("deep", 0.25), ("other", 0.1))),
)
# 각 다리가 따라갈 수 없는 클립(|θ̈| 99 백분위 × hip 관성 > 이 비율 × 착용자 토크 한계)은 그 env에서 뽑지 않는다
MOTION_ACCEL_FRAC = 0.8


@dataclass(frozen=True)
class Recipe:
    """프리셋 — 계보의 어느 단계까지 적용하는지."""

    name: str
    assist_limit: float            # --assist-limit 기본값 [N·m]
    corrected_torque_path: bool    # R0: 착용자 PD+FF 한 구동기, 보조·하중 독립
    motion_intent_per_leg: bool    # motion_leg: 실제 동작 의도 + 다리 하나만
    fixed_final_epoch: bool        # 마지막 epoch 고정 (주기 평가·best 선택 없음, 체크포인트 500 epoch마다)
    description: str


RECIPES = {
    "L1": Recipe("L1", 40.0, False, False, False, "v4 randomized pendulum (checkpoints/L1_llp_pendulum)"),
    "R0": Recipe("R0", 40.0, True, False, True, "L1 on the corrected torque path"),
    "motion_leg": Recipe("motion_leg", 20.0, True, True, True,
                         "R0 at 20 N·m with SEED motion intents, one joint assisted/observed"),
}


def leg_joint_mask(leg: str, num_dofs: int = 2):
    joint = LEGS[leg]
    return joint, tuple(1.0 if j == joint else 0.0 for j in range(num_dofs))


def resolve_recipe(args: argparse.Namespace) -> Recipe:
    """프리셋을 고르고 프리셋 의존 기본값(보조 한계)을 args에 채운다."""
    recipe = RECIPES[getattr(args, "recipe", "L1")]
    if getattr(args, "assist_limit", None) is None:
        args.assist_limit = recipe.assist_limit
    if recipe.motion_intent_per_leg and not getattr(args, "leg", None):
        raise ValueError("--recipe motion_leg requires --leg right|left")
    return recipe


# ---------------------------------------------------------------------------
# train_agent 실험 파일 계약
# ---------------------------------------------------------------------------
def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    """고정 베이스(fix_base_link) 지원 Newton 서브클래스로 교체."""
    if "newton" in simulator_cfg._target_:
        simulator_cfg._target_ = (
            "protomotions.simulator.newton.fixed_base.FixedBaseNewtonSimulator"
        )


def terrain_config(args: argparse.Namespace):
    """평지 지형 (고정 베이스라 접촉은 없지만 프레임워크상 필요)."""
    return TerrainConfig()


def scene_lib_config(args: argparse.Namespace):
    return SceneLibConfig(scene_file=None)


def motion_lib_config(args: argparse.Namespace):
    """모션 데이터 미사용 — 빈 MotionLib (--motion-file 값은 무시)."""
    return MotionLibConfig(motion_file=None)


def _pair(s: str):
    lo, hi = (float(v) for v in s.split(","))
    return (lo, hi)


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--recipe", choices=sorted(RECIPES), default="L1",
                        help="프리셋: " + "; ".join(f"{k} = {v.description}" for k, v in RECIPES.items()))
    g = parser.add_argument_group("v4 plant/disturbance randomization")
    g.add_argument("--assist-limit", type=float, default=None, help="최종 assist 토크 한계 [N·m] (기본 action 스케일도 이 값). 기본은 프리셋 값: L1·R0 40, motion_leg 20")
    g.add_argument("--action-torque-scale", type=float, default=None, help="raw action의 N·m 스케일. 기본값은 기존처럼 assist-limit")
    g.add_argument("--action-transform", choices=["tanh", "none"], default="tanh", help="none이면 선형 요청 토크를 만든 뒤 assist-limit에서만 제한")
    g.add_argument("--separate-torque-limits", action="store_true", help="보조·착용자 FF를 개별 제한하고 외란과의 합에 착용자 한계를 재적용하지 않음")
    g.add_argument("--inertia-range", type=_pair, default=(1.0, 25.0), help="대퇴 질량중심 관성 배율 lo,hi (hip 축 관성 0.42 → ×25 = 3.6 kg·m²)")
    # L2 (2026-09-29): 질량을 관성과 함께 키우고(중력/관성 비를 실제 다리처럼), 보조 한계를 env마다 다르게.
    g.add_argument("--mass-exponent", type=float, default=0.0, help="질량 배율 = 관성 배율^p (0 = 질량 고정 v4; 0.43 = ×25에서 질량 ×4)")
    g.add_argument("--assist-limit-range", type=_pair, default=None, help="env별 보조 한계 U[lo,hi] [N·m] (정책은 관측 못 함). 없으면 --assist-limit 하나")
    # L3 (2026-09-29): 하중을 사람 범위 중심으로(필요량 < 한계인 상황이 흔하게) + 유각 말기 '공짜 감속'(반전 브레이크)
    g.add_argument("--stance-prob", type=float, default=0.85, help="간격이 찼을 때 입각 사건이 실제로 오는 확률 (낮추면 가벼운 구간 ↑)")
    g.add_argument("--reversal-brake-prob", type=float, default=0.0, help="의도 반전 직전 외부 점성 브레이크 확률 (0 = 꺼짐)")
    g.add_argument("--reversal-brake-c", type=_pair, default=(2.0, 30.0), help="브레이크 점성 계수 U[lo,hi] [N·m·s/rad]")
    g.add_argument("--reversal-brake-lead", type=float, default=0.15, help="반전 전 브레이크 창 [s]")
    g.add_argument("--reversal-mode", choices=["viscous", "feedforward"], default="viscous", help="반전 사건: viscous = −c·θ̇ (L3·L4) | feedforward = κ·I·θ̈_g 를 환경이 대신 냄 (L5)")
    g.add_argument("--reversal-ff-gain", type=_pair, default=(0.5, 1.2), help="feedforward κ U[lo,hi]")
    # L4 (2026-09-29): chunk를 실제 각도에 정박 — 배포 HLP(실제 각도의 미래)와 같은 정보량
    g.add_argument("--chunk-anchor", choices=["intent", "actual"], default="intent", help="chunk 정박점: intent = θ_g 미래(v3~L3), actual = θ(t) + Δθ_g")
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
    # v5 (2026-09-28): v4는 epoch 6,000 정점 뒤 이 두 항이 이끄는 "덜 밀기"를 배워 스켈레톤에서 물러났다. 값은 '=-5e-6' 꼴로 넘길 것.
    g.add_argument("--assist-effort-weight", type=float, default=-2e-5, help="보조 토크 절약 항 Στ²_assist weight (v4 −2e-5, v5 −5e-6; motion_leg 연구의 A)")
    g.add_argument("--action-rate-weight", type=float, default=-0.005, help="action 변화율 항 weight (v4 −0.005, v5 −0.002; motion_leg 연구의 Δ)")
    # HLP 예측 형식 — 배포 HLP(flexion100: 10 ms × 6점 = 0~50 ms, 10 ms마다 갱신)와 맞춘다.
    # v3 형식(25 ms × 7점, 창 0~150 ms)으로 학습하면 스켈레톤 시험에서 50 ms 밖 창이 ZOH 상수가 된다.
    g.add_argument("--hlp-dt", type=float, default=0.01, help="chunk 예측 시점 간격 [s]")
    g.add_argument("--chunk-knots", type=int, default=6, help="chunk 예측 시점 수 (0부터)")
    g.add_argument("--chunk-refresh-steps", type=int, default=1, help="chunk 갱신 주기 [policy step] (1 = 10 ms)")
    g.add_argument("--obs-future-samples", type=int, default=3, help="LLP 관측 미래 창 샘플 수 (v10: 3 = {0,10,20 ms})")
    g.add_argument("--obs-future-dt", type=float, default=0.01, help="관측 미래 창 간격 [s]")
    m = parser.add_argument_group("motion_leg: per-leg training with motion-data intents")
    m.add_argument("--leg", choices=sorted(LEGS), default=None, help="학습할 다리 (motion_leg 프리셋에서 필수)")
    m.add_argument("--motion-zarr", default=SEED_ZARR, help="SEED 고관절 굴곡 zarr (100 fps)")
    m.add_argument("--motion-metadata", default=SEED_METADATA, help="SEED metadata parquet")
    # --- FPO++ / Temporal Transformer (2026-11-03): env·관측·보상·토크 한계는 위와 같고 정책·학습 알고리즘만 바꾼다 ---
    add_policy_arguments(parser, policy_note="진자 학습은 모두 무작위 초기화 (motion_leg 프리셋과는 함께 쓸 수 없다)")


# ---------------------------------------------------------------------------
# env config: 계보의 단계
# ---------------------------------------------------------------------------
def _env_base(robot_cfg: RobotConfig) -> AssistEnvConfig:
    """v2: 합성 sine 의도 + 10 Hz 에뮬레이션 chunk + 관측·보상·종료 뼈대."""
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.component_factories import (
        previous_actions_factory,
        action_smoothness_factory,
    )
    from protomotions.envs.control.assist_target_control import (
        AssistTargetControlConfig,
    )
    from protomotions.envs.obs.assist_obs import (
        compute_assist_proprio_obs,
        compute_assist_target_obs,
        compute_assist_privileged_obs,
    )
    from protomotions.envs.rewards.assist import (
        compute_human_power_rew,
        compute_target_tracking_rew,
        compute_assist_effort_rew,
    )
    from protomotions.envs.terminations.assist import (
        compute_tracking_failure_term,
        compute_dof_vel_failure_term,
    )
    from protomotions.envs.action import make_torque_action_config

    control_components = {
        "assist": AssistTargetControlConfig(
            # HLP 에뮬레이션: 10 Hz chunk (policy 100 Hz 기준 10 step마다)
            hlp_dt=0.1,
            chunk_knots=8,
            chunk_refresh_steps=10,
            chunk_noise_std=0.02,
            chunk_bias_std=0.03,
            chunk_delay_max=0.05,
            # 관측 미래 창: 0.0~0.4 s
            obs_future_samples=5,
            obs_future_dt=0.1,
        ),
    }

    observation_components = {
        # 관절 상태 + 히스토리 (실기기: encoder/IMU)
        "assist_proprio": MdpComponent(
            compute_func=compute_assist_proprio_obs,
            dynamic_vars={
                "dof_pos": EnvContext.current.dof_pos,
                "dof_vel": EnvContext.current.dof_vel,
                "historical_dof_pos": EnvContext.historical.dof_pos,
                "historical_dof_vel": EnvContext.historical.dof_vel,
            },
            static_params={"dof_vel_scale": 0.25},
        ),
        # HLP 목표 창 (interpolation된 chunk, 노이즈/지연 포함)
        "assist_target": MdpComponent(
            compute_func=compute_assist_target_obs,
            dynamic_vars={
                "theta_d_future": EnvContext.assist.theta_d_future,
                "dof_pos": EnvContext.current.dof_pos,
                "chunk_age": EnvContext.assist.chunk_age,
            },
        ),
        # action 히스토리
        "previous_actions": previous_actions_factory(history_steps=5),
        # critic 전용 privileged 관측 (사람 토크, 깨끗한 θ_g, 게인 배율)
        "assist_privileged": MdpComponent(
            compute_func=compute_assist_privileged_obs,
            dynamic_vars={
                "tau_agent": EnvContext.assist.tau_agent,
                "tau_assist": EnvContext.assist.tau_assist,
                "theta_g": EnvContext.assist.theta_g,
                "dof_pos": EnvContext.current.dof_pos,
                "gain_scale": EnvContext.assist.gain_scale,
            },
            static_params={"torque_scale": 0.01},
        ),
    }

    reward_components = {
        # 1차 목표: 궤적 추종 유지 (퇴화 해 차단)
        "tracking": MdpComponent(
            compute_func=compute_target_tracking_rew,
            dynamic_vars={
                "dof_pos": EnvContext.current.dof_pos,
                "theta_g": EnvContext.assist.theta_g,
            },
            static_params={"tracking_coef": 40.0, "weight": 1.0},
        ),
        # 핵심 목표: 사람 에너지 최소화 (τ_agent 파워 페널티)
        "human_power": MdpComponent(
            compute_func=compute_human_power_rew,
            dynamic_vars={
                "tau_agent": EnvContext.assist.tau_agent,
                "dof_vel": EnvContext.current.dof_vel,
            },
            static_params={
                "use_torque_squared": False,
                "weight": -0.01,
                "min_value": -3.0,
                "zero_during_grace_period": True,
            },
        ),
        # 정칙화: assist 사용량 + action rate
        "assist_effort": MdpComponent(
            compute_func=compute_assist_effort_rew,
            dynamic_vars={"tau_assist": EnvContext.assist.tau_assist},
            static_params={"weight": -2e-5},
        ),
        "action_rate": action_smoothness_factory(weight=-0.005),
    }

    termination_components = {
        "tracking_failure": MdpComponent(
            compute_func=compute_tracking_failure_term,
            dynamic_vars={
                "dof_pos": EnvContext.current.dof_pos,
                "theta_g": EnvContext.assist.theta_g,
            },
            static_params={"max_err": 1.0},
        ),
        "dof_vel_failure": MdpComponent(
            compute_func=compute_dof_vel_failure_term,
            dynamic_vars={"dof_vel": EnvContext.current.dof_vel},
            static_params={"max_vel": 25.0},
        ),
    }

    return AssistEnvConfig(
        max_episode_length=1000,  # 10 s @ 100 Hz
        reset_grace_period=5,
        num_state_history_steps=10,
        control_components=control_components,
        observation_components=observation_components,
        reward_components=reward_components,
        termination_components=termination_components,
        action_config=make_torque_action_config(
            torque_scale=ASSIST_TORQUE_LIMIT, action_transform="tanh"
        ),
        assist_torque_limit=ASSIST_TORQUE_LIMIT,
        human_gain_scale_min=0.7,
        human_gain_scale_max=1.3,
        # 사람 모델 v2: τ_agent = Kp(θ_g−θ) + Kd(θ̇_g−θ̇) — 속도 피드포워드로
        # 사람 단독(베이스라인) 추종 성능을 올려, assist 효과가 "순수 에너지
        # 절감"으로 분리되게 한다.
        human_velocity_feedforward=True,
    )


def _env_hlp_chunk_format(cfg: AssistEnvConfig) -> None:
    """v3 (2026-09-08): chunk 형식·관측 창·control 클래스만 실 HLP(150 ms)에 맞춘다.

    의도 궤적 θ_g는 v2와 같은 합성 sine을 유지한다 — 합성 궤적이 mocap보다 케이던스·진폭·좌우 상관을
    넓게 흔들어 "다양한 상황에서 목표각 도달을 돕는다"는 LLP 설계 목표에 맞고, mocap 궤적의 성능은
    일반화 시험으로 평가 단계에서 본다. chunk 오차는 실 HLP 통계(백색 0.004 + 바이어스 0.005 +
    drift 0.45 rad/s; 반정규분포 σ = MAE/0.798, 측정 MAE 0.59/1.07/2.08/3.10° @25/50/100/150 ms)에 맞춘다.
    """
    from protomotions.envs.control.assist_motion_control import (
        MotionAssistTargetControlConfig,
    )

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


def _env_randomized_plant(cfg: AssistEnvConfig, args: argparse.Namespace) -> None:
    """v4 (2026-09-22, L1): 의도·관측 계약·보상 뼈대는 v3 그대로, 플랜트와 외란의 분포를 넓힌다.

    | 축 | v3 | v4 |
    |----|----|----|
    | 하중 | 없음 | 입각 사건(외부 모멘트 0~2 N·m/kg + 착용자 굳힘 Kp×2~8) + 작은 섭동, 시점·부호 무작위 |
    | 관성 | 0.42 고정 | 질량중심 관성 ×1~25 → hip 축 0.42~3.6 kg·m² (스켈레톤 다리 3.5) |
    | 착용자 임피던스 | 게인 ±30 % | ω_n 12~24 rad/s, ζ 0.3~0.8 |
    | θ_g | sine + hold, 블렌드 0.4~1.2 s | 블렌드 0.25~2.5 s, hold 30 % |
    | 히스토리 | 100 ms / 이전 action 50 ms | 250 ms / 100 ms (관성 10배 범위의 암묵적 식별용) |
    | 비용 | \\|τ·ω\\| | \\|τ·ω\\| + τ² |
    HLP chunk 형식은 배포 HLP flexion100에 맞춘다 (10 ms × 6점 = 0~50 ms, 관측 창 {0, 10, 20 ms}).
    """
    from protomotions.envs.component_factories import action_smoothness_factory, previous_actions_factory
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.control.assist_disturbance_control import (
        DisturbedAssistTargetControlConfig,
    )
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.rewards.assist import compute_human_power_rew
    from protomotions.envs.action import make_torque_action_config

    v3 = cfg.control_components["assist"]

    # --- 의도·chunk는 v3 그대로, 외란 컴포넌트로 교체 + θ_g 다양화 ---
    assert (args.chunk_knots - 1) * args.hlp_dt >= (args.obs_future_samples - 1) * args.obs_future_dt, \
        "관측 미래 창이 chunk 예측 범위를 넘으면 ZOH 허구값이 된다"
    cfg.control_components["assist"] = DisturbedAssistTargetControlConfig(
        hlp_dt=float(args.hlp_dt),
        chunk_knots=int(args.chunk_knots),
        chunk_refresh_steps=int(args.chunk_refresh_steps),
        chunk_noise_std=v3.chunk_noise_std,
        chunk_bias_std=v3.chunk_bias_std,
        chunk_drift_std=v3.chunk_drift_std,
        chunk_delay_max=v3.chunk_delay_max,
        obs_future_samples=int(args.obs_future_samples),
        obs_future_dt=float(args.obs_future_dt),
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
        stance_prob=float(args.stance_prob),
        reversal_brake_prob=float(args.reversal_brake_prob),
        reversal_brake_c_range=tuple(args.reversal_brake_c),
        reversal_brake_lead=float(args.reversal_brake_lead),
        reversal_brake_mode=str(args.reversal_mode),
        reversal_ff_gain_range=tuple(args.reversal_ff_gain),
        chunk_anchor=str(args.chunk_anchor),
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
    cfg.reward_components["assist_effort"].static_params["weight"] = float(args.assist_effort_weight)
    cfg.reward_components["action_rate"] = action_smoothness_factory(weight=float(args.action_rate_weight))

    # --- 토크 한계 ---
    scale = args.assist_limit if args.action_torque_scale is None else args.action_torque_scale
    if not 0 < float(scale) < float("inf") or not 0 < float(args.assist_limit) < float("inf"):
        raise ValueError("Action scale and assist limit must be finite and positive")
    cfg.action_config = make_torque_action_config(
        torque_scale=float(scale),
        action_transform=None if args.action_transform == "none" else args.action_transform,
    )
    cfg.assist_torque_limit = float(args.assist_limit)
    cfg.clamp_combined_injection = not args.separate_torque_limits
    cfg.mass_scale_exponent = float(args.mass_exponent)
    if args.assist_limit_range is not None:
        cfg.assist_limit_range = tuple(args.assist_limit_range)


def _env_corrected_torque_path(cfg: AssistEnvConfig) -> None:
    """R0 (2026-10-20): 착용자 PD+FF를 구동기 하나로 한 번만 제한하고 보조·하중은 독립으로 주입한다.

    L1(v4)은 보조·착용자 FF·외란의 합에 착용자 한계를 다시 적용했고, 그것이 L1 포화 출력의 주원인이었다.
    같은 ±20 N·m에서 R0는 L1 이득의 94 %를 유지하며 저부담·최악 위상 손해를 절반 이하로 줄였다.
    """
    cfg.clamp_combined_injection = False
    cfg.human_feedforward_inside_pd = True


def _env_motion_intent_per_leg(cfg: AssistEnvConfig, robot_cfg: RobotConfig, args: argparse.Namespace) -> None:
    """motion_leg (2026-10-21): 실제 동작 의도 집단 + 다리 하나만 보조·보상·종료.

    env를 세 집단에 고정 배정한다 (`INTENT_POOLS`): 합성 20 % (R0 관성 범위), 가벼운 다리 40 %
    (관성 배율 1–3.9, 달리기 60·조깅 30·점프 10 %), 일반 40 % (걷기·조깅·점프·앉기류·기타).
    R0 관성 분포에서는 달리기를 따라갈 다리가 2 % 남짓이라 가벼운 다리 집단을 따로 둔다.
    `--leg`의 관절에만 보조를 넣고 보상·종료도 그 관절만 센다 (`joint_mask`).
    """
    from protomotions.envs.control.assist_disturbance_control import (
        DisturbedAssistTargetControlConfig,
    )

    joint, mask = leg_joint_mask(args.leg, robot_cfg.number_of_actions)
    seed = int(getattr(args, "seed", 0) or 0)

    old = cfg.control_components["assist"]
    values = {f.name: getattr(old, f.name) for f in fields(old) if f.name != "_target_"}
    values.update(
        intent_source="motion_data",
        motion_zarr_path=str(args.motion_zarr),
        motion_split="train",
        motion_metadata_parquet=str(Path(args.motion_metadata).resolve()),
        motion_seed=seed,
        intent_pools=INTENT_POOLS,
        motion_groups=MOTION_GROUPS,
        motion_accel_frac=MOTION_ACCEL_FRAC,
        motion_accel_check_dofs=(joint,),
        intent_pool_seed=seed,
    )
    cfg.control_components["assist"] = DisturbedAssistTargetControlConfig(**values)

    cfg.assist_active_joints = (joint,)
    for name in ("tracking", "human_power", "human_torque_sq", "assist_effort"):
        cfg.reward_components[name].static_params["joint_mask"] = mask
    for name in ("tracking_failure", "dof_vel_failure"):
        cfg.termination_components[name].static_params["joint_mask"] = mask


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> AssistEnvConfig:
    recipe = resolve_recipe(args)
    cfg = _env_base(robot_cfg)
    _env_hlp_chunk_format(cfg)
    _env_randomized_plant(cfg, args)
    if recipe.corrected_torque_path:
        _env_corrected_torque_path(cfg)
    if recipe.motion_intent_per_leg:
        _env_motion_intent_per_leg(cfg, robot_cfg, args)
    return cfg


# ---------------------------------------------------------------------------
# agent config
# ---------------------------------------------------------------------------
def _agent_base(robot_config: RobotConfig, args: argparse.Namespace) -> PPOAgentConfig:
    """PPO: actor는 실기기에서 측정 가능한 관측만, critic은 + privileged."""
    from protomotions.agents.common.config import MLPWithConcatConfig, MLPLayerConfig
    from protomotions.agents.ppo.config import (
        PPOActorConfig,
        PPOModelConfig,
        AdvantageNormalizationConfig,
    )
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.evaluators.config import EvaluatorConfig
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.rewards.assist import (
        compute_tracking_error_eval,
        compute_human_power_eval,
    )

    actor_in_keys = ["assist_proprio", "assist_target", "previous_actions"]
    critic_in_keys = actor_in_keys + ["assist_privileged"]

    actor_config = PPOActorConfig(
        num_out=robot_config.number_of_actions,
        actor_logstd=-2.0,
        in_keys=actor_in_keys,
        mu_key="actor_trunk_out",
        mu_model=MLPWithConcatConfig(
            in_keys=actor_in_keys,
            normalize_obs=True,
            norm_clamp_value=5,
            out_keys=["actor_trunk_out"],
            num_out=robot_config.number_of_actions,
            layers=[
                MLPLayerConfig(units=512, activation="relu"),
                MLPLayerConfig(units=256, activation="relu"),
                MLPLayerConfig(units=128, activation="relu"),
            ],
        ),
    )

    critic_config = MLPWithConcatConfig(
        in_keys=critic_in_keys,
        out_keys=["value"],
        normalize_obs=True,
        norm_clamp_value=5,
        num_out=1,
        layers=[
            MLPLayerConfig(units=512, activation="relu"),
            MLPLayerConfig(units=256, activation="relu"),
            MLPLayerConfig(units=128, activation="relu"),
        ],
    )

    evaluation_components = {
        "tracking_error": MdpComponent(
            compute_func=compute_tracking_error_eval,
            dynamic_vars={
                "dof_pos": EnvContext.current.dof_pos,
                "theta_g": EnvContext.assist.theta_g,
            },
            static_params={"threshold": 0.3, "fail_above": True},
        ),
        "human_power": MdpComponent(
            compute_func=compute_human_power_eval,
            dynamic_vars={
                "tau_agent": EnvContext.assist.tau_agent,
                "dof_vel": EnvContext.current.dof_vel,
            },
            static_params={"threshold": 1.0e9},
        ),
    }

    return PPOAgentConfig(
        model=PPOModelConfig(
            in_keys=critic_in_keys,
            out_keys=["action", "mean_action", "neglogp", "value"],
            actor=actor_config,
            critic=critic_config,
            actor_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4),
            critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=5e-4),
        ),
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        clip_critic_loss=True,
        advantage_normalization=AdvantageNormalizationConfig(
            enabled=True, shift_mean=True, use_ema=True
        ),
        evaluator=EvaluatorConfig(
            evaluation_components=evaluation_components,
            max_eval_steps=1000,
        ),
    )


def _agent_fixed_final_epoch(cfg: PPOAgentConfig) -> None:
    """R0 이후: 마지막 epoch 고정 — 주기 평가·best 체크포인트 선택 없음, 500 epoch마다 보관."""
    cfg.save_epoch_checkpoint_every = 500
    cfg.save_last_checkpoint_every = 50
    cfg.evaluator.eval_metrics_every = None


def _agent_single_joint_actor(cfg: PPOAgentConfig, robot_config: RobotConfig, args: argparse.Namespace) -> None:
    """motion_leg: actor는 학습 관절의 관측만 읽는다 (다른 관절 출력은 상수 0)."""
    from protomotions.agents.common.shared_joint_mlp import SingleJointMLPConfig

    joint, _ = leg_joint_mask(args.leg, robot_config.number_of_actions)
    old = cfg.model.actor.mu_model
    values = {f.name: getattr(old, f.name) for f in fields(old) if f.name != "_target_"}
    cfg.model.actor.mu_model = SingleJointMLPConfig(**values, joint_index=joint)


def agent_config(
    robot_config: RobotConfig, env_config: AssistEnvConfig, args: argparse.Namespace
) -> PPOAgentConfig:
    recipe = resolve_recipe(args)
    cfg = _agent_base(robot_config, args)
    if recipe.fixed_final_epoch:
        _agent_fixed_final_epoch(cfg)
    if recipe.motion_intent_per_leg:
        _agent_single_joint_actor(cfg, robot_config, args)
    if getattr(args, "policy", "mlp_ppo") != "mlp_ppo":
        # FPO++ (2026-11-03): 진자 PPO 설정(critic·정규화·500 epoch 보관)은 그대로, actor만 flow 정책 + CFM 비율 손실.
        # 토큰 시간축(히스토리 25·이전 action 10·예측 창 3)은 env 관측 구성에서 읽는다.
        if recipe.motion_intent_per_leg:
            raise ValueError("--policy mlp_fpo/tfm_fpo cannot be combined with the motion_leg preset (SingleJointMLP)")
        cfg = fpo_agent_config(cfg, args, env_cfg=env_config, actor_cfg=cfg.model.actor, actor_lr=cfg.model.actor_optimizer.lr)
    return cfg


def apply_inference_overrides(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    env_cfg,
    agent_cfg,
    terrain_cfg,
    motion_lib_cfg,
    scene_lib_cfg,
    args: argparse.Namespace,
):
    """평가 시 에피소드를 길게 (종료 조건은 유지 — 발산 감지용)."""
    if env_cfg is not None:
        env_cfg.max_episode_length = 3000
