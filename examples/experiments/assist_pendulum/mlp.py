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
"""AI-SBC LLP 1단계: 2-DOF hip pendulum assist torque RL (PPO).

사람(BUILT_IN_PD)이 의도 궤적 θ_g를 추종하는 동안 policy가 additive assist
torque(τ_assist)를 출력해 사람 에너지 |τ_agent·θ̇|를 최소화하는 태스크.
HLP(ManiFlow) 예측은 노이즈/지연이 주입된 저주파 chunk로 에뮬레이션된다.

학습:
    python protomotions/train_agent.py \
        --robot-name hip_pendulum \
        --simulator newton \
        --experiment-path examples/experiments/assist_pendulum/mlp.py \
        --experiment-name assist_pendulum \
        --motion-file none \
        --num-envs 4096 \
        --batch-size 16384

주의: --motion-file은 CLI상 필수지만 이 실험은 모션을 쓰지 않는다 —
아무 문자열이나 전달하면 되고 motion_lib_config가 무시한다.
어시스트 없는 베이스라인 평가: --overrides env.assist_torque_limit=0
"""

import argparse

from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.envs.base_env.assist_env import AssistEnvConfig
from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.components.terrains.config import TerrainConfig
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.motion_lib import MotionLibConfig

# 어시스트 토크 한계 [N·m] — action scale과 env clamp에 공통 사용
ASSIST_TORQUE_LIMIT = 25.0


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


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> AssistEnvConfig:
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


def agent_config(
    robot_config: RobotConfig, env_config: AssistEnvConfig, args: argparse.Namespace
) -> PPOAgentConfig:
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

    # actor: 실기기에서 측정 가능한 관측만 / critic: + privileged
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
