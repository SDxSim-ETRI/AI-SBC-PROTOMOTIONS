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
"""LLP 재학습 — 보행 착용자(동결 mimic tracker) 위의 스켈레톤 hip assist (Newton).

진자 v3(`assist_pendulum/mlp_v3.py`)와 관측 계약·보상 구조를 유지하고 플랜트와
착용자만 바꾼다:

| 항목 | 진자 v3 | 이 실험 |
|------|---------|---------|
| 플랜트 | 2-DOF 진자 (I=0.43 kg·m²) | suit skeleton 27 DOF, 평지 접촉 |
| 착용자 | PD(v2) + 합성 sine 의도 | 동결 mimic tracker (20 Hz) + mocap |
| 의도 θ_g | 합성 궤적 | mocap 레퍼런스 hip flexion (평가·critic용) |
| HLP chunk | 에뮬레이션 | **실 ManiFlow** (배치 예측, 20 ms마다) |
| LLP 주기 | 10 ms | 10 ms (물리 400 Hz / decimation 4) |
| 관측 | hip 2ch 상태·히스토리 10, 미래 창 4×50 ms, 이전 action 5 | 동일 |
| 보상 | hip 목표 추종(exp) + 파워 + effort + action rate | **착용자 몸체 오차 hinge 벌점**(무보조 수준 초과분만) + 파워 + effort + action rate (`--reward-variant tracker` = v1의 gt/rh 추종 보상) |
| 종료 | 추종 실패 / 과속 | mimic 몸체 오차 0.5 m(낙상 포함) / 과속 |

hip 목표 추종 항을 착용자 몸체 추종으로 바꾼 이유: 보행 착용자의 "의도"를 hip 각도
하나로 정의할 수 없다 — tracker의 PD 목표는 각도와 15~20° 차이 나는 토크 명령이고,
mocap hip 각도는 tracker 자체가 10~14° 벗어나 따라간다. 대신 "보조가 착용자의 보행
과제를 깨뜨리지 않는다"를 tracker 자신의 몸체 추종 보상으로 직접 잰다.

착용자 tracker의 관측(`max_coords_obs`, `mimic_target_poses`)·종료·motion manager
설정은 tracker 체크포인트의 resolved config에서 그대로 가져온다 (같은 객체).

실행 (sbc env):
    python protomotions/train_agent.py --robot-name skeleton_torque_suit_passive_cable \\
        --simulator newton --experiment-path examples/experiments/assist_skeleton/llp_walk.py \\
        --experiment-name llp_skel_walk \\
        --motion-file data/motion_for_trackers/skeleton_torque_suit_walk.pt \\
        --num-envs 4096 --batch-size 16384 \\
        --tracker-checkpoint checkpoints/v30_newton_suit_passive_walk/last.ckpt
평가: examples/experiments/assist_skeleton/run_system.py --human tracker --llp-checkpoint results/llp_skel_walk/last.ckpt
"""

import argparse
from pathlib import Path

import torch

from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.components.motion_lib import MotionLibConfig
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.terrains.config import TerrainConfig
from protomotions.envs.base_env.skeleton_assist_env import SkeletonAssistEnvConfig
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig

ASSIST_TORQUE_LIMIT = 25.0  # N·m 기본값 — 진자 v3와 동일 (착용 장치 현실 범위); --assist-limit로 변경
DEFAULT_TRACKER = "checkpoints/v30_newton_suit_passive_walk/last.ckpt"


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--tracker-checkpoint", type=str, default=DEFAULT_TRACKER,
                        help="착용자 대역 mimic tracker last.ckpt (쉼표로 여러 개 = 착용자 앙상블, env별 무작위 배정)")
    parser.add_argument("--hlp-checkpoint", type=str, default=None,
                        help="ManiFlow flexion ckpt (None = flexion40 run 자동 탐색)")
    parser.add_argument("--hlp-denoise", type=int, default=1,
                        help="HLP denoise 스텝 (실측: 정확도는 스텝 수와 무관, 지연만 스텝당 3 ms)")
    parser.add_argument("--obs-future-samples", type=int, default=4,
                        help="LLP가 보는 미래 창 샘플 수. HLP 예측 범위 안에 들어가야 한다 "
                             "(150 ms 모델: 4×50 ms, 20 ms 모델: 2×10 ms)")
    parser.add_argument("--obs-future-dt", type=float, default=0.05,
                        help="미래 창 샘플 간격 [s]")
    parser.add_argument("--chunk-refresh-steps", type=int, default=1,
                        help="HLP 재예측 주기 (LLP step 단위; 10 ms step에서 1 = 10 ms). "
                             "denoise 1에서 실측 4.9 ms라 10 ms 예산에 들어간다")
    parser.add_argument("--assist-limit", type=float, default=ASSIST_TORQUE_LIMIT,
                        help="assist 토크 한계 [N·m] (tanh 스케일 = clamp)")
    parser.add_argument("--human-power-weight", type=float, default=-0.003)
    parser.add_argument("--reward-variant", choices=["hinge", "tracker"], default="hinge",
                        help="hinge: 몸체 오차가 margin 초과분만 벌점(권장) / tracker: tracker의 gt·rh 추종 보상(v1)")
    parser.add_argument("--body-err-margin", type=float, default=0.09, help="hinge 시작 몸체 오차 [m]")
    parser.add_argument("--body-err-weight", type=float, default=10.0, help="hinge 벌점 weight (0.05 m 초과 → −0.5/step)")
    parser.add_argument("--assist-effort-weight", type=float, default=-5.0e-5)
    parser.add_argument("--hlp-calibration", type=str, default="motion",
                        help="HLP 영점: motion(재생 클립 평균) | none | 'p,q,r,s'(deg, 착용자 스트림 평균으로 잰 값). "
                             "착용자 자세는 mocap과 10° 가까이 다를 수 있다 — run_system.py --calibration wearer가 측정값을 찍어 준다")
    parser.add_argument("--action-rate-weight", type=float, default=-0.005,
                        help="연속 step 사이 토크 변화 벌점. ×10(−0.05)은 과했다 — 정책이 거의 일정한 "
                             "큰 토크로 수렴해 오히려 착용자를 방해했다(v7). 부드러움은 --assist-lpf-hz 쪽을 쓸 것")
    parser.add_argument("--assist-lpf-hz", type=float, default=0.0,
                        help="주입 전 1차 저역통과 [Hz] (0 = 끔). 구동기 대역을 플랜트에 넣는 방식 — "
                             "정책이 그 한계 안에서 위상을 맞추는 법을 배운다. 실행 시 "
                             "run_system.py --torque-lpf-hz 같은 값으로")


def _tracker_resolved(args: argparse.Namespace):
    """tracker의 **학습** resolved config (inference 판은 종료 조건이 비어 있다)."""
    ckpt_dir = Path(args.tracker_checkpoint.split(",")[0].strip()).parent  # 앙상블이면 첫 tracker 기준
    path = ckpt_dir / "resolved_configs.pt"
    if not path.exists():
        path = ckpt_dir / "resolved_configs_inference.pt"
    assert path.exists(), f"tracker resolved configs not found in {ckpt_dir}"
    return torch.load(path, map_location="cpu", weights_only=False)


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    """tracker와 같은 접촉 센서(발) + LLP 주기: 물리 400 Hz / decimation 4 (10 ms)."""
    robot_cfg.update_fields(contact_bodies=["all_left_foot_bodies", "all_right_foot_bodies"])
    simulator_cfg.sim.fps = 400
    simulator_cfg.sim.decimation = 4


def terrain_config(args: argparse.Namespace):
    return _tracker_resolved(args)["terrain"] or TerrainConfig()


def scene_lib_config(args: argparse.Namespace):
    return SceneLibConfig(scene_file=None)


def motion_lib_config(args: argparse.Namespace):
    """tracker의 motion_lib 설정에 이번 모션 파일만 바꿔 넣는다."""
    cfg: MotionLibConfig = _tracker_resolved(args)["motion_lib"]
    cfg.motion_file = args.motion_file
    return cfg


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> SkeletonAssistEnvConfig:
    from protomotions.envs.component_factories import (
        action_smoothness_factory,
        previous_actions_factory,
    )
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.control.skeleton_assist_control import (
        SkeletonAssistControlConfig,
        TrackerMimicControlConfig,
    )
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.obs.assist_obs import (
        compute_assist_privileged_obs,
        compute_assist_proprio_obs,
        compute_assist_target_obs,
    )
    from protomotions.envs.rewards.assist import (
        compute_assist_effort_rew,
        compute_body_error_hinge_rew,
        compute_human_power_rew,
    )
    from protomotions.envs.terminations.assist import compute_dof_vel_failure_term
    from protomotions.envs.action import make_torque_action_config

    tracker_env = _tracker_resolved(args)["env"]
    tracker_mimic = tracker_env.control_components["mimic"]

    control_components = {
        # 착용자 레퍼런스: tracker 학습 주기(50 ms) 간격의 미래 프레임
        "mimic": TrackerMimicControlConfig(
            bootstrap_on_episode_end=tracker_mimic.bootstrap_on_episode_end,
            future_steps=tracker_mimic.future_steps,
            future_dt=0.05,
        ),
        "assist": SkeletonAssistControlConfig(
            tracker_checkpoint=str(args.tracker_checkpoint),
            tracker_period_steps=5,
            maniflow_ckpt=args.hlp_checkpoint,
            maniflow_denoise_steps=args.hlp_denoise,
            hlp_calibration=str(args.hlp_calibration),
            motion_file_for_calibration=args.motion_file,
            obs_future_samples=int(args.obs_future_samples),
            obs_future_dt=float(args.obs_future_dt),
            chunk_refresh_steps=int(args.chunk_refresh_steps),
        ),
    }

    observation_components = {
        # --- 착용자 tracker 입력 (tracker config 객체 그대로; 정책은 5 step마다 소비) ---
        "max_coords_obs": tracker_env.observation_components["max_coords_obs"],
        "mimic_target_poses": tracker_env.observation_components["mimic_target_poses"],
        # --- LLP: hip 2ch 상태 + 히스토리 (진자와 같은 커널, 부분집합 컨텍스트) ---
        "assist_proprio": MdpComponent(
            compute_func=compute_assist_proprio_obs,
            dynamic_vars={
                "dof_pos": EnvContext.assist.dof_pos,
                "dof_vel": EnvContext.assist.dof_vel,
                "historical_dof_pos": EnvContext.assist.historical_dof_pos,
                "historical_dof_vel": EnvContext.assist.historical_dof_vel,
            },
            static_params={"dof_vel_scale": 0.25},
        ),
        "assist_target": MdpComponent(
            compute_func=compute_assist_target_obs,
            dynamic_vars={
                "theta_d_future": EnvContext.assist.theta_d_future,
                "dof_pos": EnvContext.assist.dof_pos,
                "chunk_age": EnvContext.assist.chunk_age,
            },
        ),
        "previous_actions": previous_actions_factory(history_steps=5),
        # critic 전용
        "assist_privileged": MdpComponent(
            compute_func=compute_assist_privileged_obs,
            dynamic_vars={
                "tau_agent": EnvContext.assist.tau_agent,
                "tau_assist": EnvContext.assist.tau_assist,
                "theta_g": EnvContext.assist.theta_g,
                "dof_pos": EnvContext.assist.dof_pos,
                "gain_scale": EnvContext.assist.gain_scale,
            },
            static_params={"torque_scale": 0.01},
        ),
    }

    # 보행 hip 파워는 진자(~17 W)의 10배 규모라 weight를 1/10 안팎으로 (항 크기 균형 유지).
    # 착용자 보행 유지:
    #   hinge  — 몸체 오차가 무보조 수준(margin)을 넘는 만큼만 벌점. v1(tracker의 exp 추종 보상)은
    #            정책이 최대 토크로 착용자를 mocap 쪽으로 끌어 "추종 개선"만 얻고 파워는 못 줄였다.
    #   tracker — tracker 자신의 gt/rh 추종 보상 (v1 재현용)
    reward_components = {
        "human_power": MdpComponent(
            compute_func=compute_human_power_rew,
            dynamic_vars={
                "tau_agent": EnvContext.assist.tau_agent,
                "dof_vel": EnvContext.assist.dof_vel,
            },
            static_params={"weight": float(args.human_power_weight)},
        ),
        "assist_effort": MdpComponent(
            compute_func=compute_assist_effort_rew,
            dynamic_vars={"tau_assist": EnvContext.assist.tau_assist},
            static_params={"weight": float(args.assist_effort_weight)},
        ),
        "action_rate": action_smoothness_factory(weight=float(args.action_rate_weight)),
    }
    if args.reward_variant == "hinge":
        reward_components["human_body_hinge"] = MdpComponent(
            compute_func=compute_body_error_hinge_rew,
            dynamic_vars={
                "current_rigid_body_pos": EnvContext.current.rigid_body_pos,
                "ref_rigid_body_pos": EnvContext.mimic.ref_state.rigid_body_pos,
            },
            static_params={"weight": float(args.body_err_weight), "margin": float(args.body_err_margin)},
        )
    else:
        reward_components["human_gt"] = tracker_env.reward_components["gt_rew"]
        reward_components["human_rh"] = tracker_env.reward_components["rh_rew"]

    termination_components = {
        # 착용자 몸체가 레퍼런스에서 0.5 m 이상 벗어나면 (낙상 포함) — tracker 설정 그대로
        "tracking_error": tracker_env.termination_components["tracking_error"],
        "dof_vel_failure": MdpComponent(
            compute_func=compute_dof_vel_failure_term,
            dynamic_vars={"dof_vel": EnvContext.assist.dof_vel},
            static_params={"max_vel": 25.0},
        ),
    }

    return SkeletonAssistEnvConfig(
        max_episode_length=1000,  # 10 s @ 100 Hz
        reset_grace_period=5,
        num_state_history_steps=10,
        motion_manager=tracker_env.motion_manager,
        ref_respawn_offset=tracker_env.ref_respawn_offset,
        ref_contact_smooth_window=tracker_env.ref_contact_smooth_window,
        skip_correct_terrain_height_on_flat=tracker_env.skip_correct_terrain_height_on_flat,
        control_components=control_components,
        observation_components=observation_components,
        reward_components=reward_components,
        termination_components=termination_components,
        action_config=make_torque_action_config(torque_scale=float(args.assist_limit), action_transform="tanh"),
        assist_torque_limit=float(args.assist_limit),
        assist_torque_lpf_hz=float(args.assist_lpf_hz),
    )


def agent_config(
    robot_config: RobotConfig, env_config: SkeletonAssistEnvConfig, args: argparse.Namespace
) -> PPOAgentConfig:
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.common.config import MLPLayerConfig, MLPWithConcatConfig
    from protomotions.agents.evaluators.config import EvaluatorConfig
    from protomotions.agents.ppo.config import (
        AdvantageNormalizationConfig,
        PPOActorConfig,
        PPOModelConfig,
    )
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.rewards.assist import compute_human_power_eval, compute_tracking_error_eval

    num_assist = len(env_config.control_components["assist"].assist_dof_names)
    actor_in_keys = ["assist_proprio", "assist_target", "previous_actions"]
    critic_in_keys = actor_in_keys + ["assist_privileged"]
    layers = [
        MLPLayerConfig(units=512, activation="relu"),
        MLPLayerConfig(units=256, activation="relu"),
        MLPLayerConfig(units=128, activation="relu"),
    ]
    actor_config = PPOActorConfig(
        num_out=num_assist,
        actor_logstd=-2.0,
        in_keys=actor_in_keys,
        mu_key="actor_trunk_out",
        mu_model=MLPWithConcatConfig(
            in_keys=actor_in_keys, normalize_obs=True, norm_clamp_value=5,
            out_keys=["actor_trunk_out"], num_out=num_assist, layers=layers,
        ),
    )
    critic_config = MLPWithConcatConfig(
        in_keys=critic_in_keys, out_keys=["value"], normalize_obs=True, norm_clamp_value=5,
        num_out=1, layers=list(layers),
    )
    evaluation_components = {
        "tracking_error": MdpComponent(
            compute_func=compute_tracking_error_eval,
            dynamic_vars={"dof_pos": EnvContext.assist.dof_pos, "theta_g": EnvContext.assist.theta_g},
            static_params={"threshold": 0.5, "fail_above": True},
        ),
        "human_power": MdpComponent(
            compute_func=compute_human_power_eval,
            dynamic_vars={"tau_agent": EnvContext.assist.tau_agent, "dof_vel": EnvContext.assist.dof_vel},
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
        advantage_normalization=AdvantageNormalizationConfig(enabled=True, shift_mean=True, use_ema=True),
        evaluator=EvaluatorConfig(evaluation_components=evaluation_components, max_eval_steps=1000),
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
    """평가 시 에피소드를 길게 (종료 조건은 유지 — 낙상·발산 감지용)."""
    if env_cfg is not None:
        env_cfg.max_episode_length = 3000
