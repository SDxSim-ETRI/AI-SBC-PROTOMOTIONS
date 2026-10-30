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

진자 v3(`assist_pendulum/mlp.py`의 `_env_hlp_chunk_format` 단계)와 관측 계약·보상 구조를 유지하고 플랜트와
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

진자 LLP actor 이어받기 추가학습 (`--initial-actor-checkpoint`, 2026-10-11 L13 방식을 옵션으로 통합):
    출발 checkpoint의 관측 구성·action 변환·토크 한계를 그대로 쓰고(actor 가중치만 이어받음, critic·optimizer는 새로),
    무보조 착용자를 같은 난수로 나란히 돌려 보상을 계산하는 `PairedSkeletonAssistEnv`를 쓴다.
    에피소드 8 s, 시작 시각은 클립의 0–`--segment-start-max` s. `--exclude-motion-ids`로 걷기(0)·달리기(1)를 고른다.
    둘을 한 모델로 같이 학습하려면 제외 없이 `--motion-weights 0.5,0.5`와 motion id별 영점
    `--hlp-calibration "0:p,q,r,s;1:p,q,r,s"`를 준다 (2026-10-29).
    python protomotions/train_agent.py --robot-name skeleton_torque_suit_human --simulator newton \\
        --experiment-path examples/experiments/assist_skeleton/llp_walk.py --experiment-name llp_C0ft_walk \\
        --motion-file data/motion_for_trackers/skeleton_torque_suit_gait.pt --num-envs 256 --batch-size 8192 \\
        --training-max-steps 1638400 --seed 13 --tracker-checkpoint checkpoints/W1_wearer_human/last.ckpt \\
        --hlp-checkpoint <H1> --decimation 4 --human-cost torque_sq --hlp-calibration "p,q,r,s" \\
        --initial-actor-checkpoint results/llp_relative_R0_s0_20261020/epoch_6000.ckpt --exclude-motion-ids 1 \\
        --finetune-assist-limit 20
"""

import argparse
import copy
from dataclasses import fields
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
    parser.add_argument("--decimation", type=int, default=2,
                        help="물리 400 Hz 기준 LLP 주기 = decimation × 2.5 ms (2 = 5 ms, 4 = 10 ms). "
                             "히스토리·이전 action·tracker 주기·HLP 갱신 주기는 시간 길이가 유지되도록 함께 맞춰진다")
    parser.add_argument("--obs-future-samples", type=int, default=3,
                        help="LLP가 보는 미래 창 샘플 수 (offset 0부터 obs_future_dt 간격). "
                             "기본 3 = {0, 10, 20 ms}: 0은 예측과 현재 측정의 차이를 보는 기준점이고 "
                             "실제 선행 정보는 10·20 ms 두 점. HLP 예측 범위(50 ms) 안이어야 한다")
    parser.add_argument("--obs-future-dt", type=float, default=0.01,
                        help="미래 창 샘플 간격 [s]")
    parser.add_argument("--chunk-refresh-steps", type=int, default=0,
                        help="HLP 재예측 주기 (LLP step 단위). 0이면 decimation에서 10 ms가 되도록 자동. "
                             "denoise 1에서 실측 4.9 ms라 10 ms 예산에 들어간다")
    parser.add_argument("--assist-limit", type=float, default=ASSIST_TORQUE_LIMIT,
                        help="assist 토크 한계 [N·m] (tanh 스케일 = clamp)")
    parser.add_argument("--human-power-weight", type=float, default=None,
                        help="착용자 비용 항 weight (None = --human-cost에 맞춘 기본값). "
                             "두 지표는 크기가 60배 달라서 같이 쓰면 안 된다")
    parser.add_argument("--human-cost", choices=["work", "torque_sq", "both"], default="work",
                        help="착용자 비용을 무엇으로 재는가. "
                             "work: |τ·ω| 기계적 파워 — **착용자 다리를 덜 움직이게 해도 내려간다** "
                             "(v10이 그 해로 수렴: 착지 구간 |ω| −42 %, 근토크는 오히려 +2.9 %). "
                             "torque_sq: τ² 근활성 프록시 — 속도로는 못 내리고 토크를 실제로 덜어야 내려간다. "
                             "both: 둘을 반씩 — τ²가 속도 꼼수를 막고, |τ·ω|가 브레이킹 일 절감을 계속 보상한다 "
                             "(β 오라클은 두 지표 모두 좋았다: P −20.3 %, τ² −32.3 %)")
    parser.add_argument("--reward-variant", choices=["hinge", "tracker"], default="hinge",
                        help="hinge: 몸체 오차가 margin 초과분만 벌점(권장) / tracker: tracker의 gt·rh 추종 보상(v1)")
    parser.add_argument("--body-err-margin", type=float, default=0.09, help="hinge 시작 몸체 오차 [m]")
    parser.add_argument("--body-err-weight", type=float, default=10.0, help="hinge 벌점 weight (0.05 m 초과 → −0.5/step)")
    parser.add_argument("--assist-effort-weight", type=float, default=None,
                        help="assist Στ² 벌점 weight (None = 토크 한계에 맞춰 자동 환산 — "
                             "한계를 올려도 이 항이 몰래 한계로 작동하지 않게)")
    parser.add_argument("--hlp-calibration", type=str, default="motion",
                        help="HLP 영점: motion(재생 클립 평균) | none | 'p,q,r,s'(deg, 착용자 스트림 평균으로 잰 값) | "
                             "'0:p,q,r,s;1:p,q,r,s'(motion id별 — 걷기·달리기 동시 학습). "
                             "착용자 자세는 mocap과 10° 가까이 다를 수 있다 — run_system.py --calibration wearer가 측정값을 찍어 준다")
    parser.add_argument("--action-rate-weight", type=float, default=-0.005,
                        help="연속 step 사이 토크 변화 벌점. ×10(−0.05)은 과했다 — 정책이 거의 일정한 "
                             "큰 토크로 수렴해 오히려 착용자를 방해했다(v7). 부드러움은 --assist-lpf-hz 쪽을 쓸 것")
    parser.add_argument("--assist-lpf-hz", type=float, default=0.0,
                        help="주입 전 1차 저역통과 [Hz] (0 = 끔). 구동기 대역을 플랜트에 넣는 방식 — "
                             "정책이 그 한계 안에서 위상을 맞추는 법을 배운다. 실행 시 "
                             "run_system.py --torque-lpf-hz 같은 값으로")
    # --- 진자 LLP actor 이어받기 추가학습 (L13 방식) ---
    parser.add_argument("--initial-actor-checkpoint", type=str, default=None,
                        help="진자 LLP checkpoint. 주면 actor 가중치만 이어받고(critic·optimizer 새로), 관측 구성·action 변환·"
                             "토크 한계는 그 checkpoint의 설정을 쓰며, 무보조 착용자를 나란히 돌리는 PairedSkeletonAssistEnv로 학습")
    parser.add_argument("--exclude-motion-ids", type=str, default=None,
                        help="추가학습에서 뺄 motion id (쉼표). gait.pt: 0=걷기, 1=달리기. 예: 걷기만 = 1")
    parser.add_argument("--motion-weights", type=str, default=None,
                        help="추가학습 motion 표본 비율 (쉼표, motion id 순). 없으면 motion 파일 가중치(gait.pt 걷기 0.75·달리기 0.25)")
    parser.add_argument("--segment-start-max", type=float, default=20.0,
                        help="에피소드 시작 시각 상한 [s] (클립 앞부분만 학습에 쓰고 뒷부분은 평가용으로 남김)")
    parser.add_argument("--local-excess-weight", type=float, default=1.0,
                        help="PairedSkeletonAssistEnv의 관절별 부담 증가 벌점 가중치 (L13 1.0, L13R 0.149 + floor 10)")
    parser.add_argument("--local-excess-floor", type=float, default=0.0,
                        help="위 벌점의 분모 바닥 [N·m] (0 = 고정 scale로 나눔)")
    parser.add_argument("--finetune-actor-lr", type=float, default=3e-5, help="이어받은 actor의 학습률 (L13 3e-5)")
    parser.add_argument("--finetune-assist-limit", type=float, default=None,
                        help="추가학습 중 보조 토크 출력 배율·한계 [N·m] (None = 출발 checkpoint 값). 40·tanh로 학습한 R0에 20을 주면 "
                             "평가의 R0 ×0.5(20·tanh)와 같은 출력에서 시작해 20 N·m 한계로 학습한다")


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
    """tracker와 같은 접촉 센서(발) + LLP 주기: 물리 400 Hz / decimation (기본 2 = 5 ms)."""
    if getattr(args, "initial_actor_checkpoint", None):
        # L13 방식: 로봇·물리 설정을 착용자 checkpoint의 저장본으로 맞춘다.
        wearer = _tracker_resolved(args)
        robot_cfg.__dict__.update(copy.deepcopy(wearer["robot"].__dict__))
        simulator_cfg.sim = copy.deepcopy(wearer["simulator"].sim)
    robot_cfg.update_fields(contact_bodies=["all_left_foot_bodies", "all_right_foot_bodies"])
    simulator_cfg.sim.fps = 400
    simulator_cfg.sim.decimation = int(args.decimation)


def terrain_config(args: argparse.Namespace):
    return _tracker_resolved(args)["terrain"] or TerrainConfig()


def scene_lib_config(args: argparse.Namespace):
    return SceneLibConfig(scene_file=None)


def motion_lib_config(args: argparse.Namespace):
    """tracker의 motion_lib 설정에 이번 모션 파일만 바꿔 넣는다."""
    cfg: MotionLibConfig = _tracker_resolved(args)["motion_lib"]
    cfg.motion_file = args.motion_file
    return cfg


def _human_cost_components(args: argparse.Namespace):
    """착용자 비용 보상 항 — 지표마다 크기가 60배 달라 weight를 따로 잡는다.

    보행 실측 규모(무보조 A, 양다리 합): |τ·ω| ≈ 180 W, τ² ≈ 11,300 (N·m)².
    각 항이 step당 −0.5 안팎(both는 반씩 −0.27)이 되도록 기본 weight를 고른다.
    `--human-power-weight`를 주면 그 값이 **work 항**에 그대로 들어간다.
    """
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.rewards.assist import compute_human_power_rew

    both = args.human_cost == "both"
    w_work = args.human_power_weight if args.human_power_weight is not None else (-0.0015 if both else -0.003)
    w_sq = -2.5e-5 if both else -5.0e-5

    def term(weight: float, torque_squared: bool):
        return MdpComponent(
            compute_func=compute_human_power_rew,
            dynamic_vars={
                "tau_agent": EnvContext.assist.tau_agent,
                "dof_vel": EnvContext.assist.dof_vel,
            },
            static_params={"weight": float(weight), "use_torque_squared": torque_squared},
        )

    out = {}
    if args.human_cost in ("work", "both"):
        out["human_power"] = term(w_work, False)
    if args.human_cost in ("torque_sq", "both"):
        out["human_torque_sq"] = term(w_sq, True)
    return out


def _rate_steps(decimation: int):
    """LLP 주기가 바뀌어도 **시간 길이**가 같도록 step 수를 환산한다 (물리 400 Hz 기준).

    decimation 4 → LLP 10 ms: 히스토리 10(100 ms), 이전 action 5(50 ms),
    tracker 5 step(50 ms), HLP 갱신 1 step(10 ms).
    decimation 2 → LLP 5 ms: 각각 20 / 10 / 10 / 2 로 두 배.
    """
    assert 4 % decimation == 0 or decimation in (1, 2, 4), "decimation은 1·2·4 중 하나"
    return {
        "history": 40 // decimation,        # 100 ms
        "prev_actions": 20 // decimation,   # 50 ms
        "tracker_period": 20 // decimation,  # 50 ms (tracker 학습 주기)
        "chunk_refresh": max(4 // decimation, 1),  # 10 ms (HLP 갱신)
    }


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
    rate = _rate_steps(int(args.decimation))

    control_components = {
        # 착용자 레퍼런스: tracker 학습 주기(50 ms) 간격의 미래 프레임
        "mimic": TrackerMimicControlConfig(
            bootstrap_on_episode_end=tracker_mimic.bootstrap_on_episode_end,
            future_steps=tracker_mimic.future_steps,
            future_dt=0.05,
        ),
        "assist": SkeletonAssistControlConfig(
            tracker_checkpoint=str(args.tracker_checkpoint),
            tracker_period_steps=rate["tracker_period"],
            maniflow_ckpt=args.hlp_checkpoint,
            maniflow_denoise_steps=args.hlp_denoise,
            hlp_calibration=str(args.hlp_calibration),
            motion_file_for_calibration=args.motion_file,
            obs_future_samples=int(args.obs_future_samples),
            obs_future_dt=float(args.obs_future_dt),
            chunk_refresh_steps=int(args.chunk_refresh_steps or rate["chunk_refresh"]),
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
        "previous_actions": previous_actions_factory(history_steps=rate["prev_actions"]),
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
        **_human_cost_components(args),
        "assist_effort": MdpComponent(
            compute_func=compute_assist_effort_rew,
            dynamic_vars={"tau_assist": EnvContext.assist.tau_assist},
            # Στ²_assist는 한계의 제곱으로 커진다 — weight를 고정해 두면 한계를 올릴수록
            # 이 항이 몰래 한계 노릇을 한다. **포화했을 때의 벌점 크기가 같도록** 환산한다
            # (40 N·m·−5e-5 기준: 양다리 포화 시 −0.16/step).
            static_params={
                "weight": float(
                    args.assist_effort_weight
                    if args.assist_effort_weight is not None
                    else -5.0e-5 * (ASSIST_TORQUE_LIMIT / float(args.assist_limit)) ** 2
                )
            },
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

    cfg = SkeletonAssistEnvConfig(
        max_episode_length=1000,  # 10 s @ 100 Hz
        reset_grace_period=5,
        num_state_history_steps=rate["history"],
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
    if args.initial_actor_checkpoint:
        cfg = _env_finetune(cfg, args)
    return cfg


def _finetune_source(args: argparse.Namespace):
    """추가학습 출발 checkpoint(진자 LLP)의 학습 resolved config."""
    path = Path(args.initial_actor_checkpoint).parent / "resolved_configs.pt"
    assert path.exists(), f"resolved_configs.pt not found next to {args.initial_actor_checkpoint}"
    return torch.load(path, map_location="cpu", weights_only=False)


def _env_finetune(cfg: SkeletonAssistEnvConfig, args: argparse.Namespace):
    """L13 방식 env: 출발 checkpoint의 관측 구성·action 변환·토크 한계를 그대로 쓰고, 무보조 착용자를 같은 난수로
    나란히 돌려 보상을 계산한다(PairedSkeletonAssistEnv). 에피소드 8 s, 시작 시각 0–segment_start_max s."""
    from protomotions.envs.base_env.paired_skeleton_assist_env import (
        PairedSkeletonAssistEnvConfig, WalkingSegmentConfig,
    )
    from protomotions.envs.context_views import EnvContext

    source = _finetune_source(args)["env"]
    out = PairedSkeletonAssistEnvConfig(**{f.name: getattr(cfg, f.name) for f in fields(cfg) if f.name != "_target_"})
    out.num_state_history_steps = source.num_state_history_steps
    cc = out.control_components["assist"]
    cc.hlp_dt = 0.01
    cc.chunk_knots = 6
    cc.target_low_override = (-1.52, -1.52)
    cc.target_high_override = (1.52, 1.52)
    for key in ("assist_proprio", "assist_target", "previous_actions", "assist_privileged"):
        out.observation_components[key] = copy.deepcopy(source.observation_components[key])
    # 출발 모델의 관측은 진자 전체 DOF를 가리킨다 — 스켈레톤에서는 두 hip 축(assist 컨텍스트)으로 바꿔 단다.
    for key in ("assist_proprio", "assist_target", "assist_privileged"):
        obs = out.observation_components[key]
        for name in ("dof_pos", "dof_vel", "historical_dof_pos", "historical_dof_vel"):
            if name in obs.dynamic_vars:
                obs.dynamic_vars[name] = getattr(EnvContext.assist, name)
    out.action_config = copy.deepcopy(source.action_config)
    out.assist_torque_limit = source.assist_torque_limit
    if args.finetune_assist_limit is not None:
        limit = float(args.finetune_assist_limit)
        assert isinstance(out.action_config, dict) and "torque_scale" in out.action_config, "source action config has no torque_scale"
        out.action_config["torque_scale"] = limit  # 예: 40·tanh → 20·tanh (R0 ×0.5와 같은 출력)
        out.assist_torque_limit = limit
    out.reward_components = {}  # 보상은 두 착용자의 물리 step 뒤 PairedSkeletonAssistEnv가 계산
    out.motion_manager = WalkingSegmentConfig(**{f.name: copy.deepcopy(getattr(out.motion_manager, f.name))
                                                for f in fields(out.motion_manager) if f.name != "_target_"})
    if args.exclude_motion_ids:
        out.motion_manager.exclude_motion_ids = [int(v) for v in args.exclude_motion_ids.split(",")]
    if getattr(args, "motion_weights", None):
        out.motion_manager.motion_weights = tuple(float(v) for v in args.motion_weights.split(","))
    out.motion_manager.init_start_prob = 0.0
    out.motion_manager.start_max_s = float(args.segment_start_max)
    out.max_episode_length = 800
    out.local_excess_weight = float(args.local_excess_weight)
    out.local_excess_floor_Nm = float(args.local_excess_floor)
    return out


def _agent_finetune(args: argparse.Namespace) -> PPOAgentConfig:
    """출발 checkpoint의 PPO 설정에 actor 이어받기(AssistFinetunePPO)만 얹는다. 평가는 run_system A/B로 따로."""
    from protomotions.agents.ppo.assist_finetune import AssistFinetuneConfig

    source = _finetune_source(args)["agent"]
    cfg = AssistFinetuneConfig(**{f.name: copy.deepcopy(getattr(source, f.name)) for f in fields(source) if f.name != "_target_"})
    cfg.initial_actor_checkpoint = str(Path(args.initial_actor_checkpoint).resolve())
    cfg.batch_size = args.batch_size
    cfg.training_max_steps = args.training_max_steps
    cfg.model.actor_optimizer.lr = float(args.finetune_actor_lr)
    cfg.model.critic_optimizer.lr = 5e-4
    cfg.save_epoch_checkpoint_every = 50
    cfg.save_last_checkpoint_every = 10
    cfg.evaluator.eval_metrics_every = None
    return cfg


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

    if args.initial_actor_checkpoint:
        return _agent_finetune(args)
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
