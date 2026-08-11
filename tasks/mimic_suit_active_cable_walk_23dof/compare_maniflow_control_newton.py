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
"""ManiFlow 토크를 실제 제어에 사용하는 A/B 비교 (Newton 전용).

같은 씬(Newton 멀티월드)에 두 agent를 겹쳐 스폰합니다:

  Agent A (env 0, 고스트) : RL policy + built-in PD — 기존과 동일한 순수 RL 보행.
  Agent B (env 1, 실체)   : ManiFlow가 예측한 토크를 estimator 채널에 직접
                            인가하고, 나머지 관절은 RL policy + built-in PD 유지.

Newton은 env마다 독립된 world를 만들므로 두 agent는 물리적으로 간섭하지 않고
완전히 겹쳐 시뮬레이션됩니다. 뷰어에서는 Agent A를 반투명 라인 스켈레톤
(고스트)으로, Agent B를 일반 메시로 그립니다.

estimator 채널: 순수 hip 6개 DOF (공통 [0,1,2,5,6,7] — 이름 기반 파생).
  채널 이름은 런타임에 robot config에서 파생합니다. 과거 first6(공통 0-5,
  수집 버그) 채널로 학습된 legacy run01 모델은 2026-07-14 삭제되어 관련
  옵션(--action-dofs)도 제거되었습니다.

동기화: 두 env는 항상 같은 모션(id 0), 같은 시작 시각(t=0), 같은 스폰 위치를
사용하고, 어느 한쪽이 done이 되면 둘 다 함께 리셋합니다(에피소드 단위 비교).
inference용 resolved config에는 termination 컴포넌트가 없는 경우가 많아,
스크립트 자체적으로 넘어짐(root 높이 유지 실패)과 A↔B 과대 발산을 감지해
에피소드를 끊습니다 (--fall-z / --fall-hold / --divergence-reset). 다만
--min-episode-seconds(기본 5s) 동안은 넘어짐/발산이 감지돼도 리셋하지 않고
쓰러진 채로 시뮬레이션을 계속합니다 — 매 에피소드가 최소한 관찰 가능한
길이를 갖도록 보장하는 grace period입니다 (env 자체의 진짜 termination은
게이트 없이 즉시 반영).

정렬(chunk offset): 수집 데이터에서 action[t]는 상태 s_t를 만든 전이
(t-1→t) 동안 인가된 토크입니다. 따라서 s_t까지 관측한 chunk의 k번째 원소는
전이 (t+k-1→t+k)용 토크이고, 다음 스텝 제어에는 chunk[1]부터 사용해야
합니다(기본 --chunk-offset 1). chunk[0]은 이미 지나간 전이의 사후 추정치라
Agent A의 수동(passive) 예측-실측 비교용으로만 기록합니다.

워밍업 핸드오버(--handover-steps K, 기본 0=비활성): 에피소드 시작 후 K스텝
동안은 B도 순수 RL+PD로 보행하고(override 해제) estimator에는 실제 관측
히스토리만 쌓다가, K스텝째에 estimator 채널 게인을 다시 0으로 만들고
ManiFlow 토크 제어로 전환합니다. 리셋 직후 상태 s0는 수집 데이터에 존재하지
않아(수집기는 물리 1스텝 후의 s1부터 기록 — contacts 미갱신·기구학 리셋
속도) 첫 chunk가 크게 어긋나는데, 핸드오버는 이를 우회해 on-distribution
보행 상태에서 피드포워드 제어의 순수한 생존 시간을 측정합니다. 워밍업 구간의
tau_b_cmd/pred_a_passive는 NaN으로 기록되고 지표에서 제외됩니다.

보조 지연 투입(--assist-start-steps K, 기본 0=비활성, handover와 배타):
에피소드 시작부터 B의 hip PD를 α로 줄인 채(잔여 근력만) ManiFlow 보조 없이
보행하다가, K스텝째부터 (1-α)·MF 보조 토크를 투입합니다 — "보조 꺼짐 →
켜짐" 시연용. 투입 전 구간도 tau_b_cmd는 NaN으로 기록됩니다.
가산 보조 모드(--assist-beta)와 함께 쓰면 투입 전 구간이 "보조 없는 온전한
에이전트"(A와 동일)가 되고, 투입 후 τ_agent가 줄어드는 과정을 한 에피소드
안에서 볼 수 있습니다.

가산 보조(--assist-beta β, 기본 0=비활성, --residual-pd-scale와 배타):
hip PD 게인을 **감쇠하지 않고**(engage(1.0), 순수 가산 주입) ManiFlow 토크를
β배로 더합니다 — hip 총토크 = τ_agent(full PD) + β·τ_exo. α 블렌드가 게인을
강제로 깎아 에이전트 토크를 줄이는 것과 달리, 여기서는 보조 토크가 트래킹
오차를 줄여 PD 오차항이 작아지는 **능동적(수동적이지 않은) 토크 감소**가
일어나는지를 측정합니다. 보행 자체는 보조 전/후 모두 정상이므로 보조 효과는
영상 대신 다음 지표로 드러납니다:
  - τ_agent RMS 감소율 (B vs A=보조 없는 동일 정책) 및 offload 효율 감소율/β
  - τ_agent·ω 관절 파워(근육 일률 대리 지표) 감소율
  - 총토크 보존 오차 rms(τ_agent^B + β·τ_exo − τ_agent^A) — 과구동 여부
  - 레퍼런스 트래킹 오차(err6)·보상이 유지/개선되는지
이 모드에서는 B의 qfrc_actuator 리드백이 곧 τ_agent이며, run04 라벨 정의
(substep 평균)과 단위를 맞추기 위해 τ_agent는 substep 평균 리드백
(get_substep_mean_dof_forces)으로 기록합니다(tau_{a,b}_mean).

잔여 PD 결합(--residual-pd-scale α, 기본 0=순수 토크 치환): estimator 채널의
built-in PD 게인을 0 대신 α·(ke,kd)로 남기고 ManiFlow 토크를 (1-α)배로
주입합니다 — hip 토크 = α·PD(substep 피드백) + (1-α)·ManiFlow. 정상 보행
매니폴드에서는 ManiFlow ≈ PD 적용 토크이므로 총토크가 A와 같아지는 convex
블렌드입니다(명목 보행 보존, 과구동 없음); 상태가 이탈하면 α 비율의
임피던스가 매 substep 교정합니다. α 스윕으로 "생존에 필요한 최소 피드백
몫"을 정량화합니다. α>0이면 B의 qfrc_actuator 잔여(tau_b_qfrc)는 0이 아닌
PD 몫을 보고하며, B의 총 hip 토크 = tau_b_cmd + tau_b_qfrc 입니다.

출력 (--output, 기본 tasks/.../maniflow_control_results/<timestamp>/):
  metrics.json / metrics.txt : 에피소드 통계(생존 스텝, 종료 원인), 트래킹
                               오차 A vs B, 채널별 토크 통계
  traces.npz                 : 토크/관절/루트/보상 전체 trace + 에피소드 경계
  torque_channels_*.png      : 채널별 B 인가 토크 vs A 적용 토크 (+A 수동 예측)
  tracking_*.png             : 레퍼런스 대비 관절 오차·루트 높이 A vs B
  sim-*/sim-*.mp4            : (--record) 정면 시뮬 녹화 (고스트 A + 실체 B)
  sim_side.mp4               : (--record, 기본) 정측면 뷰 녹화
  sim_with_torque.mp4        : (--record) 영상(정면 위 + 측면 아래) + 토크 패널
                               합성 비디오

실행:
  bash tasks/mimic_suit_active_cable_walk_23dof/compare_maniflow_control_newton.sh
  # 실시간 GUI (A=고스트 스켈레톤, B=메시; 라이브 플롯 포함):
  bash tasks/.../compare_maniflow_control_newton.sh --viewer
  # 동영상 저장:
  bash tasks/.../compare_maniflow_control_newton.sh --record --episode-steps 600
  # 매 스텝 재예측(청크 대신 최신 상태 반영):
  bash tasks/.../compare_maniflow_control_newton.sh --predict-mode every_step
"""

# ---------------------------------------------------------------------------
# argparse를 먼저 수행 (simulator import 순서 규약, infer_maniflow_newton.py와 동일)
# ---------------------------------------------------------------------------
import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

TASK_ROOT = "tasks/mimic_suit_active_cable_walk_23dof"
DEFAULT_RL_CKPT = f"{TASK_ROOT}/output_newton_flat/score_based.ckpt"
DEFAULT_MANIFLOW_RUN_DIR = os.path.join(
    str(Path.home()),
    "Projects/ManiFlow_Policy/ManiFlow/data/outputs",
    # 2026-08-06: 기본값 run02 → run04(v2 substep 평균 라벨 DAgger 재학습).
    # best topk = epoch=0180-val_loss=0.015600.ckpt (discover_best_checkpoint 자동)
    "walking_flat-maniflow_lowdim_policy_walking-newton-hips-dagger-v2-run04_seed42",
)

GHOST_ENV = 0  # Agent A: pure RL (반투명 고스트)
MANIFLOW_ENV = 1  # Agent B: ManiFlow torque on estimator channels


def _create_parser():
    p = argparse.ArgumentParser(
        description="A/B comparison: pure-RL agent vs RL+ManiFlow-torque agent (Newton)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--rl-checkpoint", default=DEFAULT_RL_CKPT,
                   help="보행 RL policy 체크포인트 (ProtoMotions)")
    p.add_argument("--maniflow-ckpt", default=None,
                   help="ManiFlow 체크포인트 경로. 미지정 시 --maniflow-run-dir에서 "
                        "best topk 자동 선택")
    p.add_argument("--maniflow-run-dir", default=DEFAULT_MANIFLOW_RUN_DIR,
                   help="ManiFlow 학습 run 디렉토리 (--maniflow-ckpt 미지정 시 사용)")
    p.add_argument("--maniflow-root", default=None,
                   help="maniflow 패키지 위치 (기본: $MANIFLOW_ROOT 또는 관례 경로)")
    p.add_argument("--motion-file",
                   default="data/motion_for_trackers/skeleton_torque_suit_walk.pt")
    p.add_argument("--episode-steps", type=int, default=1200,
                   help="총 rollout 스텝 (에피소드가 끝나면 두 env를 함께 리셋하며 "
                        "이어서 진행). 1200 = 60s @ 20Hz")
    p.add_argument("--predict-mode", choices=["receding", "every_step"],
                   default="receding",
                   help="receding: n_action_steps 청크 단위 예측(학습 배포 방식), "
                        "every_step: 매 스텝 재예측 후 다음 액션만 사용")
    p.add_argument("--denoise-steps", type=int, default=3,
                   help="추론 ODE(Euler) 스텝 수. 기본 3 — 2026-07-15 검증: "
                        "consistency 학습이 임의 dt 점프를 커버해 재학습 없이 "
                        "축소 가능하고, N=3이 체크포인트 설정(10)보다 정확하며 "
                        "3배 빠름. 이전 동작 재현은 --denoise-steps 10")
    p.add_argument("--chunk-offset", type=int, default=1, choices=[0, 1],
                   help="청크에서 제어에 사용할 시작 인덱스. 1=다음 전이용 토크"
                        "(권장, 수집 정렬과 일치), 0=한 스텝 지연된 사후 추정치")
    p.add_argument("--torque-scale", type=float, default=1.0,
                   help="Agent B에 인가할 ManiFlow 토크 배율 (0 = 해당 채널 무동력 "
                        "sanity check)")
    p.add_argument("--residual-pd-scale", type=float, default=0.0,
                   help="estimator 채널에 남길 built-in PD 게인 비율 α∈[0,1). "
                        "0=순수 토크 치환(기존 동작). α>0이면 hip 토크 = α·PD + "
                        "(1-α)·ManiFlow convex 블렌드 — PD가 substep 안정화를, "
                        "ManiFlow가 보행 토크 본체를 분담")
    p.add_argument("--assist-beta", type=float, default=0.0,
                   help="가산 보조 계수 β>0: hip PD 게인을 그대로 두고(감쇠 없음) "
                        "hip 총토크 = τ_agent(full PD) + β·ManiFlow. 보조로 "
                        "트래킹 오차가 줄어 PD(=agent) 토크가 능동적으로 "
                        "감소하는지를 측정 — --residual-pd-scale와 배타 "
                        "(0 = 비활성)")
    p.add_argument("--handover-steps", type=int, default=0,
                   help="에피소드 시작 후 이 스텝 수 동안 B도 순수 RL+PD로 "
                        "보행(워밍업)한 뒤 ManiFlow 토크 제어로 전환. 리셋 직후 "
                        "관측(s0)이 학습 분포 밖이라 첫 chunk가 어긋나는 문제를 "
                        "우회 (0 = 리셋 직후부터 ManiFlow 제어)")
    p.add_argument("--assist-start-steps", type=int, default=0,
                   help="에피소드 시작 후 이 스텝 수 동안 B는 보조 없이(α 블렌드 "
                        "모드=α·PD 단독 잔여 근력만, --assist-beta 모드=full PD "
                        "온전한 에이전트) 보행하다가 이후 ManiFlow 보조를 투입 — "
                        "'보조 off→on' 시연용. --handover-steps와 동시 사용 불가 "
                        "(0 = 비활성)")
    p.add_argument("--fall-z", type=float, default=0.5,
                   help="root 높이가 이 값[m] 아래로 --fall-hold 스텝 연속 유지되면 "
                        "넘어짐으로 판정해 두 env를 함께 리셋 (<=0 비활성)")
    p.add_argument("--fall-hold", type=int, default=10,
                   help="넘어짐 판정에 필요한 연속 스텝 수 (10 = 0.5s @ 20Hz)")
    p.add_argument("--divergence-reset", type=float, default=5.0,
                   help="A↔B root XY 거리가 이 값[m]을 넘으면 에피소드를 끊고 "
                        "함께 리셋 (<=0 비활성)")
    p.add_argument("--min-episode-seconds", type=float, default=5.0,
                   help="에피소드 최소 지속 시간[s]. 넘어짐/발산이 감지돼도 이 "
                        "시간이 지나기 전에는 리셋하지 않음(넘어진 채로 시뮬은 "
                        "계속됨) — 관찰 가능한 최소 구간을 보장 (0 = 비활성, "
                        "감지 즉시 리셋)")
    p.add_argument("--viewer", action="store_true", default=False,
                   help="뷰어 표시 (기본 headless). A=고스트 스켈레톤, B=메시")
    p.add_argument("--record", action="store_true", default=False,
                   help="시뮬 mp4 + 토크 패널 합성 sim_with_torque.mp4 저장 "
                        "(뷰어 자동 활성화)")
    p.add_argument("--record-side", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="녹화 시 정측면 뷰도 함께 캡처(sim_side.mp4) 후 합성 "
                        "영상에 정면 위 / 측면 아래로 배치. --no-record-side로 "
                        "정면만 녹화")
    p.add_argument("--side-azimuth", type=float, default=90.0,
                   help="정측면 카메라의 정면 대비 방위각[deg] (90 = 왼쪽 측면, "
                        "270 = 오른쪽 측면)")
    p.add_argument("--no-mesh", action="store_true", default=False,
                   help="뷰어/녹화 시 skeleton mesh 에셋 대신 기본(캡슐) 에셋 사용")
    p.add_argument("--ghost-alpha", type=float, default=0.5,
                   help="Agent A 고스트 라인의 투명도 (0=완전 투명, 1=불투명)")
    p.add_argument("--ghost-line-width", type=float, default=3.5,
                   help="Agent A 고스트 스켈레톤 라인 두께 [px]")
    p.add_argument("--output", default=None,
                   help="결과 디렉토리 (기본: {task}/maniflow_control_results/<ts>)")
    p.add_argument("--zoom-steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overrides", nargs="*", default=[])
    return p


_parser = _create_parser()
_args, _ = _parser.parse_known_args()

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

import_simulator_before_torch("newton")

import json  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict  # noqa: E402
from datetime import datetime  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lightning.fabric import Fabric  # noqa: E402

from protomotions.utils.hydra_replacement import get_class  # noqa: E402
from protomotions.utils.fabric_config import FabricConfig  # noqa: E402
from protomotions.maniflow import (  # noqa: E402
    JointTorqueOverride,
    ManiFlowTorqueEstimator,
    discover_best_checkpoint,
    hip_dof_indices,
    load_maniflow_policy,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
log = logging.getLogger(__name__)

# ManiFlow action 채널 (공통 DOF 인덱스). main()에서 robot config로부터
# 이름 기반으로 채워짐 — 순수 hip 6 DOF = 공통 [0,1,2,5,6,7].
ESTIMATOR_DOF_INDICES = list(range(6))


# ---------------------------------------------------------------------------
def setup_agent_and_env(args, fabric: Fabric):
    """infer_maniflow_newton.py와 동일한 순서로 초기화 (Newton 고정, 2 envs)."""
    checkpoint = Path(args.rl_checkpoint)
    resolved_path = checkpoint.parent / "resolved_configs_inference.pt"
    assert resolved_path.exists(), f"Not found: {resolved_path}"

    resolved = torch.load(resolved_path, map_location="cpu", weights_only=False)

    robot_config = resolved["robot"]
    simulator_config = resolved["simulator"]
    terrain_config = resolved.get("terrain")
    scene_lib_config = resolved["scene_lib"]
    motion_lib_config = resolved["motion_lib"]
    env_config = resolved["env"]
    agent_config = resolved["agent"]

    current_sim = simulator_config._target_.split(".")[-3]
    if current_sim != "newton":
        log.info(f"Switching simulator: {current_sim} -> newton")
        from protomotions.simulator.factory import update_simulator_config_for_test
        simulator_config = update_simulator_config_for_test(
            current_simulator_config=simulator_config,
            new_simulator="newton",
            robot_config=robot_config,
        )

    from protomotions.utils.inference_utils import apply_backward_compatibility_fixes
    apply_backward_compatibility_fixes(robot_config, simulator_config, env_config)

    simulator_config.num_envs = 2  # env 0 = Agent A (ghost), env 1 = Agent B
    simulator_config.headless = not args.viewer
    motion_lib_config.motion_file = args.motion_file
    robot_config.reset_noise = None  # 두 agent의 초기 상태를 완전히 일치시킴

    if args.overrides:
        from protomotions.utils.config_utils import parse_cli_overrides, apply_config_overrides
        apply_config_overrides(
            parse_cli_overrides(args.overrides),
            env_config, simulator_config, robot_config,
            agent_config, terrain_config, motion_lib_config, scene_lib_config,
        )

    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator
    terrain_config, simulator_config = convert_friction_for_simulator(
        terrain_config, simulator_config
    )

    from protomotions.utils.component_builder import build_all_components
    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=fabric.device,
        save_dir=getattr(env_config, "save_dir", None),
    )

    from protomotions.envs.base_env.env import BaseEnv
    EnvClass = get_class(env_config._target_)
    env: BaseEnv = EnvClass(
        config=env_config,
        robot_config=robot_config,
        device=fabric.device,
        terrain=components["terrain"],
        scene_lib=components["scene_lib"],
        motion_lib=components["motion_lib"],
        simulator=components["simulator"],
    )

    from protomotions.agents.base_agent.agent import BaseAgent
    AgentClass = get_class(agent_config._target_)
    agent: BaseAgent = AgentClass(
        config=agent_config, env=env, fabric=fabric, root_dir=checkpoint.parent
    )
    agent.setup()
    agent.load(str(checkpoint), load_env=False)

    return agent, env


# ---------------------------------------------------------------------------
def pin_spawn_location(env) -> torch.Tensor:
    """terrain 스폰 샘플링을 고정 좌표로 패치해 두 env가 항상 겹쳐 스폰되게 함."""
    fixed_xy = env.terrain.sample_valid_locations(1)[0].clone()  # (2,)

    def _fixed_locations(num_envs, sample_flat=False):
        return fixed_xy.unsqueeze(0).expand(num_envs, -1).clone()

    env.terrain.sample_valid_locations = _fixed_locations
    return fixed_xy


def synced_reset(env, estimator, override, env_ids=None):
    """두 env를 같은 모션·같은 시각·같은 위치로 함께 리셋."""
    env.motion_manager.motion_ids[:] = 0
    env.motion_manager.motion_times[:] = 0.0
    obs, _ = env.reset(env_ids, disable_motion_resample=True)
    override.zero()
    estimator.reset()
    estimator.observe(env.simulator.get_robot_state())
    return obs


class GhostSkeletonRenderer:
    """Agent A를 반투명 라인 스켈레톤으로 그리는 뷰어 훅.

    world 0의 메시는 viewer.set_visible_worlds([1])로 숨기고, 매 렌더 프레임마다
    kinematic tree의 (parent→child) 뼈대 세그먼트를 log_lines로 그립니다.
    라인 파이프라인은 블렌딩이 활성화되어 있으나 alpha 유니폼이 1.0으로
    고정되어 있어, wireframe 셰이더의 update_frame을 감싸 고스트 알파를
    적용합니다 (실패 시 불투명 라인으로 동작).
    """

    def __init__(self, simulator, env_id: int = GHOST_ENV,
                 color=(0.45, 0.80, 1.0), alpha: float = 0.5,
                 line_width: float = 3.5):
        import warp as wp

        self.sim = simulator
        self.env_id = env_id
        self.viewer = simulator.viewer
        kin = simulator.robot_config.kinematic_info
        parents = kin.parent_indices
        self._children = torch.tensor(
            [i for i, p in enumerate(parents) if p >= 0],
            dtype=torch.long, device=simulator.device,
        )
        self._parents = torch.tensor(
            [p for p in parents if p >= 0],
            dtype=torch.long, device=simulator.device,
        )
        n_bones = len(self._children)
        self._starts_t = torch.zeros(n_bones, 3, device=simulator.device)
        self._ends_t = torch.zeros(n_bones, 3, device=simulator.device)
        self._starts_wp = wp.from_torch(self._starts_t, dtype=wp.vec3)
        self._ends_wp = wp.from_torch(self._ends_t, dtype=wp.vec3)
        self._color = color

        self._patch_line_alpha(alpha)
        try:
            self.viewer.renderer.line_width = line_width
        except AttributeError:
            pass

        # world 0 메시 숨김 + 월드 오프셋 0 고정 (물리 좌표 그대로 렌더)
        self.viewer.set_visible_worlds([MANIFLOW_ENV])
        self.viewer.set_world_offsets((0.0, 0.0, 0.0))

        # NewtonSimulator.render()의 _render_hook 슬롯에 체인 등록
        prev_hook = getattr(simulator, "_render_hook", None)

        def _hook():
            if prev_hook is not None:
                prev_hook()
            self._draw()

        simulator._render_hook = _hook

    def _patch_line_alpha(self, alpha: float) -> None:
        """라인 렌더 패스의 alpha 유니폼을 고스트 값으로 강제."""
        try:
            renderer = self.viewer.renderer
            shader = renderer._wireframe_shader
            orig_update = shader.update_frame

            def _update(*a, **kw):
                kw["alpha"] = alpha
                return orig_update(*a, **kw)

            shader.update_frame = _update
        except AttributeError as e:  # newton 내부 구조 변경 대비
            log.warning(f"고스트 라인 alpha 패치 실패({e}) — 불투명 라인으로 표시합니다.")

    def _draw(self) -> None:
        # 렌더 시점의 state_0에서 직접 읽어 지연 없이 그림 (SIM→COMMON 변환)
        bodies = self.sim._get_simulator_bodies_state()
        pos_sim = bodies.rigid_body_pos[self.env_id]  # (num_bodies, 3) SIM order
        pos = pos_sim[self.sim.data_conversion.body_convert_to_common]
        self._starts_t.copy_(pos[self._parents])
        self._ends_t.copy_(pos[self._children])
        self.viewer.log_lines(
            "/ghost/agent_a", self._starts_wp, self._ends_wp, self._color
        )


class SideViewRecorder:
    """정측면 뷰를 같은 뷰어 창 안의 패널로 띄우고(선택) mp4로도 저장.

    기본 녹화(RecordingMixin)는 `simulator.step()` 끝의 `render()`마다 현재
    카메라(정면) 뷰를 PNG로 캡처한다. 이 클래스는 스텝마다 카메라 방위각만
    `delta_deg` 돌려 **한 번 더** 렌더하고 그 프레임을 직접 받는다.

    화면 깜빡임 방지: 추가 렌더는 백버퍼에만 그리고 **버퍼 스왑(present)을
    막아** 창에 보이는 화면은 정면 뷰 그대로 유지한다. `get_frame()`은 화면이
    아니라 오프스크린 FBO(`renderer._frame_fbo`)에서 읽으므로 스왑을 막아도
    측면 프레임은 정상적으로 얻어진다. 얻은 프레임은 `viewer.log_image()`로
    같은 창의 도킹 패널에 표시한다 — 창 하나에서 정면 + 정측면 두 시점.

    또 추가 렌더 동안 `_user_is_recording`을 잠시 꺼서 정면 녹화 시퀀스(PNG
    프레임 번호 연속성 — 끊기면 ffmpeg가 중간에 멈춘다)를 건드리지 않는다.

    `config.camera_offset`이 설정된 경우 방위각이 무시되므로(고정 오프셋) 그
    오프셋 벡터를 z축 기준으로 회전시키는 방식으로 측면을 만든다.

    Args:
        out_path: mp4 저장 경로. None이면 뷰어 표시만 하고 파일은 쓰지 않음.
        panel_name: 뷰어 패널 이름. None이면 표시하지 않음(녹화 전용).
    """

    def __init__(self, simulator, out_path=None, fps: int = 20,
                 delta_deg: float = 90.0, panel_name="side view (sagittal)"):
        self.sim = simulator
        self.delta = delta_deg
        self.path = out_path
        self.panel = panel_name if hasattr(simulator.viewer, "log_image") else None
        self.writer = None
        if out_path is not None:
            import imageio.v2 as imageio
            self.writer = imageio.get_writer(
                str(out_path), fps=fps, codec="libx264", pixelformat="yuv420p",
                macro_block_size=2)
        self.frames = 0
        self.size = None  # (H, W) — 인코더는 고정 크기를 요구

    def _rotate_camera(self):
        """카메라를 측면으로 돌리고, 원복 함수를 반환."""
        sim = self.sim
        offset = getattr(sim.config, "camera_offset", None)
        if offset is not None:
            saved = list(offset)
            rad = np.deg2rad(self.delta)
            c, s = np.cos(rad), np.sin(rad)
            x, y, z = saved[0], saved[1], saved[2]
            sim.config.camera_offset = [x * c - y * s, x * s + y * c, z]

            def _restore():
                sim.config.camera_offset = saved
        else:
            saved_az = sim._camera_azimuth
            sim._camera_azimuth = (saved_az + self.delta) % 360.0

            def _restore():
                sim._camera_azimuth = saved_az
        return _restore

    def capture(self) -> bool:
        """측면 프레임 1장 캡처. 녹화 상태 전환 프레임이면 건너뜀."""
        sim = self.sim
        if getattr(sim, "_user_recording_state_change", False):
            return False  # 전환 처리(시작/종료)는 정면 렌더에 맡긴다
        restore = self._rotate_camera()
        was_recording = sim._user_is_recording
        sim._user_is_recording = False  # 이 렌더는 정면 시퀀스에 넣지 않음
        renderer = getattr(sim.viewer, "renderer", None)
        saved_present = getattr(renderer, "present", None)
        if saved_present is not None:
            # 백버퍼에만 그리고 스왑하지 않음 → 창에는 정면 뷰가 그대로 남는다
            renderer.present = lambda: None
        try:
            sim.render()
            frame_wp = sim.viewer.get_frame()  # 오프스크린 FBO에서 읽음
            if self.panel is not None:
                sim.viewer.log_image(self.panel, frame_wp)  # 같은 창의 패널
            frame = frame_wp.numpy() if self.writer is not None else None
        finally:
            if saved_present is not None:
                renderer.present = saved_present
            sim._user_is_recording = was_recording
            restore()
        if frame is not None:
            if self.size is None:
                self.size = frame.shape[:2]
            elif frame.shape[:2] != self.size:
                from PIL import Image
                frame = np.asarray(Image.fromarray(frame).resize(
                    (self.size[1], self.size[0])))
            self.writer.append_data(frame)
        self.frames += 1
        return True

    def close(self):
        if self.writer is not None:
            self.writer.close()
            print(f"정측면 영상 저장: {self.path} ({self.frames} 프레임)")


# ---------------------------------------------------------------------------
@torch.no_grad()
def run_rollout(agent, env, estimator, override, args, side_recorder=None):
    """A(순수 RL)와 B(RL+ManiFlow 토크)를 같은 씬에서 굴리며 trace를 수집."""
    N = env.num_envs
    T = args.episode_steps
    Ta = estimator.n_action_steps
    Da = estimator.action_dim
    n_dofs = env.robot_config.number_of_actions
    offset = args.chunk_offset
    assert offset < Ta, f"chunk-offset({offset}) must be < n_action_steps({Ta})"
    alpha = args.residual_pd_scale
    blend = 1.0 - alpha  # ManiFlow 몫 — α·PD + (1-α)·MF ≈ 명목 총토크 유지
    beta = args.assist_beta
    additive = beta > 0.0  # 가산 보조: τ_agent(full PD) + β·τ_exo
    # 주입 배율: 가산 모드는 β, 블렌드 모드는 (1-α)
    inject_scale = args.torque_scale * (beta if additive else blend)

    steps_per_sec = round(1.0 / env.simulator.dt) if env.simulator.dt > 0 else 20
    min_episode_steps = max(1, round(args.min_episode_seconds * steps_per_sec))
    log.info(f"에피소드 최소 지속 시간: {args.min_episode_seconds}s "
             f"({min_episode_steps} steps @ {steps_per_sec}Hz)")

    viewer = getattr(env.simulator, "viewer", None)
    live_plot = viewer is not None and hasattr(viewer, "log_scalar")
    ch_names = override.dof_names
    # 라이브 플롯에 띄울 채널 (좌/우 hip flexion) — 채널 레이아웃에서 파생
    live_channels = [j for j, nm in enumerate(ch_names)
                     if nm.startswith("hip_flexion")]

    tr = {
        "tau_b_cmd": np.zeros((T, Da), np.float32),     # B에 실제 인가(클램프 후)
        "tau_a_applied": np.zeros((T, Da), np.float32),  # A의 PD 적용 토크(qfrc)
        "tau_b_qfrc": np.zeros((T, Da), np.float32),     # B의 qfrc 잔여(≈0 검증용)
        # substep 평균 qfrc = ZOH 임펄스 일관 토크(run04 라벨 정의와 동일 단위).
        # 가산 보조 모드에서 A/B의 "에이전트(PD) 토크"를 비교하는 기준 지표.
        "tau_a_mean": np.zeros((T, Da), np.float32),
        "tau_b_mean": np.zeros((T, Da), np.float32),
        "pred_a_passive": np.zeros((T, Da), np.float32),  # A 수동 예측(chunk 동일 원소)
        "dof_pos": np.zeros((T, N, n_dofs), np.float32),
        "dof_vel": np.zeros((T, N, n_dofs), np.float32),
        "ref_dof_pos": np.zeros((T, n_dofs), np.float32),
        "root_pos": np.zeros((T, N, 3), np.float32),
        "rew": np.zeros((T, N), np.float32),
    }
    episodes = []  # dict(start, end, cause)
    ep_start = 0
    fall_count = np.zeros(N, dtype=int)  # 연속 저고도 스텝 수 (자체 넘어짐 감지)

    handover = max(0, args.handover_steps)
    assist_start = max(0, args.assist_start_steps)
    assert not (handover > 0 and assist_start > 0), (
        "--handover-steps와 --assist-start-steps는 동시 사용 불가")
    warmup = handover if handover > 0 else assist_start
    obs = synced_reset(env, estimator, override)
    if handover > 0:
        override.restore()  # 워밍업: B도 순수 PD 보행, chunk는 핸드오버 시점에 생성
        chunk = None
    elif assist_start > 0:
        chunk = None  # α·PD 단독(보조 꺼짐) — engage(α)는 유지, MF 투입만 지연
    else:
        chunk = estimator.predict()  # (N, Ta, Da)
    chunk_pos = offset

    gain_check_done = False
    t_start = time.time()
    steps_done = T

    for t in range(T):
        if viewer is not None and hasattr(viewer, "is_running") and not viewer.is_running():
            log.warning(f"뷰어 창이 닫혀 step {t}에서 중단합니다.")
            steps_done = t
            break

        obs = agent.add_agent_info_to_obs(obs)
        obs_td = agent.obs_dict_to_tensordict(obs)
        model_out = agent.model(obs_td)
        action = model_out.get("mean_action", model_out["action"])

        ep_rel = t - ep_start
        if chunk is None and ep_rel >= warmup:
            # 워밍업 종료: 실제 히스토리(2프레임)로 첫 chunk를 만들고 제어 전환
            if handover > 0:
                # 가산 모드는 게인 무감쇠(1.0), 블렌드 모드는 α로 재결합
                override.engage(1.0 if additive else alpha)
                log.info(f"  handover @ step {t} (ep {len(episodes) + 1}): "
                         + (f"B에 +{beta:g}·ManiFlow 가산 보조 투입" if additive
                            else f"B를 ManiFlow 토크 제어로 전환 (α={alpha:g})"))
            elif additive:
                log.info(f"  assist ON @ step {t} (ep {len(episodes) + 1}): "
                         f"에이전트 단독(full PD) → +{beta:g}·ManiFlow 보조 투입")
            else:
                log.info(f"  assist ON @ step {t} (ep {len(episodes) + 1}): "
                         f"α·PD 단독 → (1-α)·ManiFlow 보조 투입 (α={alpha:g})")
            chunk = estimator.predict()
            chunk_pos = offset

        if chunk is not None:
            cur = chunk[:, chunk_pos]  # (N, Da) — 이번 전이(t→t+1)용 토크
            tau_b = override.set_torques(
                cur[MANIFLOW_ENV : MANIFLOW_ENV + 1] * inject_scale
            )
            tr["tau_b_cmd"][t] = tau_b[0].cpu().numpy()
            tr["pred_a_passive"][t] = cur[GHOST_ENV].cpu().numpy()
        else:
            # 워밍업 구간: B는 PD 보행 중 — ManiFlow 미사용 (지표에서 제외)
            tr["tau_b_cmd"][t] = np.nan
            tr["pred_a_passive"][t] = np.nan

        obs, rewards, dones, _terminated, _extras = env.step(action)
        if side_recorder is not None:
            side_recorder.capture()  # 같은 상태를 측면에서 한 장 더
        robot_state = env.simulator.get_robot_state()
        estimator.observe(robot_state)

        dof_forces = robot_state.dof_forces[:, ESTIMATOR_DOF_INDICES].cpu().numpy()
        tr["tau_a_applied"][t] = dof_forces[GHOST_ENV]
        tr["tau_b_qfrc"][t] = dof_forces[MANIFLOW_ENV]
        mean_forces = (env.simulator.get_substep_mean_dof_forces()
                       [:, ESTIMATOR_DOF_INDICES].cpu().numpy())
        tr["tau_a_mean"][t] = mean_forces[GHOST_ENV]
        tr["tau_b_mean"][t] = mean_forces[MANIFLOW_ENV]
        tr["dof_pos"][t] = robot_state.dof_pos.cpu().numpy()
        tr["dof_vel"][t] = robot_state.dof_vel.cpu().numpy()
        tr["root_pos"][t] = robot_state.rigid_body_pos[:, 0].cpu().numpy()
        tr["rew"][t] = rewards.cpu().numpy()
        mm = env.motion_manager
        ref_state = env.motion_lib.get_motion_state(mm.motion_ids, mm.motion_times)
        tr["ref_dof_pos"][t] = ref_state.dof_pos[0].cpu().numpy()

        if not gain_check_done and chunk is not None and ep_rel == warmup + 2:
            leak = float(np.abs(tr["tau_b_qfrc"][t]).max())
            if additive:
                a_max = float(np.abs(tr["tau_a_applied"][t]).max())
                log.info(f"가산 보조 (β={beta:g}): B qfrc = 에이전트 full PD 토크 "
                         f"|max| {leak:.2f} N·m (A {a_max:.2f} N·m — 게인 무감쇠 "
                         "확인)")
            elif alpha > 0:
                log.info(f"잔여 PD 활성 (α={alpha:g}): B qfrc PD 몫 |max| "
                         f"{leak:.2f} N·m (0이 아닌 것이 정상)")
            elif leak > 1.0:
                log.warning(
                    f"Agent B의 override 채널에서 PD 잔여 토크 감지({leak:.2f} N·m) "
                    "— gain zero-out이 적용되지 않았을 수 있습니다."
                )
            else:
                log.info(f"gain zero-out 검증 OK (B qfrc 잔여 {leak:.3f} N·m)")
            gain_check_done = True

        if chunk is not None:
            chunk_pos += 1
            need_new_chunk = (
                args.predict_mode == "every_step" or chunk_pos >= Ta
            )
            if need_new_chunk:
                chunk = estimator.predict()
                chunk_pos = offset

        if live_plot:
            for j in live_channels:  # hip_flexion_r / hip_flexion_l
                if np.isfinite(tr["tau_b_cmd"][t, j]):
                    viewer.log_scalar(f"torque/{ch_names[j]}/B_maniflow",
                                      float(tr["tau_b_cmd"][t, j]))
                    if alpha > 0 or additive:  # 총토크 = MF 주입분 + PD 몫
                        viewer.log_scalar(
                            f"torque/{ch_names[j]}/B_total",
                            float(tr["tau_b_cmd"][t, j] + tr["tau_b_qfrc"][t, j]))
                if additive:  # 능동적 토크 감소를 라이브로 확인 (같은 substep 평균 정의)
                    viewer.log_scalar(f"torque/{ch_names[j]}/A_agent_PD",
                                      float(tr["tau_a_mean"][t, j]))
                    viewer.log_scalar(f"torque/{ch_names[j]}/B_agent_PD",
                                      float(tr["tau_b_mean"][t, j]))
                viewer.log_scalar(f"torque/{ch_names[j]}/A_applied",
                                  float(tr["tau_a_applied"][t, j]))
            err = np.abs(tr["dof_pos"][t][:, ESTIMATOR_DOF_INDICES]
                         - tr["ref_dof_pos"][t][None, ESTIMATOR_DOF_INDICES])
            viewer.log_scalar("tracking/dof_err6/A_pureRL", float(err[GHOST_ENV].mean()))
            viewer.log_scalar("tracking/dof_err6/B_maniflow", float(err[MANIFLOW_ENV].mean()))
        if viewer is not None and t % 20 == 0:
            b_mode = (f"{alpha:g}·PD+{blend:g}·ManiFlow" if alpha > 0
                      else "ManiFlow torque")
            if additive:
                b_mode = f"agent PD + {beta:g}·ManiFlow (additive)"
            if assist_start > 0:
                off_mode = f"{alpha:g}·PD" if not additive else "agent PD only"
                on_mode = (f"agent PD+{beta:g}·MF" if additive
                           else f"{alpha:g}·PD+{blend:g}·MF")
                b_mode = (f"{off_mode} — assist OFF" if chunk is None
                          else f"{on_mode} — assist ON")
            env.simulator.set_window_title(
                f"A: RL+PD (ghost)  |  B: {b_mode} on {Da}ch (solid)  "
                f"|  ep {len(episodes) + 1}  step {t}/{T}")

        # ── 에피소드 종료 판정: env done + 자체 넘어짐/발산 감지 ──────────
        # (inference config에는 termination이 없는 경우가 많아 자체 감지 필요)
        root_z = tr["root_pos"][t, :, 2]
        if args.fall_z > 0:
            fall_count = np.where(root_z < args.fall_z, fall_count + 1, 0)
        fallen = fall_count >= args.fall_hold
        xy_div = float(np.linalg.norm(
            tr["root_pos"][t, GHOST_ENV, :2] - tr["root_pos"][t, MANIFLOW_ENV, :2]))

        # 최소 지속 시간 게이트: 이 구간 안에서는 넘어짐/발산이 감지돼도 리셋하지
        # 않고 그대로 시뮬레이션을 진행 (쓰러진 채로 남음) — 관찰 가능한 최소
        # 길이를 보장. env 자체의 done(진짜 termination)은 게이트 없이 즉시 반영.
        elapsed = t + 1 - ep_start
        past_grace = elapsed >= min_episode_steps

        cause = None
        if bool(dones.any()):
            term = _terminated.cpu().numpy()
            if term[MANIFLOW_ENV] and not term[GHOST_ENV]:
                cause = "B_terminated"
            elif term[GHOST_ENV] and not term[MANIFLOW_ENV]:
                cause = "A_terminated"
            elif term.all():
                cause = "both_terminated"
            else:
                cause = "clip_end"
        elif past_grace and fallen[MANIFLOW_ENV]:
            cause = "B_fall"
        elif past_grace and fallen[GHOST_ENV]:
            cause = "A_fall"
        elif past_grace and args.divergence_reset > 0 and xy_div > args.divergence_reset:
            cause = "diverged"

        if cause is not None:
            episodes.append({"start": ep_start, "end": t + 1, "cause": cause})
            log.info(f"  episode {len(episodes)}: steps {ep_start}-{t + 1} "
                     f"({t + 1 - ep_start} steps, {cause}) — 두 env 함께 리셋")
            obs = synced_reset(env, estimator, override)
            if handover > 0:
                override.restore()  # 다음 에피소드도 워밍업(PD 보행)부터
                chunk = None
            elif assist_start > 0:
                chunk = None  # 다음 에피소드도 α·PD 단독(보조 꺼짐)부터
            else:
                chunk = estimator.predict()
            chunk_pos = offset
            ep_start = t + 1
            fall_count[:] = 0

        if (t + 1) % 200 == 0:
            log.info(f"  step {t + 1:5d}/{T} | episodes {len(episodes)} "
                     f"| {time.time() - t_start:.1f}s")

    if ep_start < steps_done:
        episodes.append({"start": ep_start, "end": steps_done, "cause": "horizon"})

    if steps_done < T:
        for k in tr:
            tr[k] = tr[k][:steps_done]

    return tr, episodes


# ---------------------------------------------------------------------------
def _rms(x) -> float:
    x = np.asarray(x)
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else float("nan")


def _corr(x, y) -> float:
    with np.errstate(invalid="ignore"):
        if len(x) > 1 and x.std() > 0 and y.std() > 0:
            return float(np.corrcoef(x, y)[0, 1])
    return float("nan")


def compute_assist_metrics(tr, ch_names, beta: float, assist_start_steps: int = 0):
    """가산 보조(τ_total = τ_agent + β·τ_exo)의 에이전트 토크 감소 분석.

    A(보조 없음)와 B(보조)는 같은 정책·같은 레퍼런스 모션·같은 시각을 공유하는
    두 world이므로 시간 정렬 비교가 성립한다. 모든 에이전트 토크는 substep 평균
    qfrc(=20Hz ZOH 임펄스 일관, run04 라벨과 동일 정의)를 사용해 주입 토크
    β·τ_exo와 단위를 맞춘다.

    핵심 지표:
      agent_rms_reduction_pct  A 대비 B의 에이전트(PD) 토크 RMS 감소율 [%]
      offload_efficiency       감소율/β — 1.0이면 보조한 만큼 정확히 대체
      agent_power_reduction_pct  |τ_agent·ω| 평균 감소율 [%] (근육 일률 대리)
      conservation_rmse        rms(τ_agent^B + β·τ_exo − τ_agent^A) — 총토크
                               보존 오차(과구동/부족구동 여부)

    보조 토크 정합 지표 — **감소율을 좌우하는 양**:
      corr_exo_agentA      ρ = corr(β·τ_exo, τ_agent^A)
      exo_tracking_nrmse   rms(τ_exo − τ_agent^A)/rms(τ_agent^A) — 보조 토크가
                           "필요 토크"를 못 맞추는 정도(β 배율 제외한 원 예측)
      predicted_reduction_pct  해석 모델 100·(1 − √(1 − 2ρr + r²)),
                           r = rms(β·τ_exo)/rms(τ_agent^A). 상태가 명목에
                           머무르면 τ_agent^B ≈ τ_agent^A − β·τ_exo이므로 성립
      superposition_opt_reduction_pct  같은 ρ에서 β를 최적화한 중첩모델
                           값 100·(1 − √(1 − ρ²)) (r = ρ에서 달성)
      beta_superposition_opt  그 값을 주는 β* = ρ·rms(τ_agent^A)/rms(τ_exo)
    ρ가 낮을수록 줄일 수 있는 착용자 토크가 줄어든다. 단 이 중첩모델은
    "B의 필요 토크 = A의 필요 토크"를 가정하므로 β가 커져 B의 보행·목표각
    자체가 변하면 어긋난다(실측이 모델값을 넘을 수 있음) — 상한이 아니라
    경향 설명용 기준선으로만 쓸 것.
    """
    on = np.isfinite(tr["tau_b_cmd"]).all(axis=-1)  # 보조 투입 구간
    idx6 = ESTIMATOR_DOF_INDICES
    exo = np.nan_to_num(tr["tau_b_cmd"], nan=0.0)  # 이미 β배·클램프된 주입 토크
    a_ag, b_ag = tr["tau_a_mean"], tr["tau_b_mean"]
    b_tot = b_ag + exo
    pw_a = a_ag * tr["dof_vel"][:, GHOST_ENV][:, idx6]
    pw_b = b_ag * tr["dof_vel"][:, MANIFLOW_ENV][:, idx6]
    err6 = np.abs(
        tr["dof_pos"][:, :, idx6] - tr["ref_dof_pos"][:, None, idx6]
    ).mean(axis=-1)  # (T, N)

    def _block(mask):
        if not mask.any():
            return None
        r_a, r_b = _rms(a_ag[mask]), _rms(b_ag[mask])
        p_a = float(np.abs(pw_a[mask]).mean())
        p_b = float(np.abs(pw_b[mask]).mean())
        red = (1.0 - r_b / r_a) if r_a > 0 else float("nan")
        # 보조 토크 정합 ρ, r → 중첩모델 감소율(경향 기준선)
        r_exo = _rms(exo[mask])
        nan = float("nan")
        rho = pred = sup_opt = beta_star = exo_nrmse = nan
        if r_exo > 0 and r_a > 0 and beta > 0:
            rho = _corr(exo[mask].ravel(), a_ag[mask].ravel())
            ratio = r_exo / r_a
            exo_full = exo[mask] / beta  # β 배율 제외한 원 예측
            exo_nrmse = _rms(exo_full - a_ag[mask]) / r_a
            if np.isfinite(rho):
                pred = 100.0 * (1.0 - np.sqrt(
                    max(0.0, 1.0 - 2.0 * rho * ratio + ratio ** 2)))
                sup_opt = 100.0 * (1.0 - np.sqrt(max(0.0, 1.0 - rho ** 2)))
                beta_star = rho * r_a / (r_exo / beta)
        return {
            "steps": int(mask.sum()),
            "agent_rms": {"A_noassist": r_a, "B_assisted": r_b},
            "agent_absmean": {"A_noassist": float(np.abs(a_ag[mask]).mean()),
                              "B_assisted": float(np.abs(b_ag[mask]).mean())},
            "exo_rms": _rms(exo[mask]),
            "B_total_rms": _rms(b_tot[mask]),
            "agent_rms_reduction_pct": 100.0 * red,
            "offload_efficiency": (red / beta) if beta > 0 else float("nan"),
            "agent_power_absmean": {"A_noassist": p_a, "B_assisted": p_b},
            "agent_power_reduction_pct": (100.0 * (1.0 - p_b / p_a)
                                         if p_a > 0 else float("nan")),
            "conservation_rmse": _rms(b_tot[mask] - a_ag[mask]),
            "corr_exo_agentA": float(rho),
            "exo_tracking_nrmse": float(exo_nrmse),
            "predicted_reduction_pct": float(pred),
            "superposition_opt_reduction_pct": float(sup_opt),
            "beta_superposition_opt": float(beta_star),
            "dof_err6_mean": {"A_noassist": float(err6[mask, GHOST_ENV].mean()),
                              "B_assisted": float(err6[mask, MANIFLOW_ENV].mean())},
            "reward_mean": {
                "A_noassist": float(tr["rew"][mask, GHOST_ENV].mean()),
                "B_assisted": float(tr["rew"][mask, MANIFLOW_ENV].mean()),
            },
        }

    per_ch = {}
    for j, name in enumerate(ch_names):
        r_a, r_b = _rms(a_ag[on, j]), _rms(b_ag[on, j])
        per_ch[name] = {
            "A_agent_rms": r_a,
            "B_agent_rms": r_b,
            "exo_rms": _rms(exo[on, j]),
            "B_total_rms": _rms(b_tot[on, j]),
            "reduction_pct": (100.0 * (1.0 - r_b / r_a) if r_a > 0
                              else float("nan")),
            "corr_exo_A": _corr(exo[on, j], a_ag[on, j]),
            "conservation_rmse": _rms(b_tot[on, j] - a_ag[on, j]),
        }

    out = {"beta": beta, "overall": _block(on), "per_channel": per_ch}
    if assist_start_steps > 0 and bool((~on).any()):
        out["phase_off"] = _block(~on)   # 보조 없음 (A와 동일해야 정상)
        out["phase_on"] = out["overall"]
    return out


def compute_metrics(tr, episodes, ch_names, fall_z: float = 0.5,
                    assist_beta: float = 0.0, assist_start_steps: int = 0):
    dof_err = np.abs(tr["dof_pos"] - tr["ref_dof_pos"][:, None, :])  # (T, N, D)
    err6 = dof_err[:, :, ESTIMATOR_DOF_INDICES].mean(axis=-1)   # (T, N)
    err_all = dof_err.mean(axis=-1)

    # ManiFlow가 실제 제어한 구간 (워밍업 핸드오버 스텝은 NaN으로 기록됨)
    valid = np.isfinite(tr["tau_b_cmd"]).all(axis=-1)  # (T,)

    # 에피소드별 B의 최초 저고도 진입 시점 (에피소드-상대 스텝)
    for e in episodes:
        bz = tr["root_pos"][e["start"]:e["end"], MANIFLOW_ENV, 2]
        below = np.where(bz < fall_z)[0] if fall_z > 0 else np.array([], int)
        e["b_fall_rel"] = int(below[0]) if len(below) else None

    ep_lengths = [e["end"] - e["start"] for e in episodes]
    causes = [e["cause"] for e in episodes]

    m = {
        "episodes": episodes,
        "num_episodes": len(episodes),
        "episode_lengths": ep_lengths,
        "mean_episode_length": float(np.mean(ep_lengths)) if ep_lengths else 0.0,
        "cause_counts": {c: causes.count(c) for c in sorted(set(causes))},
        "mean_length_by_cause": {
            c: float(np.mean([e["end"] - e["start"] for e in episodes
                              if e["cause"] == c]))
            for c in sorted(set(causes))
        },
        "tracking_dof_err6_mean": {
            "A_pureRL": float(err6[:, GHOST_ENV].mean()),
            "B_maniflow": float(err6[:, MANIFLOW_ENV].mean()),
        },
        "tracking_dof_err_all_mean": {
            "A_pureRL": float(err_all[:, GHOST_ENV].mean()),
            "B_maniflow": float(err_all[:, MANIFLOW_ENV].mean()),
        },
        "reward_mean": {
            "A_pureRL": float(tr["rew"][:, GHOST_ENV].mean()),
            "B_maniflow": float(tr["rew"][:, MANIFLOW_ENV].mean()),
        },
        "root_xy_divergence_mean": float(
            np.linalg.norm(
                tr["root_pos"][:, GHOST_ENV, :2] - tr["root_pos"][:, MANIFLOW_ENV, :2],
                axis=-1,
            ).mean()
        ),
        "maniflow_active_steps": int(valid.sum()),
        "b_qfrc_residual_absmax": (
            float(np.abs(tr["tau_b_qfrc"][valid]).max()) if valid.any() else 0.0
        ),
        "channels": {},
    }
    if assist_beta > 0:
        m["assist"] = compute_assist_metrics(
            tr, ch_names, assist_beta, assist_start_steps=assist_start_steps
        )

    for j, name in enumerate(ch_names):
        a, b = tr["tau_a_applied"][valid, j], tr["tau_b_cmd"][valid, j]
        b_tot = b + tr["tau_b_qfrc"][valid, j]  # 총토크 = MF 주입 + 잔여 PD
        m["channels"][name] = {
            "A_applied_std": float(a.std()) if len(a) else float("nan"),
            "B_cmd_std": float(b.std()) if len(b) else float("nan"),
            "B_cmd_absmean": float(np.abs(b).mean()) if len(b) else float("nan"),
            "corr_A_B": _corr(a, b),
            "B_total_std": float(b_tot.std()) if len(b_tot) else float("nan"),
            "corr_A_Btotal": _corr(a, b_tot),
        }
    return m


def format_metrics_text(m, ch_names, residual_pd_scale: float = 0.0) -> str:
    lines = []
    lines.append(f"episodes: {m['num_episodes']}  "
                 f"(mean length {m['mean_episode_length']:.1f} steps @ 20Hz)")
    lines.append("end causes: " + ", ".join(
        f"{c}×{n} (mean {m['mean_length_by_cause'][c]:.0f} steps)"
        for c, n in m["cause_counts"].items()))
    fall_rel = [e.get("b_fall_rel") for e in m["episodes"]]
    if any(v is not None for v in fall_rel):
        lines.append("B fall step (ep-relative, root z<fall_z): "
                     + ", ".join("-" if v is None else str(v) for v in fall_rel))
    lines.append("")
    lines.append(f"{'':24s} {'A (pure RL)':>14s} {'B (ManiFlow)':>14s}")
    lines.append(f"{'dof err (6ch, rad)':24s} "
                 f"{m['tracking_dof_err6_mean']['A_pureRL']:14.4f} "
                 f"{m['tracking_dof_err6_mean']['B_maniflow']:14.4f}")
    lines.append(f"{'dof err (all, rad)':24s} "
                 f"{m['tracking_dof_err_all_mean']['A_pureRL']:14.4f} "
                 f"{m['tracking_dof_err_all_mean']['B_maniflow']:14.4f}")
    lines.append(f"{'mean reward':24s} "
                 f"{m['reward_mean']['A_pureRL']:14.4f} "
                 f"{m['reward_mean']['B_maniflow']:14.4f}")
    lines.append("")
    lines.append(f"root XY divergence (A↔B) mean: {m['root_xy_divergence_mean']:.3f} m")
    if residual_pd_scale > 0:
        lines.append(f"B qfrc PD share |max| (α={residual_pd_scale:g}): "
                     f"{m['b_qfrc_residual_absmax']:.3f} N·m")
    else:
        lines.append("B qfrc residual |max| (≈0 expected): "
                     f"{m['b_qfrc_residual_absmax']:.3f} N·m")
    lines.append("")
    hdr = (f"{'channel':>18s} | {'A std':>8s} | {'B std':>8s} | {'B |mean|':>8s} "
           f"| {'corr':>6s} | {'Btot std':>8s} | {'corrT':>6s}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for name in ch_names:
        c = m["channels"][name]
        lines.append(f"{name:>18s} | {c['A_applied_std']:8.2f} | {c['B_cmd_std']:8.2f} "
                     f"| {c['B_cmd_absmean']:8.2f} | {c['corr_A_B']:6.3f} "
                     f"| {c['B_total_std']:8.2f} | {c['corr_A_Btotal']:6.3f}")
    if "assist" in m:
        lines.append("")
        lines.append(format_assist_text(m["assist"], ch_names))
    return "\n".join(lines)


def format_assist_text(asst, ch_names) -> str:
    """가산 보조 분석 블록의 사람이 읽는 표현."""
    beta = asst["beta"]
    o = asst["overall"]
    lines = [f"=== 가산 보조 분석 (β={beta:g}): hip 총토크 = τ_agent(full PD) "
             f"+ {beta:g}·τ_exo ===",
             "  (에이전트 토크는 substep 평균 qfrc — 주입 토크와 같은 정의)"]
    if o is None:
        lines.append("  보조 구간 없음")
        return "\n".join(lines)
    lines.append(f"{'':32s} {'A (no assist)':>14s} {'B (assisted)':>14s}")
    lines.append(f"{'agent torque RMS [N·m]':32s} "
                 f"{o['agent_rms']['A_noassist']:14.2f} "
                 f"{o['agent_rms']['B_assisted']:14.2f}   "
                 f"({-o['agent_rms_reduction_pct']:+.1f}%)")
    lines.append(f"{'agent |torque| mean [N·m]':32s} "
                 f"{o['agent_absmean']['A_noassist']:14.2f} "
                 f"{o['agent_absmean']['B_assisted']:14.2f}")
    lines.append(f"{'agent |power| mean [W]':32s} "
                 f"{o['agent_power_absmean']['A_noassist']:14.2f} "
                 f"{o['agent_power_absmean']['B_assisted']:14.2f}   "
                 f"({-o['agent_power_reduction_pct']:+.1f}%)")
    lines.append(f"{'dof err 6ch [rad]':32s} "
                 f"{o['dof_err6_mean']['A_noassist']:14.4f} "
                 f"{o['dof_err6_mean']['B_assisted']:14.4f}")
    lines.append(f"{'mean reward':32s} "
                 f"{o['reward_mean']['A_noassist']:14.4f} "
                 f"{o['reward_mean']['B_assisted']:14.4f}")
    lines.append("")
    lines.append(f"exo torque RMS (β·τ_exo)      : {o['exo_rms']:8.2f} N·m")
    lines.append(f"B total RMS (agent + exo)     : {o['B_total_rms']:8.2f} N·m "
                 f"(A {o['agent_rms']['A_noassist']:.2f})")
    lines.append(f"offload 효율 (RMS 감소율 / β) : {o['offload_efficiency']:8.3f} "
                 "(1.0 = 보조한 만큼 정확히 대체)")
    lines.append(f"총토크 보존 오차 rms(Btot-A)  : {o['conservation_rmse']:8.2f} N·m")
    lines.append("")
    lines.append("보조 토크 정합 (= 감소율을 좌우하는 양):")
    lines.append(f"  ρ = corr(β·τ_exo, τ_agent^A) : {o['corr_exo_agentA']:8.3f}")
    lines.append(f"  τ_exo 정규화 오차 rms/rms   : "
                 f"{o['exo_tracking_nrmse']:8.3f}  (τ_agent−τ_exo tracking error)")
    lines.append(f"  중첩모델 예측 감소율        : "
                 f"{-o['predicted_reduction_pct']:+8.1f}%  (실측 "
                 f"{-o['agent_rms_reduction_pct']:+.1f}%)")
    lines.append(f"  중첩모델 β 최적화 시         : "
                 f"{-o['superposition_opt_reduction_pct']:+8.1f}%  (β* ≈ "
                 f"{o['beta_superposition_opt']:.2f}) — 상한 아님, 경향 기준선")
    lines.append("")
    hdr = (f"{'channel':>18s} | {'A rms':>8s} | {'B rms':>8s} | {'β·exo':>8s} "
           f"| {'Btot rms':>8s} | {'Δagent%':>7s} | {'corr(exo,A)':>11s}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for name in ch_names:
        c = asst["per_channel"][name]
        lines.append(f"{name:>18s} | {c['A_agent_rms']:8.2f} "
                     f"| {c['B_agent_rms']:8.2f} | {c['exo_rms']:8.2f} "
                     f"| {c['B_total_rms']:8.2f} | {-c['reduction_pct']:7.1f} "
                     f"| {c['corr_exo_A']:11.3f}")
    if "phase_off" in asst:
        lines.append("")
        lines.append("보조 off→on 구간 비교 (같은 에이전트, 같은 에피소드):")
        for tag, key in [("off", "phase_off"), ("on ", "phase_on")]:
            p = asst[key]
            if p is None:
                continue
            lines.append(
                f"  {tag} ({p['steps']:4d} steps): agent RMS "
                f"A {p['agent_rms']['A_noassist']:6.2f} / "
                f"B {p['agent_rms']['B_assisted']:6.2f} "
                f"({-p['agent_rms_reduction_pct']:+6.1f}%) | "
                f"|power| A {p['agent_power_absmean']['A_noassist']:6.2f} / "
                f"B {p['agent_power_absmean']['B_assisted']:6.2f} "
                f"({-p['agent_power_reduction_pct']:+6.1f}%) | "
                f"err6 A {p['dof_err6_mean']['A_noassist']:.4f} / "
                f"B {p['dof_err6_mean']['B_assisted']:.4f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def _add_episode_lines(ax, episodes, t_end):
    for e in episodes:
        if 0 < e["start"] < t_end:
            ax.axvline(e["start"], color="gray", lw=0.8, ls=":", alpha=0.7)


def save_plots(out_dir: Path, tr, episodes, ch_names, zoom_steps: int,
               residual_pd_scale: float = 0.0, assist_steps=None):
    T = tr["tau_b_cmd"].shape[0]
    # 보조 지연 투입 모드: 꺼짐 구간의 MF cmd(NaN)는 0으로 그려 "보조 없음"을 명시
    b_cmd = np.nan_to_num(tr["tau_b_cmd"], nan=0.0) if assist_steps else tr["tau_b_cmd"]

    def _add_assist_lines(ax, t_end, first_label=False):
        for si, s in enumerate(assist_steps or []):
            if s < t_end:
                ax.axvline(s, color="blue", lw=1.5, alpha=0.9,
                           label="assist ON" if (first_label and si == 0) else None)
    for tag, t_end in [("full", T), ("zoom", min(zoom_steps, T))]:
        # 1) 채널별 토크: B 인가 vs A 적용 (+A 수동 예측 점선)
        fig, axes = plt.subplots(3, 2, figsize=(16, 9), sharex=True)
        t_axis = np.arange(t_end)
        for j, ax in enumerate(axes.flat):
            ax.plot(t_axis, tr["tau_a_applied"][:t_end, j], color="black", lw=1.0,
                    label="A: applied (RL+PD)")
            ax.plot(t_axis, b_cmd[:t_end, j], color="tab:red", lw=1.0,
                    alpha=0.85, label="B: ManiFlow cmd (applied)")
            if residual_pd_scale > 0:
                ax.plot(t_axis,
                        b_cmd[:t_end, j] + tr["tau_b_qfrc"][:t_end, j],
                        color="tab:green", lw=0.9, alpha=0.8,
                        label=f"B: total (MF+{residual_pd_scale:g}·PD)")
            ax.plot(t_axis, tr["pred_a_passive"][:t_end, j], color="tab:orange",
                    lw=0.8, ls="--", alpha=0.6, label="A: ManiFlow passive pred")
            _add_episode_lines(ax, episodes, t_end)
            _add_assist_lines(ax, t_end, first_label=(j == 0))
            ax.set_title(ch_names[j], fontsize=10)
            if j == 0:
                ax.legend(fontsize=8, loc="upper right")
        fig.suptitle("Torque on estimator channels — A applied vs B commanded "
                     f"({tag})")
        fig.supxlabel("policy step (20Hz)")
        fig.supylabel("torque [N·m]")
        fig.tight_layout()
        fig.savefig(out_dir / f"torque_channels_{tag}.png", dpi=120)
        plt.close(fig)

        # 2) 트래킹: 관절 오차 / 루트 높이 / A-B 발산
        dof_err6 = np.abs(
            tr["dof_pos"][:t_end][:, :, ESTIMATOR_DOF_INDICES]
            - tr["ref_dof_pos"][:t_end][:, None, ESTIMATOR_DOF_INDICES]
        ).mean(axis=-1)
        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        axes[0].plot(t_axis, dof_err6[:, GHOST_ENV], color="black", lw=1.0,
                     label="A: pure RL")
        axes[0].plot(t_axis, dof_err6[:, MANIFLOW_ENV], color="tab:red", lw=1.0,
                     label="B: RL+ManiFlow")
        axes[0].set_ylabel("mean |dof err| 6ch [rad]")
        axes[0].legend(fontsize=8)
        axes[1].plot(t_axis, tr["root_pos"][:t_end, GHOST_ENV, 2], color="black",
                     lw=1.0, label="A root z")
        axes[1].plot(t_axis, tr["root_pos"][:t_end, MANIFLOW_ENV, 2], color="tab:red",
                     lw=1.0, label="B root z")
        axes[1].set_ylabel("root height [m]")
        axes[1].legend(fontsize=8)
        div = np.linalg.norm(
            tr["root_pos"][:t_end, GHOST_ENV, :2]
            - tr["root_pos"][:t_end, MANIFLOW_ENV, :2], axis=-1)
        axes[2].plot(t_axis, div, color="tab:blue", lw=1.0)
        axes[2].set_ylabel("A↔B root XY dist [m]")
        for ax in axes:
            _add_episode_lines(ax, episodes, t_end)
            _add_assist_lines(ax, t_end)
        fig.suptitle(f"Tracking vs reference & A/B divergence ({tag})")
        fig.supxlabel("policy step (20Hz)")
        fig.tight_layout()
        fig.savefig(out_dir / f"tracking_{tag}.png", dpi=120)
        plt.close(fig)


def _rolling_stat(x, w: int, rms: bool = True):
    """(T, D) → (T,) 폭 w 이동 RMS(또는 평균). 앞 구간은 가능한 만큼만 평균."""
    v = np.square(x).mean(axis=-1) if rms else np.asarray(x, np.float64)
    c = np.cumsum(np.insert(v, 0, 0.0))
    T = len(v)
    hi = np.arange(T) + 1
    lo = np.maximum(hi - w, 0)
    out = (c[hi] - c[lo]) / (hi - lo)
    return np.sqrt(out) if rms else out


def _rolling_rms_channels(x, w: int):
    """(T, Da) → (T, Da) 채널별 폭 w 이동 RMS (진폭 포락선)."""
    x = np.asarray(x)
    return np.stack([_rolling_stat(x[:, j:j + 1], w) for j in range(x.shape[1])],
                    axis=-1)


def save_assist_plots(out_dir: Path, tr, episodes, ch_names, zoom_steps: int,
                      beta: float, assist_steps=None, assist_metrics=None,
                      fps: int = 20):
    """가산 보조 모드 전용 플롯 — 에이전트 토크가 실제로 줄었는지 보여준다."""
    T = tr["tau_a_mean"].shape[0]
    exo = np.nan_to_num(tr["tau_b_cmd"], nan=0.0)  # β배·클램프된 주입 토크
    a_ag, b_ag = tr["tau_a_mean"], tr["tau_b_mean"]
    b_tot = b_ag + exo

    def _lines(ax, t_end, first=False):
        _add_episode_lines(ax, episodes, t_end)
        for si, s in enumerate(assist_steps or []):
            if s < t_end:
                ax.axvline(s, color="blue", lw=1.5, alpha=0.9,
                           label="assist ON" if (first and si == 0) else None)

    # 1) 채널별: A 에이전트 토크 vs B 에이전트 토크 vs 보조 토크 vs 총합
    for tag, t_end in [("full", T), ("zoom", min(zoom_steps, T))]:
        fig, axes = plt.subplots(3, 2, figsize=(16, 9), sharex=True)
        t_axis = np.arange(t_end)
        for j, ax in enumerate(axes.flat):
            ax.plot(t_axis, a_ag[:t_end, j], color="black", lw=1.0,
                    label="A: agent torque (no assist)")
            ax.plot(t_axis, b_ag[:t_end, j], color="tab:red", lw=1.0,
                    label="B: agent torque (assisted)")
            ax.plot(t_axis, exo[:t_end, j], color="tab:green", lw=1.0,
                    alpha=0.85, label=f"B: exo assist ({beta:g}·ManiFlow)")
            ax.plot(t_axis, b_tot[:t_end, j], color="0.55", lw=0.8, ls="--",
                    alpha=0.9, label="B: total (agent + exo)")
            _lines(ax, t_end, first=(j == 0))
            ax.set_title(ch_names[j], fontsize=10)
            if j == 0:
                ax.legend(fontsize=8, loc="upper right")
        fig.suptitle(f"Additive assist beta={beta:g} — agent torque offload "
                     f"(substep-mean qfrc, {tag})")
        fig.supxlabel("policy step (20Hz)")
        fig.supylabel("torque [N·m]")
        fig.tight_layout()
        fig.savefig(out_dir / f"assist_offload_{tag}.png", dpi=120)
        plt.close(fig)

    # 2) 이동 RMS 추이 — 보조 투입 후 에이전트 토크가 내려가는지 한눈에
    w = max(2, int(round(2.0 * fps)))  # 2초 창
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    t_axis = np.arange(T)
    axes[0].plot(t_axis, _rolling_stat(a_ag, w), color="black", lw=1.4,
                 label="A: agent torque RMS (no assist)")
    axes[0].plot(t_axis, _rolling_stat(b_ag, w), color="tab:red", lw=1.4,
                 label="B: agent torque RMS (assisted)")
    axes[0].plot(t_axis, _rolling_stat(exo, w), color="tab:green", lw=1.2,
                 alpha=0.85, label=f"B: exo assist RMS ({beta:g}·MF)")
    axes[0].plot(t_axis, _rolling_stat(b_tot, w), color="0.55", lw=1.0, ls="--",
                 label="B: total RMS (agent + exo)")
    axes[0].set_ylabel(f"6ch torque RMS [N·m] ({w / fps:g}s window)")
    axes[0].legend(fontsize=8, loc="upper right")
    err6 = np.abs(
        tr["dof_pos"][:, :, ESTIMATOR_DOF_INDICES]
        - tr["ref_dof_pos"][:, None, ESTIMATOR_DOF_INDICES]
    ).mean(axis=-1)
    axes[1].plot(t_axis, _rolling_stat(err6[:, GHOST_ENV], w, rms=False),
                 color="black", lw=1.4, label="A: dof err6 (no assist)")
    axes[1].plot(t_axis, _rolling_stat(err6[:, MANIFLOW_ENV], w, rms=False),
                 color="tab:red", lw=1.4, label="B: dof err6 (assisted)")
    axes[1].set_ylabel("mean |dof err| 6ch [rad]")
    axes[1].legend(fontsize=8, loc="upper right")
    for ax in axes:
        _lines(ax, T)
    fig.suptitle(f"Additive assist beta={beta:g} — agent torque & tracking error "
                 "vs assist injection")
    fig.supxlabel("policy step (20Hz)")
    fig.tight_layout()
    fig.savefig(out_dir / "assist_rms_trace.png", dpi=120)
    plt.close(fig)

    # 3) 채널별 RMS 막대 요약
    if assist_metrics is not None:
        per = assist_metrics["per_channel"]
        x = np.arange(len(ch_names))
        wid = 0.2
        fig, ax = plt.subplots(figsize=(12, 5))
        for k, (key, color, lbl) in enumerate([
            ("A_agent_rms", "black", "A: agent (no assist)"),
            ("B_agent_rms", "tab:red", "B: agent (assisted)"),
            ("exo_rms", "tab:green", f"B: exo ({beta:g}·MF)"),
            ("B_total_rms", "0.6", "B: total"),
        ]):
            ax.bar(x + (k - 1.5) * wid, [per[n][key] for n in ch_names],
                   wid, color=color, label=lbl)
        for i, n in enumerate(ch_names):
            ax.text(i, max(per[n]["A_agent_rms"], per[n]["B_total_rms"]) * 1.02,
                    f"{-per[n]['reduction_pct']:.0f}%", ha="center", fontsize=9,
                    color="tab:red")
        ax.set_xticks(x)
        ax.set_xticklabels(ch_names, fontsize=9)
        ax.set_ylabel("torque RMS [N·m]")
        o = assist_metrics["overall"]
        ax.set_title(f"Additive assist beta={beta:g} — per-channel torque RMS "
                     f"(total -{o['agent_rms_reduction_pct']:.1f}%, "
                     f"offload eff {o['offload_efficiency']:.2f}, "
                     f"power -{o['agent_power_reduction_pct']:.1f}%)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "assist_summary.png", dpi=120)
        plt.close(fig)


def compose_torque_video(sim_mp4, tr, out_path, ch_names, fps=20, window_s=8.0,
                         event_times=None, fill_zero=False, curves=None,
                         side_mp4=None):
    """녹화 영상 옆에 6채널 토크 스크롤 플롯을 붙인 mp4.

    event_times: 보조 투입 시점[s] 목록 — 파란 세로선으로 표시.
    fill_zero: MF cmd의 NaN(보조 꺼짐 구간)을 0으로 그려 "보조 없음"을 명시.
    curves: [(array(T, Da), color, label), ...] — 미지정 시 기본
        (A 적용 토크, B ManiFlow 인가 토크). 가산 보조 모드에서는 호출자가
        A/B 에이전트 토크와 보조 토크를 넘겨 "능동적 감소"를 보이게 한다.
    side_mp4: 정측면 뷰 mp4 — 주어지면 정면 위 / 측면 아래로 세로 결합하고
        토크 패널 폭도 그에 맞춰 키운다.
    """
    import imageio.v2 as imageio

    if curves is None:
        b = tr["tau_b_cmd"]
        if fill_zero:
            b = np.nan_to_num(b, nan=0.0)
        curves = [(tr["tau_a_applied"], "black", "A applied"),
                  (b, "tab:red", "B ManiFlow")]
    T = curves[0][0].shape[0]
    # 뷰별 reader — 여러 개면 세로로 이어붙인다 (정면 위 / 정측면 아래).
    paths = [sim_mp4] + [p for p in (side_mp4,) if p and Path(p).exists()]
    readers = [imageio.get_reader(str(p)) for p in paths]
    counts = [r.count_frames() for r in readers]
    # 뷰마다 프레임 수가 1 정도 다를 수 있어(녹화 시작/종료 프레임) 개별 정렬
    offsets = [max(0, c - T) for c in counts]
    n_out = min(min(c - o for c, o in zip(counts, offsets)), T)
    if any(o not in (0, 1) for o in offsets):
        log.warning(f"프레임 수{counts}와 스텝 수({T}) 불일치 — offsets={offsets}")

    def _stack(k):
        frames = [r.get_data(k + o) for r, o in zip(readers, offsets)]
        w = frames[0].shape[1]
        for i, f in enumerate(frames[1:], start=1):
            if f.shape[1] != w:  # 뷰 폭이 다르면 첫 뷰 폭으로 맞춤
                from PIL import Image
                h = round(f.shape[0] * w / f.shape[1])
                frames[i] = np.asarray(Image.fromarray(f).resize((w, h)))
        return np.vstack(frames) if len(frames) > 1 else frames[0]

    H = _stack(0).shape[0]
    dpi = 100
    plot_w = max(720, int(round(H * 2 / 3)))  # 뷰가 늘면 패널도 같이 키움
    fs = plot_w / 720  # 폰트 스케일
    fig, axes = plt.subplots(3, 2, figsize=(plot_w / dpi, H / dpi), dpi=dpi,
                             sharex=True)
    tt = np.arange(T) / fps
    cursors = []
    for j, ax in enumerate(axes.flat):
        for y, color, label in curves:
            ax.plot(tt, y[:, j], color=color, lw=1.0 * fs, alpha=0.9, label=label)
        for ei, et in enumerate(event_times or []):
            ax.axvline(et, color="blue", lw=1.8 * fs, alpha=0.9,
                       label="assist ON" if (j == 0 and ei == 0) else None)
        lo, hi = np.nanpercentile(
            np.concatenate([y[:, j] for y, _c, _l in curves]), [1, 99])
        pad = 0.15 * max(hi - lo, 1e-3)
        ax.set_ylim(lo - pad, hi + pad)
        cursors.append(ax.axvline(0.0, color="0.45", lw=1.0 * fs, ls="--"))
        ax.set_title(ch_names[j], fontsize=9 * fs)
        ax.tick_params(labelsize=7 * fs)
        if j == 0:
            ax.legend(fontsize=7 * fs, loc="upper right")
    fig.supxlabel("time [s]", fontsize=9 * fs)
    fig.supylabel("torque [N·m]", fontsize=9 * fs)
    fig.tight_layout()

    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264",
                                pixelformat="yuv420p", macro_block_size=2)
    for k in range(n_out):
        t_now = k / fps
        for ax in axes.flat:
            ax.set_xlim(max(0.0, t_now - window_s), max(window_s, t_now + 0.5))
        for cur in cursors:
            cur.set_xdata([t_now, t_now])
        fig.canvas.draw()
        plot_rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        if plot_rgb.shape[0] != H:
            from PIL import Image
            plot_rgb = np.asarray(Image.fromarray(plot_rgb).resize(
                (plot_rgb.shape[1], H)))
        writer.append_data(np.hstack([_stack(k), plot_rgb]))
        if (k + 1) % 200 == 0:
            log.info(f"  합성 {k + 1}/{n_out} 프레임")
    writer.close()
    for r in readers:
        r.close()
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    args = _args
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    assert args.assist_beta >= 0.0, "--assist-beta는 0 이상"
    assert not (args.assist_beta > 0 and args.residual_pd_scale > 0), (
        "--assist-beta(가산 보조: PD 무감쇠 + β·MF)와 --residual-pd-scale"
        "(convex 블렌드: α·PD + (1-α)·MF)는 서로 다른 결합 방식이라 "
        "동시 사용할 수 없습니다")

    if args.record and not args.viewer:
        log.info("--record: 프레임 캡처에 GL 컨텍스트가 필요해 뷰어를 자동 활성화합니다.")
        args.viewer = True
    if args.viewer and not args.no_mesh:
        args.overrides = list(args.overrides) + [
            "robot.asset.asset_file_name=mjcf/skeleton_torque_suit_mesh.xml"
        ]
        log.info("skeleton mesh 에셋으로 시각화합니다 (--no-mesh로 비활성화).")

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.output or f"{TASK_ROOT}/maniflow_control_results/{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    fabric_config = FabricConfig(accelerator="gpu", devices=1, num_nodes=1,
                                 loggers=[], callbacks=[])
    fabric: Fabric = Fabric(**asdict(fabric_config))
    fabric.launch()

    agent, env = setup_agent_and_env(args, fabric)
    sim = env.simulator

    # ManiFlow 추정기 로드
    maniflow_ckpt = args.maniflow_ckpt or discover_best_checkpoint(args.maniflow_run_dir)
    policy, _mf_cfg, mf_info = load_maniflow_policy(
        maniflow_ckpt, device=str(fabric.device), maniflow_root=args.maniflow_root
    )
    estimator = ManiFlowTorqueEstimator(policy, num_envs=env.num_envs,
                                        device=fabric.device)
    assert args.denoise_steps >= 1
    if args.denoise_steps != policy.num_inference_steps:
        log.info(f"denoise(ODE) steps: {policy.num_inference_steps}(ckpt 설정) -> "
                 f"{args.denoise_steps}")
        policy.num_inference_steps = args.denoise_steps

    # Agent B: estimator 채널(순수 hip 6 DOF)을 ManiFlow 토크로 구동
    dof_names = list(env.robot_config.kinematic_info.dof_names)
    ESTIMATOR_DOF_INDICES[:] = hip_dof_indices(dof_names)
    override = JointTorqueOverride(
        sim,
        env_ids=[MANIFLOW_ENV],
        common_dof_indices=ESTIMATOR_DOF_INDICES,
    )
    if args.assist_beta > 0:
        override.engage(1.0)  # PD 게인 무변경 — 순수 가산 주입
        log.info(f"가산 보조 모드: hip 총토크 = τ_agent(full PD) + "
                 f"{args.assist_beta:g}·ManiFlow — 보조로 트래킹 오차가 줄어 "
                 "에이전트 PD 토크가 능동적으로 감소하는지 측정")
    else:
        override.engage(args.residual_pd_scale)
        if args.residual_pd_scale > 0:
            log.info(f"잔여 PD 결합: hip 토크 = {args.residual_pd_scale:g}·PD + "
                     f"{1.0 - args.residual_pd_scale:g}·ManiFlow (convex 블렌드)")
    log.info(f"ManiFlow ckpt: {maniflow_ckpt} (epoch={mf_info['epoch']}, "
             f"n_obs_steps={estimator.n_obs_steps}, "
             f"n_action_steps={estimator.n_action_steps})")
    log.info(f"Override channels (hips, COMMON DOF "
             f"{ESTIMATOR_DOF_INDICES}): {override.dof_names}")
    log.info(f"Torque limits: {override.torque_limits.cpu().numpy()}")

    fixed_xy = pin_spawn_location(env)
    log.info(f"고정 스폰 위치: ({fixed_xy[0]:.2f}, {fixed_xy[1]:.2f}) — "
             "두 agent가 겹쳐 스폰됩니다 (Newton world 분리로 상호 충돌 없음)")

    viewer = getattr(sim, "viewer", None)
    if args.viewer and viewer is not None:
        viewer.show_ui = True
        GhostSkeletonRenderer(sim, env_id=GHOST_ENV, alpha=args.ghost_alpha,
                              line_width=args.ghost_line_width)
        sim._camera_target = {"env": GHOST_ENV, "element": 0}
        log.info("Agent A = 반투명 고스트 스켈레톤(라인), Agent B = 메시로 표시합니다.")
    policy_fps = round(1.0 / sim.dt) if getattr(sim, "dt", 0) > 0 else 20
    side_recorder = None
    side_mp4 = None
    if args.record:
        sim._user_recording_video_path = str(out_dir / "sim-%s")
        sim._toggle_video_record()
    # 정측면 뷰: 녹화 시 mp4로도 저장, 뷰어만 켠 경우엔 창 안 패널로만 표시
    if args.record_side and (args.record or args.viewer) and viewer is not None:
        side_mp4 = (out_dir / "sim_side.mp4") if args.record else None
        side_recorder = SideViewRecorder(sim, side_mp4, fps=policy_fps,
                                        delta_deg=args.side_azimuth)
        print(f"정측면 뷰 활성화 (정면 대비 {args.side_azimuth:g}°, 스텝마다 "
              "렌더 1회 추가) — 뷰어 창에는 "
              + ("'side view (sagittal)' 패널로 표시"
                 if side_recorder.panel else "표시 불가(log_image 없음)")
              + (f", 영상 {side_mp4.name} 저장" if side_mp4 else ""))

    print(f"\nA/B rollout 시작: {args.episode_steps} steps | "
          f"predict_mode={args.predict_mode} | chunk_offset={args.chunk_offset} | "
          f"torque_scale={args.torque_scale} | "
          + (f"assist_beta={args.assist_beta}" if args.assist_beta > 0
             else f"residual_pd_scale={args.residual_pd_scale}")
          + (" | recording" if args.record else "")
          + (" (+side view)" if side_recorder is not None else ""))
    tr, episodes = run_rollout(agent, env, estimator, override, args,
                               side_recorder=side_recorder)

    if side_recorder is not None:
        side_recorder.close()
    if args.record:
        sim._toggle_video_record()
        sim.render()

    # ── 저장 ────────────────────────────────────────────────────────────
    ch_names = override.dof_names
    m = compute_metrics(tr, episodes, ch_names, fall_z=args.fall_z,
                        assist_beta=args.assist_beta,
                        assist_start_steps=args.assist_start_steps)
    text = format_metrics_text(m, ch_names,
                               residual_pd_scale=args.residual_pd_scale)
    if args.assist_beta > 0:
        b_desc = f"RL+PD(full) + {args.assist_beta:g}·ManiFlow (가산 보조)"
    else:
        b_desc = (f"RL+{args.residual_pd_scale:g}·PD + "
                  f"{1.0 - args.residual_pd_scale:g}·ManiFlow"
                  if args.residual_pd_scale > 0 else "RL+PD + ManiFlow torque")
    print(f"\n=== A (pure RL+PD, ghost) vs B ({b_desc}, solid) ===")
    print(text + "\n")

    np.savez_compressed(out_dir / "traces.npz", **tr,
                        episode_starts=np.array([e["start"] for e in episodes]),
                        episode_ends=np.array([e["end"] for e in episodes]))

    result = {
        "rl_checkpoint": str(args.rl_checkpoint),
        "maniflow_ckpt": str(maniflow_ckpt),
        "maniflow_epoch": mf_info["epoch"],
        "predict_mode": args.predict_mode,
        "denoise_steps": int(policy.num_inference_steps),
        "chunk_offset": args.chunk_offset,
        "torque_scale": args.torque_scale,
        "residual_pd_scale": args.residual_pd_scale,
        "assist_beta": args.assist_beta,
        "handover_steps": args.handover_steps,
        "assist_start_steps": args.assist_start_steps,
        "episode_steps": int(tr["tau_b_cmd"].shape[0]),
        "ghost_env": GHOST_ENV,
        "maniflow_env": MANIFLOW_ENV,
        "action_dofs": "hips",
        "action_dof_indices": list(ESTIMATOR_DOF_INDICES),
        "override_channels": ch_names,
        "spawn_xy": [float(fixed_xy[0]), float(fixed_xy[1])],
        "metrics": m,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(out_dir / "metrics.txt", "w") as f:
        f.write(f"rl ckpt:  {args.rl_checkpoint}\n"
                f"mf ckpt:  {maniflow_ckpt} (epoch {mf_info['epoch']})\n"
                f"predict:  {args.predict_mode} (denoise_steps="
                f"{policy.num_inference_steps}, "
                f"chunk_offset={args.chunk_offset}, "
                f"torque_scale={args.torque_scale}, "
                f"residual_pd_scale={args.residual_pd_scale}, "
                f"assist_beta={args.assist_beta}, "
                f"handover_steps={args.handover_steps}, "
                f"assist_start_steps={args.assist_start_steps})\n"
                f"channels: {ch_names}\n\n" + text + "\n")

    assist_steps = None
    if args.assist_start_steps > 0:
        T_tr = tr["tau_b_cmd"].shape[0]
        assist_steps = [e["start"] + args.assist_start_steps for e in episodes
                        if e["start"] + args.assist_start_steps < min(e["end"], T_tr)]
    save_plots(out_dir, tr, episodes, ch_names, args.zoom_steps,
               residual_pd_scale=args.residual_pd_scale,
               assist_steps=assist_steps)
    if args.assist_beta > 0:
        save_assist_plots(out_dir, tr, episodes, ch_names, args.zoom_steps,
                          args.assist_beta, assist_steps=assist_steps,
                          assist_metrics=m.get("assist"), fps=policy_fps)

    # ── 녹화 영상 + 토크 패널 합성 ──────────────────────────────────────
    if args.record:
        rec_name = getattr(sim, "_curr_user_recording_name", None)
        sim_mp4 = Path(rec_name) / f"{Path(rec_name).name}.mp4" if rec_name else None
        if sim_mp4 is None or not sim_mp4.exists():
            log.warning("녹화 mp4가 없어 합성 비디오를 건너뜁니다.")
        else:
            out_mp4 = out_dir / "sim_with_torque.mp4"
            log.info(f"토크 합성 비디오 생성 중... ({sim_mp4.name} + traces)")
            # 가산 보조 모드: 영상 패널에 A/B 에이전트 토크와 보조 토크를 그려
            # "보조가 들어오면 에이전트 토크가 줄어든다"를 직접 보이게 한다.
            curves = None
            if args.assist_beta > 0:
                exo = np.nan_to_num(tr["tau_b_cmd"], nan=0.0)
                curves = [
                    (tr["tau_a_mean"], "black", "A agent (no assist)"),
                    (tr["tau_b_mean"], "tab:red", "B agent (assisted)"),
                    (exo, "tab:green", f"B exo ({args.assist_beta:g}·MF)"),
                ]
            event_times = ([s / policy_fps for s in assist_steps]
                           if assist_steps else None)
            compose_torque_video(
                sim_mp4, tr, out_mp4, ch_names, fps=policy_fps,
                event_times=event_times,
                fill_zero=args.assist_start_steps > 0, curves=curves,
                side_mp4=side_mp4)
            print(f"시뮬 영상(정면):  {sim_mp4}")
            if side_mp4 is not None and Path(side_mp4).exists():
                print(f"시뮬 영상(측면):  {side_mp4}")
            print(f"토크 합성 비디오: {out_mp4}")
            if args.assist_beta > 0:
                # 원파형은 20Hz 진동이 커 진폭 감소가 잘 안 보인다 — 채널별
                # 2초 이동 RMS 포락선 버전을 추가로 합성 (감소가 한눈에 보임).
                w = max(2, int(round(2.0 * policy_fps)))
                rms_mp4 = out_dir / "sim_with_torque_rms.mp4"
                log.info(f"이동 RMS 포락선 비디오 생성 중... (창 {w / policy_fps:g}s)")
                compose_torque_video(
                    sim_mp4, tr, rms_mp4, ch_names, fps=policy_fps,
                    event_times=event_times, side_mp4=side_mp4,
                    curves=[
                        (_rolling_rms_channels(tr["tau_a_mean"], w), "black",
                         "A agent RMS (no assist)"),
                        (_rolling_rms_channels(tr["tau_b_mean"], w), "tab:red",
                         "B agent RMS (assisted)"),
                        (_rolling_rms_channels(
                            np.nan_to_num(tr["tau_b_cmd"], nan=0.0), w),
                         "tab:green", f"B exo RMS ({args.assist_beta:g}·MF)"),
                    ])
                print(f"RMS 포락선 비디오: {rms_mp4}")

    print(f"결과 저장: {out_dir}")


if __name__ == "__main__":
    main()
