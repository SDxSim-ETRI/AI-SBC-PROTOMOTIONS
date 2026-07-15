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
  sim-*/sim-*.mp4            : (--record) 시뮬 녹화 (고스트 A + 실체 B)
  sim_with_torque.mp4        : (--record) 영상 + 토크 패널 합성 비디오

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
    "walking_flat-maniflow_lowdim_policy_walking-newton-hips-run02_seed42",
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
    p.add_argument("--handover-steps", type=int, default=0,
                   help="에피소드 시작 후 이 스텝 수 동안 B도 순수 RL+PD로 "
                        "보행(워밍업)한 뒤 ManiFlow 토크 제어로 전환. 리셋 직후 "
                        "관측(s0)이 학습 분포 밖이라 첫 chunk가 어긋나는 문제를 "
                        "우회 (0 = 리셋 직후부터 ManiFlow 제어)")
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


# ---------------------------------------------------------------------------
@torch.no_grad()
def run_rollout(agent, env, estimator, override, args):
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
        "pred_a_passive": np.zeros((T, Da), np.float32),  # A 수동 예측(chunk 동일 원소)
        "dof_pos": np.zeros((T, N, n_dofs), np.float32),
        "ref_dof_pos": np.zeros((T, n_dofs), np.float32),
        "root_pos": np.zeros((T, N, 3), np.float32),
        "rew": np.zeros((T, N), np.float32),
    }
    episodes = []  # dict(start, end, cause)
    ep_start = 0
    fall_count = np.zeros(N, dtype=int)  # 연속 저고도 스텝 수 (자체 넘어짐 감지)

    handover = max(0, args.handover_steps)
    obs = synced_reset(env, estimator, override)
    if handover > 0:
        override.restore()  # 워밍업: B도 순수 PD 보행, chunk는 핸드오버 시점에 생성
        chunk = None
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
        if chunk is None and ep_rel >= handover:
            # 워밍업 종료: 실제 히스토리(2프레임)로 첫 chunk를 만들고 제어 전환
            override.engage(alpha)
            chunk = estimator.predict()
            chunk_pos = offset
            log.info(f"  handover @ step {t} (ep {len(episodes) + 1}): "
                     f"B를 ManiFlow 토크 제어로 전환 (α={alpha:g})")

        if chunk is not None:
            cur = chunk[:, chunk_pos]  # (N, Da) — 이번 전이(t→t+1)용 토크
            tau_b = override.set_torques(
                cur[MANIFLOW_ENV : MANIFLOW_ENV + 1] * (args.torque_scale * blend)
            )
            tr["tau_b_cmd"][t] = tau_b[0].cpu().numpy()
            tr["pred_a_passive"][t] = cur[GHOST_ENV].cpu().numpy()
        else:
            # 워밍업 구간: B는 PD 보행 중 — ManiFlow 미사용 (지표에서 제외)
            tr["tau_b_cmd"][t] = np.nan
            tr["pred_a_passive"][t] = np.nan

        obs, rewards, dones, _terminated, _extras = env.step(action)
        robot_state = env.simulator.get_robot_state()
        estimator.observe(robot_state)

        dof_forces = robot_state.dof_forces[:, ESTIMATOR_DOF_INDICES].cpu().numpy()
        tr["tau_a_applied"][t] = dof_forces[GHOST_ENV]
        tr["tau_b_qfrc"][t] = dof_forces[MANIFLOW_ENV]
        tr["dof_pos"][t] = robot_state.dof_pos.cpu().numpy()
        tr["root_pos"][t] = robot_state.rigid_body_pos[:, 0].cpu().numpy()
        tr["rew"][t] = rewards.cpu().numpy()
        mm = env.motion_manager
        ref_state = env.motion_lib.get_motion_state(mm.motion_ids, mm.motion_times)
        tr["ref_dof_pos"][t] = ref_state.dof_pos[0].cpu().numpy()

        if not gain_check_done and chunk is not None and ep_rel == handover + 2:
            leak = float(np.abs(tr["tau_b_qfrc"][t]).max())
            if alpha > 0:
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
                    if alpha > 0:  # 총토크 = MF 주입분 + 잔여 PD 몫
                        viewer.log_scalar(
                            f"torque/{ch_names[j]}/B_total",
                            float(tr["tau_b_cmd"][t, j] + tr["tau_b_qfrc"][t, j]))
                viewer.log_scalar(f"torque/{ch_names[j]}/A_applied",
                                  float(tr["tau_a_applied"][t, j]))
            err = np.abs(tr["dof_pos"][t][:, ESTIMATOR_DOF_INDICES]
                         - tr["ref_dof_pos"][t][None, ESTIMATOR_DOF_INDICES])
            viewer.log_scalar("tracking/dof_err6/A_pureRL", float(err[GHOST_ENV].mean()))
            viewer.log_scalar("tracking/dof_err6/B_maniflow", float(err[MANIFLOW_ENV].mean()))
        if viewer is not None and t % 20 == 0:
            b_mode = (f"{alpha:g}·PD+{blend:g}·ManiFlow" if alpha > 0
                      else "ManiFlow torque")
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
def compute_metrics(tr, episodes, ch_names, fall_z: float = 0.5):
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
    def _corr(x, y):
        with np.errstate(invalid="ignore"):
            if len(x) > 1 and x.std() > 0 and y.std() > 0:
                return float(np.corrcoef(x, y)[0, 1])
        return float("nan")

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
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def _add_episode_lines(ax, episodes, t_end):
    for e in episodes:
        if 0 < e["start"] < t_end:
            ax.axvline(e["start"], color="gray", lw=0.8, ls=":", alpha=0.7)


def save_plots(out_dir: Path, tr, episodes, ch_names, zoom_steps: int,
               residual_pd_scale: float = 0.0):
    T = tr["tau_b_cmd"].shape[0]
    for tag, t_end in [("full", T), ("zoom", min(zoom_steps, T))]:
        # 1) 채널별 토크: B 인가 vs A 적용 (+A 수동 예측 점선)
        fig, axes = plt.subplots(3, 2, figsize=(16, 9), sharex=True)
        t_axis = np.arange(t_end)
        for j, ax in enumerate(axes.flat):
            ax.plot(t_axis, tr["tau_a_applied"][:t_end, j], color="black", lw=1.0,
                    label="A: applied (RL+PD)")
            ax.plot(t_axis, tr["tau_b_cmd"][:t_end, j], color="tab:red", lw=1.0,
                    alpha=0.85, label="B: ManiFlow cmd (applied)")
            if residual_pd_scale > 0:
                ax.plot(t_axis,
                        tr["tau_b_cmd"][:t_end, j] + tr["tau_b_qfrc"][:t_end, j],
                        color="tab:green", lw=0.9, alpha=0.8,
                        label=f"B: total (MF+{residual_pd_scale:g}·PD)")
            ax.plot(t_axis, tr["pred_a_passive"][:t_end, j], color="tab:orange",
                    lw=0.8, ls="--", alpha=0.6, label="A: ManiFlow passive pred")
            _add_episode_lines(ax, episodes, t_end)
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
        fig.suptitle(f"Tracking vs reference & A/B divergence ({tag})")
        fig.supxlabel("policy step (20Hz)")
        fig.tight_layout()
        fig.savefig(out_dir / f"tracking_{tag}.png", dpi=120)
        plt.close(fig)


def compose_torque_video(sim_mp4, tr, out_path, ch_names, fps=20, window_s=8.0):
    """녹화 영상 옆에 6채널 토크(B 인가 vs A 적용) 스크롤 플롯을 붙인 mp4."""
    import imageio.v2 as imageio

    a, b = tr["tau_a_applied"], tr["tau_b_cmd"]
    T = a.shape[0]
    reader = imageio.get_reader(str(sim_mp4))
    n_frames = reader.count_frames()
    offset = max(0, n_frames - T)
    if offset not in (0, 1):
        log.warning(f"프레임 수({n_frames})와 스텝 수({T}) 불일치 — offset={offset}")

    first = reader.get_data(0)
    H = first.shape[0]
    dpi = 100
    plot_w = 720
    fig, axes = plt.subplots(3, 2, figsize=(plot_w / dpi, H / dpi), dpi=dpi,
                             sharex=True)
    tt = np.arange(T) / fps
    cursors = []
    for j, ax in enumerate(axes.flat):
        ax.plot(tt, a[:, j], color="black", lw=1.0, label="A applied")
        ax.plot(tt, b[:, j], color="tab:red", lw=1.0, alpha=0.85, label="B ManiFlow")
        lo, hi = np.nanpercentile(np.concatenate([a[:, j], b[:, j]]), [1, 99])
        pad = 0.15 * max(hi - lo, 1e-3)
        ax.set_ylim(lo - pad, hi + pad)
        cursors.append(ax.axvline(0.0, color="tab:blue", lw=1.2))
        ax.set_title(ch_names[j], fontsize=9)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.legend(fontsize=7, loc="upper right")
    fig.supxlabel("time [s]", fontsize=9)
    fig.supylabel("torque [N·m]", fontsize=9)
    fig.tight_layout()

    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264",
                                pixelformat="yuv420p", macro_block_size=2)
    for k in range(min(n_frames - offset, T)):
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
        sim_frame = reader.get_data(k + offset)
        writer.append_data(np.hstack([sim_frame, plot_rgb]))
        if (k + 1) % 200 == 0:
            log.info(f"  합성 {k + 1}/{min(n_frames - offset, T)} 프레임")
    writer.close()
    reader.close()
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    args = _args
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
    if args.record:
        sim._user_recording_video_path = str(out_dir / "sim-%s")
        sim._toggle_video_record()

    print(f"\nA/B rollout 시작: {args.episode_steps} steps | "
          f"predict_mode={args.predict_mode} | chunk_offset={args.chunk_offset} | "
          f"torque_scale={args.torque_scale} | "
          f"residual_pd_scale={args.residual_pd_scale}"
          + (" | recording" if args.record else ""))
    tr, episodes = run_rollout(agent, env, estimator, override, args)

    if args.record:
        sim._toggle_video_record()
        sim.render()

    # ── 저장 ────────────────────────────────────────────────────────────
    ch_names = override.dof_names
    m = compute_metrics(tr, episodes, ch_names, fall_z=args.fall_z)
    text = format_metrics_text(m, ch_names,
                               residual_pd_scale=args.residual_pd_scale)
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
        "handover_steps": args.handover_steps,
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
                f"handover_steps={args.handover_steps})\n"
                f"channels: {ch_names}\n\n" + text + "\n")

    save_plots(out_dir, tr, episodes, ch_names, args.zoom_steps,
               residual_pd_scale=args.residual_pd_scale)

    # ── 녹화 영상 + 토크 패널 합성 ──────────────────────────────────────
    if args.record:
        rec_name = getattr(sim, "_curr_user_recording_name", None)
        sim_mp4 = Path(rec_name) / f"{Path(rec_name).name}.mp4" if rec_name else None
        if sim_mp4 is None or not sim_mp4.exists():
            log.warning("녹화 mp4가 없어 합성 비디오를 건너뜁니다.")
        else:
            policy_fps = (round(1.0 / sim.dt) if getattr(sim, "dt", 0) > 0 else 20)
            out_mp4 = out_dir / "sim_with_torque.mp4"
            log.info(f"토크 합성 비디오 생성 중... ({sim_mp4.name} + traces)")
            compose_torque_video(sim_mp4, tr, out_mp4, ch_names, fps=policy_fps)
            print(f"시뮬 영상:        {sim_mp4}")
            print(f"토크 합성 비디오: {out_mp4}")

    print(f"결과 저장: {out_dir}")


if __name__ == "__main__":
    main()
