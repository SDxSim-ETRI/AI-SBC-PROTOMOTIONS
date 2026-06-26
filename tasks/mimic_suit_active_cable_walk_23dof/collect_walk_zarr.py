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
"""Walk episode zarr collector.

정책 추론으로 walk 에피소드를 수집하고 zarr 포맷으로 저장합니다.
낙상/경계이탈(done=True)이 발생한 에피소드는 제외합니다.

저장 데이터 (에피소드당 T=1200 steps, 1분 @ 20Hz):
  hip_torque  (N, T,  6) : DOF 0-5 적용 토크 [N·m]
  dof_pos     (N, T, 27) : 전체 관절 위치 [rad]
  dof_vel     (N, T, 27) : 전체 관절 속도 [rad/s]
  actions     (N, T, 27) : 정책 출력 위치 타겟 (정규화됨)
  root_pos    (N, T,  3) : pelvis 월드 위치 [m]
  root_vel    (N, T,  3) : pelvis 월드 선속도 [m/s]
  contacts    (N, T, 28) : rigid body 접촉 플래그 (bool)

출력 경로:
  zarr_data/{terrain}/{terrain}-YYYY-MM-DD-HH-MM-SS.zarr

지원 terrain 프리셋:
  단일: flat, smooth_slope, rough_slope, stairs_up, stairs_down,
        discrete, stepping, poles
  복합: slope_discrete, mixed

실행:
  cd /home/user/ProtoMotions
  bash tasks/mimic_suit_active_cable_walk_23dof/collect_walk_zarr.sh
  bash tasks/mimic_suit_active_cable_walk_23dof/collect_walk_zarr.sh score_based.ckpt flat
  bash tasks/mimic_suit_active_cable_walk_23dof/collect_walk_zarr.sh score_based.ckpt discrete
"""

# ---------------------------------------------------------------------------
# terrain 프리셋 정의 (argparse 전에 선언)
# 인덱스 순서: [smooth_slope, rough_slope, stairs_up, stairs_down,
#               discrete, stepping, poles, flat]
# ---------------------------------------------------------------------------
TERRAIN_PRESETS = {
    # [smooth_slope, rough_slope, stairs_up, stairs_down, discrete, stepping, poles, flat]
    "flat":         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    "smooth_slope": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "rough_slope":  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "stairs_up":    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "stairs_down":  [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "discrete":     [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "stepping":     [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    "poles":        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
}

# ---------------------------------------------------------------------------
# IsaacLab은 argparse 직후, torch 임포트 전에 호출해야 함
# ---------------------------------------------------------------------------
import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _create_parser():
    p = argparse.ArgumentParser(
        description="Collect walk episodes to zarr",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--output", default=None,
        help="출력 경로 (미지정 시 zarr_data/{terrain}/{terrain}-{timestamp}.zarr 자동 생성)",
    )
    p.add_argument(
        "--motion-file",
        default="data/motion_for_trackers/skeleton_torque_suit_walk.pt",
    )
    p.add_argument(
        "--terrain",
        default="flat",
        choices=list(TERRAIN_PRESETS.keys()),
        help="terrain 종류 (프리셋에서 선택)",
    )
    p.add_argument("--num-envs",        type=int, default=10)
    p.add_argument("--target-episodes", type=int, default=1000)
    p.add_argument("--episode-steps",   type=int, default=1200,
                   help="1200 = 60s @ 20Hz (fps=120/decimation=6)")
    p.add_argument("--overrides", nargs="*", default=[])
    return p


_parser = _create_parser()
_args, _ = _parser.parse_known_args()

# IsaacLab을 torch보다 먼저 임포트
from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch("isaaclab")

# 이후 torch 및 나머지 임포트 안전
import logging  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict  # noqa: E402
from datetime import datetime  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402

from protomotions.utils.hydra_replacement import get_class  # noqa: E402
from protomotions.utils.fabric_config import FabricConfig  # noqa: E402
from lightning.fabric import Fabric  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 데이터 필드 정의  (key, dim, dtype)
# ---------------------------------------------------------------------------
FIELDS = [
    ("hip_torque",  6,  "float32"),   # DOF 0-5
    ("dof_pos",    27,  "float32"),
    ("dof_vel",    27,  "float32"),
    ("actions",    27,  "float32"),
    ("root_pos",    3,  "float32"),   # pelvis world pos
    ("root_vel",    3,  "float32"),   # pelvis world lin vel
    ("contacts",   28,  "bool"),
]

HIP_DOF_SLICE = slice(0, 6)
ROOT_BODY_IDX = 0  # pelvis

TASK_ROOT = "tasks/mimic_suit_active_cable_walk_23dof"


def _resolve_output_path(args) -> str:
    """args.output 미지정 시 zarr_data/{terrain}/{terrain}-{timestamp}.zarr 반환."""
    if args.output is not None:
        return args.output
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    terrain = args.terrain
    return f"{TASK_ROOT}/zarr_data/{terrain}/{terrain}-{ts}.zarr"


def _terrain_overrides(terrain: str) -> list:
    """terrain 프리셋에 해당하는 config override 문자열 목록 반환."""
    if terrain == "flat":
        return []  # 기본 TerrainConfig는 이미 flat
    proportions = TERRAIN_PRESETS[terrain]
    prop_str = "[" + ",".join(f"{v:.1f}" for v in proportions) + "]"
    return [f"terrain.terrain_proportions={prop_str}"]


# ---------------------------------------------------------------------------
def setup_agent_and_env(args, fabric: Fabric, app_launcher):
    """inference_agent.py와 동일한 순서로 초기화합니다."""
    checkpoint    = Path(args.checkpoint)
    resolved_path = checkpoint.parent / "resolved_configs_inference.pt"
    assert resolved_path.exists(), f"Not found: {resolved_path}"

    resolved = torch.load(resolved_path, map_location="cpu", weights_only=False)

    robot_config      = resolved["robot"]
    simulator_config  = resolved["simulator"]
    terrain_config    = resolved.get("terrain")
    scene_lib_config  = resolved["scene_lib"]
    motion_lib_config = resolved["motion_lib"]
    env_config        = resolved["env"]
    agent_config      = resolved["agent"]

    # simulator 고정 (항상 isaaclab)
    current_sim = simulator_config._target_.split(".")[-3]
    if current_sim != "isaaclab":
        from protomotions.simulator.factory import update_simulator_config_for_test
        simulator_config = update_simulator_config_for_test(
            current_simulator_config=simulator_config,
            new_simulator="isaaclab",
            robot_config=robot_config,
        )

    from protomotions.utils.inference_utils import apply_backward_compatibility_fixes
    apply_backward_compatibility_fixes(robot_config, simulator_config, env_config)

    simulator_config.num_envs     = args.num_envs
    simulator_config.headless     = True
    motion_lib_config.motion_file = args.motion_file

    # terrain override: 사용자 --overrides + terrain 프리셋 자동 추가
    effective_overrides = list(args.overrides) + _terrain_overrides(args.terrain)
    if effective_overrides:
        from protomotions.utils.config_utils import parse_cli_overrides, apply_config_overrides
        apply_config_overrides(
            parse_cli_overrides(effective_overrides),
            env_config, simulator_config, robot_config,
            agent_config, terrain_config, motion_lib_config, scene_lib_config,
        )

    # friction 변환
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
        simulation_app=app_launcher.app,
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
def init_zarr(output_path: str, target: int, T: int, terrain: str, checkpoint: str) -> zarr.Group:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    store = zarr.open(output_path, mode="w")
    for name, dim, dtype in FIELDS:
        store.create_dataset(name, shape=(target, T, dim), dtype=dtype,
                             chunks=(1, T, dim), fill_value=0)
    store.attrs["episode_steps"]        = T
    store.attrs["policy_hz"]            = 20
    store.attrs["terrain"]              = terrain
    store.attrs["terrain_proportions"]  = TERRAIN_PRESETS[terrain]
    store.attrs["checkpoint"]           = str(checkpoint)
    store.attrs["dof_order"] = (
        "hip_flexion_r hip_adduction_r hip_rotation_r "
        "hip_flexion_l hip_adduction_l hip_rotation_l "
        "lumbar_extension lumbar_bending lumbar_rotation "
        "knee_angle_r knee_angle_l "
        "arm_flex_r arm_add_r arm_rot_r arm_flex_l arm_add_l arm_rot_l "
        "slide1 slide2 slide3 slide4 "
        "ankle_angle_r ankle_angle_l "
        "elbow_flex_r elbow_flex_l pro_sup_r pro_sup_l"
    )
    return store


# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_episodes(agent, env, store: zarr.Group, target: int, T: int):
    device    = env.device
    N         = env.num_envs
    collected = 0
    batch_idx = 0

    bufs = {name: np.zeros((N, T, dim), dtype=dtype) for name, dim, dtype in FIELDS}

    print(f"\n수집 시작: target={target} episodes | {N} envs | T={T} steps")
    t_start = time.time()

    while collected < target:
        batch_idx += 1
        obs, _      = env.reset()
        env_failed  = np.zeros(N, dtype=bool)
        done_indices = None

        for t in range(T):
            if done_indices is not None and len(done_indices) > 0:
                env_failed[done_indices.cpu().numpy()] = True
                obs, _ = env.reset(done_indices)

            obs       = agent.add_agent_info_to_obs(obs)
            obs_td    = agent.obs_dict_to_tensordict(obs)
            model_out = agent.model(obs_td)
            action    = model_out.get("mean_action", model_out["action"])

            obs, _rewards, dones, _terminated, _extras = env.step(action)
            robot_state = env.simulator.get_robot_state()

            active_np = ~env_failed
            if active_np.any():
                active_th = torch.from_numpy(active_np).to(device)
                bufs["hip_torque"][active_np, t] = (
                    robot_state.dof_forces[active_th, HIP_DOF_SLICE].cpu().numpy()
                )
                bufs["dof_pos"][active_np, t] = (
                    robot_state.dof_pos[active_th].cpu().numpy()
                )
                bufs["dof_vel"][active_np, t] = (
                    robot_state.dof_vel[active_th].cpu().numpy()
                )
                bufs["actions"][active_np, t] = action[active_th].cpu().numpy()
                bufs["root_pos"][active_np, t] = (
                    robot_state.rigid_body_pos[active_th, ROOT_BODY_IDX].cpu().numpy()
                )
                bufs["root_vel"][active_np, t] = (
                    robot_state.rigid_body_vel[active_th, ROOT_BODY_IDX].cpu().numpy()
                )
                bufs["contacts"][active_np, t] = (
                    robot_state.rigid_body_contacts[active_th].cpu().numpy()
                )

            done_indices = dones.nonzero(as_tuple=False).flatten()

        if done_indices is not None and len(done_indices) > 0:
            env_failed[done_indices.cpu().numpy()] = True

        for e in range(N):
            if not env_failed[e] and collected < target:
                for name, _, _ in FIELDS:
                    store[name][collected] = bufs[name][e]
                collected += 1

        success = int((~env_failed).sum())
        elapsed = time.time() - t_start
        eta     = (elapsed / max(collected, 1)) * (target - collected)
        print(
            f"  batch {batch_idx:4d} | ok {success}/{N} "
            f"| total {collected:4d}/{target} "
            f"| {elapsed/60:.1f}m elapsed | ETA {eta/60:.1f}m"
        )

    store.attrs["collected"] = collected
    print(f"\n완료: {collected} 에피소드 → {store.store.path}")


# ---------------------------------------------------------------------------
def main():
    args = _args

    output_path = _resolve_output_path(args)
    log.info(f"terrain: {args.terrain}  |  출력: {output_path}")

    # inference_agent.py와 동일한 순서: Fabric 먼저, 그 다음 AppLauncher
    fabric_config = FabricConfig(accelerator="gpu", devices=1, num_nodes=1,
                                 loggers=[], callbacks=[])
    fabric: Fabric = Fabric(**asdict(fabric_config))
    fabric.launch()

    app_launcher = AppLauncher({"headless": True, "device": str(fabric.device)})

    import carb
    carb.settings.get_settings().set(
        "/persistent/physics/visualizationSimulationOutput", False
    )

    agent, env = setup_agent_and_env(args, fabric, app_launcher)
    store = init_zarr(output_path, args.target_episodes, args.episode_steps,
                      args.terrain, args.checkpoint)
    collect_episodes(agent, env, store, args.target_episodes, args.episode_steps)


if __name__ == "__main__":
    main()
