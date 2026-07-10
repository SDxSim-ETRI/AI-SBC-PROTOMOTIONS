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
"""Walk episode zarr collector (ManiFlow hip-torque 학습 데이터).

정책 추론으로 walk 에피소드를 수집하고 zarr 포맷으로 저장합니다.
기본 시뮬레이터는 Newton입니다 (RL 보행 체크포인트가 Newton에서 학습됨 —
inference 도메인과 일치시켜 sim2sim 갭을 제거).

저장 데이터 (에피소드당 T=1200 steps, 1분 @ 20Hz):
  hip_torque  (N, T,  6) : 순수 hip 6개 DOF 적용 토크 [N·m]
                           = [hip_flexion_r, hip_adduction_r, hip_rotation_r,
                              hip_flexion_l, hip_adduction_l, hip_rotation_l]
                           = 공통 DOF [0, 1, 2, 5, 6, 7] (이름 기반 파생 —
                           protomotions.maniflow.channels.hip_dof_indices).
                           ⚠️ 과거 수집본은 slice(0,6)을 잘못 사용해 오른다리
                           전체+왼 hip flexion(knee/ankle 포함)을 기록했음.
                           attrs의 action_dof_names/action_dof_indices로 구분.
  dof_pos     (N, T, 27) : 전체 관절 위치 [rad]
  dof_vel     (N, T, 27) : 전체 관절 속도 [rad/s]
  actions     (N, T, 27) : 정책 출력 위치 타겟 (정규화됨)
  root_pos    (N, T,  3) : pelvis 월드 위치 [m]
  root_vel    (N, T,  3) : pelvis 월드 선속도 [m/s]
  contacts    (N, T, 28) : rigid body 접촉 플래그 (bool)

에피소드 필터링:
  inference resolved config에는 termination 컴포넌트가 없어 env done만으로는
  낙상을 걸러낼 수 없습니다 (과거 수집본에 root z가 -4000 m대까지 추락한
  에피소드가 172/1000개 섞여 normalizer를 오염시킨 원인). 그래서 수집기가
  자체적으로 낙상을 감지합니다: root 높이 < --fall-z 가 --fall-hold 스텝 연속
  유지되거나 상태에 비유한값(NaN/Inf)이 나오면 해당 env의 에피소드를 버리고
  리셋합니다. env done(있다면)도 기존대로 실패 처리.

출력 경로:
  zarr_data/{terrain}/{terrain}-{simulator}-YYYY-MM-DD-HH-MM-SS.zarr

지원 terrain 프리셋 (비평지는 IsaacLab 전용 — Newton은 flat만):
  단일: flat, smooth_slope, rough_slope, stairs_up, stairs_down,
        discrete, stepping, poles

실행:
  bash tasks/mimic_suit_active_cable_walk_23dof/collect_walk_zarr.sh
  bash tasks/mimic_suit_active_cable_walk_23dof/collect_walk_zarr.sh \
      tasks/mimic_suit_active_cable_walk_23dof/output_newton_flat/score_based.ckpt flat 64
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
# IsaacLab 사용 시 argparse 직후, torch 임포트 전에 호출해야 함
# ---------------------------------------------------------------------------
import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

TASK_ROOT = "tasks/mimic_suit_active_cable_walk_23dof"
DEFAULT_CKPT = f"{TASK_ROOT}/output_newton_flat/score_based.ckpt"


def _create_parser():
    p = argparse.ArgumentParser(
        description="Collect walk episodes to zarr",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", default=DEFAULT_CKPT,
                   help="보행 RL policy 체크포인트 (기본: Newton 학습본)")
    p.add_argument(
        "--output", default=None,
        help="출력 경로 (미지정 시 zarr_data/{terrain}/{terrain}-{simulator}-"
             "{timestamp}.zarr 자동 생성)",
    )
    p.add_argument(
        "--motion-file",
        default="data/motion_for_trackers/skeleton_torque_suit_walk.pt",
    )
    p.add_argument(
        "--simulator",
        default="newton",
        choices=["newton", "isaaclab"],
        help="수집 시뮬레이터. RL ckpt 학습 도메인과 일치시킬 것 "
             "(output_newton_flat → newton)",
    )
    p.add_argument(
        "--terrain",
        default="flat",
        choices=list(TERRAIN_PRESETS.keys()),
        help="terrain 종류 (비평지는 IsaacLab 전용)",
    )
    p.add_argument("--num-envs",        type=int, default=10)
    p.add_argument("--target-episodes", type=int, default=1000)
    p.add_argument("--episode-steps",   type=int, default=1200,
                   help="1200 = 60s @ 20Hz (fps=120/decimation=6)")
    p.add_argument("--fall-z",    type=float, default=0.5,
                   help="root 높이가 이 값[m] 아래로 --fall-hold 스텝 연속이면 "
                        "낙상으로 판정해 에피소드 폐기 (<=0 비활성)")
    p.add_argument("--fall-hold", type=int, default=10,
                   help="낙상 판정에 필요한 연속 스텝 수 (10 = 0.5s @ 20Hz)")
    p.add_argument("--overrides", nargs="*", default=[])
    return p


_parser = _create_parser()
_args, _ = _parser.parse_known_args()

if _args.terrain != "flat" and _args.simulator == "newton":
    _parser.error("Newton은 비평지 terrain을 지원하지 않습니다 — "
                  "--simulator isaaclab을 사용하세요.")

# 시뮬레이터(isaaclab)를 torch보다 먼저 임포트
from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch(_args.simulator)

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
from protomotions.maniflow.channels import HIP_DOF_NAMES, hip_dof_indices  # noqa: E402
from lightning.fabric import Fabric  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 데이터 필드 정의  (key, dim, dtype)
# ---------------------------------------------------------------------------
FIELDS = [
    ("hip_torque",  6,  "float32"),   # 순수 hip 6 DOF (공통 [0,1,2,5,6,7])
    ("dof_pos",    27,  "float32"),
    ("dof_vel",    27,  "float32"),
    ("actions",    27,  "float32"),
    ("root_pos",    3,  "float32"),   # pelvis world pos
    ("root_vel",    3,  "float32"),   # pelvis world lin vel
    ("contacts",   28,  "bool"),
]

ROOT_BODY_IDX = 0  # pelvis


def _resolve_output_path(args) -> str:
    """args.output 미지정 시 zarr_data/{terrain}/{terrain}-{sim}-{ts}.zarr 반환."""
    if args.output is not None:
        return args.output
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    terrain = args.terrain
    return f"{TASK_ROOT}/zarr_data/{terrain}/{terrain}-{args.simulator}-{ts}.zarr"


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

    current_sim = simulator_config._target_.split(".")[-3]
    if current_sim != args.simulator:
        log.info(f"Switching simulator: {current_sim} -> {args.simulator}")
        from protomotions.simulator.factory import update_simulator_config_for_test
        simulator_config = update_simulator_config_for_test(
            current_simulator_config=simulator_config,
            new_simulator=args.simulator,
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

    simulator_extra_params = {}
    if app_launcher is not None:
        simulator_extra_params["simulation_app"] = app_launcher.app

    from protomotions.utils.component_builder import build_all_components
    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=fabric.device,
        save_dir=getattr(env_config, "save_dir", None),
        **simulator_extra_params,
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
def init_zarr(output_path: str, target: int, T: int, args,
              dof_names: list, action_dof_indices: list) -> zarr.Group:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    # zarr v2 포맷 고정: 학습 파이프라인(ManiFlow_Policy, zarr 2.x)이 읽어야 함
    # (zarr-python 3은 기본으로 v3 포맷을 쓰므로 명시 필요; zarr 2에선 인자 없음)
    open_kwargs = {"zarr_format": 2} if zarr.__version__.startswith("3") else {}
    store = zarr.open(output_path, mode="w", **open_kwargs)
    for name, dim, dtype in FIELDS:
        store.create_dataset(name, shape=(target, T, dim), dtype=dtype,
                             chunks=(1, T, dim), fill_value=0)
    store.attrs["episode_steps"]        = T
    store.attrs["policy_hz"]            = 20
    store.attrs["simulator"]            = args.simulator
    store.attrs["terrain"]              = args.terrain
    store.attrs["terrain_proportions"]  = TERRAIN_PRESETS[args.terrain]
    store.attrs["checkpoint"]           = str(args.checkpoint)
    store.attrs["fall_z"]               = args.fall_z
    store.attrs["fall_hold"]            = args.fall_hold
    # 실제 공통 DOF 순서(kinematic tree 순서)와 action 채널 매핑을 기록.
    # hip_torque 필드는 순수 hip 6 DOF = 공통 [0,1,2,5,6,7] (이름 기반 파생).
    # 과거 수집본(action_dof_indices attr 없음)은 slice(0,6)을 잘못 기록했음.
    store.attrs["dof_order"] = " ".join(dof_names)
    store.attrs["action_dof_names"]   = [dof_names[i] for i in action_dof_indices]
    store.attrs["action_dof_indices"] = list(action_dof_indices)
    return store


# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_episodes(agent, env, store: zarr.Group, target: int, T: int,
                     action_dof_indices: list, fall_z: float, fall_hold: int,
                     output_path: str = ""):
    device    = env.device
    N         = env.num_envs
    collected = 0
    batch_idx = 0
    discarded = 0

    action_idx_th = torch.tensor(action_dof_indices, dtype=torch.long, device=device)
    bufs = {name: np.zeros((N, T, dim), dtype=dtype) for name, dim, dtype in FIELDS}

    print(f"\n수집 시작: target={target} episodes | {N} envs | T={T} steps "
          f"| fall filter z<{fall_z} x{fall_hold}")
    t_start = time.time()

    while collected < target:
        batch_idx += 1
        obs, _       = env.reset()
        env_failed   = np.zeros(N, dtype=bool)
        reset_indices = None
        fall_count   = torch.zeros(N, dtype=torch.long, device=device)

        for t in range(T):
            if reset_indices is not None and len(reset_indices) > 0:
                env_failed[reset_indices.cpu().numpy()] = True
                obs, _ = env.reset(reset_indices)
                fall_count[reset_indices] = 0

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
                    robot_state.dof_forces[active_th][:, action_idx_th].cpu().numpy()
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

            # ── 자체 낙상/비유한값 감지 (inference config에 termination이 없어
            #    env done만으로는 낙상을 못 거름 — root z 쓰레기값 방지) ──────
            root_pos_now = robot_state.rigid_body_pos[:, ROOT_BODY_IDX]
            bad = dones.clone()
            if fall_z > 0:
                fall_count = torch.where(
                    root_pos_now[:, 2] < fall_z,
                    fall_count + 1,
                    torch.zeros_like(fall_count),
                )
                bad |= fall_count >= fall_hold
            bad |= ~torch.isfinite(root_pos_now).all(dim=-1)
            bad |= ~torch.isfinite(robot_state.dof_pos).all(dim=-1)
            reset_indices = bad.nonzero(as_tuple=False).flatten()

        # 배치 종료 시점 처리: 마지막 스텝의 done/낙상 + 판정 중이던 저고도
        # 스트릭(홀드 미충족 낙상 꼬리)도 실패로 간주해 꼬리 오염을 방지
        if reset_indices is not None and len(reset_indices) > 0:
            env_failed[reset_indices.cpu().numpy()] = True
        env_failed |= (fall_count > 0).cpu().numpy()

        for e in range(N):
            if not env_failed[e] and collected < target:
                for name, _, _ in FIELDS:
                    store[name][collected] = bufs[name][e]
                collected += 1
        discarded += int(env_failed.sum())

        success = int((~env_failed).sum())
        elapsed = time.time() - t_start
        eta     = (elapsed / max(collected, 1)) * (target - collected)
        print(
            f"  batch {batch_idx:4d} | ok {success}/{N} "
            f"| total {collected:4d}/{target} "
            f"| {elapsed/60:.1f}m elapsed | ETA {eta/60:.1f}m"
        )

    store.attrs["collected"] = collected
    store.attrs["discarded_episodes"] = discarded
    print(f"\n완료: {collected} 에피소드 (폐기 {discarded}) → {output_path}")


# ---------------------------------------------------------------------------
def main():
    args = _args

    output_path = _resolve_output_path(args)
    log.info(f"simulator: {args.simulator}  |  terrain: {args.terrain}  |  "
             f"출력: {output_path}")

    # inference_agent.py와 동일한 순서: Fabric 먼저, 필요 시 AppLauncher
    fabric_config = FabricConfig(accelerator="gpu", devices=1, num_nodes=1,
                                 loggers=[], callbacks=[])
    fabric: Fabric = Fabric(**asdict(fabric_config))
    fabric.launch()

    app_launcher = None
    if args.simulator == "isaaclab":
        app_launcher = AppLauncher({"headless": True, "device": str(fabric.device)})
        import carb
        carb.settings.get_settings().set(
            "/persistent/physics/visualizationSimulationOutput", False
        )

    agent, env = setup_agent_and_env(args, fabric, app_launcher)

    dof_names = list(env.robot_config.kinematic_info.dof_names)
    action_dof_indices = hip_dof_indices(dof_names)
    log.info(f"action 채널 (hip 6 DOF): indices={action_dof_indices} "
             f"names={[dof_names[i] for i in action_dof_indices]}")
    assert [dof_names[i] for i in action_dof_indices] == HIP_DOF_NAMES

    store = init_zarr(output_path, args.target_episodes, args.episode_steps,
                      args, dof_names=dof_names,
                      action_dof_indices=action_dof_indices)
    collect_episodes(agent, env, store, args.target_episodes, args.episode_steps,
                     action_dof_indices=action_dof_indices,
                     fall_z=args.fall_z, fall_hold=args.fall_hold,
                     output_path=output_path)


if __name__ == "__main__":
    main()
