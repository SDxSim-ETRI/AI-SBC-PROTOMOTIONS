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
"""DAgger/DART식 walk 에피소드 zarr 수집기 (ManiFlow hip-torque 재학습용, Newton 전용).

기존 `collect_walk_zarr.py`(순수 RL+PD 궤적 = 보행 매니폴드 위 데이터)를 보완해,
**매니폴드에서 살짝 이탈한 상태 + 그 상태에서 전문가(full-gain PD)가 내는 교정
토크** 쌍을 수집합니다. 목적: ManiFlow가 off-manifold 복원 토크까지 배우게 해
잔여 PD 결합(`compare_maniflow_control_newton.py --residual-pd-scale α`)의
생존 문턱(현재 α ∈ (0.25, 0.375])을 낮추는 것.

두 가지 상태 교란 모드 (조합 가능):

1. **외란 주입 (DART식, `--perturb-scale c > 0`)**: 전 관절 full-gain PD를
   유지한 채(α=1) hip 6채널에 ZOH 랜덤 토크 외란(채널별 σ = c×기준 std,
   `--perturb-hold` 스텝 유지)을 `qfrc_applied`로 가산 주입. 배포 시 ManiFlow
   ZOH 토크 오차와 같은 구조의 교란이며, PD가 매 substep 반작용해 상태가
   매니폴드 주변 튜브를 훑음.
2. **잔여 PD 블렌드 on-policy (`--residual-pd-scale α < 1`)**: hip 게인을 α배로
   낮추고 ManiFlow 예측 토크를 (1-α)배 주입한 채(배포와 동일한 하이브리드)
   수집 — estimator 자신이 만든 상태 분포(리셋 직후 첫 chunk 오예측이 PD에
   흡수되는 과도 구간 포함)를 데이터에 넣는 DAgger 본체.

라벨 규약 (hip_torque 필드):
  label = qfrc_actuator readback / α   (α=1이면 readback 그대로 = 기존 수집기와
  동일), effort limit으로 클램프. readback은 α·(implicit PD 응답)이고 주입
  토크(qfrc_applied)는 포함되지 않으므로, 나누기 α는 "이 상태에서 full-gain
  PD(전문가)가 냈을 토크"의 근사 질의(DAgger expert query)입니다. 외란/MF
  주입분은 상태를 만든 원인일 뿐 라벨에 들어가지 않습니다.

정렬은 기존 수집기와 동일: env.step() 직후의 robot_state에서 obs[t]와
hip_torque[t]를 같은 시점에 기록 (에피소드 첫 프레임 = s₁). 필드 레이아웃도
동일 + 디버그용 hip_dist / hip_mf_cmd 추가 (변환 스크립트는 무시).

에피소드 필터링(자체 낙상/비유한값 감지·폐기)도 기존 수집기와 동일 — 외란이
과해 넘어진 에피소드는 데이터에 남지 않습니다.

실행 (기본: run02 estimator, Newton):
  # DART: full PD + hip 외란 (σ = 0.3×기준 std, 150ms hold)
  bash tasks/mimic_suit_active_cable_walk_23dof/collect_walk_zarr_dagger.sh \
      dart 384 --perturb-scale 0.3
  # DAgger on-policy: α=0.5 블렌드 + 약한 외란
  bash tasks/mimic_suit_active_cable_walk_23dof/collect_walk_zarr_dagger.sh \
      blend050 192 --residual-pd-scale 0.5 --perturb-scale 0.15

출력: zarr_data/flat/flat-newton-dagger-{tag}-{timestamp}.zarr (zarr v2)
"""

# ---------------------------------------------------------------------------
# argparse를 먼저 수행 (simulator import 순서 규약)
# ---------------------------------------------------------------------------
import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

TASK_ROOT = "tasks/mimic_suit_active_cable_walk_23dof"
DEFAULT_CKPT = f"{TASK_ROOT}/output_newton_flat/score_based.ckpt"
DEFAULT_MANIFLOW_RUN_DIR = os.path.join(
    str(Path.home()),
    "Projects/ManiFlow_Policy/ManiFlow/data/outputs",
    "walking_flat-maniflow_lowdim_policy_walking-newton-hips-run02_seed42",
)

# 채널별 기준 토크 std [N·m] — 2026-07-14 잔여 PD 스윕의 Agent A(순수 RL+PD)
# 실측 (HIP_DOF_NAMES 순서: flex_r, add_r, rot_r, flex_l, add_l, rot_l).
# 외란 σ = --perturb-scale × 이 값.
HIP_TORQUE_STD_REF = [64.0, 50.0, 17.0, 57.0, 52.0, 18.0]


def _create_parser():
    p = argparse.ArgumentParser(
        description="Collect DAgger/DART walk episodes to zarr (Newton)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", default=DEFAULT_CKPT,
                   help="보행 RL policy 체크포인트 (Newton 학습본)")
    p.add_argument("--tag", default="dagger",
                   help="출력 파일명 태그 (flat-newton-dagger-{tag}-{ts}.zarr)")
    p.add_argument("--output", default=None,
                   help="출력 경로 (미지정 시 zarr_data/flat/에 자동 생성)")
    p.add_argument("--motion-file",
                   default="data/motion_for_trackers/skeleton_torque_suit_walk.pt")
    p.add_argument("--num-envs",        type=int, default=64)
    p.add_argument("--target-episodes", type=int, default=256)
    p.add_argument("--episode-steps",   type=int, default=1200,
                   help="1200 = 60s @ 20Hz (fps=120/decimation=6)")
    # ── 상태 교란 모드 ────────────────────────────────────────────────────
    p.add_argument("--residual-pd-scale", type=float, default=1.0,
                   help="hip 채널에 남길 PD 게인 비율 α. 1.0(기본)=full PD"
                        "(DART 모드 — --perturb-scale 필요). α<1이면 ManiFlow "
                        "estimator가 (1-α)배 토크를 주입하는 배포 동일 블렌드로 "
                        "수집(DAgger on-policy). 라벨은 항상 readback/α")
    p.add_argument("--perturb-scale", type=float, default=0.0,
                   help="hip ZOH 토크 외란의 채널별 σ 배율 (σ = 이 값 × 기준 "
                        "std). 0 = 외란 없음")
    p.add_argument("--perturb-hold", type=int, default=3,
                   help="외란 값을 유지할 control 스텝 수 (3 = 150ms — 배포 시 "
                        "ManiFlow chunk 주기와 유사한 ZOH 구조)")
    p.add_argument("--perturb-clamp", type=float, default=3.0,
                   help="외란을 ±(이 값)×σ로 클램프")
    # ── ManiFlow estimator (α<1일 때 사용) ───────────────────────────────
    p.add_argument("--maniflow-ckpt", default=None,
                   help="ManiFlow 체크포인트 경로 (미지정 시 run-dir에서 best "
                        "topk 자동)")
    p.add_argument("--maniflow-run-dir", default=DEFAULT_MANIFLOW_RUN_DIR)
    p.add_argument("--maniflow-root", default=None)
    p.add_argument("--chunk-offset", type=int, default=1, choices=[0, 1],
                   help="receding chunk에서 제어에 쓸 시작 인덱스 (배포 기본 1)")
    # ── 낙상 필터 (collect_walk_zarr.py와 동일) ──────────────────────────
    p.add_argument("--fall-z",    type=float, default=0.5)
    p.add_argument("--fall-hold", type=int, default=10)
    p.add_argument("--overrides", nargs="*", default=[])
    return p


_parser = _create_parser()
_args, _ = _parser.parse_known_args()

if _args.residual_pd_scale >= 1.0 and _args.perturb_scale <= 0.0:
    _parser.error("α=1(full PD)이고 외란도 없으면 기존 collect_walk_zarr.py와 "
                  "동일한 데이터입니다 — --perturb-scale 또는 "
                  "--residual-pd-scale을 지정하세요.")
if not (0.0 < _args.residual_pd_scale <= 1.0):
    _parser.error(f"--residual-pd-scale은 (0, 1] 범위여야 합니다 "
                  f"(라벨 = readback/α): {_args.residual_pd_scale}")

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

import_simulator_before_torch("newton")

import logging  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict  # noqa: E402
from datetime import datetime  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402

from protomotions.utils.hydra_replacement import get_class  # noqa: E402
from protomotions.utils.fabric_config import FabricConfig  # noqa: E402
from protomotions.maniflow import (  # noqa: E402
    HIP_DOF_NAMES,
    JointTorqueOverride,
    ManiFlowTorqueEstimator,
    discover_best_checkpoint,
    hip_dof_indices,
    load_maniflow_policy,
)
from lightning.fabric import Fabric  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 데이터 필드 정의 — collect_walk_zarr.py와 동일 + 디버그 2필드
# (hip_dist/hip_mf_cmd는 process_data_walking.py가 읽지 않는 부가 정보)
# ---------------------------------------------------------------------------
FIELDS = [
    ("hip_torque",  6,  "float32"),   # 라벨: full-PD 전문가 질의 (readback/α)
    ("dof_pos",    27,  "float32"),
    ("dof_vel",    27,  "float32"),
    ("actions",    27,  "float32"),
    ("root_pos",    3,  "float32"),
    ("root_vel",    3,  "float32"),
    ("contacts",   28,  "bool"),
    ("hip_dist",    6,  "float32"),   # 주입한 외란 (클램프 전 raw)
    ("hip_mf_cmd",  6,  "float32"),   # 주입한 (1-α)·ManiFlow 토크 (클램프 전)
]

ROOT_BODY_IDX = 0  # pelvis


def _resolve_output_path(args) -> str:
    if args.output is not None:
        return args.output
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return f"{TASK_ROOT}/zarr_data/flat/flat-newton-dagger-{args.tag}-{ts}.zarr"


# ---------------------------------------------------------------------------
def setup_agent_and_env(args, fabric: Fabric):
    """collect_walk_zarr.py와 동일한 순서로 초기화 (Newton 고정)."""
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

    simulator_config.num_envs     = args.num_envs
    simulator_config.headless     = True
    motion_lib_config.motion_file = args.motion_file

    if args.overrides:
        from protomotions.utils.config_utils import parse_cli_overrides, apply_config_overrides
        apply_config_overrides(
            parse_cli_overrides(list(args.overrides)),
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
    agent.load(str(args.checkpoint), load_env=False)

    return agent, env


# ---------------------------------------------------------------------------
def init_zarr(output_path: str, target: int, T: int, args,
              dof_names: list, action_dof_indices: list,
              maniflow_ckpt: str) -> zarr.Group:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    # zarr v2 포맷 고정 (학습 파이프라인 zarr 2.x 호환 — collect_walk_zarr.py 참고)
    open_kwargs = {"zarr_format": 2} if zarr.__version__.startswith("3") else {}
    store = zarr.open(output_path, mode="w", **open_kwargs)
    for name, dim, dtype in FIELDS:
        store.create_dataset(name, shape=(target, T, dim), dtype=dtype,
                             chunks=(1, T, dim), fill_value=0)
    store.attrs["episode_steps"]       = T
    store.attrs["policy_hz"]           = 20
    store.attrs["simulator"]           = "newton"
    store.attrs["terrain"]             = "flat"
    store.attrs["checkpoint"]          = str(args.checkpoint)
    store.attrs["fall_z"]              = args.fall_z
    store.attrs["fall_hold"]           = args.fall_hold
    store.attrs["dof_order"] = " ".join(dof_names)
    store.attrs["action_dof_names"]   = [dof_names[i] for i in action_dof_indices]
    store.attrs["action_dof_indices"] = list(action_dof_indices)
    # ── DAgger/DART 수집 메타 ────────────────────────────────────────────
    store.attrs["collection_mode"]     = ("blend" if args.residual_pd_scale < 1.0
                                          else "dart")
    store.attrs["residual_pd_scale"]   = args.residual_pd_scale
    store.attrs["label_rule"]          = "qfrc_actuator_readback/alpha (clamped)"
    store.attrs["perturb_scale"]       = args.perturb_scale
    store.attrs["perturb_hold_steps"]  = args.perturb_hold
    store.attrs["perturb_clamp_sigma"] = args.perturb_clamp
    store.attrs["perturb_std_ref"]     = list(HIP_TORQUE_STD_REF)
    store.attrs["maniflow_ckpt"]       = maniflow_ckpt or ""
    store.attrs["chunk_offset"]        = args.chunk_offset
    return store


# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_episodes(agent, env, store: zarr.Group, target: int, T: int,
                     action_dof_indices: list, args,
                     override: JointTorqueOverride,
                     estimator, output_path: str = ""):
    device    = env.device
    N         = env.num_envs
    collected = 0
    batch_idx = 0
    discarded = 0

    alpha  = args.residual_pd_scale
    blend  = 1.0 - alpha
    offset = args.chunk_offset
    Ta     = estimator.n_action_steps if estimator is not None else 0

    action_idx_th = torch.tensor(action_dof_indices, dtype=torch.long, device=device)
    torque_limits = override.torque_limits  # (6,) COMMON hip 채널 순서
    sigma = (torch.tensor(HIP_TORQUE_STD_REF, device=device, dtype=torch.float32)
             * args.perturb_scale)
    bufs = {name: np.zeros((N, T, dim), dtype=dtype) for name, dim, dtype in FIELDS}

    mode = "blend" if alpha < 1.0 else "dart"
    print(f"\nDAgger 수집 시작 [{mode}]: target={target} eps | {N} envs | T={T} "
          f"| α={alpha:g} | perturb σ×{args.perturb_scale:g} "
          f"hold={args.perturb_hold} | fall filter z<{args.fall_z}×{args.fall_hold}")
    t_start = time.time()

    zeros6 = torch.zeros(N, 6, device=device)

    while collected < target:
        batch_idx += 1
        obs, _        = env.reset()
        env_failed    = np.zeros(N, dtype=bool)
        reset_indices = None
        fall_count    = torch.zeros(N, dtype=torch.long, device=device)
        dist          = zeros6.clone()

        chunk, chunk_pos = None, offset
        if estimator is not None:
            # 배포(h0)와 동일: 리셋 상태 s₀로 히스토리 프라이밍 후 첫 chunk.
            # 첫 chunk 오예측은 잔여 PD가 흡수하며, 그 과도 상태가 바로
            # 수집하려는 on-policy 데이터임.
            estimator.reset()
            estimator.observe(env.simulator.get_robot_state())
            chunk = estimator.predict()

        for t in range(T):
            if reset_indices is not None and len(reset_indices) > 0:
                env_failed[reset_indices.cpu().numpy()] = True
                obs, _ = env.reset(reset_indices)
                fall_count[reset_indices] = 0
                if estimator is not None:
                    # 다음 observe에서 edge-padding으로 재프라이밍.
                    # (해당 env의 남은 chunk는 stale이지만 그 에피소드는 이미
                    #  폐기 대상 + set_torques가 effort limit으로 클램프)
                    estimator.reset(reset_indices)

            obs       = agent.add_agent_info_to_obs(obs)
            obs_td    = agent.obs_dict_to_tensordict(obs)
            model_out = agent.model(obs_td)
            action    = model_out.get("mean_action", model_out["action"])

            # ── hip 주입 토크 = (1-α)·ManiFlow + ZOH 외란 ────────────────
            if args.perturb_scale > 0 and t % args.perturb_hold == 0:
                dist = torch.clamp(
                    torch.randn(N, 6, device=device) * sigma,
                    -args.perturb_clamp * sigma, args.perturb_clamp * sigma,
                )
            mf_cmd = chunk[:, chunk_pos] * blend if chunk is not None else zeros6
            override.set_torques(mf_cmd + dist)

            obs, _rewards, dones, _terminated, _extras = env.step(action)
            robot_state = env.simulator.get_robot_state()

            active_np = ~env_failed
            if active_np.any():
                active_th = torch.from_numpy(active_np).to(device)
                # 전문가(full-gain PD) 질의: readback = α·PD_implicit
                label = robot_state.dof_forces[:, action_idx_th] / alpha
                label = torch.clamp(label, -torque_limits, torque_limits)
                bufs["hip_torque"][active_np, t] = label[active_th].cpu().numpy()
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
                bufs["hip_dist"][active_np, t] = dist[active_th].cpu().numpy()
                bufs["hip_mf_cmd"][active_np, t] = mf_cmd[active_th].cpu().numpy()

            if estimator is not None:
                estimator.observe(robot_state)
                chunk_pos += 1
                if chunk_pos >= Ta:
                    chunk = estimator.predict()
                    chunk_pos = offset

            # ── 자체 낙상/비유한값 감지 (collect_walk_zarr.py와 동일) ────
            root_pos_now = robot_state.rigid_body_pos[:, ROOT_BODY_IDX]
            bad = dones.clone()
            if args.fall_z > 0:
                fall_count = torch.where(
                    root_pos_now[:, 2] < args.fall_z,
                    fall_count + 1,
                    torch.zeros_like(fall_count),
                )
                bad |= fall_count >= args.fall_hold
            bad |= ~torch.isfinite(root_pos_now).all(dim=-1)
            bad |= ~torch.isfinite(robot_state.dof_pos).all(dim=-1)
            reset_indices = bad.nonzero(as_tuple=False).flatten()

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
            f"| {elapsed/60:.1f}m elapsed | ETA {eta/60:.1f}m",
            flush=True,
        )

    store.attrs["collected"] = collected
    store.attrs["discarded_episodes"] = discarded
    print(f"\n완료: {collected} 에피소드 (폐기 {discarded}) → {output_path}",
          flush=True)


# ---------------------------------------------------------------------------
def main():
    args = _args

    output_path = _resolve_output_path(args)
    log.info(f"mode: {'blend' if args.residual_pd_scale < 1.0 else 'dart'} "
             f"(α={args.residual_pd_scale:g}, perturb×{args.perturb_scale:g}) "
             f"| 출력: {output_path}")

    fabric_config = FabricConfig(accelerator="gpu", devices=1, num_nodes=1,
                                 loggers=[], callbacks=[])
    fabric: Fabric = Fabric(**asdict(fabric_config))
    fabric.launch()

    agent, env = setup_agent_and_env(args, fabric)

    dof_names = list(env.robot_config.kinematic_info.dof_names)
    action_dof_indices = hip_dof_indices(dof_names)
    assert [dof_names[i] for i in action_dof_indices] == HIP_DOF_NAMES
    log.info(f"action 채널 (hip 6 DOF): indices={action_dof_indices}")

    # 전 env의 hip 채널에 override 장착. α=1(DART)이면 게인 무변경 + 외란만
    # 가산, α<1(blend)이면 게인 α배 + (1-α)·ManiFlow 주입.
    override = JointTorqueOverride(
        env.simulator,
        env_ids=list(range(env.num_envs)),
        common_dof_indices=action_dof_indices,
    )
    override.engage(args.residual_pd_scale)

    estimator, maniflow_ckpt = None, None
    if args.residual_pd_scale < 1.0:
        maniflow_ckpt = (args.maniflow_ckpt
                         or discover_best_checkpoint(args.maniflow_run_dir))
        policy, _cfg, mf_info = load_maniflow_policy(
            maniflow_ckpt, device=str(fabric.device),
            maniflow_root=args.maniflow_root,
        )
        estimator = ManiFlowTorqueEstimator(policy, num_envs=env.num_envs,
                                            device=fabric.device)
        log.info(f"ManiFlow ckpt: {maniflow_ckpt} (epoch={mf_info['epoch']}, "
                 f"n_action_steps={estimator.n_action_steps})")

    store = init_zarr(output_path, args.target_episodes, args.episode_steps,
                      args, dof_names=dof_names,
                      action_dof_indices=action_dof_indices,
                      maniflow_ckpt=maniflow_ckpt)
    collect_episodes(agent, env, store, args.target_episodes,
                     args.episode_steps,
                     action_dof_indices=action_dof_indices, args=args,
                     override=override, estimator=estimator,
                     output_path=output_path)


if __name__ == "__main__":
    main()
