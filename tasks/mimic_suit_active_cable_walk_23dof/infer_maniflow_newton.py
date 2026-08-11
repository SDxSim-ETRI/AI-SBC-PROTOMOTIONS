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
"""ManiFlow hip-torque 추정기 폐루프 inference (기본: Newton).

RL walking policy(ProtoMotions 체크포인트)로 시뮬레이터에서 보행을 실행하면서,
학습된 ManiFlow lowdim 정책이 센서 상태만으로 hip torque를 receding-horizon
방식으로 예측하고, 시뮬레이터가 실제로 가한 토크와 비교합니다.

관측 구성 (collect_walk_zarr.py와 동일, common ordering):
  obs(88) = dof_pos(27) + dof_vel(27) + root_pos(3) + root_vel(3) + contacts(28)

action 채널: 순수 hip 6개 DOF (공통 [0,1,2,5,6,7] — 이름 기반 파생)
  = [hip_flexion_r, hip_adduction_r, hip_rotation_r,
     hip_flexion_l, hip_adduction_l, hip_rotation_l]
  과거 first6(공통 0-5 = 오른다리 전체+왼 hip flexion, 수집 버그) 채널로
  학습된 legacy run01 모델은 2026-07-14 삭제되어 관련 옵션(--action-dofs)도
  제거되었습니다.

정렬(alignment): 매 스텝 env.step 직후의 robot_state로 obs[t]와 gt_torque[t]를
같은 시점에 기록 — 수집 스크립트와 동일. predict()가 반환하는 청크의 첫 스텝은
마지막 관측 시점 t에 해당하며, receding 모드에서는 n_action_steps 간격으로
예측하여 전 구간을 빈틈없이 채웁니다.

출력 (--output, 기본 tasks/.../maniflow_infer_results/<timestamp>/):
  metrics.json / metrics.txt : per-joint MAE/RMSE/R^2/corr (+ run 설정)
  traces.npz                 : pred/gt hip torque + obs 전체 trace + env_failed
  env*_{full,zoom}.png       : 관절별 pred vs gt trace 플롯
  sim-<ts>/sim-<ts>.mp4      : (--record) 시뮬레이션 녹화 영상
  sim_with_torque.mp4        : (--record) 영상 + 예측/실제 토크 플롯 합성 비디오

실행:
  bash tasks/mimic_suit_active_cable_walk_23dof/infer_maniflow_newton.sh
  # 실시간 GUI로 보기 (뷰어 UI에 hip_flexion pred/gt 라이브 플롯 표시):
  bash tasks/mimic_suit_active_cable_walk_23dof/infer_maniflow_newton.sh --viewer
  # 동영상 저장 (뷰어 자동 활성화; 시뮬 mp4 + 토크 합성 mp4 생성):
  bash tasks/mimic_suit_active_cable_walk_23dof/infer_maniflow_newton.sh \
      --record --num-envs 2 --episode-steps 600
  # IsaacLab(학습 도메인)에서 sanity check:
  bash tasks/mimic_suit_active_cable_walk_23dof/infer_maniflow_newton.sh \
      --simulator isaaclab \
      --rl-checkpoint tasks/mimic_suit_active_cable_walk_23dof/output_isaaclab_flat/score_based.ckpt
"""

# ---------------------------------------------------------------------------
# IsaacLab 사용 시 torch 임포트 전에 호출해야 하므로 argparse를 먼저 수행
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


def _create_parser():
    p = argparse.ArgumentParser(
        description="Closed-loop ManiFlow hip-torque inference in simulation",
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
    p.add_argument("--simulator", default="newton",
                   help="시뮬레이터 (newton / isaaclab / ...)")
    p.add_argument("--motion-file",
                   default="data/motion_for_trackers/skeleton_torque_suit_walk.pt")
    p.add_argument("--num-envs", type=int, default=2)
    p.add_argument("--episode-steps", type=int, default=1200,
                   help="1200 = 60s @ 20Hz (fps=120/decimation=6)")
    p.add_argument("--predict-mode", choices=["receding", "every_step"],
                   default="receding",
                   help="receding: n_action_steps 간격 청크 예측(오프라인 eval과 동일), "
                        "every_step: 매 스텝 예측 후 첫 액션만 사용")
    p.add_argument("--denoise-steps", type=int, default=3,
                   help="추론 ODE(Euler) 스텝 수. 기본 3 — 2026-07-15 검증: "
                        "consistency 학습(임의 dt 점프, target_t 조건) 덕분에 "
                        "재학습 없이 축소 가능하고, N=3이 체크포인트 설정(10)보다 "
                        "정확하며 3배 빠름(MAE 0.32 vs 1.40 N·m, 5.8 vs 18.9 ms). "
                        "이전 동작 재현은 --denoise-steps 10")
    p.add_argument("--control-mode", choices=["config", "proportional", "built_in_pd"],
                   default="config",
                   help="RL policy 액추에이션 모드 (config = 체크포인트 설정 그대로, "
                        "학습과 동일). Newton BUILT_IN_PD의 적용 토크는 MuJoCo 솔버의 "
                        "qfrc_actuator readback으로 읽음. proportional(explicit PD)은 "
                        "이 로봇(active cable)에서 발산 이력 있음 — 주의")
    p.add_argument("--viewer", action="store_true", default=False,
                   help="뷰어 표시 (기본 headless). 뷰어 UI에 hip_flexion pred/gt "
                        "실시간 플롯이 함께 표시됨")
    p.add_argument("--record", action="store_true", default=False,
                   help="시뮬레이션 mp4 저장 + 예측/실제 토크 플롯을 옆에 합성한 "
                        "sim_with_torque.mp4 생성 (뷰어 자동 활성화 — 프레임 캡처에 "
                        "GL 컨텍스트 필요)")
    p.add_argument("--no-mesh", action="store_true", default=False,
                   help="뷰어/녹화 시 skeleton mesh 에셋 대신 기본(캡슐) 에셋 사용 — "
                        "수집 조건과 완전히 동일한 물리(hip ring 충돌 포함)가 필요할 때. "
                        "기본은 play/record_newton.sh와 같은 skeleton mesh 시각화")
    p.add_argument("--output", default=None,
                   help="결과 디렉토리 (기본: {task}/maniflow_infer_results/<timestamp>)")
    p.add_argument("--n-plot-envs", type=int, default=2)
    p.add_argument("--zoom-steps", type=int, default=300)
    p.add_argument("--overrides", nargs="*", default=[])
    return p


_parser = _create_parser()
_args, _ = _parser.parse_known_args()

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch(_args.simulator)

# 이후 torch 및 나머지 임포트 안전
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
    ManiFlowTorqueEstimator,
    discover_best_checkpoint,
    hip_dof_indices,
    load_maniflow_policy,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
log = logging.getLogger(__name__)

# ManiFlow action 채널 (공통 DOF 인덱스/이름). main()에서 robot config로부터
# 이름 기반으로 채워집니다 (하드코딩 금지 — 순수 hip 6 DOF = 공통 [0,1,2,5,6,7]).
ACTION_DOF_INDICES = list(range(6))
HIP_JOINT_NAMES = [
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
]


# ---------------------------------------------------------------------------
def setup_agent_and_env(args, fabric: Fabric, app_launcher):
    """collect_walk_zarr.py / inference_agent.py와 동일한 순서로 초기화합니다."""
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

    if args.control_mode != "config":
        from protomotions.robot_configs.base import ControlType
        robot_config.control.control_type = ControlType[args.control_mode.upper()]
        log.info(f"Control mode override: {robot_config.control.control_type.name} "
                 "(explicit PD는 적용 토크를 dof_forces로 읽을 수 있음)")

    simulator_config.num_envs = args.num_envs
    simulator_config.headless = not args.viewer
    motion_lib_config.motion_file = args.motion_file

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
@torch.no_grad()
def run_rollout(agent, env, estimator: ManiFlowTorqueEstimator, T: int,
                predict_mode: str, record: bool = False):
    """RL policy로 보행하며 ManiFlow 예측/실제 hip torque trace를 수집합니다.

    record=True면 초기 reset 직후 녹화를 시작해 스텝과 프레임을 1:1로 맞춥니다
    (simulator.step()이 물리 스텝 직후 render()를 호출하며 그때 프레임 캡처됨).
    뷰어가 있으면 hip_flexion 좌/우의 pred/gt를 실시간 플롯으로 스트리밍합니다.
    """
    N = env.num_envs
    Ta = estimator.n_action_steps
    Da = estimator.action_dim
    viewer = getattr(env.simulator, "viewer", None)
    live_plot = viewer is not None and hasattr(viewer, "log_scalar")

    pred_trace = np.full((N, T, Da), np.nan, dtype=np.float32)
    gt_trace = np.zeros((N, T, Da), dtype=np.float32)
    obs_trace = np.zeros((N, T, estimator.obs_dim), dtype=np.float32)
    env_failed = np.zeros(N, dtype=bool)
    # 라이브 플롯에 띄울 채널 (좌/우 hip flexion) — 채널 레이아웃에서 파생
    live_channels = [j for j, nm in enumerate(HIP_JOINT_NAMES)
                     if nm.startswith("hip_flexion")]

    obs, _ = env.reset()
    estimator.reset()
    done_indices = None

    if record:
        env.simulator._toggle_video_record()  # 다음 render()에서 녹화 시작

    abs_err_sum, err_count = 0.0, 0
    t_start = time.time()
    steps_done = T
    for t in range(T):
        if viewer is not None and hasattr(viewer, "is_running") and not viewer.is_running():
            log.warning(f"뷰어 창이 닫혀 step {t}에서 중단합니다.")
            steps_done = t
            break

        if done_indices is not None and len(done_indices) > 0:
            env_failed[done_indices.cpu().numpy()] = True
            obs, _ = env.reset(done_indices)
            estimator.reset(done_indices)

        obs = agent.add_agent_info_to_obs(obs)
        obs_td = agent.obs_dict_to_tensordict(obs)
        model_out = agent.model(obs_td)
        action = model_out.get("mean_action", model_out["action"])

        obs, _rewards, dones, _terminated, _extras = env.step(action)
        robot_state = env.simulator.get_robot_state()

        obs_now = estimator.observe(robot_state)
        obs_trace[:, t] = obs_now.cpu().numpy()
        gt_trace[:, t] = robot_state.dof_forces[:, ACTION_DOF_INDICES].cpu().numpy()

        if predict_mode == "receding":
            if t % Ta == 0:
                pred = estimator.predict().cpu().numpy()  # (N, Ta, Da)
                end = min(t + Ta, T)
                pred_trace[:, t:end] = pred[:, : end - t]
        else:  # every_step: 매 스텝 예측, 현재 스텝 액션만 사용
            pred = estimator.predict().cpu().numpy()
            pred_trace[:, t] = pred[:, 0]

        if live_plot:
            for j in live_channels:  # hip_flexion_r / hip_flexion_l
                nm = HIP_JOINT_NAMES[j]
                viewer.log_scalar(f"torque/{nm}/applied", float(gt_trace[0, t, j]))
                viewer.log_scalar(f"torque/{nm}/maniflow", float(pred_trace[0, t, j]))
        if viewer is not None:
            abs_err_sum += float(np.abs(pred_trace[:, t] - gt_trace[:, t]).mean())
            err_count += 1
            if t % 20 == 0:
                env.simulator.set_window_title(
                    f"ManiFlow hip-torque  step {t}/{T}  "
                    f"running MAE {abs_err_sum / max(err_count, 1):.1f} N·m")

        done_indices = dones.nonzero(as_tuple=False).flatten()

        if (t + 1) % 200 == 0:
            log.info(f"  step {t + 1:5d}/{T} | failed {int(env_failed.sum())}/{N} "
                     f"| {time.time() - t_start:.1f}s")

    if done_indices is not None and len(done_indices) > 0:
        env_failed[done_indices.cpu().numpy()] = True

    if record:
        env.simulator._toggle_video_record()  # 녹화 종료 표시
        env.simulator.render()  # mp4 컴파일 + 임시 프레임 정리 flush

    if steps_done < T:  # 뷰어 조기 종료 시 뒷부분 잘라냄
        pred_trace = pred_trace[:, :steps_done]
        gt_trace = gt_trace[:, :steps_done]
        obs_trace = obs_trace[:, :steps_done]

    return pred_trace, gt_trace, obs_trace, env_failed


# ---------------------------------------------------------------------------
def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """pred, gt: (M, Da) raw torque. per-joint MAE/RMSE/R^2/corr."""
    err = pred - gt
    mae = np.abs(err).mean(axis=0)
    rmse = np.sqrt((err ** 2).mean(axis=0))
    ss_res = (err ** 2).sum(axis=0)
    ss_tot = ((gt - gt.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / ss_tot
    with np.errstate(invalid="ignore"):
        corr = np.array([
            np.corrcoef(pred[:, j], gt[:, j])[0, 1] for j in range(gt.shape[1])
        ])
    return {"mae": mae, "rmse": rmse, "r2": r2, "corr": corr}


def format_metrics_table(m: dict, gt_std: np.ndarray) -> str:
    hdr = (f"{'joint':>16} | {'GT std':>8} | {'MAE':>8} | {'RMSE':>8} | "
           f"{'R2':>7} | {'corr':>6}")
    lines = [hdr, "-" * len(hdr)]
    for j, name in enumerate(HIP_JOINT_NAMES):
        lines.append(
            f"{name:>16} | {gt_std[j]:8.3f} | {m['mae'][j]:8.3f} | "
            f"{m['rmse'][j]:8.3f} | {m['r2'][j]:7.3f} | {m['corr'][j]:6.3f}")
    lines.append("-" * len(hdr))
    lines.append(
        f"{'mean':>16} | {gt_std.mean():8.3f} | {m['mae'].mean():8.3f} | "
        f"{m['rmse'].mean():8.3f} | {m['r2'].mean():7.3f} | {m['corr'].mean():6.3f}")
    return "\n".join(lines)


def save_trace_plots(out_dir: Path, pred_trace, gt_trace, env_ids, zoom_steps):
    T = gt_trace.shape[1]
    for e in env_ids:
        for tag, t_end in [("full", T), ("zoom", min(zoom_steps, T))]:
            fig, axes = plt.subplots(3, 2, figsize=(16, 9), sharex=True)
            t_axis = np.arange(t_end)
            for j, ax in enumerate(axes.flat):
                ax.plot(t_axis, gt_trace[e, :t_end, j], color="black", lw=1.0,
                        label="applied (simulator)")
                ax.plot(t_axis, pred_trace[e, :t_end, j], color="tab:red", lw=1.0,
                        alpha=0.8, label="ManiFlow predicted")
                valid = ~np.isnan(pred_trace[e, :t_end, j])
                mae_j = np.abs(pred_trace[e, :t_end, j][valid]
                               - gt_trace[e, :t_end, j][valid]).mean()
                ax.set_title(f"{HIP_JOINT_NAMES[j]}  (MAE {mae_j:.3f})", fontsize=10)
                if j == 0:
                    ax.legend(fontsize=8, loc="upper right")
            fig.suptitle(f"env {e} — ManiFlow predicted vs applied hip torque ({tag})")
            fig.supxlabel("policy step (20Hz)")
            fig.supylabel("torque [N·m]")
            fig.tight_layout()
            fig.savefig(out_dir / f"env{e:02d}_{tag}.png", dpi=120)
            plt.close(fig)


def compose_torque_video(sim_mp4, pred, gt, out_path, fps=20, window_s=8.0):
    """녹화된 시뮬 영상 옆에 예측/실제 hip torque 스크롤 플롯을 붙인 mp4 생성.

    sim_mp4의 프레임은 rollout 스텝과 1:1 대응합니다(simulator.step()당 render
    1회). pred/gt는 카메라가 따라가는 env 0의 (T, 6) trace.
    """
    import imageio.v2 as imageio

    pred, gt = np.asarray(pred), np.asarray(gt)
    T = gt.shape[0]
    reader = imageio.get_reader(str(sim_mp4))
    n_frames = reader.count_frames()
    offset = max(0, n_frames - T)  # 여분 프레임은 시작(리셋 시점)에 생긴 것으로 간주
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
        ax.plot(tt, gt[:, j], color="black", lw=1.0, label="applied")
        ax.plot(tt, pred[:, j], color="tab:red", lw=1.0, alpha=0.85, label="ManiFlow")
        lo, hi = np.nanpercentile(np.concatenate([gt[:, j], pred[:, j]]), [1, 99])
        pad = 0.15 * max(hi - lo, 1e-3)
        ax.set_ylim(lo - pad, hi + pad)
        cursors.append(ax.axvline(0.0, color="tab:blue", lw=1.2))
        ax.set_title(HIP_JOINT_NAMES[j], fontsize=9)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.legend(fontsize=7, loc="upper right")
    fig.supxlabel("time [s]", fontsize=9)
    fig.supylabel("hip torque [N·m]", fontsize=9)
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
        if plot_rgb.shape[0] != H:  # dpi 반올림 편차 방어
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

    if args.record and not args.viewer:
        log.info("--record: 프레임 캡처에 GL 컨텍스트가 필요해 뷰어를 자동 활성화합니다.")
        args.viewer = True
    if args.viewer and not args.no_mesh:
        # 시각화 모드 기본값: skeleton mesh 에셋 (play/record_newton.sh와 동일)
        args.overrides = list(args.overrides) + [
            "robot.asset.asset_file_name=mjcf/skeleton_torque_suit_mesh.xml"
        ]
        log.info("skeleton mesh 에셋으로 시각화합니다 (--no-mesh로 비활성화).")

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.output or f"{TASK_ROOT}/maniflow_infer_results/{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # inference_agent.py와 동일한 순서: Fabric 먼저, 필요 시 AppLauncher
    fabric_config = FabricConfig(accelerator="gpu", devices=1, num_nodes=1,
                                 loggers=[], callbacks=[])
    fabric: Fabric = Fabric(**asdict(fabric_config))
    fabric.launch()

    app_launcher = None
    if args.simulator == "isaaclab":
        app_launcher = AppLauncher(
            {"headless": not args.viewer, "device": str(fabric.device)}
        )
        import carb
        carb.settings.get_settings().set(
            "/persistent/physics/visualizationSimulationOutput", False
        )

    agent, env = setup_agent_and_env(args, fabric, app_launcher)

    # 채널 인덱스/이름을 robot config에서 이름 기반으로 파생 (하드코딩 금지)
    dof_names = list(env.robot_config.kinematic_info.dof_names)
    ACTION_DOF_INDICES[:] = hip_dof_indices(dof_names)
    HIP_JOINT_NAMES[:] = [dof_names[i] for i in ACTION_DOF_INDICES]
    log.info(f"Estimator channels (hips, COMMON DOF "
             f"{ACTION_DOF_INDICES}): {HIP_JOINT_NAMES}")

    # ManiFlow 추정기 로드 (RL agent 이후: 시뮬레이터 초기화와 분리)
    maniflow_ckpt = args.maniflow_ckpt or discover_best_checkpoint(args.maniflow_run_dir)
    policy, mf_cfg, mf_info = load_maniflow_policy(
        maniflow_ckpt, device=str(fabric.device), maniflow_root=args.maniflow_root
    )
    estimator = ManiFlowTorqueEstimator(policy, num_envs=env.num_envs,
                                        device=fabric.device)
    assert args.denoise_steps >= 1
    if args.denoise_steps != policy.num_inference_steps:
        log.info(f"denoise(ODE) steps: {policy.num_inference_steps}(ckpt 설정) -> "
                 f"{args.denoise_steps}")
        policy.num_inference_steps = args.denoise_steps
    log.info(f"ManiFlow ckpt: {maniflow_ckpt} (epoch={mf_info['epoch']}, "
             f"n_obs_steps={estimator.n_obs_steps}, "
             f"n_action_steps={estimator.n_action_steps}, "
             f"denoise_steps={policy.num_inference_steps})")

    viewer = getattr(env.simulator, "viewer", None)
    if args.viewer and viewer is not None:
        viewer.show_ui = True  # 실시간 torque 플롯(Plots 창) 표시
    if args.record:
        env.simulator._user_recording_video_path = str(out_dir / "sim-%s")

    T = args.episode_steps
    print(f"\nInference 시작: simulator={args.simulator} | {env.num_envs} envs "
          f"| T={T} steps | predict_mode={args.predict_mode}"
          + (" | recording" if args.record else ""))
    pred_trace, gt_trace, obs_trace, env_failed = run_rollout(
        agent, env, estimator, T, args.predict_mode, record=args.record
    )
    T = gt_trace.shape[1]  # 뷰어 조기 종료 시 실제 수집 길이

    # ── 지표 계산 (낙상 없이 완주한 env만) ────────────────────────────────
    ok_envs = np.where(~env_failed)[0]
    if len(ok_envs) == 0:
        log.error("모든 env가 done(낙상/이탈) — 지표를 계산할 수 없습니다. "
                  "trace만 저장합니다.")
    np.savez_compressed(
        out_dir / "traces.npz",
        pred=pred_trace, gt=gt_trace, obs=obs_trace, env_failed=env_failed,
    )

    result = {
        "simulator": args.simulator,
        "rl_checkpoint": str(args.rl_checkpoint),
        "maniflow_ckpt": str(maniflow_ckpt),
        "maniflow_epoch": mf_info["epoch"],
        "predict_mode": args.predict_mode,
        "denoise_steps": int(policy.num_inference_steps),
        "num_envs": int(env.num_envs),
        "episode_steps": int(T),
        "ok_envs": ok_envs.tolist(),
        "env_failed": env_failed.tolist(),
        "action_dofs": "hips",
        "action_dof_indices": list(ACTION_DOF_INDICES),
        "joint_names": HIP_JOINT_NAMES,
    }

    if len(ok_envs) > 0:
        pred_ok = pred_trace[ok_envs]           # (n_ok, T, 6)
        gt_ok = gt_trace[ok_envs]
        valid = ~np.isnan(pred_ok[..., 0])      # (n_ok, T)
        pred_flat = pred_ok[valid]
        gt_flat = gt_ok[valid]

        m = compute_metrics(pred_flat, gt_flat)
        gt_std = gt_flat.std(axis=0)
        table = format_metrics_table(m, gt_std)
        print(f"\nPer-joint hip-torque prediction on {args.simulator} "
              f"({pred_flat.shape[0]} points, {len(ok_envs)}/{env.num_envs} envs):")
        print(table + "\n")

        result["n_points"] = int(pred_flat.shape[0])
        result["gt_std"] = gt_std.tolist()
        result["metrics"] = {k: v.tolist() for k, v in m.items()}
        with open(out_dir / "metrics.txt", "w") as f:
            f.write(f"simulator: {args.simulator}\n"
                    f"rl ckpt:   {args.rl_checkpoint}\n"
                    f"mf  ckpt:  {maniflow_ckpt} (epoch {mf_info['epoch']})\n"
                    f"predict:   {args.predict_mode} "
                    f"(denoise_steps={policy.num_inference_steps})\n\n"
                    + table + "\n")

        save_trace_plots(out_dir, pred_trace, gt_trace,
                         ok_envs[: args.n_plot_envs], args.zoom_steps)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    # ── 녹화 영상 + 토크 플롯 합성 비디오 ────────────────────────────────
    if args.record:
        rec_name = getattr(env.simulator, "_curr_user_recording_name", None)
        sim_mp4 = Path(rec_name) / f"{Path(rec_name).name}.mp4" if rec_name else None
        if sim_mp4 is None or not sim_mp4.exists():
            log.warning("녹화 mp4가 없어 합성 비디오를 건너뜁니다.")
        else:
            policy_fps = (round(1.0 / env.simulator.dt)
                          if getattr(env.simulator, "dt", 0) > 0 else 20)
            out_mp4 = out_dir / "sim_with_torque.mp4"
            log.info(f"토크 합성 비디오 생성 중... ({sim_mp4.name} + traces)")
            compose_torque_video(sim_mp4, pred_trace[0], gt_trace[0], out_mp4,
                                 fps=policy_fps)
            print(f"시뮬 영상:        {sim_mp4}")
            print(f"토크 합성 비디오: {out_mp4}")

    print(f"결과 저장: {out_dir}")


if __name__ == "__main__":
    main()
