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
"""스켈레톤 착용자 위에서 HLP + LLP 통합 시스템을 돌리는 폐루프 러너 (Newton).

같은 씬에 env 2개를 겹쳐 띄운다 (Newton은 env마다 독립 world라 간섭 없음):

    env 0 (A) = 보조 없음 — 착용자 단독 (대조군)
    env 1 (B) = 보조 — HLP(ManiFlow) → LLP(assist RL) → hip flexion qfrc 주입

두 env는 같은 mocap 클립·같은 초기 상태를 받고 착용자 대역이 결정적이므로,
B의 차이는 전부 assist 토크에서 온다 (paired 비교).

착용자 대역 (--human):
    mocap_pd  골반 고정(fix_base_link + z 상승) + 관절 built-in PD가 mocap 각도를
              100 Hz로 추종(+속도 피드포워드). tracker 없이 즉시 실행. 지면 반력 없음
              → 진자와 같은 조건을 전신으로 확장한 것.
    tracker   mimic tracker 정책이 걷게 함 (--tracker-checkpoint 필요). env 20 Hz.

시간 구조: 물리 --physics-fps(400 Hz) / env dt --env-dt(50 ms, tracker 학습 주기) /
LLP --llp-dt(10 ms, 학습 주기). LLP는 `SubstepHookMixin`으로 서브스텝 루프 안에서
돈다 — tracker의 관측이 env dt에 묶여 있어 env 주기를 바꿀 수 없기 때문.

HLP 관측: hip flexion = 관절각, trunk pitch/roll = pelvis 자세 × lumbar FK
(`protomotions.assist.skeleton_features`, 학습 데이터 추출기와 동일). 영점 보정은
재생 클립 평균을 SEED 평균에 맞추는 오프셋 (--calibration motion).

출력 (--out/<timestamp>_<human>/): traces.npz, metrics.json, report.md, plots.png,
(--record) mp4.

실행 (sbc env):
    python examples/experiments/assist_skeleton/run_system.py --human mocap_pd --seconds 30
    python examples/experiments/assist_skeleton/run_system.py --human tracker \\
        --tracker-checkpoint results/mimic_suit_passive_walk/last.ckpt --seconds 30
    # 뷰어(A=고스트 라인, B=메시) / 녹화
    python examples/experiments/assist_skeleton/run_system.py --human mocap_pd --viewer --record
"""

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

import_simulator_before_torch("newton")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from lightning.fabric import Fabric  # noqa: E402

from protomotions.assist import AssistController, AssistControllerConfig  # noqa: E402
from protomotions.assist.skeleton_features import (  # noqa: E402
    SEED_CHANNEL_MEAN_DEG,
    calibration_offset_from_motion,
    hip_flexion_indices,
    hlp_channels,
    lumbar_indices,
)
from protomotions.maniflow.hybrid_control import JointTorqueOverride  # noqa: E402
from protomotions.utils.hydra_replacement import get_class  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis import analyze  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
MIMIC_EXPERIMENT = REPO / "examples" / "experiments" / "mimic" / "mlp.py"
HOOK_SIM = "protomotions.simulator.newton.substep_hook.SubstepHookNewtonSimulator"
HOOK_SIM_FIXED = "protomotions.simulator.newton.substep_hook.SubstepHookFixedBaseNewtonSimulator"
ENV_A, ENV_B = 0, 1  # A = 보조 없음, B = 보조
PELVIS_BODY = 0


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--human", choices=["mocap_pd", "tracker"], default="mocap_pd")
    p.add_argument("--tracker-checkpoint", default="results/mimic_suit_passive_walk/last.ckpt")
    p.add_argument("--robot-name", default="skeleton_torque_suit_passive_cable")
    p.add_argument("--motion-file", default="data/motion_for_trackers/skeleton_torque_suit_walk.pt")
    p.add_argument("--motion-id", type=int, default=0)
    p.add_argument("--start-time", type=float, default=0.0, help="클립 시작 시각 [s]")
    p.add_argument("--llp-checkpoint", default="results/assist_pendulum_v3/last.ckpt")
    p.add_argument("--hlp-checkpoint", default=None, help="None이면 flexion40 run 자동 탐색")
    p.add_argument("--denoise", type=int, default=3)
    p.add_argument("--calibration", default="motion",
                   help="motion(선택 클립 평균) | motion_all(파일 전 클립 평균 — 다중 모션 학습 LLP와 동일) | none | 'p,q,r,s' (deg)")
    p.add_argument("--physics-fps", type=int, default=400)
    p.add_argument("--env-dt", type=float, default=0.05, help="착용자 env 주기 [s] (tracker 학습 50 ms)")
    p.add_argument("--llp-dt", type=float, default=0.01, help="LLP 주기 [s] (학습 10 ms)")
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--warmup", type=float, default=1.0, help="지표에서 제외할 초반 구간 [s]")
    p.add_argument("--assist-scale", type=float, default=1.0, help="B에 주입하는 τ_assist 배율 (0 = 차단)")
    p.add_argument("--torque-limit", type=float, default=None, help="assist 토크 한계 [N·m] (None = LLP 학습값)")
    p.add_argument("--torque-lpf-hz", type=float, default=0.0,
                   help="assist 출력 1차 저역통과 차단주파수 [Hz] (0 = 끔) — 구동기 대역 모사")
    p.add_argument("--torque-slew-limit", type=float, default=None,
                   help="assist 출력 변화율 한계 [N·m/s] (None = 끔)")
    p.add_argument("--fixed-base-z", type=float, default=0.6, help="mocap_pd: 골반을 올릴 높이 [m]")
    p.add_argument("--no-feedforward", action="store_true", help="mocap_pd: 속도 피드포워드 Kd·θ̇_g 끄기")
    p.add_argument("--human-kp-scale", type=float, default=12.4,
                   help="mocap_pd: 착용자 PD 강성 배율 (기본값은 hip 고유진동수를 진자 학습 플랜트와 맞춘 값)")
    p.add_argument("--human-kd-scale", type=float, default=6.2,
                   help="mocap_pd: 착용자 PD 감쇠 배율 (기본값은 감쇠비를 진자와 맞춘 값)")
    p.add_argument("--lock-knee-ankle", action="store_true",
                   help="mocap_pd 진단: 무릎·발목 목표를 0으로 고정해 다리를 강체 진자로 만든다")
    p.add_argument("--fall-z", type=float, default=0.5, help="tracker: 루트 높이 이하면 낙상")
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--record", action="store_true", help="뷰어 프레임을 mp4로 저장 (--viewer 자동)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/assist_skeleton")
    p.add_argument("--tag", default="")
    args = p.parse_args()
    if args.record:
        args.viewer = True
    return args


def load_experiment_module(path: Path):
    spec = importlib.util.spec_from_file_location("experiment_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# config 조립
# ---------------------------------------------------------------------------
def build_configs(args):
    """(robot, simulator, terrain, scene_lib, motion_lib, env, agent) config."""
    if args.human == "tracker":
        ckpt = Path(args.tracker_checkpoint)
        resolved_path = ckpt.parent / "resolved_configs_inference.pt"
        assert resolved_path.exists(), f"tracker resolved configs not found: {resolved_path}"
        resolved = torch.load(resolved_path, map_location="cpu", weights_only=False)
        cfgs = {k: resolved[k] for k in ("robot", "simulator", "terrain", "scene_lib", "motion_lib", "env", "agent")}
        current_sim = cfgs["simulator"]._target_.split(".")[-3]
        if current_sim != "newton":
            from protomotions.simulator.factory import update_simulator_config_for_test

            cfgs["simulator"] = update_simulator_config_for_test(
                current_simulator_config=cfgs["simulator"],
                new_simulator="newton",
                robot_config=cfgs["robot"],
            )
        from protomotions.utils.inference_utils import apply_backward_compatibility_fixes

        apply_backward_compatibility_fixes(cfgs["robot"], cfgs["simulator"], cfgs["env"])
        return cfgs

    # mocap_pd: tracker 체크포인트 없이 mimic 실험 파일로 env config를 만든다
    from protomotions.utils.config_builder import build_standard_configs

    mod = load_experiment_module(MIMIC_EXPERIMENT)
    ns = SimpleNamespace(
        robot_name=args.robot_name,
        simulator="newton",
        headless=not args.viewer,
        num_envs=2,
        experiment_name="assist_skeleton",
        motion_file=args.motion_file,
        scenes_file=None,
        batch_size=1,
        training_max_steps=1,
    )
    cfgs = build_standard_configs(
        args=ns,
        terrain_config_fn=mod.terrain_config,
        scene_lib_config_fn=mod.scene_lib_config,
        motion_lib_config_fn=mod.motion_lib_config,
        env_config_fn=mod.env_config,
        configure_robot_and_simulator_fn=getattr(mod, "configure_robot_and_simulator", None),
        agent_config_fn=getattr(mod, "agent_config", None),
    )
    # 평가용 override: 종료 조건 제거 + 긴 에피소드 (mimic/mlp.py의 규칙 그대로)
    mod.apply_inference_overrides(
        cfgs["robot"], cfgs["simulator"], cfgs["env"], cfgs.get("agent"),
        cfgs["terrain"], cfgs["motion_lib"], cfgs["scene_lib"], ns,
    )
    return cfgs


def configure_for_system(cfgs, args):
    sim = cfgs["simulator"]
    fps = int(args.physics_fps)
    decimation = int(round(args.env_dt * fps))
    hook_every = int(round(args.llp_dt * fps))
    assert abs(decimation / fps - args.env_dt) < 1e-9, "env_dt must be a multiple of the physics step"
    assert abs(hook_every / fps - args.llp_dt) < 1e-9, "llp_dt must be a multiple of the physics step"
    assert decimation % hook_every == 0, "env_dt must be a multiple of llp_dt"
    sim.sim.fps = fps
    sim.sim.decimation = decimation
    sim.num_envs = 2
    sim.headless = not args.viewer
    sim.experiment_name = "assist_skeleton"
    fixed = args.human == "mocap_pd"
    sim._target_ = HOOK_SIM_FIXED if fixed else HOOK_SIM
    setattr(sim, "substep_hook_every", hook_every)
    setattr(sim, "fixed_base_z_offset", float(args.fixed_base_z) if fixed else 0.0)
    cfgs["robot"].asset.fix_base_link = fixed
    cfgs["robot"].reset_noise = None  # 두 env 초기 상태 동일
    cfgs["motion_lib"].motion_file = args.motion_file
    return decimation, hook_every


def build_env_agent(cfgs, args, fabric):
    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator
    from protomotions.utils.component_builder import build_all_components

    terrain_cfg, sim_cfg = convert_friction_for_simulator(cfgs["terrain"], cfgs["simulator"])
    components = build_all_components(
        terrain_config=terrain_cfg,
        scene_lib_config=cfgs["scene_lib"],
        motion_lib_config=cfgs["motion_lib"],
        simulator_config=sim_cfg,
        robot_config=cfgs["robot"],
        device=fabric.device,
        save_dir=None,
    )
    EnvClass = get_class(cfgs["env"]._target_)
    env = EnvClass(
        config=cfgs["env"],
        robot_config=cfgs["robot"],
        device=fabric.device,
        terrain=components["terrain"],
        scene_lib=components["scene_lib"],
        motion_lib=components["motion_lib"],
        simulator=components["simulator"],
    )
    agent = None
    if args.human == "tracker":
        ckpt = Path(args.tracker_checkpoint)
        AgentClass = get_class(cfgs["agent"]._target_)
        agent = AgentClass(config=cfgs["agent"], env=env, fabric=fabric, root_dir=ckpt.parent)
        agent.setup()
        agent.load(str(ckpt), load_env=False)
        agent.model.eval()
    return env, agent


def scale_human_gains(sim, kp_scale: float, kd_scale: float) -> None:
    """착용자 대역(built-in PD)의 ke/kd를 모든 env·DOF에 대해 배율 조정.

    골반 고정 대역에서 로봇 config의 게인(hip Kp 200/Kd 20)은 다리 관성
    (hip 기준 3.5 kg·m²)에 비해 너무 물러서(ω_n≈7.5 rad/s, ζ≈0.38) mocap을
    12° 이상 오버슈트하며 따라간다 — 사람 같지 않은 궤적이라 HLP가 분포 밖으로
    나간다. 기본 배율은 LLP가 학습된 진자(I=0.43 kg·m², Kp 300/Kd 15 →
    ω_n≈26.5 rad/s, ζ≈0.66)와 같은 동역학 영역이 되도록 고른 값이다:
    Kp = ω_n²·I ≈ 2480 (×12.4), Kd = 2ζ√(Kp·I) ≈ 124 (×6.2).
    """
    if kp_scale == 1.0 and kd_scale == 1.0:
        return
    import warp as wp
    from newton.solvers import SolverNotifyFlags

    ke_wp = sim.robot_view.get_attribute("joint_target_ke", sim.model)
    kd_wp = sim.robot_view.get_attribute("joint_target_kd", sim.model)
    ke = wp.to_torch(ke_wp)
    kd = wp.to_torch(kd_wp)
    ke *= float(kp_scale)
    kd *= float(kd_scale)
    sim.robot_view.set_attribute("joint_target_ke", sim.model, ke_wp)
    sim.robot_view.set_attribute("joint_target_kd", sim.model, kd_wp)
    sim.solver.notify_model_changed(SolverNotifyFlags.JOINT_DOF_PROPERTIES)


def pin_spawn_location(env):
    """두 env가 같은 위치에 스폰되도록 terrain 샘플링을 고정 (뷰어 겹침용)."""
    fixed_xy = env.terrain.sample_valid_locations(1)[0].clone()

    def _fixed(num_envs, sample_flat=False):
        return fixed_xy.unsqueeze(0).expand(num_envs, -1).clone()

    env.terrain.sample_valid_locations = _fixed


def resolve_calibration(args, dof_names):
    """HLP 관측 4ch 영점 오프셋 [rad] 또는 None."""
    spec = args.calibration.strip().lower()
    if spec == "none":
        return None
    if spec in ("motion", "motion_all"):
        return calibration_offset_from_motion(
            args.motion_file, dof_names, fixed_root=(args.human == "mocap_pd"),
            motion_index=None if spec == "motion_all" else args.motion_id,
        )
    vals = [float(v) for v in args.calibration.split(",")]
    assert len(vals) == 4, "--calibration 'p,q,r,s' needs 4 values (deg)"
    return torch.deg2rad(torch.tensor(vals, dtype=torch.float32))


# ---------------------------------------------------------------------------
# 서브스텝 훅: 착용자 목표(옵션) + HLP/LLP + qfrc 주입 + 기록
# ---------------------------------------------------------------------------
class SystemHook:
    def __init__(self, env, ctrl: AssistController, override: JointTorqueOverride, args, n_calls: int):
        self.env = env
        self.sim = env.simulator
        self.ctrl = ctrl
        self.override = override
        self.args = args
        self.mocap_pd = args.human == "mocap_pd"
        self.llp_dt = args.llp_dt
        dev = env.device
        robot = env.robot_config
        names = robot.kinematic_info.dof_names
        self.hip_idx = list(hip_flexion_indices(names))
        self.lumbar_idx = list(lumbar_indices(names))
        self.n_dofs = robot.number_of_actions
        self.conv_to_sim = self.sim.data_conversion.dof_convert_to_sim
        kd = torch.tensor(
            [robot.control.control_info[n].damping for n in names], device=dev, dtype=torch.float32
        )
        kd_scale = float(args.human_kd_scale) if self.mocap_pd else 1.0
        self.kd = (kd * kd_scale).unsqueeze(0).expand(2, -1)
        self.use_ff = self.mocap_pd and not args.no_feedforward
        self.assist_scale = float(args.assist_scale)
        self.motion_ids = torch.full((2,), int(args.motion_id), device=dev, dtype=torch.long)
        self.lock_idx = None
        if self.mocap_pd and args.lock_knee_ankle:
            self.lock_idx = torch.tensor(
                [names.index(n) for n in ("knee_angle_r", "ankle_angle_r", "knee_angle_l", "ankle_angle_l")],
                device=dev, dtype=torch.long,
            )
        self.need_reset = True
        self.prev_tau_ff = torch.zeros(2, self.n_dofs, device=dev)
        self.hip_idx_t = torch.tensor(self.hip_idx, device=dev, dtype=torch.long)

        # 기록 버퍼 (hook 호출 단위 = LLP step)
        S = ctrl.layout.obs_future_samples
        K = ctrl.num_knots
        self.i = 0
        self.n_calls = n_calls
        z = lambda *shape: torch.zeros(n_calls, *shape, device=dev)  # noqa: E731
        self.log = {
            "t": torch.zeros(n_calls, device=dev),
            "hip_pos": z(2, 2), "hip_vel": z(2, 2), "trunk": z(2, 2), "ref_hip": z(2, 2),
            "tau_assist": z(2, 2), "tau_pd_prev": z(2, 2), "tau_ff_prev": z(2, 2),
            "raw_action": z(2, 2), "theta_d_future": z(2, S, 2), "chunk_age": z(2),
            "chunk": z(2, K, 2), "chunk_t0": z(2), "root_z": z(2),
        }
        self.t_sim = 0.0

    @torch.no_grad()
    def __call__(self, sim, window: int) -> None:
        if self.i >= self.n_calls:
            return
        env = self.env
        dof_pos, dof_vel = sim.hook_dof_state()  # (2, D) COMMON
        pelvis_q = sim.hook_body_rot(PELVIS_BODY)  # (2, 4) xyzw
        ch = hlp_channels(dof_pos, pelvis_q, self.hip_idx, self.lumbar_idx)
        hip_pos, trunk = ch[:, :2], ch[:, 2:]
        hip_vel = dof_vel[:, self.hip_idx]

        if self.need_reset:
            self.ctrl.reset(None, hip_pos, hip_vel, trunk)
            self.need_reset = False
        else:
            self.ctrl.observe(hip_pos, hip_vel, trunk)
        tau = self.ctrl.act()  # (2, 2) 관절 규약

        # 착용자 대역 목표 (mocap_pd) / 기록용 레퍼런스 (tracker)
        t_motion = env.motion_manager.motion_times
        if self.mocap_pd:
            t_ref = t_motion + (window + 1) * self.llp_dt
        else:
            t_ref = t_motion + window * self.llp_dt
        ref = env.motion_lib.get_motion_state(self.motion_ids, t_ref)
        if self.lock_idx is not None:
            ref.dof_pos[:, self.lock_idx] = 0.0
            ref.dof_vel[:, self.lock_idx] = 0.0
        tau_ff = torch.zeros(2, self.n_dofs, device=dof_pos.device)
        if self.mocap_pd:
            sim._apply_simulator_pd_targets(ref.dof_pos[:, self.conv_to_sim])
            if self.use_ff:
                tau_ff = self.kd * ref.dof_vel

        inject = tau_ff.clone()
        inject[ENV_B, self.hip_idx_t] += self.assist_scale * tau[ENV_B]
        self.override.set_torques(inject)

        # 기록
        i = self.i
        L = self.log
        L["t"][i] = self.t_sim
        L["hip_pos"][i] = hip_pos
        L["hip_vel"][i] = hip_vel
        L["trunk"][i] = trunk
        L["ref_hip"][i] = ref.dof_pos[:, self.hip_idx]
        L["tau_assist"][i] = inject[:, self.hip_idx_t] - tau_ff[:, self.hip_idx_t]
        L["tau_pd_prev"][i] = sim.window_tau_pd_common[:, self.hip_idx]
        L["tau_ff_prev"][i] = self.prev_tau_ff[:, self.hip_idx]
        L["raw_action"][i] = self.ctrl.last_raw_action
        L["theta_d_future"][i] = self.ctrl.target_future_joint()
        L["chunk_age"][i] = self.ctrl.chunk_age
        L["chunk"][i] = self.ctrl.chunk_joint()
        L["chunk_t0"][i] = self.ctrl._chunk_t0
        if not self.mocap_pd:
            L["root_z"][i] = sim._get_simulator_root_state().root_pos[:, 2]
        self.prev_tau_ff = tau_ff
        self.i += 1
        self.t_sim += self.llp_dt

    def traces(self):
        n = self.i
        return {k: v[:n].cpu().numpy() for k, v in self.log.items()}


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    out_dir = Path(args.out) / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{args.human}" + (f"_{args.tag}" if args.tag else ""))
    out_dir.mkdir(parents=True, exist_ok=True)

    cfgs = build_configs(args)
    decimation, hook_every = configure_for_system(cfgs, args)
    windows_per_step = decimation // hook_every
    num_steps = int(round(args.seconds / args.env_dt))
    n_calls = num_steps * windows_per_step

    fabric = Fabric(accelerator="gpu", devices=1, num_nodes=1, loggers=[], callbacks=[])
    fabric.launch()
    env, agent = build_env_agent(cfgs, args, fabric)
    sim = env.simulator
    robot = env.robot_config
    dof_names = robot.kinematic_info.dof_names
    print(
        f"[system] human={args.human} physics={args.physics_fps} Hz decimation={decimation} "
        f"(env {args.env_dt*1e3:.0f} ms) hook_every={hook_every} (LLP {args.llp_dt*1e3:.0f} ms) "
        f"fixed_base={robot.asset.fix_base_link} sim={type(sim).__name__}"
    )

    # HLP 영점 보정
    calib = resolve_calibration(args, dof_names)
    if calib is not None:
        print("[system] HLP obs offset (deg):", [round(math.degrees(v), 2) for v in calib.tolist()])

    ctrl = AssistController(
        AssistControllerConfig(
            llp_checkpoint=args.llp_checkpoint,
            hlp_checkpoint=args.hlp_checkpoint,
            hlp_denoise_steps=args.denoise,
            hlp_obs_offset=None if calib is None else calib.tolist(),
            joint_sign=1.0,  # skeleton hinge: +θ = 굴곡 = 데이터 규약
            dt=args.llp_dt,
            torque_limit=args.torque_limit,
            torque_lpf_hz=args.torque_lpf_hz,
            torque_slew_limit=args.torque_slew_limit,
            measure_timing=True,
        ),
        num_envs=2,
        device=env.device,
    )

    # additive 주입 경로: 모든 env·DOF에 대해 게인 무변경(1.0)으로 engage
    override = JointTorqueOverride(sim, env_ids=range(2), common_dof_indices=range(robot.number_of_actions))
    override.engage(gain_scale=1.0)

    if args.human == "mocap_pd":
        scale_human_gains(sim, args.human_kp_scale, args.human_kd_scale)
        hip_kp = robot.control.control_info["hip_flexion_r"].stiffness * args.human_kp_scale
        hip_kd = robot.control.control_info["hip_flexion_r"].damping * args.human_kd_scale
        print(f"[system] human PD (hip): Kp {hip_kp:.0f} Kd {hip_kd:.0f} (scale ×{args.human_kp_scale}/×{args.human_kd_scale})")

    hook = SystemHook(env, ctrl, override, args, n_calls)
    sim.substep_hook = hook

    # 동기 리셋: 같은 클립, 같은 시작 시각, 같은 위치
    pin_spawn_location(env)
    torch.manual_seed(args.seed)
    env.motion_manager.motion_ids[:] = args.motion_id
    env.motion_manager.motion_times[:] = args.start_time
    obs, _ = env.reset(disable_motion_resample=True)
    override.zero()
    hook.need_reset = True

    ghost = None
    if args.viewer:
        from viz import GhostSkeletonRenderer, HipTorqueArrows

        ghost = GhostSkeletonRenderer(sim, ghost_env=ENV_A, visible_env=ENV_B)
        HipTorqueArrows(sim, hook, env_id=ENV_B)
        if args.record:
            sim._toggle_video_record()

    # 루프 (env 주기)
    fall_steps = 0
    survived_s = None
    t_wall = time.perf_counter()
    for step in range(num_steps):
        if agent is not None:
            td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))
            with torch.no_grad():
                out = agent.model(td)
            action = out.get("mean_action", out["action"])
        else:
            action = torch.zeros(2, robot.number_of_actions, device=env.device)
        obs, _rew, _done, _term, _extras = env.step(action)

        if args.human == "tracker":
            root_z = sim._get_simulator_root_state().root_pos[:, 2]
            if bool((root_z < args.fall_z).any()):
                fall_steps += 1
                if fall_steps * args.env_dt >= 0.5:
                    survived_s = (step + 1) * args.env_dt
                    print(f"[system] fall detected at {survived_s:.2f} s (root z {root_z.tolist()})")
                    break
            else:
                fall_steps = 0
        if (step + 1) % int(round(5.0 / args.env_dt)) == 0:
            print(f"[system] {(step+1)*args.env_dt:5.1f} s / {args.seconds:.0f} s  ({time.perf_counter()-t_wall:.0f} s wall)")

    if args.record:
        sim._toggle_video_record()
        sim.render()  # 마지막 프레임에서 인코딩 트리거 (RecordingMixin → ffmpeg)
        rec_dir = getattr(sim, "_curr_user_recording_name", None)
        if rec_dir and Path(rec_dir).exists():
            import shutil

            for mp4 in Path(rec_dir).glob("*.mp4"):
                shutil.copy(mp4, out_dir / "sim.mp4")
                print(f"[system] video → {out_dir / 'sim.mp4'}")

    traces = hook.traces()
    meta = {
        "human": args.human,
        "robot": args.robot_name if args.human == "mocap_pd" else type(robot).__name__,
        "motion_file": args.motion_file,
        "motion_id": args.motion_id,
        "start_time": args.start_time,
        "llp_checkpoint": str(Path(args.llp_checkpoint).resolve()),
        "llp_epoch": ctrl.llp.epoch,
        "hlp_checkpoint": ctrl.hlp_checkpoint,
        "denoise": args.denoise,
        "calibration_deg": None if calib is None else [math.degrees(v) for v in calib.tolist()],
        "seed_mean_deg": list(SEED_CHANNEL_MEAN_DEG),
        "physics_fps": args.physics_fps,
        "env_dt": args.env_dt,
        "llp_dt": args.llp_dt,
        "seconds": args.seconds,
        "warmup": args.warmup,
        "assist_scale": args.assist_scale,
        "torque_limit": ctrl.torque_limit,
        "torque_lpf_hz": args.torque_lpf_hz or None,
        "torque_slew_limit": args.torque_slew_limit,
        "feedforward": hook.use_ff,
        "human_kp_scale": args.human_kp_scale if args.human == "mocap_pd" else None,
        "human_kd_scale": args.human_kd_scale if args.human == "mocap_pd" else None,
        "lock_knee_ankle": bool(args.lock_knee_ankle) if args.human == "mocap_pd" else None,
        "fixed_base_z": args.fixed_base_z if args.human == "mocap_pd" else None,
        "survived_s": survived_s,
        "hlp_ms": {"mean": float(np.mean(ctrl.hlp_ms)) if ctrl.hlp_ms else None,
                   "p95": float(np.percentile(ctrl.hlp_ms, 95)) if ctrl.hlp_ms else None},
        "llp_ms": {"mean": float(np.mean(ctrl.llp_ms)) if ctrl.llp_ms else None,
                   "p95": float(np.percentile(ctrl.llp_ms, 95)) if ctrl.llp_ms else None},
        "hlp_dt": ctrl.hlp_dt,
        "num_knots": ctrl.num_knots,
        "obs_future_dt": ctrl.layout.obs_future_dt,
        "tracker_checkpoint": str(Path(args.tracker_checkpoint).resolve()) if args.human == "tracker" else None,
        "command": " ".join(sys.argv),
    }
    np.savez_compressed(out_dir / "traces.npz", **traces)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    metrics = analyze(traces, meta, out_dir)
    print(json.dumps(metrics["summary"], indent=2, ensure_ascii=False))
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
