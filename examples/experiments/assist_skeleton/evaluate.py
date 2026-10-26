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
"""스켈레톤 착용자 평가 도구 — run_system을 여러 번 돌리거나 착용자만 따로 재는 것들.

  `MotionSweep`     (`motions`)        모션 클립마다 `run_system.py --human tracker`를 별도 프로세스로 돌려
                                       (Newton은 한 프로세스에서 env를 다시 만들지 않음) metrics.json을 표로 모은다 —
                                       LLP 일반화(학습 모션 vs 미학습 모션) 비교용.
  `NativeWearerEval`(`native-wearer`)  동결 W1을 짧은 실제 보행 클립에서 무보조로 평가. 클립을 이어 붙이지 않고
                                       끝 100 ms 전에 멈춘다 (현재 참조와 50 ms 미래 관측이 원본 안에 있도록).
                                       env 2개는 모두 무보조 반복 점검이다.
  `WearerTable`     (`wearer`)         기록의 무보조 env(A)로 착용자 자체가 얼마나 사람 같은가 (`analysis.WearerStats`).

    python examples/experiments/assist_skeleton/evaluate.py motions --llp-checkpoint <ckpt> --tracker-checkpoint <ckpt> \\
        --motion-file data/motion_for_trackers/skeleton_torque_suit_motions_11+koo_4.pt --calibration motion_all --seconds 20 --out <dir>
    python examples/experiments/assist_skeleton/evaluate.py native-wearer --motion-file <pt> --starts 0 0.5 --max-seconds 8 --out <dir>
    python examples/experiments/assist_skeleton/evaluate.py wearer results/oracle/*_tracker_v33*_m0
"""

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from protomotions.utils.simulator_imports import import_simulator_before_torch

import_simulator_before_torch("newton")

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


class MotionSweep:
    @staticmethod
    def motion_names(motion_file: str):
        m = torch.load(motion_file, map_location="cpu", weights_only=False)
        names = [os.path.basename(f).replace(".npz", "") for f in m["motion_files"]]
        lengths = [float(x) for x in m["motion_lengths"]]
        return names, lengths

    @classmethod
    def run(cls, args):
        names, lengths = cls.motion_names(args.motion_file)
        ids = list(range(len(names))) if args.motion_ids == "all" else [int(v) for v in args.motion_ids.split(",")]
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)

        rows = []
        for mid in ids:
            # 클립이 요청 길이보다 짧으면 클립 길이의 90 %만 (tracker의 clip 종료 리셋을 피한다)
            seconds = min(args.seconds, max(3.0, 0.9 * lengths[mid]))
            tag = f"m{mid:02d}_{names[mid]}"
            cmd = [
                sys.executable, str(HERE / "run_system.py"), "--human", "tracker",
                "--tracker-checkpoint", args.tracker_checkpoint, "--llp-checkpoint", args.llp_checkpoint,
                "--motion-file", args.motion_file, "--motion-id", str(mid), "--seconds", f"{seconds:.2f}",
                "--calibration", args.calibration, "--seed", str(args.seed), "--out", str(out), "--tag", tag,
            ] + (args.extra.split() if args.extra else [])
            print(f"[eval_motions] {tag} ({seconds:.1f} s) …", flush=True)
            res = subprocess.run(cmd, capture_output=True, text=True)
            run_dirs = sorted(out.glob(f"*_{tag}"))
            if res.returncode != 0 or not run_dirs:
                print(f"  FAILED (rc={res.returncode})\n" + "\n".join(res.stderr.splitlines()[-8:]))
                rows.append({"id": mid, "name": names[mid], "failed": True})
                continue
            metrics = json.loads((run_dirs[-1] / "metrics.json").read_text())
            meta = json.loads((run_dirs[-1] / "meta.json").read_text())
            s = metrics["summary"]
            rows.append({
                "id": mid, "name": names[mid], "seconds": seconds,
                "power_A": s["human_power_W_A"], "power_B": s["human_power_W_B"],
                "reduction_pct": s["human_power_reduction_pct"], "corr": s["assist_vs_agentA_corr"],
                "assist_rms": s["assist_rms_Nm"], "track_A": s["tracking_err_deg_A"], "track_B": s["tracking_err_deg_B"],
                "survived_s": meta.get("survived_s"), "hlp_150ms": metrics["hlp_lead_mae_deg"].get("150ms"),
            })
            print(f"  A {s['human_power_W_A']:.1f} W → B {s['human_power_W_B']:.1f} W ({s['human_power_reduction_pct']:+.1f} %), corr {s['assist_vs_agentA_corr']:.2f}, fall {meta.get('survived_s')}")

        ok = [r for r in rows if not r.get("failed")]
        lines = [
            f"# 모션별 A/B 평가 — LLP `{args.llp_checkpoint}`",
            "",
            f"tracker `{args.tracker_checkpoint}`, {args.seconds:.0f} s/클립(짧은 클립은 90 %), calibration {args.calibration}, seed {args.seed}",
            "",
            "| id | 모션 | 착용자 파워 A→B [W] | 절감(+ = 감소) | corr | assist RMS | hip 오차 A→B [deg] | HLP 150 ms MAE | 낙상 |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            if r.get("failed"):
                lines.append(f"| {r['id']} | {r['name']} | 실패 | | | | | | |")
                continue
            fall = "-" if r["survived_s"] is None else f"{r['survived_s']:.1f} s"
            hlp = "-" if r["hlp_150ms"] is None else f"{r['hlp_150ms']:.1f}°"
            lines.append(
                f"| {r['id']} | {r['name']} | {r['power_A']:.1f} → {r['power_B']:.1f} | **{r['reduction_pct']:+.1f} %** | {r['corr']:.2f} | "
                f"{r['assist_rms']:.1f} | {r['track_A']:.1f} → {r['track_B']:.1f} | {hlp} | {fall} |"
            )
        if ok:
            pa = sum(r["power_A"] * r["seconds"] for r in ok) / sum(r["seconds"] for r in ok)
            pb = sum(r["power_B"] * r["seconds"] for r in ok) / sum(r["seconds"] for r in ok)
            lines += ["", f"시간 가중 평균: {pa:.1f} → {pb:.1f} W (**{100*(1-pb/pa):+.1f} %**), 낙상 {sum(1 for r in ok if r['survived_s'] is not None)}/{len(ok)}"]
        (out / "summary.md").write_text("\n".join(lines) + "\n")
        (out / "summary.json").write_text(json.dumps(rows, indent=2))
        print("\n".join(lines))
        return rows


class NativeWearerEval:
    @staticmethod
    def run(args):
        from lightning.fabric import Fabric

        from run_system import build_configs, build_env_agent, configure_for_system, pin_spawn_location

        args.out.mkdir(parents=True, exist_ok=True)
        if (args.out / "summary.json").exists():
            raise FileExistsError(args.out / "summary.json")
        run = SimpleNamespace(human="tracker", tracker_checkpoint=args.tracker_checkpoint,
                              motion_file=args.motion_file, physics_fps=400, env_dt=0.05,
                              llp_dt=0.01, viewer=False, fixed_base_z=0.0)
        cfgs = build_configs(run)
        configure_for_system(cfgs, run)
        torch.manual_seed(args.seed)
        fabric = Fabric(accelerator="gpu", devices=1, num_nodes=1, loggers=[], callbacks=[])
        fabric.launch()
        env, agent = build_env_agent(cfgs, run, fabric)
        pin_spawn_location(env)
        sim = env.simulator
        ids = args.motion_ids if args.motion_ids is not None else range(len(env.motion_lib.motion_files))
        results = {"motion_file": args.motion_file, "tracker_checkpoint": args.tracker_checkpoint,
                   "seed": args.seed, "physics_hz": 400, "policy_hz": 20, "assist_torque_Nm": 0,
                   "body_error_definition": "mean world position error across all 28 bodies, after first 0.5 seconds",
                   "fall_definition": "pelvis z <0.5m for 0.5 seconds; stop on either copy",
                   "short_clip_boundary_margin_s": 0.1,
                   "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "trials": []}
        for mid in ids:
            for start in args.starts:
                duration = min(args.max_seconds, float(env.motion_lib.motion_lengths[mid])-start-0.1)
                nsteps = int(np.floor((duration+1e-6)/0.05))
                if nsteps < 20:
                    raise ValueError("Less than one second of valid source remaining")
                torch.manual_seed(args.seed)
                env.motion_manager.motion_ids[:] = mid
                env.motion_manager.motion_times[:] = start
                obs, _ = env.reset(disable_motion_resample=True)
                initial = sim.get_robot_state()
                initial_arrays = {
                    f"initial_{key}": getattr(initial, key).detach().cpu().numpy().copy()
                    for key in ("rigid_body_pos", "rigid_body_rot", "rigid_body_vel",
                                "rigid_body_ang_vel", "dof_pos", "dof_vel")
                }
                log = {k: [] for k in ("time", "motion_time", "body_error", "root_pos", "ref_root_pos", "dof_pos", "ref_dof_pos", "dof_vel")}
                fallen = torch.zeros(2, dtype=torch.long, device=env.device)
                stopped = "clip_window_complete"
                with torch.no_grad():
                    for step in range(nsteps):
                        td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))
                        output = agent.model(td)
                        obs, _, done, _, _ = env.step(output["mean_action"])
                        ctx = env._current_context
                        state, ref = ctx.current, ctx.mimic.ref_state
                        root = state.rigid_body_pos[:, 0]
                        t = (step+1)*0.05
                        values = {"time": np.array(t), "motion_time": env.motion_manager.motion_times,
                                  "body_error": (state.rigid_body_pos-ref.rigid_body_pos).norm(dim=-1).mean(-1),
                                  "root_pos": root, "ref_root_pos": ref.rigid_body_pos[:, 0],
                                  "dof_pos": state.dof_pos, "ref_dof_pos": ref.dof_pos, "dof_vel": state.dof_vel}
                        for key, value in values.items():
                            log[key].append(value.detach().cpu().numpy().copy() if torch.is_tensor(value) else value)
                        if bool(done.any()) or not torch.allclose(env.motion_manager.motion_times, torch.full_like(env.motion_manager.motion_times, start+t), atol=2e-4):
                            raise RuntimeError("Unexpected reset/end or reference time mismatch")
                        if not torch.isfinite(root).all():
                            stopped = "nonfinite_state"
                            break
                        fallen = torch.where(root[:, 2] < 0.5, fallen+1, 0)
                        if int(fallen.max()) >= 10:
                            stopped = "fall"
                            break
                if stopped == "clip_window_complete" and int(fallen.max()) > 0:
                    stopped = "clip_end_low_pelvis"
                arrays = {k: np.asarray(v) for k, v in log.items()}
                name = f"motion_{mid:02d}_start_{start:g}"
                np.savez_compressed(args.out / f"{name}.npz", **arrays, **initial_arrays)
                valid = arrays["time"] >= 0.5
                hip_error = arrays["dof_pos"]-arrays["ref_dof_pos"]
                trial = {"motion_id": int(mid), "source": env.motion_lib.motion_files[mid], "start_s": start,
                         "planned_s": nsteps*0.05, "elapsed_s": float(arrays["time"][-1]), "status": stopped,
                         "body_error_mean_m": arrays["body_error"][valid].mean(0).tolist(),
                         "body_error_p95_m": np.percentile(arrays["body_error"][valid], 95, axis=0).tolist(),
                         "hip_rmse_deg": np.rad2deg(np.sqrt((hip_error[valid][:, :, [0, 5]]**2).mean(axis=(0, 2)))).tolist(),
                         "root_min_z_m": arrays["root_pos"][:, :, 2].min(0).tolist(),
                         "root_end_z_m": arrays["root_pos"][-1, :, 2].tolist(),
                         "trailing_low_pelvis_s": (fallen * 0.05).cpu().tolist(),
                         "root_xy_displacement_m": np.linalg.norm(arrays["root_pos"][-1, :, :2]-arrays["root_pos"][0, :, :2], axis=-1).tolist(),
                         "ref_root_xy_displacement_m": np.linalg.norm(arrays["ref_root_pos"][-1, :, :2]-arrays["ref_root_pos"][0, :, :2], axis=-1).tolist(),
                         "trace": str((args.out / f"{name}.npz").resolve())}
                results["trials"].append(trial)
                (args.out / "summary.json").write_text(json.dumps(results, indent=2))
                print(json.dumps(trial), flush=True)
        print("Unassisted native-clip evaluation complete", flush=True)
        return results


class WearerTable:
    @staticmethod
    def run(args):
        from analysis import WearerStats

        runs = sorted({Path(h) for pat in args.runs for h in glob.glob(pat)})
        WearerStats.print_table(runs, args.mass)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("motions", help="모션 클립마다 run_system A/B 평가 → 표")
    s.add_argument("--llp-checkpoint", required=True)
    s.add_argument("--tracker-checkpoint", required=True)
    s.add_argument("--motion-file", required=True)
    s.add_argument("--motion-ids", default="all", help="'all' 또는 '0,5,10'")
    s.add_argument("--seconds", type=float, default=20.0)
    s.add_argument("--calibration", default="motion_all")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", required=True)
    s.add_argument("--extra", default="", help="run_system.py에 그대로 넘길 추가 인자")

    s = sub.add_parser("native-wearer", help="동결 W1의 무보조 실제 보행 클립 평가")
    s.add_argument("--motion-file", required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--tracker-checkpoint", default="checkpoints/W1_wearer_human/last.ckpt")
    s.add_argument("--motion-ids", nargs="+", type=int)
    s.add_argument("--starts", nargs="+", type=float, default=[0.0])
    s.add_argument("--max-seconds", type=float, default=8.0)
    s.add_argument("--seed", type=int, default=42)

    s = sub.add_parser("wearer", help="착용자(tracker)가 얼마나 사람 같은가 — 무보조 env로 잰다")
    s.add_argument("runs", nargs="+", help="run_system.py 실행 디렉터리(glob 가능)")
    s.add_argument("--mass", type=float, default=90.7)

    args = p.parse_args(argv)
    if args.command == "motions":
        MotionSweep.run(args)
    elif args.command == "native-wearer":
        torch.set_num_threads(1)
        NativeWearerEval.run(args)
    else:
        WearerTable.run(args)


if __name__ == "__main__":
    main()
