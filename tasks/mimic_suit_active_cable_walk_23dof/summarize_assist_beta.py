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
"""가산 보조 β 스윕 요약 — compare_maniflow_control_newton.py 결과 집계.

여러 실행 디렉토리의 ``metrics.json``(``assist`` 블록 포함)을 읽어

  1. 마크다운 표 (MANIFLOW_CONTROL_ANALYSIS.md에 붙여넣는 용도)
  2. ``assist_beta_sweep.png`` — 3패널 요약 그림
       (a) β vs 토크 RMS: A 에이전트(보조 없음) / B 에이전트 / 보조 / B 총합
       (b) β vs 에이전트 토크·파워 감소율 + 이상선(=100β) + offload 효율
       (c) β vs 트래킹 오차(err6)·보상·생존(평균 에피소드 길이)

를 만든다. 보조 토크를 더할수록 에이전트(=착용자) 토크가 실제로 줄어드는지와
그 대가(과구동·트래킹 열화·생존)를 한 장으로 보기 위한 스크립트.

실행:
  python tasks/mimic_suit_active_cable_walk_23dof/summarize_assist_beta.py \
      --dirs tasks/.../maniflow_control_results/2026-08-06_run04_b*_add60s \
      --output tasks/.../maniflow_control_results/2026-08-06_assist_beta_sweep
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def load_runs(dirs):
    """metrics.json을 읽어 β 오름차순 레코드 리스트로 반환."""
    runs = []
    for d in dirs:
        path = Path(d) / "metrics.json"
        if not path.exists():
            print(f"  skip (metrics.json 없음): {d}")
            continue
        with open(path) as f:
            r = json.load(f)
        asst = r.get("metrics", {}).get("assist")
        if asst is None or asst.get("overall") is None:
            print(f"  skip (assist 블록 없음 — 가산 보조 실행이 아님): {d}")
            continue
        # 주입 배율이 0인 대조군은 실효 β = 0
        eff_beta = float(asst["beta"]) * float(r.get("torque_scale", 1.0))
        m = r["metrics"]
        o = asst["overall"]
        runs.append({
            "dir": str(d),
            "name": Path(d).name,
            "beta": eff_beta,
            "declared_beta": float(asst["beta"]),
            "agent_rms_A": o["agent_rms"]["A_noassist"],
            "agent_rms_B": o["agent_rms"]["B_assisted"],
            "exo_rms": o["exo_rms"],
            "total_rms_B": o["B_total_rms"],
            "reduction_pct": o["agent_rms_reduction_pct"],
            "offload_eff": o["offload_efficiency"],
            "power_A": o["agent_power_absmean"]["A_noassist"],
            "power_B": o["agent_power_absmean"]["B_assisted"],
            "power_reduction_pct": o["agent_power_reduction_pct"],
            "conservation_rmse": o["conservation_rmse"],
            "rho": o.get("corr_exo_agentA", float("nan")),
            "exo_nrmse": o.get("exo_tracking_nrmse", float("nan")),
            "pred_reduction_pct": o.get("predicted_reduction_pct", float("nan")),
            "sup_opt_pct": o.get("superposition_opt_reduction_pct", float("nan")),
            "beta_star": o.get("beta_superposition_opt", float("nan")),
            "err6_A": o["dof_err6_mean"]["A_noassist"],
            "err6_B": o["dof_err6_mean"]["B_assisted"],
            "rew_A": o["reward_mean"]["A_noassist"],
            "rew_B": o["reward_mean"]["B_assisted"],
            "num_episodes": m["num_episodes"],
            "mean_ep_len": m["mean_episode_length"],
            "cause_counts": m["cause_counts"],
            "steps": r["episode_steps"],
        })
    runs.sort(key=lambda r: r["beta"])
    return runs


def markdown_table(runs) -> str:
    head = ("| β | 에이전트 토크 RMS (A→B) | Δagent | offload 효율 | "
            "\\|파워\\| (A→B) | Δpower | 보조 RMS | B 총 RMS | 보존오차 | "
            "ρ | 해석예측 | err6 (A→B) | reward (A→B) | 생존 |")
    sep = "|" + "---|" * 14
    rows = [head, sep]
    for r in runs:
        surv = ("완주" if r["cause_counts"].get("horizon", 0) == r["num_episodes"]
                else ", ".join(f"{c}×{n}" for c, n in r["cause_counts"].items())
                + f" (평균 {r['mean_ep_len']:.0f}스텝)")
        rows.append(
            f"| **{r['beta']:g}** | {r['agent_rms_A']:.1f} → "
            f"{r['agent_rms_B']:.1f} | **{-r['reduction_pct']:+.1f}%** | "
            f"{r['offload_eff']:.2f} | {r['power_A']:.0f} → {r['power_B']:.0f} | "
            f"{-r['power_reduction_pct']:+.1f}% | {r['exo_rms']:.1f} | "
            f"{r['total_rms_B']:.1f} | {r['conservation_rmse']:.1f} | "
            f"{r['rho']:.2f} | {-r['pred_reduction_pct']:+.1f}% | "
            f"{r['err6_A']:.4f} → {r['err6_B']:.4f} | "
            f"{r['rew_A']:.3f} → {r['rew_B']:.3f} | {surv} |")
    return "\n".join(rows)


def plot_sweep(runs, out_png: Path):
    b = np.array([r["beta"] for r in runs])
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    ax = axes[0]
    ax.plot(b, [r["agent_rms_A"] for r in runs], "o-", color="black",
            label="A: agent torque (no assist)")
    ax.plot(b, [r["agent_rms_B"] for r in runs], "o-", color="tab:red",
            label="B: agent torque (assisted)")
    ax.plot(b, [r["exo_rms"] for r in runs], "o-", color="tab:green",
            label="B: exo assist")
    ax.plot(b, [r["total_rms_B"] for r in runs], "o--", color="0.6",
            label="B: total (agent + exo)")
    ax.set_xlabel("assist gain beta"), ax.set_ylabel("hip 6ch torque RMS [N·m]")
    ax.set_title("Torque split vs assist gain")
    ax.legend(fontsize=8), ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(b, [r["reduction_pct"] for r in runs], "o-", color="tab:red",
            label="agent torque RMS reduction")
    ax.plot(b, [r["power_reduction_pct"] for r in runs], "s-",
            color="tab:purple", label="agent |power| reduction")
    ax.plot(b, [r["pred_reduction_pct"] for r in runs], "--", color="tab:red",
            alpha=0.6, label="analytic: 1-sqrt(1-2*rho*r+r^2)")
    ax.plot(b, [r["sup_opt_pct"] for r in runs], "-.", color="tab:gray",
            lw=1.2, label="superposition opt at measured rho: 1-sqrt(1-rho^2)")
    ax.plot(b, 100 * b, ":", color="black",
            label="ideal (reduction = 100·beta)")
    ax.set_xlabel("assist gain beta"), ax.set_ylabel("reduction [%]")
    ax.set_title("Active offload of agent effort")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(b, [r["offload_eff"] for r in runs], "^--", color="tab:blue",
             alpha=0.8, label="offload efficiency")
    ax2.set_ylabel("offload efficiency (reduction / beta)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")

    ax = axes[2]
    ax.plot(b, [r["err6_A"] for r in runs], "o-", color="black",
            label="A: dof err6")
    ax.plot(b, [r["err6_B"] for r in runs], "o-", color="tab:red",
            label="B: dof err6")
    ax.set_xlabel("assist gain beta"), ax.set_ylabel("mean |dof err| 6ch [rad]")
    ax.set_title("Cost side: tracking / reward / survival")
    ax.grid(alpha=0.3)
    ax3 = ax.twinx()
    ax3.plot(b, [r["rew_B"] for r in runs], "s--", color="tab:orange",
             label="B: mean reward")
    ax3.plot(b, [r["mean_ep_len"] / r["steps"] for r in runs], "^:",
             color="tab:blue", label="B: mean ep len / horizon")
    ax3.set_ylabel("reward / survival fraction")
    h1, l1 = ax.get_legend_handles_labels()
    h3, l3 = ax3.get_legend_handles_labels()
    ax.legend(h1 + h3, l1 + l3, fontsize=8, loc="lower left")

    fig.suptitle("Additive assist sweep: tau_total = tau_agent(full PD) "
                 "+ beta * tau_exo (ManiFlow), 60 s each")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dirs", nargs="+", required=True,
                   help="가산 보조 실행 결과 디렉토리들 (metrics.json 포함)")
    p.add_argument("--output", required=True, help="요약 저장 디렉토리")
    args = p.parse_args()

    runs = load_runs(args.dirs)
    assert runs, "assist 블록을 가진 실행 결과가 없습니다"
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    table = markdown_table(runs)
    betas = ", ".join("{:g}".format(r["beta"]) for r in runs)
    print(f"\n{len(runs)}개 실행 집계 (β = {betas})\n")
    print(table)
    (out_dir / "assist_beta_sweep.md").write_text(table + "\n")
    with open(out_dir / "assist_beta_sweep.json", "w") as f:
        json.dump(runs, f, indent=2)
    plot_sweep(runs, out_dir / "assist_beta_sweep.png")
    print(f"\n저장: {out_dir}/assist_beta_sweep.{{md,json,png}}")


if __name__ == "__main__":
    main()
