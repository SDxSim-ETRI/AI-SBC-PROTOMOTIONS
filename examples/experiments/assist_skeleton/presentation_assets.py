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
"""Presentation figures and evidence for L1/R0 inference demonstrations."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

from report_output_scale import align, analyze_traces, masks_from_reference

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "results/llp_relative_20261020/walking/comparison.json"
FONT = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
COLORS = ["#4274b8", "#93a9c5", "#12866e"]
KEYS = ["L1_20", "L1_10", "R0_s0_20"]
LABELS = ["L1 · ±20 N·m", "L1 · ±10 N·m", "R0 · ±20 N·m"]


def save_figure(figure, folder, name, title, note):
    figure.suptitle(title, fontsize=27, fontweight="bold", x=.08, ha="left", y=.96)
    figure.text(.08, .025, note, fontsize=12, color="#586777")
    figure.subplots_adjust(left=.09, right=.97, top=.83, bottom=.16, wspace=.28)
    for extension in ("png", "svg"):
        figure.savefig(folder / f"{name}.{extension}", dpi=180, facecolor="white")
    plt.close(figure)


def charts(folder, archive):
    plt.rcParams.update({
        "font.family": FONT.get_name(), "font.size": 16, "axes.titlesize": 19,
        "axes.labelsize": 17, "xtick.labelsize": 15, "ytick.labelsize": 15,
        "axes.unicode_minus": False, "svg.fonttype": "path",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#9ca9b4", "axes.labelcolor": "#253545",
        "text.color": "#253545", "grid.color": "#e5e9ed", "grid.alpha": 1,
    })
    groups = archive["groups"]
    note = "기존 걷기 40초 · 평가 seed 42/7 평균(막대: 두 실행의 범위) · W1/H1 동결 · 토크²는 시뮬레이션 부담 지표"
    figure, axes = plt.subplots(1, 2, figsize=(16, 9))
    for axis, metric, label, upper in zip(
        axes, ["overall_tau_sq_reduction", "assist_rms"],
        ["착용자 고관절 τ² 감소 [(N·m)²]", "보조 토크 RMS [N·m]"], [340, 21],
    ):
        values = [groups[key]["metrics"][metric] for key in KEYS]
        means = np.array([v["mean"] for v in values])
        errors = np.array([[v["mean"] - v["min"] for v in values], [v["max"] - v["mean"] for v in values]])
        axis.bar(np.arange(3), means, color=COLORS, width=.58, zorder=3)
        axis.errorbar(np.arange(3), means, yerr=errors, fmt="none", capsize=5, color="#253545", zorder=4)
        for i, value in enumerate(means):
            axis.text(i, value + upper*.05, f"{value:.2f}", ha="center", fontsize=22, fontweight="bold")
        axis.set_xticks(np.arange(3), LABELS)
        axis.set_ylim(0, upper)
        axis.set_title(label, loc="left", pad=24)
        axis.grid(axis="y", zorder=0)
    save_figure(figure, folder, "graph_01_burden_and_assistance", "기존 보행에서 R0의 보조 효율", note)

    figure, axis = plt.subplots(figsize=(16, 9))
    x, y = [0.], [0.]
    for limit in (10, 15, 20):
        m = groups[f"L1_{limit}"]["metrics"]
        x.append(m["assist_rms"]["mean"])
        y.append(m["overall_tau_sq_reduction"]["mean"])
    axis.plot(x, y, "o--", color=COLORS[0], lw=2.6, ms=9, label="L1 출력 축소 곡선")
    for i, limit in enumerate((10, 15, 20), 1):
        axis.annotate(f"L1 ±{limit}", (x[i], y[i]), xytext=(12, -27), textcoords="offset points", color=COLORS[0])
    r = groups["R0_s0_20"]["metrics"]
    rx, ry = r["assist_rms"]["mean"], r["overall_tau_sq_reduction"]["mean"]
    axis.scatter([rx], [ry], s=190, color=COLORS[2], marker="D", zorder=5, label="R0 ±20 N·m")
    baseline = np.interp(rx, x, y)
    axis.plot([rx, rx], [baseline, ry], color=COLORS[2], lw=3)
    axis.annotate(f"R0\n{ry:.1f} (N·m)² / {rx:.2f} N·m", (rx, ry), xytext=(-200, 18), textcoords="offset points", color=COLORS[2])
    axis.annotate(f"같은 RMS의 L1 곡선보다\n+{ry-baseline:.1f} (N·m)²", (rx, (ry+baseline)/2),
                  xytext=(125, -90), textcoords="offset points", arrowprops={"arrowstyle": "-", "color": COLORS[2]}, color=COLORS[2])
    axis.set(xlabel="보조 토크 RMS [N·m]", ylabel="착용자 고관절 τ² 감소 [(N·m)²]", xlim=(0, 19), ylim=(0, 340))
    axis.grid(True)
    axis.legend(loc="lower right", frameon=False)
    save_figure(figure, folder, "graph_02_efficiency_curve", "같은 보조 크기에서의 부담 감소", note.replace("(막대: 두 실행의 범위)", "") + "\nL1 곡선은 평가점을 선형 보간 · R0 학습 seed 1개 · 달리기에서 같은 우위는 확인되지 않음")

    figure, axes = plt.subplots(1, 2, figsize=(16, 9))
    phase_x = (np.arange(25) + .5) * 4
    for side, axis in enumerate(axes):
        axis.axhspan(-20, 0, color="#f9e7e5", zorder=0)
        axis.axhline(0, color="#6d7d8d", lw=1)
        for key, color, label in zip(KEYS, COLORS, LABELS):
            selected = [r for r in archive["runs"].values()
                        if r["record"]["model"] == ("R0_s0" if key.startswith("R0") else "L1")
                        and r["record"]["limit_Nm"] == int(key.rsplit("_", 1)[1])]
            values = np.array([[r["rows"][side][b]["reduction"] for b in range(25)] for r in selected])
            axis.plot(phase_x, values.mean(axis=0), color=color, label=label, lw=2.6)
        axis.set(title=["오른쪽 고관절", "왼쪽 고관절"][side], xlim=(0, 100), ylim=(-20, 35),
                 xlabel="보행 위상 [%]", ylabel="착용자 RMS 감소 [N·m]")
        axis.set_xticks(np.arange(0, 101, 20))
        axis.grid(axis="y")
    axes[0].legend(loc="upper right", frameon=False, fontsize=13)
    save_figure(figure, folder, "graph_03_phase_benefit_and_interference", "도움과 방해가 나타나는 보행 구간", 
                "양수: 부담 감소 / 음수: 부담 증가 · 위상 0%는 고관절 최대 굴곡(발 착지 아님)\n기존 걷기 40초 · 평가 seed 42/7 평균 · 그래프 전 구간 표시")

    figure, axes = plt.subplots(1, 2, figsize=(16, 9))
    for axis, metric, title in zip(axes, ["low10_rms_increase", "worst_phase_rms_increase"],
                                   ["저부담 감속 구간", "좌우 전체 중 최악 위상"]):
        values = [groups[key]["metrics"][metric]["mean"] for key in KEYS]
        axis.bar(np.arange(3), values, color=COLORS, width=.58, zorder=3)
        for i, value in enumerate(values):
            axis.text(i, value+.6, f"{value:.2f}", ha="center", fontsize=23, fontweight="bold")
        axis.set_xticks(np.arange(3), LABELS)
        axis.set(title=title, ylabel="착용자 RMS 증가 [N·m]", ylim=(0, 18))
        axis.grid(axis="y", zorder=0)
    save_figure(figure, folder, "graph_04_remaining_interference", "남은 과제: 저부담 구간의 방해", 
                "작을수록 좋음 · 저부담 마스크는 무보조 L1 ±20 평가의 50 ms RMS ≤10 N·m에서 고정\n평가 seed 42/7 평균 · 비슷한 보조 RMS의 L1 ±10보다 R0의 방해는 큼")

    figure, axis = plt.subplots(figsize=(16, 9))
    categories = ["무릎", "발목", "몸통"]
    for i, (key, color, label) in enumerate(zip(KEYS, COLORS, LABELS)):
        selected = [r for r in archive["runs"].values()
                    if r["record"]["model"] == ("R0_s0" if key.startswith("R0") else "L1")
                    and r["record"]["limit_Nm"] == int(key.rsplit("_", 1)[1])]
        changes = [np.mean([next(row["B"] - row["A"] for row in r["other_joints"] if row["group"] == group)
                            for r in selected]) for group in categories]
        xs = np.arange(3) + (i-1)*.24
        axis.bar(xs, changes, width=.22, color=color, label=label, zorder=3)
        for x, value in zip(xs, changes):
            axis.text(x, value + (8 if value > 0 else -8), f"{value:+.0f}",
                      ha="center", va="bottom" if value > 0 else "top", fontsize=18)
    axis.set_xticks(np.arange(3), categories)
    axis.set(ylabel="관절별 τ² 평균 변화 [(N·m)²]", ylim=(-260, 190))
    axis.axhline(0, color="#647282", lw=1)
    axis.grid(axis="y", zorder=0)
    axis.legend(loc="upper left", frameon=False, ncol=3)
    save_figure(figure, folder, "graph_06_other_joints", "고관절 외의 부담 변화도 함께 확인", 
                "음수: 부담 감소 / 양수: 부담 증가 · 보조−무보조 · 각 부위 DOF의 τ² 평균\n기존 걷기 40초 · 평가 seed 42/7 평균 · R0에서도 몸통 부담 증가는 남음")


def fresh_evidence(study, folder):
    runs = {}
    for label in ("L1_20", "R0_20", "L1_10"):
        paths = [p for p in (study / "inference").glob(f"*_{label}_seed42") if (p / "traces.npz").exists()]
        if len(paths) != 1:
            raise ValueError(f"Need exactly one completed {label} run, found {paths}")
        path = paths[0]
        with np.load(path / "traces.npz") as data:
            tr = dict(data)
        runs[label] = dict(path=path, traces=tr, meta=json.loads((path / "meta.json").read_text()))
    ref = masks_from_reference(runs["L1_20"]["traces"])
    evidence = {}
    for label, run in runs.items():
        tr, meta = run["traces"], run["meta"]
        metrics = analyze_traces(tr, ref)
        assert meta["survived_s"] is None
        assert len(tr["t"]) == 4000 and np.allclose(np.diff(tr["t"]), .01, atol=2e-5)
        assert np.isfinite(tr["presentation_body_pos"]).all()
        assert np.allclose(np.linalg.norm(tr["presentation_body_rot"], axis=-1), 1, atol=2e-4)
        limit = 10 if label == "L1_10" else 20
        assert np.abs(tr["tau_assist"]).max() <= limit + 1e-4
        assert np.abs(tr["tau_assist"][:, 0]).max() == 0
        names = tr["presentation_dof_names"].tolist()
        hips = [names.index("hip_flexion_" + side) for side in ("r", "l")]
        assert np.allclose(tr["presentation_human_tau_prev"][:, :, hips], tr["tau_pd_prev"], atol=1e-4)
        selected = tr["t"] >= 1.5
        other = {}
        for group, token in [("knee", "knee"), ("ankle", "ankle"), ("lumbar", "lumbar")]:
            indices = [i for i, name in enumerate(names) if token in name]
            tau = tr["presentation_human_tau_prev"][selected][:, :, indices]
            other[group] = float(np.mean(tau[:, 1] ** 2) - np.mean(tau[:, 0] ** 2))
        pose_delta = tr["presentation_body_pos"][:, 1] - tr["presentation_body_pos"][:, 0]
        evidence[label] = dict(
            source_run=str(run["path"].relative_to(ROOT)), metrics=metrics,
            other_joint_tau_sq_change_Nm2=other, survived_s=40,
            motion_pair_position_difference_rms_m=float(np.sqrt(np.mean(pose_delta**2))),
            display_start_s=5, display_end_s=25,
        )
    figure, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True)
    for side in (0, 1):
        for label, color in [("L1_20", COLORS[0]), ("R0_20", COLORS[2])]:
            a = align(runs[label]["traces"])
            selected = (a["t"] >= 5) & (a["t"] < 7.5)
            x = a["t"][selected]
            if label == "L1_20":
                axes[0, side].plot(x, a["A"][selected, side], color="#78828c", label="무보조", lw=2)
            axes[0, side].plot(x, a["B"][selected, side], color=color, label=label.replace("_20", " ±20"), lw=2)
            axes[1, side].plot(x, a["assist"][selected, side], color=color, lw=2)
        axes[0, side].set(title=["오른쪽 고관절", "왼쪽 고관절"][side], ylabel="착용자 토크 [N·m]", ylim=(-170, 95))
        axes[1, side].set(ylabel="보조 토크 [N·m]", xlabel="시뮬레이션 시간 [s]", ylim=(-23, 23))
        for row in (0, 1):
            axes[row, side].axhline(0, color="#acb5bf", lw=.8)
            axes[row, side].grid(axis="y")
    axes[0, 0].legend(loc="lower right", frameon=False, fontsize=12)
    save_figure(figure, folder, "graph_05_demo_torques", "시연 영상과 같은 시점의 실제 토크", 
                "새 추론 seed 42 · 영상 시작 시각 5초부터 2.5초 · 100 Hz 원기록(평활화 없음)\n착용자 반응과 직전 10 ms 창의 보조 토크를 정렬 · 부호는 굴곡/신전 방향")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "results/presentation_walk_20261022")
    args = parser.parse_args()
    folder = args.study / "deliverables"
    folder.mkdir(parents=True, exist_ok=True)
    archive = json.loads(ARCHIVE.read_text())
    charts(folder, archive)
    evidence = fresh_evidence(args.study, folder)
    evidence["historical_summary_source"] = str(ARCHIVE.relative_to(ROOT))
    evidence["historical_summary_sha256"] = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    evidence["scope"] = "One fixed wearer and existing walking clip; simulated joint torque burden, not muscle activity or metabolism. No training."
    (folder / "evidence.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False))
    print(json.dumps({key: {"reduction_Nm2": -value["metrics"]["overall"]["tau_sq_increase"],
                           "assist_rms_Nm": value["metrics"]["overall"]["assist_rms"]}
                      for key, value in evidence.items() if isinstance(value, dict) and "metrics" in value}, indent=2))


if __name__ == "__main__":
    main()
