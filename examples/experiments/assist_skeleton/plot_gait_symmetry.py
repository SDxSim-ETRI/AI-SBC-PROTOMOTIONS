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
"""착용자 좌우 고관절 궤적 비교 그림 (관절각 추종 항 추가 전/후).

`run_system.py`가 남긴 `traces.npz` 두 개를 읽어 왼쪽·오른쪽 고관절 굴곡 각도와
가동범위를 나란히 그린다. 보는 것은 **보조를 끈 A env**라 순수한 착용자 보행이다.

실행 (sbc env):
    python examples/experiments/assist_skeleton/plot_gait_symmetry.py
"""
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False
import numpy as np

RUNS = [("기존 v31", "results/gait_eval/*_v6final_m0"),
        ("재학습 v32", "results/sym_check/*_final_m0")]
fig, axes = plt.subplots(2, 2, figsize=(13, 6.4), gridspec_kw={"width_ratios": [3, 1]})
for row, (tag, pat) in enumerate(RUNS):
    d = sorted(glob.glob(pat))[-1]
    tr = np.load(d + "/traces.npz")
    t = tr["t"]; m = (t >= 4.0) & (t <= 9.0)
    hp = np.degrees(tr["hip_pos"][m, 0]); ref = np.degrees(tr["ref_hip"][m, 0]); tt = t[m]
    ax = axes[row, 0]
    ax.plot(tt, ref[:, 0], color="0.6", lw=1.0, ls="--", label="mocap 오른쪽")
    ax.plot(tt, ref[:, 1], color="0.6", lw=1.0, ls=":", label="mocap 왼쪽")
    ax.plot(tt, hp[:, 0], color="#c2410c", lw=1.8, label="착용자 오른쪽")
    ax.plot(tt, hp[:, 1], color="#0e7a74", lw=1.8, label="착용자 왼쪽")
    ax.set_title(f"{tag} — 고관절 굴곡 각도", loc="left", fontsize=11)
    ax.set_ylabel("각도 [deg]"); ax.set_ylim(-45, 60); ax.grid(alpha=.25)
    if row == 1:
        ax.set_xlabel("시간 [s]")
    ax.legend(ncol=4, fontsize=8, loc="upper right", framealpha=.9)

    # 가동범위 막대
    m2 = t >= 2.0
    hp2 = np.degrees(tr["hip_pos"][m2, 0]); ref2 = np.degrees(tr["ref_hip"][m2, 0])
    rom = [hp2[:, i].max() - hp2[:, i].min() for i in range(2)]
    rr = [ref2[:, i].max() - ref2[:, i].min() for i in range(2)]
    axb = axes[row, 1]
    x = np.arange(2)
    axb.bar(x - 0.2, rr, 0.36, color="0.75", label="mocap")
    axb.bar(x + 0.2, rom, 0.36, color=["#c2410c", "#0e7a74"], label="착용자")
    axb.set_xticks(x); axb.set_xticklabels(["오른쪽", "왼쪽"])
    axb.set_ylim(0, 80); axb.set_ylabel("가동범위 [deg]"); axb.grid(alpha=.25, axis="y")
    axb.set_title(f"좌우 차이 {abs(rom[0]-rom[1]):.1f}°  (mocap {abs(rr[0]-rr[1]):.1f}°)",
                  loc="left", fontsize=10)
    for xi, v in zip(x + 0.2, rom):
        axb.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=9)

fig.suptitle("보조 없는 상태의 착용자 보행 — 관절각 추종 항 추가 전/후", fontsize=13, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = "results/sym_check/symmetry_compare.png"
fig.savefig(out, dpi=130)
print("saved", out)
