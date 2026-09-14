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
"""HLP(느린 예측기) + LLP(빠른 제어기) 타이밍 그림 → `docs/hlp_llp_timing.png`.

예측 범위를 정하는 부등식을 그림으로 옮긴 것:
    예측 범위 ≥ 갱신 주기 + 연산 시간 + 원하는 선행

실행 (sbc env):
    python examples/experiments/assist_skeleton/plot_timing_diagram.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

T0, T1 = 85, 185
fig, ax = plt.subplots(figsize=(13.5, 6.6))
ax.set_xlim(T0, T1); ax.set_ylim(0, 10.4); ax.axis("off")

INK="#141a21"; SOFT="#5d6a78"; TEAL="#0e7a74"; ORANGE="#c2410c"; GREY="#9aa8b6"

# 시간축
ax.annotate("", xy=(T1, 0.7), xytext=(T0, 0.7), arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
for t in range(90, 181, 10):
    ax.plot([t, t], [0.55, 0.85], color=INK, lw=1)
    ax.text(t, 0.15, f"{t}", ha="center", fontsize=9, color=SOFT)
ax.text(T1, 0.15, "ms", ha="right", fontsize=9, color=SOFT)

def lane(y, label, sub=""):
    ax.text(T0-1.0, y, label, ha="right", va="center", fontsize=10.5, color=INK, fontweight="bold")
    if sub:
        ax.text(T0-1.0, y-0.42, sub, ha="right", va="center", fontsize=8.5, color=SOFT)

# --- 1. 물리/센서 (2.5 ms) ---
lane(9.4, "물리 substep", "2.5 ms")
for t in [T0 + 2.5*i for i in range(int((T1-T0)/2.5)+1)]:
    ax.plot([t, t], [9.15, 9.65], color=GREY, lw=0.8)

# --- 2. LLP (10 ms) ---
lane(8.0, "LLP 추론", "10 ms · 0.25 ms 연산")
for t in range(90, 181, 10):
    ax.add_patch(Rectangle((t, 7.7), 1.2, 0.6, facecolor=TEAL, edgecolor="none"))
    ax.text(t+0.6, 8.55, "τ", ha="center", fontsize=8, color=TEAL)

# --- 3. HLP 주기 A/B ---
lane(6.0, "HLP 연산", "20 ms마다 · 5 ms")
for t0, col in ((100, ORANGE), (120, "#7c3aed"), (140, "#0369a1")):
    ax.add_patch(Rectangle((t0, 5.7), 5, 0.6, facecolor=col, edgecolor="none", alpha=.85))
    ax.text(t0+2.5, 6.55, f"입력 마감 {t0}", ha="center", fontsize=8, color=col)
    ax.plot([t0, t0], [0.7, 5.7], color=col, lw=0.7, ls=":", alpha=.6)
    ax.annotate("", xy=(t0+5, 5.35), xytext=(t0+5, 5.75),
                arrowprops=dict(arrowstyle="->", color=col, lw=1.4))
    ax.text(t0+5.4, 5.2, f"도착 {t0+5}", fontsize=8, color=col, va="center")

# --- 4. chunk 유효 구간 ---
lane(3.6, "chunk 유효 범위", "기준시각 +0…+50 ms")
for i, (t0, col, lbl) in enumerate(((100, ORANGE, "A"), (120, "#7c3aed", "B"), (140, "#0369a1", "C"))):
    y = 3.9 - i*0.75
    ax.add_patch(Rectangle((t0, y-0.22), 50, 0.44, facecolor=col, alpha=.18, edgecolor=col, lw=1))
    for k in range(6):
        ax.plot([t0+10*k], [y], marker="|", color=col, ms=9, mew=1.6)
    ax.text(t0+50+1.0, y, f"chunk {lbl}", fontsize=9, color=col, va="center")
    ax.plot([t0, t0+5], [y, y], color=col, lw=6, alpha=.35)   # 아직 도착 전 구간
    ax.text(t0+2.5, y+0.30, "연산중", fontsize=7, color=col, ha="center")

# --- 5. 워스트 케이스 표시 ---
ax.add_patch(Rectangle((120, 0.95), 5, 0.42, facecolor="#fde68a", edgecolor="#b45309", lw=1))
ax.text(122.5, 1.16, "B 연산중", fontsize=7.5, ha="center", va="center", color="#b45309")
ax.annotate("", xy=(140, 1.55), xytext=(120, 1.55),
            arrowprops=dict(arrowstyle="<->", color="#b45309", lw=1.3))
ax.text(130, 1.70, "가장 불리한 순간: t=120에서도 chunk A가 140까지 덮어야 한다",
        ha="center", fontsize=9, color="#b45309")

ax.text(T0, 10.1,
        "예측 범위 ≥ 갱신 주기(20) + 연산 시간(5) + 원하는 선행(20) = 45 ms  →  10 ms 격자 6점 = 50 ms",
        fontsize=11.5, color=INK, fontweight="bold")
fig.tight_layout()
out = "docs/hlp_llp_timing.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved", out)
