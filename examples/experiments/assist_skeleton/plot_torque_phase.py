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
"""보행 한 주기로 접은 토크·파워 그림 — 보조 토크가 언제 나오고 언제 쉬는지.

`run_system.py`의 traces.npz를 읽어 최대 굴곡을 위상 0 %로 잡고 주기 평균한다.
보조가 파워(|τ·ω|)를 목적으로 하기 때문에, 착용자 토크가 가장 큰 순간이 아니라
파워가 큰 순간에 토크를 쓴다는 것을 보여 준다.

실행 (sbc env):
    python examples/experiments/assist_skeleton/plot_torque_phase.py <run_dir>
"""

import sys
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

N = 100


def cycle_average(x, peaks, dt):
    out = []
    for a, b in zip(peaks[:-1], peaks[1:]):
        if not (0.7 < (b - a) * dt < 1.6):
            continue
        out.append(np.interp(np.linspace(0, 1, N), np.linspace(0, 1, b - a), x[a:b]))
    return np.array(out).mean(0)


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("results/gait_eval/*_v10_m0"))[-1]
    tr = np.load(run + "/traces.npz")
    t = tr["t"]
    dt = float(t[1] - t[0])
    m = t >= 2.0
    q = tr["hip_pos"][m, :, 0]
    w = tr["hip_vel"][m, :, 0]
    tau = (tr["tau_pd_prev"] + tr["tau_ff_prev"])[m, :, 0]
    assist = tr["tau_assist"][m, 1, 0]
    pk, _ = find_peaks(q[:, 0], distance=int(0.7 / dt))

    Q = cycle_average(np.degrees(q[:, 0]), pk, dt)
    W = cycle_average(w[:, 0], pk, dt)
    TA = cycle_average(tau[:, 0], pk, dt)
    TB = cycle_average(tau[:, 1], pk, dt)
    S = cycle_average(assist, pk, dt)
    PA = cycle_average(np.abs(tau[:, 0] * w[:, 0]), pk, dt)
    x = np.arange(N)

    fig, ax = plt.subplots(3, 1, figsize=(11.5, 8.2), sharex=True,
                           gridspec_kw={"height_ratios": [1, 1.5, 1]})
    INK, TEAL, ORANGE, GREY = "#141a21", "#0e7a74", "#c2410c", "#9aa8b6"

    ax[0].plot(x, Q, color=INK, lw=2, label="고관절 각도 (+굴곡)")
    ax[0].plot(x, np.degrees(W) / 10, color=GREY, lw=1.4, ls="--", label="각속도 ÷10 [deg/s]")
    ax[0].axhline(0, color=GREY, lw=.6)
    ax[0].set_ylabel("각도 [deg]")
    ax[0].legend(fontsize=8.5, ncol=2, loc="upper right")

    ax[1].plot(x, TA, color=ORANGE, lw=2, label="착용자 토크 (보조 없음)")
    ax[1].plot(x, TB, color=ORANGE, lw=1.2, ls=":", label="착용자 토크 (보조 켬)")
    ax[1].plot(x, S, color=TEAL, lw=2.2, label="보조 토크")
    ax[1].axhline(0, color=GREY, lw=.6)
    ax[1].axhline(40, color=TEAL, lw=.8, ls="--")
    ax[1].axhline(-40, color=TEAL, lw=.8, ls="--")
    ax[1].text(99, 41, "보조 한계 ±40 N·m", ha="right", fontsize=8, color=TEAL)
    ax[1].set_ylabel("토크 [N·m]")
    ax[1].legend(fontsize=8.5, ncol=3, loc="lower right")

    ax[2].fill_between(x, 0, PA, color=ORANGE, alpha=.25)
    ax[2].plot(x, PA, color=ORANGE, lw=1.8, label="착용자 파워 |τ·ω| (보조 없음)")
    ax[2].plot(x, np.abs(S) * 2, color=TEAL, lw=1.4, ls="--", label="|보조 토크| ×2 [N·m]")
    ax[2].set_ylabel("파워 [W]")
    ax[2].set_xlabel("보행 주기 [%] — 0 % = 최대 굴곡")
    ax[2].legend(fontsize=8.5, loc="upper right")

    i = TA.argmin()
    for a in ax:
        a.axvspan(max(i - 4, 0), i + 4, color="#fde68a", alpha=.45, zorder=0)
        a.grid(alpha=.25)
    ax[1].annotate(f"착용자 토크 음의 피크 {TA[i]:.0f} N·m\n하지만 ω≈{W[i]:+.1f} rad/s → 파워 {PA[i]:.0f} W\n보조는 {S[i]:+.0f} N·m",
                   xy=(i, TA[i]), xytext=(i + 6, TA[i] - 20), fontsize=9, color=INK,
                   arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))

    fig.suptitle("보조 토크는 착용자 토크가 큰 곳이 아니라 파워가 큰 곳에서 나온다", fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = "docs/torque_phase.png"
    fig.savefig(out, dpi=135)
    print("saved", out)


if __name__ == "__main__":
    main()
