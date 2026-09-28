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
"""착용자(tracker) 자체가 얼마나 사람 같은가 — run_system.py 실행 기록의 무보조 env(A)로 잰다.

v33(사람 범위 근력) 판정용. 같은 실행의 보조 env(B)에서 LLP의 τ² 변화도 같이 찍는다.

| 지표 | 뜻 | v32 (참고) | v33 목표 |
|---|---|---|---|
| hip 스윙 진폭 A / mocap | 각도 진폭이 mocap에 맞는가 | 54.5° / 46° | mocap ±3° |
| hip 토크 스트라이드 피크 (N·m, N·m/kg) | 착지 슬램 크기 | −332 (3.7) | ≤ 200 (2.2) |
| 주기 평균 토크 피크 | 문헌 ≈ 1 N·m/kg | −157 (1.7) | ≈ 1 |
| 피크/RMS | bang-bang 정도 | 4.3 | < 3 |
| 몸체 오차 A [m] | mocap 몸체 추종 | 0.069 | ≤ 0.08 |
| B: τ² 변화 | 얹은 LLP의 효과 | v13 −30.5 % | — |

실행 (sbc env):
    python examples/experiments/assist_skeleton/eval_wearer.py results/oracle/*_tracker_v33*_m0 results/oracle/*_tracker_v33*_m1
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_html import A, B, load, metrics  # noqa: E402

WEARER_MASS_KG = 90.7


def wearer_stats(run: Path):
    tr, meta, idx = load(run)
    tau = (tr["tau_pd_prev"] + tr["tau_ff_prev"])[idx]  # (n, env, dof)
    ph = tr["gait_phase"][idx]                             # (n, leg)
    m = metrics(run)
    out = {"run": run.name, "survived": meta.get("survived_s"), "body_A": m.get("body_A"), "tau2_B": m["tau2"],
           "omega_B": m["omega"], "amp_A": m.get("amp_A"), "amp_B": m.get("amp_B"), "amp_ref": m.get("amp_ref"),
           "assist_rms": m["assist_rms"], "corr": m["corr"]}
    tA = tau[:, A]
    out["rms_A"] = float(np.sqrt((tA ** 2).mean()))
    # 스트라이드별 피크 (신전 = 음수): 위상 0 % 경계로 스트라이드를 자른다 (오른다리, dof 0)
    wraps = np.where(np.diff(ph[:, 0]) < -0.5)[0] + 1
    ext, flex = [], []
    for a, b in zip(wraps[:-1], wraps[1:]):
        seg = tA[a:b, 0]
        if len(seg) > 10:
            ext.append(seg.min()); flex.append(seg.max())
    out["stride_ext_peak"] = float(np.median(ext)) if ext else float("nan")
    out["stride_ext_p95"] = float(np.percentile(ext, 5)) if ext else float("nan")   # 가장 센 5 %
    out["stride_flex_peak"] = float(np.median(flex)) if flex else float("nan")
    out["peak_over_rms"] = float(np.abs(np.median(ext)) / out["rms_A"]) if ext else float("nan")
    # 주기 평균 프로파일 피크 (양쪽 다리)
    bins = 50
    prof = np.array([tA[(ph >= k / bins) & (ph < (k + 1) / bins)].mean() for k in range(bins)])
    out["cycle_ext_peak"], out["cycle_flex_peak"] = float(prof.min()), float(prof.max())
    out["n_strides"] = len(ext)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="run_system.py 실행 디렉터리(glob 가능)")
    p.add_argument("--mass", type=float, default=WEARER_MASS_KG)
    args = p.parse_args()
    runs = sorted({Path(h) for pat in args.runs for h in glob.glob(pat)})
    hdr = f"{'run':38s} {'생존':>8s} {'몸체A[m]':>8s} {'스윙A/B/mocap[°]':>18s} {'RMS_A':>6s} {'스트라이드 신전피크':>16s} {'(N·m/kg)':>8s} {'p5':>6s} {'주기평균 신전/굴곡':>16s} {'피크/RMS':>8s} {'B τ²':>7s} {'보조RMS':>7s} {'상관':>6s}"
    print(hdr)
    for r in runs:
        s = wearer_stats(r)
        surv = f"{s['survived']:.1f}s 낙상" if s["survived"] else "완주"
        amp = f"{s['amp_A']:.1f}/{s['amp_B']:.1f}/{s['amp_ref']:.1f}" if s.get("amp_A") is not None else "—"
        print(f"{s['run'][:38]:38s} {surv:>8s} {s['body_A']:8.3f} {amp:>18s} {s['rms_A']:6.1f} {s['stride_ext_peak']:16.0f} "
              f"{abs(s['stride_ext_peak'])/args.mass:8.2f} {s['stride_ext_p95']:6.0f} {s['cycle_ext_peak']:8.0f}/{s['cycle_flex_peak']:<7.0f} "
              f"{s['peak_over_rms']:8.2f} {s['tau2_B']:+7.1f} {s['assist_rms']:7.1f} {s['corr']:+6.2f}")
    print("스트라이드 신전피크 = 오른쪽 hip 스트라이드별 최소 토크의 중앙값(p5 = 가장 센 5 %), 주기평균 = 위상 50 bin 평균 프로파일의 극값, B τ² = 보조 켠 env의 착용자 τ² 변화 [%]")


if __name__ == "__main__":
    main()
