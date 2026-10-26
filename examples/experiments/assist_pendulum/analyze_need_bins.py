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
"""Stored-record analysis: how L1 behaves where the unassisted wearer needs little torque.

Bins stored pendulum (real H1) and W1 walking records by the unassisted wearer
torque |tau_A| and reports the assistance magnitude, saturation, sign agreement and
the assisted wearer torque squared per bin. No simulation is run. Also reports the
mean of the R1 relative excess term on the L1 pendulum record for weight calibration.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BINS = [0, 5, 10, 20, 40, 80, 400]


def binned(abs_need, assist, tau_a, tau_b, limit):
    rows = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        s = (abs_need >= lo) & (abs_need < hi)
        if not s.any():
            continue
        rows.append(dict(need_lo_Nm=lo, need_hi_Nm=hi, sample_fraction=float(s.mean()),
            assist_rms_Nm=float(np.sqrt(np.mean(assist[s] ** 2))),
            saturation_fraction=float(np.mean(np.abs(assist[s]) > .95 * limit)),
            sign_agreement=float(np.mean(np.sign(assist[s]) == np.sign(tau_a[s]))),
            A_tau_sq_Nm2=float(np.mean(tau_a[s] ** 2)), B_tau_sq_Nm2=float(np.mean(tau_b[s] ** 2))))
    return rows


def pendulum(path):
    z = np.load(path)
    mask = z["valid"].astype(bool) & (z["time_s"] > .5)
    m = np.repeat(mask[:, :, None], 2, 2)
    ta, tb, assist = z["reference_tau"][m], z["assisted_tau"][m], z["assist_tau"][m]
    result = dict(source=str(Path(path).relative_to(ROOT)), joint_samples=int(m.sum()),
        assist_rms_Nm=float(np.sqrt(np.mean(assist ** 2))), saturation_fraction=float(np.mean(np.abs(assist) > 19.)),
        corr_assist_vs_unassisted_wearer=float(np.corrcoef(assist, ta)[0, 1]),
        sign_agreement=float(np.mean(np.sign(assist) == np.sign(ta))),
        bins=binned(np.abs(ta), assist, ta, tb, 20.))
    ta_sq = z["reference_tau"] ** 2
    for floor in (5., 10.):
        for window in (1, 5):
            c = np.cumsum(ta_sq, axis=0)
            if window > 1:
                c[window:] = c[window:] - c[:-window]
                den = c / np.minimum(np.arange(1, len(ta_sq) + 1), window)[:, None, None]
            else:
                den = ta_sq
            excess = (np.clip(z["assisted_tau"] ** 2 - ta_sq, 0, None) / (den + floor ** 2)).mean(-1)[mask]
            result[f"excess_floor{floor:g}_window{window}"] = dict(mean=float(excess.mean()), p50=float(np.median(excess)),
                p90=float(np.quantile(excess, .9)), p99=float(np.quantile(excess, .99)), max=float(excess.max()))
    hp = np.abs(z["assisted_tau"] * z["assisted_velocity"]).sum(-1)[mask].mean()
    tq = (z["assisted_tau"] ** 2).sum(-1)[mask].mean()
    result["stored_L1_costs_per_step"] = dict(human_power=float(.0016 * hp), human_torque_sq=float(3e-5 * tq), total=float(.0016 * hp + 3e-5 * tq))
    return result


def walking(path, limit):
    z = np.load(path)
    t = z["t"]; idx = np.arange(2, len(t))
    wearer = z["tau_pd_prev"] + z["tau_ff_prev"]
    a, b = wearer[idx, 0], wearer[idx, 1]
    assist = z["tau_assist"][idx - 1, 1]
    m = t[idx] >= 1.5
    a, b, assist = a[m], b[m], assist[m]
    return dict(source=str(Path(path).relative_to(ROOT)), joint_samples=int(a.size),
        A_rms_Nm=float(np.sqrt(np.mean(a ** 2))), B_rms_Nm=float(np.sqrt(np.mean(b ** 2))),
        assist_rms_Nm=float(np.sqrt(np.mean(assist ** 2))), saturation_fraction=float(np.mean(np.abs(assist) > .95 * limit)),
        corr_assist_vs_unassisted_wearer=float(np.corrcoef(assist.ravel(), a.ravel())[0, 1]),
        sign_agreement=float(np.mean(np.sign(assist) == np.sign(a))),
        bins=binned(np.abs(a), assist, a, b, limit))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results/llp_relative_20261020/l1_need_bins.json")
    args = p.parse_args()
    runs = json.loads((ROOT / "results/llp_output_scale_20261016/runs.json").read_text())
    report = dict(pendulum_L1_H1_legacy=pendulum(ROOT / "results/pendulum_research_followup_20261015/eval_L1_H1_legacy/rollout.npz"),
        walking={key: walking(ROOT / runs[key]["path"] / "traces.npz", runs[key]["limit_Nm"])
                 for key in ("L1_20Nm_seed42", "L1_20Nm_seed7", "L1_15Nm_seed42", "L1_15Nm_seed7")},
        note="Stored records only; bins by the unassisted wearer torque of the paired A run. Pendulum limit 20 N·m (L1 x0.5), real H1.")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for key, w in report["walking"].items():
        low = w["bins"][0]
        print(f"{key}: |A|<5 share {100*low['sample_fraction']:.1f}% assist RMS {low['assist_rms_Nm']:.1f} sat {100*low['saturation_fraction']:.0f}% sign {100*low['sign_agreement']:.0f}% A^2->B^2 {low['A_tau_sq_Nm2']:.1f}->{low['B_tau_sq_Nm2']:.1f}")


if __name__ == "__main__":
    main()
