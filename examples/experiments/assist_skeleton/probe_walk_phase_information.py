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
"""Information probe: can measured hip windows distinguish walking phase?

Chronological split on ONE existing walking clip, with a 1 s exclusion gap.
Labels are privileged and used only for this diagnostic. This is not an LLP,
a deployment gate, or evidence of generalization to other gaits/wearers.
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def probe(path):
    with np.load(Path(path)/"traces.npz") as tr:
        t, q, v, phase = tr["t"], tr["hip_pos"][:, 1], tr["hip_vel"][:, 1], tr["gait_phase"]
    wraps = np.flatnonzero(np.diff(phase[:, 0]) < -.5)+1
    period = np.diff(t[wraps])
    idx = np.flatnonzero((t >= 2) & (t < 39.99))
    train = np.repeat(t[idx] < 20, 2)
    test = np.repeat(t[idx] >= 21, 2)
    target = phase[idx].reshape(-1)
    results = {}
    for name, offsets in (
        ("single_hip_current", [0]), ("both_hips_current", [0]),
        ("both_hips_250ms", list(range(26))),
        ("both_hips_1000ms", list(range(101)))):
        blocks = []
        for off in offsets:
            blocks += [q[idx-off, :, None], v[idx-off, :, None]]
            if name != "single_hip_current":
                blocks += [q[idx-off, ::-1, None], v[idx-off, ::-1, None]]
        x = np.concatenate(blocks, axis=-1).reshape(len(idx)*2, -1)
        mean, std = x[train].mean(axis=0), x[train].std(axis=0).clip(.001)
        _, neighbors = cKDTree((x[train]-mean)/std).query((x[test]-mean)/std, k=8)
        y = target[train][neighbors]
        predicted = np.angle(np.exp(2j*np.pi*y).mean(axis=1))/(2*np.pi) % 1
        truth = target[test]
        error = np.abs((predicted-truth+.5) % 1-.5)*100
        actual_positive = (truth >= .7) & (truth < .9)
        positive = (predicted >= .7) & (predicted < .9)
        tp = (positive & actual_positive).sum()
        results[name] = dict(features=x.shape[1], phase_mae_percent_cycle=float(error.mean()),
                             phase_p95_percent_cycle=float(np.percentile(error, 95)),
                             precision_70_90=float(tp/max(1, positive.sum())),
                             recall_70_90=float(tp/max(1, actual_positive.sum())))
    return dict(run_dir=str(path), train_interval_s=[2, 20], test_interval_s=[21, 39.99],
                train_hip_samples=int(train.sum()), test_hip_samples=int(test.sum()),
                period_median_s=float(np.median(period)),
                period_p05_p95_s=np.percentile(period, [5, 95]).tolist(),
                history_250ms_fraction_of_cycle=float(.25/np.median(period)),
                future_20ms_fraction_of_cycle=float(.02/np.median(period)),
                model="8-nearest-neighbor circular phase regression; standardization fit on train only",
                inputs="Measured B hip q and velocity only. No time, reference, torque, phase or contact input.",
                results=results,
                limitation="Single repeated gait: tests presence of phase information only, not LLP use of it or generalization. Longer-window results also depend on this probe estimator.")


def main():
    records = json.loads(Path("results/llp_L11_opposition_sequence/comparison.json").read_text())
    results = {name: probe(records[key]["run_dir"]) for name, key in (
        ("L1 x0.5", "reference_L1_x0.5_m0"), ("L9 @2000", "eval_llp_L9_opposition_e2000_m0"))}
    out = Path("results/cli_walk_diagnostic_20261009/phase_information.json")
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
