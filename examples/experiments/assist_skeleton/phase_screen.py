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
"""Development screen against L1 x0.5; passing is NOT generalization evidence.

Thresholds are declared before the L8 results: each 10% phase may regress at most
1 Nm in RMS change, but worst phase and the two suspect intervals must not regress.
Body tracking tolerance is 1 mm above the run's unassisted control. Angular-speed
change magnitude may exceed the L1 change by at most two percentage points.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

SCREEN = {"phase_regression_tolerance_Nm": 1.0, "body_error_increase_limit_m": .001,
          "omega_extra_change_limit_pp": 2.0, "speed_change_limit_pct": 2.0,
          "duration_s": 40.0}


def screen_motion(row, ref, diag, ref_diag, completed, gain, limit):
    """Keep individual failures: good phases/metrics cannot cancel bad ones."""
    checks = []
    def add(name, value, maximum):
        finite = value is not None and maximum is not None and math.isfinite(value) and math.isfinite(maximum)
        checks.append({"name": name, "value": value if value is not None and math.isfinite(value) else None,
                       "maximum": maximum if maximum is not None and math.isfinite(maximum) else None,
                       "pass": bool(finite and value <= maximum)})
    checks.append({"name": "40s_complete", "pass": bool(completed)})
    checks.append({"name": "gain1_limit100", "pass": gain == 1.0 and limit == 100.0})
    add("global_tau2_change_pct", row.get("τ²"), ref.get("τ²"))
    if "phase" not in diag or "phase" not in ref_diag:
        checks.append({"name": "phase_data_available", "pass": False})
        return {"pass": False, "checks": checks}
    checks.append({"name": "ten_phase_bins", "pass": len(diag["phase"]) == len(ref_diag["phase"]) == 10})
    for i, (phase, base) in enumerate(zip(diag["phase"], ref_diag["phase"])):
        if not phase["samples"] or not base["samples"]:
            checks.append({"name": f"phase_{i*10}_{(i+1)*10}_samples", "pass": False})
            continue
        # New (B-A) minus baseline (B-A), not signed torque averages.
        regression = base["rms_reduction_Nm"] - phase["rms_reduction_Nm"]
        add(f"phase_{i*10}_{(i+1)*10}_RMS_regression_Nm", regression, SCREEN["phase_regression_tolerance_Nm"])
    add("worst_phase_RMS_increase_Nm", diag.get("worst_phase_rms_increase_Nm"), ref_diag.get("worst_phase_rms_increase_Nm"))
    for interval in ("20-40", "70-90"):
        current, base = diag["intervals"].get(interval, {}), ref_diag["intervals"].get(interval, {})
        if "agent_B_rms_Nm" not in current or "agent_B_rms_Nm" not in base:
            checks.append({"name": f"interval_{interval}_available", "pass": False})
            continue
        add(f"interval_{interval}_RMS_increase_Nm",
            current["agent_B_rms_Nm"] - current["agent_A_rms_Nm"],
            base["agent_B_rms_Nm"] - base["agent_A_rms_Nm"])
    add("body_error_increase_m", row.get("body_B", math.inf) - row.get("body_A", 0), SCREEN["body_error_increase_limit_m"])
    add("absolute_omega_change_pct", abs(row.get("|ω|", math.inf)), abs(ref.get("|ω|", 0)) + SCREEN["omega_extra_change_limit_pp"])
    add("absolute_speed_change_pct", abs(row.get("speed", math.inf)), SCREEN["speed_change_limit_pct"])
    return {"pass": all(c["pass"] for c in checks), "checks": checks}


def screen_records(records, diagnostics):
    refs = {int(k.rsplit("_m",1)[1]): k for k in records if k.startswith("reference_L1_")}
    result, groups = {}, {}
    for key, row in records.items():
        if key.startswith("reference"):
            continue
        run = Path(row["run_dir"])
        meta = json.loads((run / "meta.json").read_text())
        with np.load(run / "traces.npz") as tr:
            t = tr["t"]
        completed = (meta["survived_s"] is None and meta["seconds"] >= SCREEN["duration_s"]
                     and len(t) >= round(meta["seconds"] / meta["llp_dt"]))
        motion = meta["motion_id"]
        ref_key = refs.get(motion)
        if ref_key is None:
            result[key] = {"pass": False, "checks": [{"name": "matching_L1_reference", "pass": False}]}
        else:
            effective_limit = abs(meta["assist_scale"])*min(meta["torque_limit"],meta.get("policy_torque_limit",meta["torque_limit"]))
            result[key] = screen_motion(row, records[ref_key], diagnostics[key], diagnostics[ref_key],
                                        completed, meta["assist_scale"], effective_limit)
        groups.setdefault(key.rsplit("_m",1)[0], {})[motion] = result[key]["pass"]
    return {"scope": "Development screen only; multiple seeds and held-out conditions still required.",
            "thresholds": SCREEN, "evaluations": result,
            "candidate_pass": {k: set(m) == {0,1} and all(m.values()) for k,m in groups.items()}}


def main():
    from report_torque_html import rank_candidates
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="results/llp_wide100_sequence/comparison.json")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    path = Path(args.manifest)
    records = json.loads(path.read_text())
    _, diagnostics = rank_candidates(records)
    result = screen_records(records, diagnostics["evaluations"])
    output = Path(args.out) if args.out else path.with_name("phase_screen.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    for key, passed in result["candidate_pass"].items():
        print(key, "PASS (development only)" if passed else "FAIL")


if __name__ == "__main__":
    main()
