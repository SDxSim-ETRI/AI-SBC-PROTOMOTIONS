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
"""LLP 20 N·m 단일 관절·공유 연구(2026-10-21/22) 기록 무결성 점검 → verification.json.

    python examples/experiments/assist_pendulum/verify_motion_leg.py
"""

import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "results/llp20_motion_20261021"
CONDITIONS = ("A0D0", "A1D1", "A1D0", "A0D1")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    manifest = json.loads((STUDY / "manifest.json").read_text())
    eval_manifest = json.loads((STUDY / "eval_manifest.json").read_text())
    execution = json.loads((STUDY / "execution_right.json").read_text())
    runs = json.loads((STUDY / "eval_runs.json").read_text())
    checks = dict(training_sources_unchanged={k: sha(ROOT / k) == v for k, v in manifest["training_sources"].items()},
                  evaluation_sources_unchanged={k: sha(ROOT / k) == v for k, v in eval_manifest["sources"].items()},
                  evaluation_assets_unchanged={k: sha(ROOT / v["path"]) == v["sha256"] for k, v in eval_manifest["assets"].items()})
    training = {}
    for c in CONDITIONS:
        rec = execution[f"{c}_right"]
        ckpt = torch.load(ROOT / rec["checkpoint"], map_location="cpu", weights_only=False)
        env = torch.load(ROOT / rec["checkpoint"].replace("epoch_6000.ckpt", "resolved_configs.pt"),
                         map_location="cpu", weights_only=False)["env"]
        training[c] = dict(checkpoint=rec["checkpoint"], epoch=ckpt["epoch"], sha256_matches_record=sha(ROOT / rec["checkpoint"]) == rec["checkpoint_sha256"],
                           all_parameters_finite=all(not torch.is_tensor(v) or bool(torch.isfinite(v).all()) for v in ckpt["model"].values()),
                           active_joints=list(env.assist_active_joints), limit=float(env.assist_torque_limit),
                           reward_weights={k: float(env.reward_components[k].static_params["weight"]) for k in ("assist_effort", "action_rate")},
                           elapsed_h=rec["elapsed_s"] / 3600, attempts=rec["attempt"])
    walking, pendulum = {}, {}
    for identity, r in runs.items():
        path = ROOT / r["path"]
        if r["kind"] == "w1":
            meta = json.loads((path / "meta.json").read_text())
            walking[identity] = dict(fall=meta["survived_s"], share_legs=meta.get("llp_share_legs"), assist_scale=meta["assist_scale"],
                                     traces=(path / "traces.npz").exists(), fullbody=(path / "fullbody.npz").exists(),
                                     lowpass_hz=meta.get("torque_lpf_hz"))
        else:
            s = json.loads((path / "summary.json").read_text())
            pendulum[identity] = dict(failures=s["failures"], rollout=(path / "rollout.npz").exists(),
                                      zero_action_max_q_difference_rad=s.get("zero_action_max_q_difference_rad"),
                                      injection_error_max_Nm=s["injection_error_max_Nm"], assist_max_abs_Nm=s["assist_max_abs_Nm"])
    new_w1 = {k: v for k, v in walking.items() if "_M_" in k}
    checks.update(
        training=training,
        all_training_epoch_6000=all(t["epoch"] == 6000 and t["all_parameters_finite"] and t["sha256_matches_record"] for t in training.values()),
        evaluation_records=len(runs), w1_runs=len(walking), pendulum_runs=len(pendulum),
        w1_falls=sum(v["fall"] is not None for v in walking.values()),
        new_models_shared_on_both_legs=all(v["share_legs"] for v in new_w1.values()),
        w1_outputs_complete=all(v["traces"] and v["fullbody"] for v in walking.values()),
        pendulum_outputs_complete=all(v["rollout"] for v in pendulum.values()),
        pendulum_assist_within_20Nm=all(v["assist_max_abs_Nm"] <= 20.0001 for v in pendulum.values()),
        zero_action_identity={k: v["zero_action_max_q_difference_rad"] for k, v in pendulum.items() if "zero_action" in k},
        diagnostics=sorted(p.name for p in (STUDY / "diagnostics").glob("chunk_*.json")),
        runtime_failures=dict(training=json.loads((STUDY / "runtime_failures_right.json").read_text())
                              if (STUDY / "runtime_failures_right.json").exists() else [],
                              evaluation=json.loads((STUDY / "eval_failures.json").read_text())
                              if (STUDY / "eval_failures.json").exists() else []),
        walking=walking, pendulum=pendulum)
    checks["passed"] = (all(checks["training_sources_unchanged"].values()) and all(checks["evaluation_sources_unchanged"].values())
                        and all(checks["evaluation_assets_unchanged"].values()) and checks["all_training_epoch_6000"]
                        and checks["w1_falls"] == 0 and checks["new_models_shared_on_both_legs"] and checks["w1_outputs_complete"]
                        and checks["pendulum_outputs_complete"] and checks["pendulum_assist_within_20Nm"]
                        and len(checks["diagnostics"]) == 12 and checks["evaluation_records"] == 74)
    (STUDY / "verification.json").write_text(json.dumps(checks, indent=2, ensure_ascii=False))
    print(json.dumps({k: checks[k] for k in ("passed", "all_training_epoch_6000", "evaluation_records", "w1_runs", "pendulum_runs",
                                              "w1_falls", "new_models_shared_on_both_legs", "zero_action_identity")}, indent=1))
    assert checks["passed"]


if __name__ == "__main__":
    main()
