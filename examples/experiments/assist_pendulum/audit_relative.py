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
"""Saved-configuration contrast for R0/R1 against L1, G0, G4 and F1 (no rollouts)."""
import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_feasible import canonical, differences  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
VOLATILE = ("save_dir", "experiment_name", "_root", "run_name", "wandb")


def load(path):
    return canonical(torch.load(ROOT / path / "resolved_configs.pt", map_location="cpu", weights_only=False))


def relevant(items):
    return [d for d in items if not any(v in d["path"] for v in VOLATILE)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--r0", default="results/llp_relative_R0_s0_20261020")
    p.add_argument("--r1", default="results/llp_relative_R1_s0_20261020")
    p.add_argument("--out", default="results/llp_relative_20261020/config_audit.json")
    args = p.parse_args()
    paths = {"L1": "checkpoints/L1_llp_pendulum", "G0": "results/llp_general23_G0_s0_20261015_all",
             "G4": "results/llp_general23_G4_s0_20261015_all", "F1": "results/llp_feasible_F1_s0_20261020",
             "R0": args.r0, "R1": args.r1}
    configs = {name: load(path) for name, path in paths.items()}
    report = dict(paths=paths, configs=configs,
        differences_from_L1={name: relevant(differences(configs["L1"], cfg)) for name, cfg in configs.items() if name != "L1"},
        R1_minus_R0=relevant(differences(configs["R0"], configs["R1"])),
        R0_minus_F1=relevant(differences(configs["F1"], configs["R0"])))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    for key in ("R0", "R1"):
        print(f"== {key} vs L1: {len(report['differences_from_L1'][key])} differences")
        for d in report["differences_from_L1"][key]:
            print("  ", d["path"], "|", json.dumps(d["old"], ensure_ascii=False)[:90], "->", json.dumps(d["new"], ensure_ascii=False)[:90])
    print(f"== R1 vs R0: {len(report['R1_minus_R0'])} differences")
    for d in report["R1_minus_R0"]:
        print("  ", d["path"], "|", json.dumps(d["old"], ensure_ascii=False)[:90], "->", json.dumps(d["new"], ensure_ascii=False)[:90])


if __name__ == "__main__":
    main()
