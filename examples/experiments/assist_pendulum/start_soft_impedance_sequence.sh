#!/usr/bin/env bash
# L10 changes only the synthetic wearer impedance distribution from L9.
set -euo pipefail
AISBC_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
AISBC_PYTHON="$HOME/miniconda3/envs/sbc/bin/python"
cd "$AISBC_ROOT"
AISBC_WEIGHT=$("$AISBC_PYTHON" -c 'import json; print(json.load(open("results/llp_L9_opposition_sequence/calibration.json"))["matched_relative_weight"])')
AISBC_ARGS=(--label L10 --relative-opposition --soft-impedance --opposition-weight "$AISBC_WEIGHT"
  --opposition-torque-floor 10 --comparison-source results/llp_L9_opposition_sequence/comparison.json)
"$AISBC_PYTHON" examples/experiments/assist_pendulum/run_opposition_sequence.py "${AISBC_ARGS[@]}" --check-only
systemctl --user show-environment >/dev/null
if systemctl --user is-active --quiet aisbc-llp-L9-opposition.service; then
  echo "L9 is still active; refusing overlapping training." >&2
  exit 1
fi
mkdir -p results/_pids
systemd-run --user --unit=aisbc-llp-L10-opposition --working-directory="$AISBC_ROOT" \
  --property=Restart=on-failure --property=RestartSec=15 --property=StartLimitIntervalSec=0 \
  --property="StandardOutput=append:$AISBC_ROOT/results/_pids/llp_L10_opposition.log" \
  --property="StandardError=append:$AISBC_ROOT/results/_pids/llp_L10_opposition.log" \
  "$AISBC_PYTHON" -u examples/experiments/assist_pendulum/run_opposition_sequence.py "${AISBC_ARGS[@]}"
