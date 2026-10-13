#!/usr/bin/env bash
# Start only after CUDA and the user's systemd bus are accessible.
set -euo pipefail
AISBC_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
AISBC_PYTHON="$HOME/miniconda3/envs/sbc/bin/python"
cd "$AISBC_ROOT"
"$AISBC_PYTHON" examples/experiments/assist_pendulum/run_opposition_sequence.py --check-only
systemctl --user show-environment >/dev/null
mkdir -p results/_pids
systemd-run --user --unit=aisbc-llp-L8-opposition --working-directory="$AISBC_ROOT" \
  --property=Restart=on-failure --property=RestartSec=15 --property=StartLimitIntervalSec=0 \
  --property="StandardOutput=append:$AISBC_ROOT/results/_pids/llp_L8_opposition.log" \
  --property="StandardError=append:$AISBC_ROOT/results/_pids/llp_L8_opposition.log" \
  "$AISBC_PYTHON" -u examples/experiments/assist_pendulum/run_opposition_sequence.py
