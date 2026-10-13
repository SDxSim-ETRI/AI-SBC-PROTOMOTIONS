#!/usr/bin/env bash
# L12: physics checks and four PPO epochs must pass before the 2k walking trial.
set -euo pipefail
AISBC_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
AISBC_PYTHON="$HOME/miniconda3/envs/sbc/bin/python"
cd "$AISBC_ROOT"
"$AISBC_PYTHON" examples/experiments/assist_pendulum/run_paired_walk_sequence.py --check-only
AISBC_RUNNING=$(systemctl --user list-units 'aisbc*' --state=running --no-legend --no-pager)
if [[ -n "$AISBC_RUNNING" ]]; then
  echo "An AI-SBC service is already running; refusing overlapping work." >&2
  echo "$AISBC_RUNNING" >&2
  exit 1
fi
AISBC_GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
if [[ -n "$AISBC_GPU_PIDS" ]]; then
  echo "GPU compute processes exist; inspect ownership before starting." >&2
  echo "$AISBC_GPU_PIDS" >&2
  exit 1
fi
mkdir -p results/_pids
systemd-run --user --unit=aisbc-llp-L12-paired-walk --working-directory="$AISBC_ROOT" \
  --property=Restart=no \
  --property="StandardOutput=append:$AISBC_ROOT/results/_pids/llp_L12_paired_walk.log" \
  --property="StandardError=append:$AISBC_ROOT/results/_pids/llp_L12_paired_walk.log" \
  --setenv=PYTHONFAULTHANDLER=1 \
  "$AISBC_PYTHON" -u examples/experiments/assist_pendulum/run_paired_walk_sequence.py
