#!/usr/bin/env bash
set -euo pipefail

# Serial launcher for all 4 non-sort/non-rope tasks in plan+dialog modes.
# For parallel runs on shared storage, prefer launching run_baseline_one.sh in
# separate containers/processes.

NUM_RUNS="${1:-2}"
SEED="${2:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for task in cabinet sweep sandwich pack; do
  for mode in plan dialog; do
    "$SCRIPT_DIR/run_baseline_one.sh" "$task" "$mode" "$NUM_RUNS" "$SEED"
  done
done
