#!/usr/bin/env bash
set -euo pipefail

# Serially run baseline experiments for the 4 tasks excluding sort and rope.
# Tasks: cabinet, sweep, sandwich, pack
# Methods: plan, dialog
# Runs: 3 per (task, method)
#
# Usage:
#   ./scripts/run_4tasks_2methods_3runs_serial.sh [seed]
# Example:
#   ./scripts/run_4tasks_2methods_3runs_serial.sh 0
#
# Optional environment variables:
#   ROCO_LLM_MODEL=llama3.3:70b
#   OLLAMA_BASE_URL=http://127.0.0.1:11434
#   START_OLLAMA=1

SEED="${1:-0}"
NUM_RUNS=3
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/inspire/qb-ilm2/project/26summer-camp-09/26220478"
DATA_DIR="$PROJECT_ROOT/data"
LOG_DIR="$DATA_DIR/baseline_4tasks_2methods_logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/run_4tasks_2methods_3runs_${STAMP}.log"

mkdir -p "$LOG_DIR"

TASKS=(cabinet sweep sandwich pack)
MODES=(plan dialog)

echo "[all4] start serial baseline"
echo "[all4] tasks=${TASKS[*]}"
echo "[all4] modes=${MODES[*]}"
echo "[all4] num_runs=$NUM_RUNS seed=$SEED"
echo "[all4] log=$LOG_FILE"
echo

# Save full stdout/stderr to log while still printing to terminal.
exec > >(tee -a "$LOG_FILE") 2>&1

for task in "${TASKS[@]}"; do
  for mode in "${MODES[@]}"; do
    echo
    echo "============================================================"
    echo "[all4] RUN task=$task mode=$mode num_runs=$NUM_RUNS seed=$SEED"
    echo "============================================================"

    before_list="$(mktemp)"
    after_list="$(mktemp)"
    find "$DATA_DIR" -maxdepth 1 -type d -name "baseline_${task}_${mode}_*" | sort > "$before_list" || true

    job_status="OK"
    if ! "$SCRIPT_DIR/run_baseline_one.sh" "$task" "$mode" "$NUM_RUNS" "$SEED"; then
      job_status="FAILED"
      echo "[all4][ERROR] task=$task mode=$mode failed; continuing to next job"
    fi

    find "$DATA_DIR" -maxdepth 1 -type d -name "baseline_${task}_${mode}_*" | sort > "$after_list" || true
    new_dirs="$(comm -13 "$before_list" "$after_list" || true)"
    rm -f "$before_list" "$after_list"

    echo "[all4] finished task=$task mode=$mode status=$job_status"
    if [ -n "$new_dirs" ]; then
      echo "[all4] new output dirs:"
      echo "$new_dirs"
      echo "[all4] result files:"
      while IFS= read -r d; do
        find "$d" -maxdepth 2 -name 'steps*_success_*.json' -print | sort || true
      done <<< "$new_dirs"
    else
      echo "[all4][WARN] no new output dir detected for baseline_${task}_${mode}_*"
    fi
  done
done

echo
echo "============================================================"
echo "[all4] ALL DONE"
echo "[all4] log=$LOG_FILE"
echo "[all4] latest baseline summary:"
python - <<'PY'
import glob, os, re
DATA_DIR = "/inspire/qb-ilm2/project/26summer-camp-09/26220478/data"
wanted_tasks = {"cabinet", "sweep", "sandwich", "pack"}
wanted_modes = {"plan", "dialog"}
rows = []
for d in glob.glob(os.path.join(DATA_DIR, "baseline_*")):
    base = os.path.basename(d)
    m = re.match(r"baseline_([^_]+)_([^_]+)_", base)
    if not m:
        continue
    task, mode = m.group(1), m.group(2)
    if task not in wanted_tasks or mode not in wanted_modes:
        continue
    files = glob.glob(os.path.join(d, "run_*", "steps*_success_*.json"))
    run_dirs = glob.glob(os.path.join(d, "run_*"))
    ok = sum("success_True" in os.path.basename(f) for f in files)
    rows.append((os.path.getmtime(d), task, mode, ok, len(files), len(run_dirs), base))
for _, task, mode, ok, total, run_dirs, base in sorted(rows):
    print(f"{task:9s} {mode:6s} {ok}/{total} completed, run_dirs={run_dirs}  {base}")
PY
