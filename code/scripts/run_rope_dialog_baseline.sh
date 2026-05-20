#!/usr/bin/env bash
set -euo pipefail

# Run Rope dialog baseline.
# Usage:
#   ./scripts/run_rope_dialog_baseline.sh [num_runs] [seed]
# Example:
#   ./scripts/run_rope_dialog_baseline.sh 2 0
#
# Environment knobs:
#   ROCO_LLM_MODEL=llama3.3:70b
#   OLLAMA_BASE_URL=http://127.0.0.1:11434
#   START_OLLAMA=1       # start `ollama serve` if endpoint is not reachable
#   RUN_TIMEOUT=600

NUM_RUNS="${1:-2}"
SEED="${2:-0}"
PROJECT_ROOT="/inspire/qb-ilm2/project/26summer-camp-09/26220478"
CODE_DIR="$PROJECT_ROOT/code"
DATA_DIR="$PROJECT_ROOT/data"
MODEL="${ROCO_LLM_MODEL:-llama3.3:70b}"
OLLAMA_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
START_OLLAMA="${START_OLLAMA:-1}"
RUN_TIMEOUT="${RUN_TIMEOUT:-600}"
STAMP="$(date +%Y%m%d_%H%M%S)"
HOST_TAG="$(hostname | tr -c 'A-Za-z0-9_' '_')"
RUN_NAME="baseline_rope_dialog_${STAMP}_${HOST_TAG}_pid$$"
RUNTIME_DIR="/tmp/runtime-${USER:-root}"

# Activate conda env if available.
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /root/miniconda3/etc/profile.d/conda.sh
elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /opt/conda/etc/profile.d/conda.sh
fi
if command -v conda >/dev/null 2>&1; then
  conda activate roco
fi

export ROCO_LLM_MODEL="$MODEL"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$PROJECT_ROOT/ollama_models}"
mkdir -p "$OLLAMA_MODELS"
export OLLAMA_BASE_URL="$OLLAMA_URL"
export OLLAMA_HOST="$OLLAMA_URL"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-${OLLAMA_URL%/}/v1/}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-ollama}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MPLCONFIGDIR="$DATA_DIR/.matplotlib"
export XDG_RUNTIME_DIR="$RUNTIME_DIR"
mkdir -p "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR" || true

if [ "$START_OLLAMA" = "1" ]; then
  if ! curl -fsS "$OLLAMA_BASE_URL/api/tags" >/dev/null 2>&1; then
    echo "[rope-dialog] Ollama not reachable at $OLLAMA_BASE_URL; starting: ollama serve"
    nohup ollama serve > "/tmp/ollama_serve_${HOST_TAG}_$$.log" 2>&1 &
    sleep 5
  fi
fi

if ! curl -fsS "$OLLAMA_BASE_URL/api/tags" >/dev/null 2>&1; then
  echo "[rope-dialog][ERROR] Ollama is not reachable at $OLLAMA_BASE_URL" >&2
  echo "Start it manually with: conda activate roco && ollama serve" >&2
  exit 3
fi

cd "$CODE_DIR"

echo "[rope-dialog] task=rope mode=dialog num_runs=$NUM_RUNS seed=$SEED"
echo "[rope-dialog] run_name=$RUN_NAME"
echo "[rope-dialog] model=$ROCO_LLM_MODEL ollama=$OLLAMA_BASE_URL openai_base=$OPENAI_BASE_URL"

# run_dialog.py automatically forces Rope to action_and_path, split_parsed_plans,
# control_freq=20, and max_failed_waypoints=0.
xvfb-run -a python run_dialog.py \
  --task rope \
  --comm_mode dialog \
  --run_name "$RUN_NAME" \
  --num_runs "$NUM_RUNS" \
  --data_dir "$DATA_DIR" \
  --skip_display \
  --llm_source "$ROCO_LLM_MODEL" \
  --seed "$SEED" \
  --run_timeout "$RUN_TIMEOUT"

echo "[rope-dialog] done: $DATA_DIR/$RUN_NAME"
echo "[rope-dialog] summarize with: find '$DATA_DIR/$RUN_NAME' -name 'steps*_success_*.json' -print"
