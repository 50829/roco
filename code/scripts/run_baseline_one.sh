#!/usr/bin/env bash
set -euo pipefail

# Run one baseline job from the original code/ tree.
# Usage:
#   ./scripts/run_baseline_one.sh <task> <mode> [num_runs] [seed]
# Example:
#   ./scripts/run_baseline_one.sh cabinet plan 2 0
# Tasks intended for this baseline: cabinet, sweep, sandwich, pack
# Modes: plan, dialog
#
# Environment knobs:
#   ROCO_LLM_MODEL=llama3.3:70b
#   OLLAMA_BASE_URL=http://127.0.0.1:11434
#   START_OLLAMA=1     # start `ollama serve` if the endpoint is not reachable

TASK="${1:?task required: cabinet|sweep|sandwich|pack}"
MODE="${2:?mode required: plan|dialog}"
NUM_RUNS="${3:-2}"
SEED="${4:-0}"

case "$TASK" in
  cabinet|sweep|sandwich|pack) ;;
  *) echo "Unsupported task for this baseline: $TASK" >&2; exit 2 ;;
esac
case "$MODE" in
  plan|dialog) ;;
  *) echo "Unsupported mode: $MODE" >&2; exit 2 ;;
esac

PROJECT_ROOT="/inspire/qb-ilm2/project/26summer-camp-09/26220478"
CODE_DIR="$PROJECT_ROOT/code"
DATA_DIR="$PROJECT_ROOT/data"
MODEL="${ROCO_LLM_MODEL:-llama3.3:70b}"
OLLAMA_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
START_OLLAMA="${START_OLLAMA:-1}"
STAMP="$(date +%Y%m%d_%H%M%S)"
HOST_TAG="$(hostname | tr -c 'A-Za-z0-9_' '_')"
RUN_NAME="baseline_${TASK}_${MODE}_${STAMP}_${HOST_TAG}_pid$$"
RUNTIME_DIR="/tmp/runtime-${USER:-root}"

# Activate the expected conda env even when the script is launched from a fresh shell.
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

# Export model/server settings globally, not only inline for Python.
export ROCO_LLM_MODEL="$MODEL"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$PROJECT_ROOT/ollama_models}"
mkdir -p "$OLLAMA_MODELS"
export OLLAMA_BASE_URL="$OLLAMA_URL"
# Some tools use OLLAMA_HOST instead of OLLAMA_BASE_URL; keep both available.
export OLLAMA_HOST="$OLLAMA_URL"
# For OpenAI-compatible clients, if any original path is used.
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-${OLLAMA_URL%/}/v1/}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-ollama}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MPLCONFIGDIR="$DATA_DIR/.matplotlib"
export XDG_RUNTIME_DIR="$RUNTIME_DIR"

mkdir -p "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR" || true

# Start Ollama inside this container if requested and not already reachable.
if [ "$START_OLLAMA" = "1" ]; then
  if ! curl -fsS "$OLLAMA_BASE_URL/api/tags" >/dev/null 2>&1; then
    echo "[baseline] Ollama not reachable at $OLLAMA_BASE_URL; starting: ollama serve"
    nohup ollama serve > "/tmp/ollama_serve_${HOST_TAG}_$$.log" 2>&1 &
    sleep 5
  fi
fi

# Fail early with a clear message if the model server is still unavailable.
if ! curl -fsS "$OLLAMA_BASE_URL/api/tags" >/dev/null 2>&1; then
  echo "[baseline][ERROR] Ollama is not reachable at $OLLAMA_BASE_URL" >&2
  echo "Start it manually with: conda activate roco && ollama serve" >&2
  exit 3
fi

cd "$CODE_DIR"

echo "[baseline] task=$TASK mode=$MODE num_runs=$NUM_RUNS seed=$SEED"
echo "[baseline] run_name=$RUN_NAME"
echo "[baseline] model=$ROCO_LLM_MODEL ollama=$OLLAMA_BASE_URL openai_base=$OPENAI_BASE_URL"

xvfb-run -a python run_dialog.py \
  --task "$TASK" \
  --comm_mode "$MODE" \
  --run_name "$RUN_NAME" \
  --num_runs "$NUM_RUNS" \
  --data_dir "$DATA_DIR" \
  --skip_display \
  --llm_source "$ROCO_LLM_MODEL" \
  --seed "$SEED" \
  --run_timeout 600

echo "[baseline] done: $DATA_DIR/$RUN_NAME"
