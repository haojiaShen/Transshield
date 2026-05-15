#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi
RUN_NAME="${RUN_NAME:-}"
RUN_DIR="${RUN_DIR:-}"
DEFAULT_TRAIN_RUN_DIR=""
if [[ -n "$RUN_NAME" ]]; then
  DEFAULT_TRAIN_RUN_DIR="$REPO_ROOT/artifacts/train_runs/$RUN_NAME"
fi

if [[ -n "$DEFAULT_TRAIN_RUN_DIR" && ( -z "$RUN_DIR" || "$RUN_DIR" == "$REPO_ROOT/artifacts/server_runs/"* ) ]]; then
  RUN_DIR="$DEFAULT_TRAIN_RUN_DIR"
fi

if [[ -z "$RUN_DIR" ]]; then
  if [[ -z "$RUN_NAME" ]]; then
    echo "[protocol-aware-pruning-report] set RUN_NAME or RUN_DIR first" >&2
    exit 1
  fi
  RUN_DIR="$REPO_ROOT/artifacts/train_runs/$RUN_NAME"
fi

TRAIN_LOG="${TRAIN_LOG:-$RUN_DIR/train_stdout.log}"
if [[ ! -f "$TRAIN_LOG" ]]; then
  echo "[protocol-aware-pruning-report] missing TRAIN_LOG: $TRAIN_LOG" >&2
  exit 2
fi

if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="$(basename "$RUN_DIR")"
fi

PROTOCOL_AWARE_PROFILE="${PROTOCOL_AWARE_PROFILE:-conservative}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/protocol_aware_pruning_objective/$RUN_NAME}"
OUTPUT_JSON="${OUTPUT_JSON:-$OUTPUT_DIR/pruning_margin_log_report.json}"
OUTPUT_MD="${OUTPUT_MD:-$OUTPUT_DIR/pruning_margin_log_report.md}"
RECIPE_JSON="${RECIPE_JSON:-$OUTPUT_DIR/protocol_aware_pruning_recipe.json}"

ARGS=(
  --train-log "$TRAIN_LOG"
  --output-json "$OUTPUT_JSON"
  --output-md "$OUTPUT_MD"
  --profile "$PROTOCOL_AWARE_PROFILE"
)

if [[ -f "$RECIPE_JSON" ]]; then
  ARGS+=(--recipe-json "$RECIPE_JSON")
fi

echo "[protocol-aware-pruning-report] run_dir=$RUN_DIR"
echo "[protocol-aware-pruning-report] train_log=$TRAIN_LOG"
echo "[protocol-aware-pruning-report] output_json=$OUTPUT_JSON"
echo "[protocol-aware-pruning-report] output_md=$OUTPUT_MD"

"$PYTHON_BIN" tools/transshield_pruning_margin_log_report.py "${ARGS[@]}"

echo "[protocol-aware-pruning-report] 完成："
echo "[protocol-aware-pruning-report] JSON: $OUTPUT_JSON"
echo "[protocol-aware-pruning-report] MD:   $OUTPUT_MD"
