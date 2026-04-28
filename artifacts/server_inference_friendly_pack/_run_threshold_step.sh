#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"

STEP="${1:-search}"

REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="${RUN_NAME:-transshield_inference_friendly_deits}"
VAL_DATA_PATH="${VAL_DATA_PATH:-}"
RUN_DIR="${RUN_DIR:-artifacts/server_runs/${RUN_NAME}}"
THRESHOLD_EVAL_DEVICE="${THRESHOLD_EVAL_DEVICE:-cuda}"
THRESHOLD_EVAL_BATCH_SIZE="${THRESHOLD_EVAL_BATCH_SIZE:-32}"
THRESHOLD_EVAL_NUM_WORKERS="${THRESHOLD_EVAL_NUM_WORKERS:-4}"

if [[ -z "$VAL_DATA_PATH" ]]; then
  echo "请先设置 VAL_DATA_PATH 再运行。" >&2
  exit 1
fi

case "$STEP" in
  search)
    echo "[archive] 运行 modified 模型 threshold search。"
    echo "[archive] 该脚本属于 full rebuild 流程，不是默认比赛展示入口。"
    "$PYTHON_BIN" tools/transshield_binary_threshold_search.py search \
      --checkpoint "$RUN_DIR/checkpoint-best.pth" \
      --data-path "$VAL_DATA_PATH" \
      --device "$THRESHOLD_EVAL_DEVICE" \
      --batch-size "$THRESHOLD_EVAL_BATCH_SIZE" \
      --num-workers "$THRESHOLD_EVAL_NUM_WORKERS" \
      --output-json "$RUN_DIR/threshold_best.json"
    ;;
  eval)
    echo "[archive] 运行 modified 模型 threshold eval。"
    echo "[archive] 该脚本属于 full rebuild 流程，不是默认比赛展示入口。"
    "$PYTHON_BIN" tools/transshield_binary_threshold_search.py eval \
      --checkpoint "$RUN_DIR/checkpoint-best.pth" \
      --threshold-json "$RUN_DIR/threshold_best.json" \
      --data-path "$VAL_DATA_PATH" \
      --device "$THRESHOLD_EVAL_DEVICE" \
      --batch-size "$THRESHOLD_EVAL_BATCH_SIZE" \
      --num-workers "$THRESHOLD_EVAL_NUM_WORKERS" \
      --output-json "$RUN_DIR/threshold_eval.json"
    ;;
  *)
    echo "Usage: $0 [search|eval]" >&2
    exit 1
    ;;
esac
