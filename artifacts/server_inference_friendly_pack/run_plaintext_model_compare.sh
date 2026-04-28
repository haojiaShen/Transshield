#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="${RUN_NAME:-transshield_inference_friendly_deits}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-}"
VAL_DATA_PATH="${VAL_DATA_PATH:-}"
RUN_DIR="${RUN_DIR:-artifacts/server_runs/${RUN_NAME}}"
BUNDLE_DIR="${BUNDLE_DIR:-artifacts/server_bundles/${RUN_NAME}_bundle}"
SECURE_RUN_DIR="${SECURE_RUN_DIR:-artifacts/server_pipeline_run/${RUN_NAME}}"
CONFIG_PATH="${CONFIG_PATH:-configs/openbumblebee/2pc.json}"
BASELINE_LABEL="${BASELINE_LABEL:-baseline_plaintext}"
MODIFIED_LABEL="${MODIFIED_LABEL:-modified_plaintext}"

echo "[compare] 对比 baseline plaintext 与 modified plaintext。"
echo "[compare] 这里回答的问题是：当前任务上 baseline 和 modified 谁更好。"

"$PYTHON_BIN" tools/transshield_plaintext_eval_compare.py \
  --eval-a "$SECURE_RUN_DIR/plaintext_baseline_eval.json" \
  --eval-b "$SECURE_RUN_DIR/plaintext_modified_eval.json" \
  --label-a "$BASELINE_LABEL" \
  --label-b "$MODIFIED_LABEL" \
  --output-json "$SECURE_RUN_DIR/plaintext_model_compare.json"
