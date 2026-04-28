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
BASELINE_REPO_ROOT="${BASELINE_REPO_ROOT:-}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-}"
BASELINE_THRESHOLD_JSON="${BASELINE_THRESHOLD_JSON:-}"
BASELINE_LABEL="${BASELINE_LABEL:-baseline_plaintext}"
PLAINTEXT_EVAL_DEVICE="${PLAINTEXT_EVAL_DEVICE:-cpu}"
PLAINTEXT_EVAL_BATCH_SIZE="${PLAINTEXT_EVAL_BATCH_SIZE:-32}"
PLAINTEXT_EVAL_NUM_WORKERS="${PLAINTEXT_EVAL_NUM_WORKERS:-0}"
SECURE_BASELINE_PROFILE_JSON="${SECURE_BASELINE_PROFILE_JSON:-}"
SECURE_BASELINE_LABEL="${SECURE_BASELINE_LABEL:-baseline_secure}"

if [[ -z "$SECURE_BASELINE_PROFILE_JSON" ]]; then
  echo "请先设置 SECURE_BASELINE_PROFILE_JSON 再运行。" >&2
  exit 1
fi

echo "[profile] 对比当前 secure profile 与给定基线 profile。"
echo "[profile] 该脚本是可选分析项，不影响 baseline / modified / secure 主结论。"

"$PYTHON_BIN" tools/transshield_secure_profile_compare.py \
  --summary-a "$SECURE_BASELINE_PROFILE_JSON" \
  --summary-b "$SECURE_RUN_DIR/secure_profile_summary.json" \
  --label-a "$SECURE_BASELINE_LABEL" \
  --label-b modified_secure \
  --output-json "$SECURE_RUN_DIR/secure_profile_compare.json"
