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
BUNDLE_DIR="${BUNDLE_DIR:-artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_20260430}"
SECURE_RUN_DIR="${SECURE_RUN_DIR:-artifacts/server_pipeline_run/${RUN_NAME}}"
CONFIG_PATH="${CONFIG_PATH:-configs/openbumblebee/2pc.json}"

echo "[compare] 对比 modified plaintext 与 secure replay 结果。"
echo "[compare] 这里的一致指逐样本预测语义一致，不表示准确率 100% 或 logits 逐位完全相同。"

"$PYTHON_BIN" tools/transshield_plaintext_secure_score_compare.py \
  --bundle-dir "$BUNDLE_DIR" \
  --secure-replay-json "$SECURE_RUN_DIR/pipeline_inference_replay_summary.json" \
  --device cpu \
  --batch-size 16 \
  --num-workers 0 \
  --output-json "$SECURE_RUN_DIR/plaintext_vs_secure_score_compare.json" \
  --output-csv "$SECURE_RUN_DIR/plaintext_vs_secure_score_compare.csv"
