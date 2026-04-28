#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="${RUN_NAME:-transshield_selected_image_secure}"
SECURE_RUN_DIR="${SECURE_RUN_DIR:-artifacts/server_pipeline_run/${RUN_NAME}}"
CLASS_NAMES="${CLASS_NAMES:-class_0,class_1}"

echo "[diagnosis] 汇总 selected-image secure replay 结果。"
echo "[diagnosis] 该输出用于说明 secure 侧预测语义，不代表整体验证集准确率。"

"$PYTHON_BIN" tools/transshield_secure_diagnosis_report.py \
  --secure-replay-json "$SECURE_RUN_DIR/pipeline_inference_replay_summary.json" \
  --class-names "$CLASS_NAMES" \
  --output-json "$SECURE_RUN_DIR/selected_image_secure_diagnosis.json" \
  --output-csv "$SECURE_RUN_DIR/selected_image_secure_diagnosis.csv"
