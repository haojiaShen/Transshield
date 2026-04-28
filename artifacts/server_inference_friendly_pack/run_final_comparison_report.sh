#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="${RUN_NAME:-transshield_comp_full_compare_YYYYMMDD}"
SECURE_RUN_DIR="${SECURE_RUN_DIR:-$REPO_ROOT/artifacts/server_pipeline_run/${RUN_NAME}}"

echo "[report] 生成最终 comparison report。"
echo "[report] 该报告用于汇总 baseline vs modified，以及 modified vs secure 的结果。"

"$PYTHON_BIN" tools/transshield_comparison_report.py \
  --run-dir "$SECURE_RUN_DIR" \
  --output-json "$SECURE_RUN_DIR/comparison_report_summary.json" \
  --output-txt "$SECURE_RUN_DIR/comparison_report_summary.txt"

echo "[report] 输出已写入:"
echo "  - $SECURE_RUN_DIR/comparison_report_summary.txt"
echo "  - $SECURE_RUN_DIR/comparison_report_summary.json"
