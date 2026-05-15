#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="${RUN_NAME:-transshield_token_pruning}"
SECURE_RUN_DIR="${SECURE_RUN_DIR:-artifacts/server_pipeline_run/${RUN_NAME}}"
BUNDLE_DIR="${BUNDLE_DIR:-artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_20260430}"
PLAINTEXT_EVAL_DEVICE="${PLAINTEXT_EVAL_DEVICE:-cpu}"
INPUT_IMAGE="${INPUT_IMAGE:-}"

if [[ -z "$INPUT_IMAGE" ]]; then
  echo "请先设置 INPUT_IMAGE 再运行。" >&2
  exit 1
fi

echo "[pruning-vis] 生成单图 token pruning 可视化。"
echo "[pruning-vis] 该脚本输出图片、trace JSON 和 Markdown 说明。"

"$PYTHON_BIN" tools/transshield_token_pruning_visualization.py \
  --bundle-dir "$BUNDLE_DIR" \
  --image-path "$INPUT_IMAGE" \
  --device "$PLAINTEXT_EVAL_DEVICE" \
  --output-dir "$SECURE_RUN_DIR/token_pruning_visualization"
