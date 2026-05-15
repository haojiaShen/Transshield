#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="${RUN_NAME:-transshield_single_image_compare}"
SECURE_RUN_DIR="${SECURE_RUN_DIR:-artifacts/server_pipeline_run/${RUN_NAME}}"
BUNDLE_DIR="${BUNDLE_DIR:-artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_20260430}"
BASELINE_REPO_ROOT="${BASELINE_REPO_ROOT:-$REPO_ROOT/references/original_plaintext_runtime}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-$REPO_ROOT/artifacts/baselines/baseline_plaintext_eval_checkpoint_light.pth}"
BASELINE_THRESHOLD_JSON="${BASELINE_THRESHOLD_JSON:-$REPO_ROOT/artifacts/baselines/original_plaintext_threshold_best_fix3.json}"
PLAINTEXT_EVAL_DEVICE="${PLAINTEXT_EVAL_DEVICE:-cpu}"
INPUT_IMAGE="${INPUT_IMAGE:-}"
CLASS_NAMES="${CLASS_NAMES:-class_0,class_1}"

if [[ -z "$INPUT_IMAGE" ]]; then
  echo "请先设置 INPUT_IMAGE 再运行。" >&2
  exit 1
fi

echo "[single-compare] 生成 baseline vs modified 单图对照。"
echo "[single-compare] 输出摘要图、JSON 和 Markdown 说明。"

"$PYTHON_BIN" tools/transshield_single_image_comparison.py \
  --image-path "$INPUT_IMAGE" \
  --baseline-repo-root "$BASELINE_REPO_ROOT" \
  --baseline-checkpoint "$BASELINE_CHECKPOINT" \
  --baseline-threshold-json "$BASELINE_THRESHOLD_JSON" \
  --bundle-dir "$BUNDLE_DIR" \
  --device "$PLAINTEXT_EVAL_DEVICE" \
  --class-names "$CLASS_NAMES" \
  --output-dir "$SECURE_RUN_DIR/single_image_comparison"
