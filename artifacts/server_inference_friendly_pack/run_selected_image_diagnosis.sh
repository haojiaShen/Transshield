#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="${RUN_NAME:-transshield_selected_image_infer}"
SECURE_RUN_DIR="${SECURE_RUN_DIR:-artifacts/server_pipeline_run/${RUN_NAME}}"
BUNDLE_DIR="${BUNDLE_DIR:-artifacts/frozen_bundle_verified_tracka_lr3e5_20260414}"
PLAINTEXT_EVAL_DEVICE="${PLAINTEXT_EVAL_DEVICE:-cpu}"
PLAINTEXT_EVAL_BATCH_SIZE="${PLAINTEXT_EVAL_BATCH_SIZE:-16}"
PLAINTEXT_EVAL_NUM_WORKERS="${PLAINTEXT_EVAL_NUM_WORKERS:-0}"
INPUT_IMAGE="${INPUT_IMAGE:-}"
INPUT_IMAGE_LIST="${INPUT_IMAGE_LIST:-}"
INPUT_IMAGE_DIR="${INPUT_IMAGE_DIR:-}"
INPUT_GLOB_PATTERN="${INPUT_GLOB_PATTERN:-*}"
CLASS_NAMES="${CLASS_NAMES:-class_0,class_1}"

if [[ -z "$INPUT_IMAGE" && -z "$INPUT_IMAGE_LIST" && -z "$INPUT_IMAGE_DIR" ]]; then
  echo "请先设置 INPUT_IMAGE、INPUT_IMAGE_LIST 或 INPUT_IMAGE_DIR 之一再运行。" >&2
  exit 1
fi

echo "[diagnosis] 运行 modified 明文选图诊断。"
echo "[diagnosis] 该脚本用于本地展示与解释，不属于 secure 执行。"

INPUT_ARGS=()
if [[ -n "$INPUT_IMAGE" ]]; then
  INPUT_ARGS+=(--image "$INPUT_IMAGE")
fi
if [[ -n "$INPUT_IMAGE_LIST" ]]; then
  INPUT_ARGS+=(--image-list "$INPUT_IMAGE_LIST")
fi
if [[ -n "$INPUT_IMAGE_DIR" ]]; then
  INPUT_ARGS+=(--input-dir "$INPUT_IMAGE_DIR" --glob-pattern "$INPUT_GLOB_PATTERN")
fi

"$PYTHON_BIN" tools/transshield_selected_image_report.py \
  --bundle-dir "$BUNDLE_DIR" \
  --device "$PLAINTEXT_EVAL_DEVICE" \
  --batch-size "$PLAINTEXT_EVAL_BATCH_SIZE" \
  --num-workers "$PLAINTEXT_EVAL_NUM_WORKERS" \
  --class-names "$CLASS_NAMES" \
  --output-json "$SECURE_RUN_DIR/selected_image_diagnosis.json" \
  --output-csv "$SECURE_RUN_DIR/selected_image_diagnosis.csv" \
  "${INPUT_ARGS[@]}"
