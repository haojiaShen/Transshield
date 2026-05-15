#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"

VARIANT="${1:-modified}"

REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="${RUN_NAME:-transshield_selected_image_infer}"
SECURE_RUN_DIR="${SECURE_RUN_DIR:-artifacts/server_pipeline_run/${RUN_NAME}}"
PLAINTEXT_EVAL_DEVICE="${PLAINTEXT_EVAL_DEVICE:-cpu}"
PLAINTEXT_EVAL_BATCH_SIZE="${PLAINTEXT_EVAL_BATCH_SIZE:-32}"
PLAINTEXT_EVAL_NUM_WORKERS="${PLAINTEXT_EVAL_NUM_WORKERS:-0}"
INPUT_IMAGE="${INPUT_IMAGE:-}"
INPUT_IMAGE_LIST="${INPUT_IMAGE_LIST:-}"
INPUT_IMAGE_DIR="${INPUT_IMAGE_DIR:-}"
INPUT_GLOB_PATTERN="${INPUT_GLOB_PATTERN:-*}"

if [[ -z "$INPUT_IMAGE" && -z "$INPUT_IMAGE_LIST" && -z "$INPUT_IMAGE_DIR" ]]; then
  echo "请先设置 INPUT_IMAGE、INPUT_IMAGE_LIST 或 INPUT_IMAGE_DIR 之一再运行。" >&2
  exit 1
fi

case "$VARIANT" in
  baseline)
    TARGET_REPO_ROOT="${BASELINE_REPO_ROOT:-$REPO_ROOT/references/original_plaintext_runtime}"
    TARGET_CHECKPOINT="${BASELINE_CHECKPOINT:-$REPO_ROOT/artifacts/baselines/baseline_plaintext_eval_checkpoint_light.pth}"
    TARGET_THRESHOLD_JSON="${BASELINE_THRESHOLD_JSON:-$REPO_ROOT/artifacts/baselines/original_plaintext_threshold_best_fix3.json}"
    TARGET_LABEL="${BASELINE_LABEL:-baseline_plaintext}"
    OUTPUT_PREFIX="plaintext_baseline_selected_infer"
    HUMAN_LABEL="baseline"
    ;;
  modified)
    BUNDLE_DIR="${BUNDLE_DIR:-artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_20260430}"
    TARGET_REPO_ROOT="$REPO_ROOT"
    TARGET_CHECKPOINT="${MODIFIED_CHECKPOINT:-$BUNDLE_DIR/modified_plaintext_eval_checkpoint_light.pth}"
    TARGET_THRESHOLD_JSON="${MODIFIED_THRESHOLD_JSON:-$BUNDLE_DIR/threshold_best.json}"
    TARGET_LABEL="${MODIFIED_LABEL:-modified_plaintext}"
    OUTPUT_PREFIX="plaintext_modified_selected_infer"
    HUMAN_LABEL="modified"
    ;;
  *)
    echo "Usage: $0 [baseline|modified]" >&2
    exit 1
    ;;
esac

echo "[predict] 运行 ${HUMAN_LABEL} 明文单图/选图推理。"
echo "[predict] 该脚本用于展示或调试 ${HUMAN_LABEL} 输出，不属于 secure 执行。"

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

"$PYTHON_BIN" tools/transshield_plaintext_checkpoint_infer.py \
  --repo-root "$TARGET_REPO_ROOT" \
  --checkpoint "$TARGET_CHECKPOINT" \
  --device "$PLAINTEXT_EVAL_DEVICE" \
  --batch-size "$PLAINTEXT_EVAL_BATCH_SIZE" \
  --num-workers "$PLAINTEXT_EVAL_NUM_WORKERS" \
  --threshold-json "$TARGET_THRESHOLD_JSON" \
  --label "$TARGET_LABEL" \
  --output-json "$SECURE_RUN_DIR/${OUTPUT_PREFIX}.json" \
  --output-csv "$SECURE_RUN_DIR/${OUTPUT_PREFIX}.csv" \
  "${INPUT_ARGS[@]}"
