#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"

VARIANT="${1:-modified}"

REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="${RUN_NAME:-transshield_inference_friendly_deits}"
VAL_DATA_PATH="${VAL_DATA_PATH:-}"
SECURE_RUN_DIR="${SECURE_RUN_DIR:-artifacts/server_pipeline_run/${RUN_NAME}}"
PLAINTEXT_EVAL_DEVICE="${PLAINTEXT_EVAL_DEVICE:-cpu}"
PLAINTEXT_EVAL_BATCH_SIZE="${PLAINTEXT_EVAL_BATCH_SIZE:-32}"
PLAINTEXT_EVAL_NUM_WORKERS="${PLAINTEXT_EVAL_NUM_WORKERS:-0}"
PLAINTEXT_MAX_SAMPLES="${PLAINTEXT_MAX_SAMPLES:-0}"

if [[ -z "$VAL_DATA_PATH" ]]; then
  echo "请先设置 VAL_DATA_PATH 再运行。" >&2
  exit 1
fi

case "$VARIANT" in
  baseline)
    TARGET_REPO_ROOT="${BASELINE_REPO_ROOT:-$REPO_ROOT/references/original_plaintext_runtime}"
    TARGET_CHECKPOINT="${BASELINE_CHECKPOINT:-$REPO_ROOT/artifacts/baselines/baseline_plaintext_eval_checkpoint_light.pth}"
    TARGET_THRESHOLD_JSON="${BASELINE_THRESHOLD_JSON:-$REPO_ROOT/artifacts/baselines/original_plaintext_threshold_best_fix3.json}"
    TARGET_LABEL="${BASELINE_LABEL:-baseline_plaintext}"
    OUTPUT_PREFIX="plaintext_baseline_eval"
    HUMAN_LABEL="baseline"
    if [[ ! -f "$TARGET_CHECKPOINT" ]]; then
      echo "缺少 BASELINE_CHECKPOINT：$TARGET_CHECKPOINT" >&2
      exit 1
    fi
    ;;
  modified)
    BUNDLE_DIR="${BUNDLE_DIR:-artifacts/frozen_bundle_verified_tracka_lr3e5_20260414}"
    TARGET_REPO_ROOT="$REPO_ROOT"
    TARGET_CHECKPOINT="${MODIFIED_CHECKPOINT:-$BUNDLE_DIR/modified_plaintext_eval_checkpoint_light.pth}"
    TARGET_THRESHOLD_JSON="${MODIFIED_THRESHOLD_JSON:-$BUNDLE_DIR/threshold_best.json}"
    TARGET_LABEL="${MODIFIED_LABEL:-modified_plaintext}"
    OUTPUT_PREFIX="plaintext_modified_eval"
    HUMAN_LABEL="modified"
    ;;
  *)
    echo "Usage: $0 [baseline|modified]" >&2
    exit 1
    ;;
esac

echo "[plaintext] 运行 ${HUMAN_LABEL} 明文评估。"
if [[ "$VARIANT" == "baseline" ]]; then
  echo "[plaintext] 该结果用于 baseline vs modified 的对照，不属于 secure 执行。"
else
  echo "[plaintext] 该结果既用于 baseline 对照，也用于后续 secure 一致性比较。"
fi

THRESHOLD_ARGS=()
if [[ -n "${TARGET_THRESHOLD_JSON:-}" ]]; then
  THRESHOLD_ARGS+=(--threshold-json "$TARGET_THRESHOLD_JSON")
fi

"$PYTHON_BIN" tools/transshield_plaintext_checkpoint_eval.py \
  --repo-root "$TARGET_REPO_ROOT" \
  --checkpoint "$TARGET_CHECKPOINT" \
  --data-path "$VAL_DATA_PATH" \
  --device "$PLAINTEXT_EVAL_DEVICE" \
  --batch-size "$PLAINTEXT_EVAL_BATCH_SIZE" \
  --num-workers "$PLAINTEXT_EVAL_NUM_WORKERS" \
  --max-samples "$PLAINTEXT_MAX_SAMPLES" \
  --label "$TARGET_LABEL" \
  --output-json "$SECURE_RUN_DIR/${OUTPUT_PREFIX}.json" \
  --output-csv "$SECURE_RUN_DIR/${OUTPUT_PREFIX}.csv" \
  "${THRESHOLD_ARGS[@]}"
