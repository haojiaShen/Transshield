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
    BUNDLE_DIR="${BUNDLE_DIR:-artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_20260430}"
    DEFAULT_LIGHT_CHECKPOINT="$BUNDLE_DIR/modified_plaintext_eval_checkpoint_light.pth"
    DEFAULT_BUNDLE_STATE_DICT="$BUNDLE_DIR/modified_plaintext_model_state_dict.pth"
    DEFAULT_THRESHOLD_JSON="$BUNDLE_DIR/threshold_best.json"
    COMPAT_REPO_ROOT="$REPO_ROOT/training_compat"
    COMPAT_THRESHOLD_JSON="$BUNDLE_DIR/threshold_best_compat.json"
    COMPAT_EVAL_JSON="$BUNDLE_DIR/plaintext_eval_full_compat.json"
    USE_BUNDLE_DIRECT=0
    CUSTOM_MODIFIED_CHECKPOINT=""
    CUSTOM_MODIFIED_THRESHOLD_JSON=""
    if [[ -n "${MODIFIED_CHECKPOINT:-}" && "${MODIFIED_CHECKPOINT:-}" != "$DEFAULT_LIGHT_CHECKPOINT" ]]; then
      CUSTOM_MODIFIED_CHECKPOINT="$MODIFIED_CHECKPOINT"
    fi
    if [[ -n "${MODIFIED_THRESHOLD_JSON:-}" && "${MODIFIED_THRESHOLD_JSON:-}" != "$DEFAULT_THRESHOLD_JSON" ]]; then
      CUSTOM_MODIFIED_THRESHOLD_JSON="$MODIFIED_THRESHOLD_JSON"
    fi
    if [[ -n "${MODIFIED_REPO_ROOT:-}" ]]; then
      TARGET_REPO_ROOT="$MODIFIED_REPO_ROOT"
    elif [[ -f "$COMPAT_THRESHOLD_JSON" || -f "$COMPAT_EVAL_JSON" ]]; then
      TARGET_REPO_ROOT="$COMPAT_REPO_ROOT"
    elif [[ -z "$CUSTOM_MODIFIED_CHECKPOINT" && -f "$DEFAULT_BUNDLE_STATE_DICT" && -f "$BUNDLE_DIR/args_snapshot.json" ]]; then
      USE_BUNDLE_DIRECT=1
      TARGET_REPO_ROOT=""
    else
      TARGET_REPO_ROOT="$REPO_ROOT"
    fi
    TARGET_CHECKPOINT="${CUSTOM_MODIFIED_CHECKPOINT:-${MODIFIED_CHECKPOINT:-$DEFAULT_LIGHT_CHECKPOINT}}"
    if [[ -n "${MODIFIED_THRESHOLD_JSON:-}" ]]; then
      TARGET_THRESHOLD_JSON="$MODIFIED_THRESHOLD_JSON"
    elif [[ -f "$COMPAT_THRESHOLD_JSON" ]]; then
      TARGET_THRESHOLD_JSON="$COMPAT_THRESHOLD_JSON"
    else
      TARGET_THRESHOLD_JSON="$DEFAULT_THRESHOLD_JSON"
    fi
    TARGET_LABEL="${MODIFIED_LABEL:-modified_plaintext}"
    OUTPUT_PREFIX="plaintext_modified_eval"
    HUMAN_LABEL="modified"
    ;;
  *)
    echo "Usage: $0 [baseline|modified]" >&2
    exit 1
    ;;
esac

if [[ "$VARIANT" == "modified" && "${USE_BUNDLE_DIRECT:-0}" != "1" && ! -f "$TARGET_CHECKPOINT" ]]; then
  FALLBACK_CHECKPOINT="$BUNDLE_DIR/checkpoint-best.pth"
  if [[ -f "$FALLBACK_CHECKPOINT" && ( -z "${MODIFIED_CHECKPOINT:-}" || "${MODIFIED_CHECKPOINT:-}" == "$DEFAULT_LIGHT_CHECKPOINT" ) ]]; then
    echo "[plaintext] modified plaintext light checkpoint 缺失，回退到 $FALLBACK_CHECKPOINT"
    TARGET_CHECKPOINT="$FALLBACK_CHECKPOINT"
  else
    echo "缺少 MODIFIED_CHECKPOINT：$TARGET_CHECKPOINT" >&2
    exit 1
  fi
fi

if [[ -n "${TARGET_THRESHOLD_JSON:-}" && ! -f "$TARGET_THRESHOLD_JSON" ]]; then
  if [[ -n "${CUSTOM_MODIFIED_THRESHOLD_JSON:-}" ]]; then
    echo "缺少 MODIFIED_THRESHOLD_JSON：$TARGET_THRESHOLD_JSON" >&2
    exit 1
  fi
  echo "[plaintext] threshold json 文件缺失；如 bundle manifest 内仍保留 threshold metadata，则会自动回退到该冻结阈值。"
  TARGET_THRESHOLD_JSON=""
fi

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

if [[ "$VARIANT" == "modified" && "${USE_BUNDLE_DIRECT:-0}" == "1" ]]; then
  echo "[plaintext] modified full-val 默认直接评估 frozen bundle 的 state_dict，与 secure replay / fairness 保持同口径。"
  "$PYTHON_BIN" tools/transshield_plaintext_checkpoint_eval.py \
    --bundle-dir "$BUNDLE_DIR" \
    --data-path "$VAL_DATA_PATH" \
    --device "$PLAINTEXT_EVAL_DEVICE" \
    --batch-size "$PLAINTEXT_EVAL_BATCH_SIZE" \
    --num-workers "$PLAINTEXT_EVAL_NUM_WORKERS" \
    --max-samples "$PLAINTEXT_MAX_SAMPLES" \
    --label "$TARGET_LABEL" \
    --output-json "$SECURE_RUN_DIR/${OUTPUT_PREFIX}.json" \
    --output-csv "$SECURE_RUN_DIR/${OUTPUT_PREFIX}.csv" \
    "${THRESHOLD_ARGS[@]}"
else
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
fi
