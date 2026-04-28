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
BUNDLE_DIR="${BUNDLE_DIR:-artifacts/frozen_bundle_verified_tracka_lr3e5_20260414}"
SECURE_RUN_DIR="${SECURE_RUN_DIR:-artifacts/server_pipeline_run/${RUN_NAME}}"
CONFIG_PATH="${CONFIG_PATH:-configs/openbumblebee/2pc.json}"
SECURE_MAX_SAMPLES="${SECURE_MAX_SAMPLES:-0}"
SECURE_EXPORT_DEVICE="${SECURE_EXPORT_DEVICE:-cpu}"
SECURE_EXPORT_BATCH_SIZE="${SECURE_EXPORT_BATCH_SIZE:-32}"
SECURE_EXPORT_NUM_WORKERS="${SECURE_EXPORT_NUM_WORKERS:-0}"
INPUT_IMAGE="${INPUT_IMAGE:-}"
INPUT_IMAGE_LIST="${INPUT_IMAGE_LIST:-}"
INPUT_IMAGE_DIR="${INPUT_IMAGE_DIR:-}"
INPUT_GLOB_PATTERN="${INPUT_GLOB_PATTERN:-*}"
if [[ -z "$VAL_DATA_PATH" && -z "$INPUT_IMAGE" && -z "$INPUT_IMAGE_LIST" && -z "$INPUT_IMAGE_DIR" ]]; then
  echo "请先设置 VAL_DATA_PATH，或设置 INPUT_IMAGE / INPUT_IMAGE_LIST / INPUT_IMAGE_DIR 之一再运行。" >&2
  exit 1
fi

echo "[secure-export] 导出 secure sidecar 输入。"
echo "[secure-export] 这一步只负责导出 pruning 决策边界相关输入，不执行 secure 协议本身。"
echo "[secure-export] 注意：输出文件名中保留了历史上的 *_smoke8 命名，但真实样本数量由 SECURE_MAX_SAMPLES 控制。"

INPUT_ARGS=()
if [[ -n "$VAL_DATA_PATH" ]]; then
  INPUT_ARGS+=(--data-path "$VAL_DATA_PATH")
fi
if [[ -n "$INPUT_IMAGE" ]]; then
  INPUT_ARGS+=(--image "$INPUT_IMAGE")
fi
if [[ -n "$INPUT_IMAGE_LIST" ]]; then
  INPUT_ARGS+=(--image-list "$INPUT_IMAGE_LIST")
fi
if [[ -n "$INPUT_IMAGE_DIR" ]]; then
  INPUT_ARGS+=(--input-dir "$INPUT_IMAGE_DIR" --glob-pattern "$INPUT_GLOB_PATTERN")
fi
mkdir -p "$SECURE_RUN_DIR"
"$PYTHON_BIN" tools/transshield_secure_sidecar_export_suite.py --bundle-dir "$BUNDLE_DIR" --device "$SECURE_EXPORT_DEVICE" --batch-size "$SECURE_EXPORT_BATCH_SIZE" --num-workers "$SECURE_EXPORT_NUM_WORKERS" --max-samples "$SECURE_MAX_SAMPLES" --input-output-pt "$SECURE_RUN_DIR/stage2_secure_network_kth_input_smoke8.pt" --input-output-json "$SECURE_RUN_DIR/stage2_secure_network_kth_input_smoke8.json" --kth-output-pt "$SECURE_RUN_DIR/stage2_secure_network_kth_reference_smoke8.pt" --kth-output-json "$SECURE_RUN_DIR/stage2_secure_network_kth_reference_smoke8.json" --tie-output-pt "$SECURE_RUN_DIR/stage2_secure_tie_policy_lowest_smoke8.pt" --tie-output-json "$SECURE_RUN_DIR/stage2_secure_tie_policy_lowest_smoke8.json" "${INPUT_ARGS[@]}"
"$PYTHON_BIN" tools/transshield_secure_network_kth.py manifest --bundle-dir "$BUNDLE_DIR" --output-json "$SECURE_RUN_DIR/stage2_secure_network_kth_manifest.json"
