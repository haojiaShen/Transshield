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
SECURE_REPLAY_BATCH_SIZE="${SECURE_REPLAY_BATCH_SIZE:-32}"
SECURE_REPLAY_NUM_WORKERS="${SECURE_REPLAY_NUM_WORKERS:-0}"

echo "[replay] 将 secure sidecar 输出接回 modified 明文模型剩余前向。"
echo "[replay] replay 的目标是验证 secure payload 是否足以恢复与明文一致的预测语义。"

"$PYTHON_BIN" tools/transshield_openbumblebee_pipeline.py replay --output-dir "$SECURE_RUN_DIR" --bundle-dir "$BUNDLE_DIR" --device cpu --batch-size "$SECURE_REPLAY_BATCH_SIZE" --num-workers "$SECURE_REPLAY_NUM_WORKERS" --enable-model-replay
