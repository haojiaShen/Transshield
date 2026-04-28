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
BUNDLE_DIR="${BUNDLE_DIR:-artifacts/server_bundles/${RUN_NAME}_bundle}"
SECURE_RUN_DIR="${SECURE_RUN_DIR:-artifacts/server_pipeline_run/${RUN_NAME}}"
CONFIG_PATH="${CONFIG_PATH:-configs/openbumblebee/2pc.json}"

echo "[profile] 汇总当前 secure 运行的 profile 信息。"
echo "[profile] 该脚本用于补充时间/通信分析，不属于默认结果主链路。"

FASTPATH_PROFILE_JSON="$SECURE_RUN_DIR/fastpath_profile_summary.json"
FASTPATH_PROFILE_MD="$SECURE_RUN_DIR/fastpath_profile_summary.md"
if "$PYTHON_BIN" tools/transshield_fastpath_profile_summary.py \
  logs/spu_nodes \
  "$SECURE_RUN_DIR" \
  "$SECURE_RUN_DIR/step_logs" \
  --output-json "$FASTPATH_PROFILE_JSON" \
  --output-md "$FASTPATH_PROFILE_MD"; then
  echo "[profile] Python fastpath 通信摘要：$FASTPATH_PROFILE_JSON"
else
  echo "[profile] Python fastpath 通信摘要生成失败，将继续使用已有 SPU profile 信息。" >&2
fi

PROFILE_ARGS=(
  --secure-run-dir "$SECURE_RUN_DIR"
  --spu-state-json logs/spu_runtime_ports.json
  --spu-log-dir logs/spu_nodes
  --output-json "$SECURE_RUN_DIR/secure_profile_summary.json"
)
if [[ -s "$FASTPATH_PROFILE_JSON" ]]; then
  PROFILE_ARGS+=(--fastpath-profile-json "$FASTPATH_PROFILE_JSON")
fi

"$PYTHON_BIN" tools/transshield_secure_profile_summary.py "${PROFILE_ARGS[@]}"
