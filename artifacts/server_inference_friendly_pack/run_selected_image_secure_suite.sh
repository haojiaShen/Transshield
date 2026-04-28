#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"
SECURE_RUNTIME="${SECURE_RUNTIME:-spu}"

echo "[selected-suite] 运行选图 secure 闭环。"
if [[ "$SECURE_RUNTIME" == "cpu" ]]; then
  echo "[selected-suite] 当前使用 CPU 参考后端：便于调试，不是真正 2PC。"
else
  echo "[selected-suite] 当前使用 SPU 后端：真实 secure 执行。"
fi

bash "$SCRIPT_DIR/run_secure_export_inputs.sh"
if [[ "$SECURE_RUNTIME" == "cpu" ]]; then
  bash "$SCRIPT_DIR/run_secure_pipeline.sh" cpu
else
  bash "$SCRIPT_DIR/run_secure_pipeline.sh" spu
fi
bash "$SCRIPT_DIR/run_secure_replay.sh"
bash "$SCRIPT_DIR/run_selected_image_secure_diagnosis.sh"
