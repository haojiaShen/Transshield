#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"

echo "[legacy] 运行 modified 明文训练 + threshold + bundle 全流程。"
echo "[legacy] 该脚本用于重建资产，不是默认比赛展示入口。"

"$SCRIPT_DIR/run_train.sh"
"$SCRIPT_DIR/run_threshold.sh" search
"$SCRIPT_DIR/run_threshold.sh" eval
"$SCRIPT_DIR/run_freeze_export.sh"
"$SCRIPT_DIR/run_verify_bundle.sh"
