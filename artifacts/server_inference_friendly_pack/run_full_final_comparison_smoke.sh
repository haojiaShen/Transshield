#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"
SMOKE_MAX_SAMPLES="${SMOKE_MAX_SAMPLES:-8}"
export PLAINTEXT_MAX_SAMPLES="${PLAINTEXT_MAX_SAMPLES:-$SMOKE_MAX_SAMPLES}"
export SECURE_MAX_SAMPLES="${SECURE_MAX_SAMPLES:-$SMOKE_MAX_SAMPLES}"

echo "[smoke] 运行小样本链路验证。"
echo "[smoke] 本次将 plaintext 与 secure 输入都截断到前 ${SMOKE_MAX_SAMPLES} 个样本。"
echo "[smoke] smoke 只用于验证脚本和闭环能否跑通，不用于判断最终模型性能。"

"$SCRIPT_DIR/run_full_final_comparison_suite.sh"
