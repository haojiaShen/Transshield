#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"

RUNTIME="${1:-cpu}"

case "$RUNTIME" in
  cpu)
    SECURE_PIPELINE_ARGS=("$SCRIPT_DIR/run_secure_pipeline.sh" cpu)
    HUMAN_LABEL="CPU"
    ;;
  spu)
    SECURE_PIPELINE_ARGS=("$SCRIPT_DIR/run_secure_pipeline.sh" spu)
    HUMAN_LABEL="SPU"
    ;;
  *)
    echo "Usage: $0 [cpu|spu]" >&2
    exit 1
    ;;
esac

echo "[legacy] 运行“从训练到 secure”的完整 ${HUMAN_LABEL} 重建链。"
echo "[legacy] 该脚本偏向研发/重建，不是 README 默认比赛展示入口。"

"$SCRIPT_DIR/run_full_vit_pipeline.sh"
"$SCRIPT_DIR/run_plaintext_eval.sh" modified
"$SCRIPT_DIR/run_secure_export_inputs.sh"
"${SECURE_PIPELINE_ARGS[@]}"
"$SCRIPT_DIR/run_secure_replay.sh"
"$SCRIPT_DIR/run_secure_score_compare.sh"
"$SCRIPT_DIR/run_secure_profile_summary.sh"
