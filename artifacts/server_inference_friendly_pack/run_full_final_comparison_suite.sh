#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"
SECURE_RUNTIME="${SECURE_RUNTIME:-spu}"

if [[ "$SECURE_RUNTIME" != "cpu" && "$SECURE_RUNTIME" != "spu" ]]; then
  echo "Invalid SECURE_RUNTIME: $SECURE_RUNTIME (expected: cpu or spu)" >&2
  exit 1
fi

echo "[suite] 开始完整 comparison suite。"
echo "[suite] 结果主线：baseline vs modified；modified vs secure。"
if [[ "$SECURE_RUNTIME" == "cpu" ]]; then
  echo "[suite] 当前 secure runtime = cpu：本地明文参考执行，用于调试与链路验证，不是真正 2PC。"
else
  echo "[suite] 当前 secure runtime = spu：真实 secure 执行，将调用 SPU / OpenBumbleBee。"
fi

echo "[suite] Step 1/8：baseline 明文评估"
"$SCRIPT_DIR/run_plaintext_eval.sh" baseline
echo "[suite] Step 2/8：modified 明文评估"
"$SCRIPT_DIR/run_plaintext_eval.sh" modified
echo "[suite] Step 3/8：baseline vs modified 对比"
"$SCRIPT_DIR/run_plaintext_model_compare.sh"
echo "[suite] Step 4/8：secure sidecar 输入导出"
"$SCRIPT_DIR/run_secure_export_inputs.sh"
if [[ "$SECURE_RUNTIME" == "cpu" ]]; then
  echo "[suite] Step 5/8：secure pipeline（CPU 参考后端）"
  "$SCRIPT_DIR/run_secure_pipeline.sh" cpu
else
  echo "[suite] Step 5/8：secure pipeline（SPU 后端）"
  "$SCRIPT_DIR/run_secure_pipeline.sh" spu
fi
echo "[suite] Step 6/8：secure replay"
"$SCRIPT_DIR/run_secure_replay.sh"
echo "[suite] Step 7/8：modified plaintext vs secure 对比"
"$SCRIPT_DIR/run_secure_score_compare.sh"

if [[ -n "${SECURE_BASELINE_PROFILE_JSON:-}" ]]; then
  echo "[suite] 可选步骤：secure profile 对比"
  "$SCRIPT_DIR/run_secure_profile_compare.sh"
else
  echo "[skip] secure profile compare：未设置 SECURE_BASELINE_PROFILE_JSON。"
fi

echo "[suite] Step 8/8：生成最终 comparison report"
bash "$SCRIPT_DIR/run_final_comparison_report.sh"
echo "[suite] 完整 comparison suite 运行完成。"
