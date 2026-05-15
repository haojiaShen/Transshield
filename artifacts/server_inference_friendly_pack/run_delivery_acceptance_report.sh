#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="${RUN_NAME:-delivery_acceptance_$(date +%Y%m%d_%H%M%S)}"
BUNDLE_DIR="${BUNDLE_DIR:-$REPO_ROOT/artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_20260430}"
SECURE_RUN_DIR="${SECURE_RUN_DIR:-$REPO_ROOT/artifacts/server_pipeline_run/${RUN_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/delivery_acceptance/${RUN_NAME}}"
OUTPUT_JSON="${OUTPUT_JSON:-$OUTPUT_DIR/delivery_acceptance_report.json}"
OUTPUT_MD="${OUTPUT_MD:-$OUTPUT_DIR/delivery_acceptance_report.md}"

PLAINTEXT_EVAL_JSON="${PLAINTEXT_EVAL_JSON:-$SECURE_RUN_DIR/plaintext_modified_eval.json}"
FAIR_COMPARISON_JSON="${FAIR_COMPARISON_JSON:-}"
NETWORK_KTH_CHECK_JSON="${NETWORK_KTH_CHECK_JSON:-$SECURE_RUN_DIR/stage2_secure_network_kth_candidate_check.json}"
TIE_CHECK_JSON="${TIE_CHECK_JSON:-$SECURE_RUN_DIR/stage2_secure_tie_candidate_check.json}"
PLAINTEXT_SECURE_COMPARE_JSON="${PLAINTEXT_SECURE_COMPARE_JSON:-$SECURE_RUN_DIR/plaintext_vs_secure_score_compare.json}"
E2E_VERIFY_JSON="${E2E_VERIFY_JSON:-}"
SECRET_ISOLATED_SUMMARY_JSON="${SECRET_ISOLATED_SUMMARY_JSON:-$SECURE_RUN_DIR/secret_isolated_eval_summary.json}"

ARGS=(
  --bundle-dir "$BUNDLE_DIR"
  --output-json "$OUTPUT_JSON"
  --output-md "$OUTPUT_MD"
)

if [[ -f "$PLAINTEXT_EVAL_JSON" ]]; then
  ARGS+=(--plaintext-eval-json "$PLAINTEXT_EVAL_JSON")
fi
if [[ -n "$FAIR_COMPARISON_JSON" && -f "$FAIR_COMPARISON_JSON" ]]; then
  ARGS+=(--fair-comparison-json "$FAIR_COMPARISON_JSON")
fi
if [[ -f "$NETWORK_KTH_CHECK_JSON" ]]; then
  ARGS+=(--network-kth-check-json "$NETWORK_KTH_CHECK_JSON")
fi
if [[ -f "$TIE_CHECK_JSON" ]]; then
  ARGS+=(--tie-check-json "$TIE_CHECK_JSON")
fi
if [[ -f "$PLAINTEXT_SECURE_COMPARE_JSON" ]]; then
  ARGS+=(--plaintext-secure-compare-json "$PLAINTEXT_SECURE_COMPARE_JSON")
fi
if [[ -n "$E2E_VERIFY_JSON" && -f "$E2E_VERIFY_JSON" ]]; then
  ARGS+=(--e2e-verify-json "$E2E_VERIFY_JSON")
fi
if [[ -f "$SECRET_ISOLATED_SUMMARY_JSON" ]]; then
  ARGS+=(--secret-isolated-summary-json "$SECRET_ISOLATED_SUMMARY_JSON")
fi

echo "[acceptance] 生成当前 delivery line 验收汇总。"
echo "[acceptance] BUNDLE_DIR=$BUNDLE_DIR"
echo "[acceptance] SECURE_RUN_DIR=$SECURE_RUN_DIR"
echo "[acceptance] OUTPUT_JSON=$OUTPUT_JSON"
echo "[acceptance] OUTPUT_MD=$OUTPUT_MD"

"$PYTHON_BIN" tools/transshield_delivery_acceptance_report.py "${ARGS[@]}"

echo "[acceptance] 完成："
echo "[acceptance] JSON: $OUTPUT_JSON"
echo "[acceptance] MD:   $OUTPUT_MD"
