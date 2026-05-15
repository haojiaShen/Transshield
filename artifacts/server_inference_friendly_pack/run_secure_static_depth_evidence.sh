#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="${RUN_NAME:-delivery_line_suite_$(date +%Y%m%d_%H%M%S)}"
BUNDLE_DIR="${BUNDLE_DIR:-$REPO_ROOT/artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_20260430}"
SECURE_RUN_DIR="${SECURE_RUN_DIR:-$REPO_ROOT/artifacts/server_pipeline_run/$RUN_NAME}"
SECRET_RUN_DIR="${SECRET_RUN_DIR:-${RUN_ROOT:-$REPO_ROOT/artifacts/server_pipeline_run/secret_depth6_clip0_guarded_eval_20260505_clean}}"
DEFAULT_ACCEPTANCE_JSON="$REPO_ROOT/results/delivery_acceptance/delivery_acceptance_20260510_full/delivery_acceptance_report.json"
FALLBACK_ACCEPTANCE_JSON="$REPO_ROOT/results/delivery_acceptance/${RUN_NAME}/delivery_acceptance_report.json"
ACCEPTANCE_JSON="${ACCEPTANCE_JSON:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/secure_static_train_depth_evidence/$RUN_NAME}"
OUTPUT_JSON="${OUTPUT_JSON:-$OUTPUT_DIR/secure_static_train_depth_evidence.json}"
OUTPUT_MD="${OUTPUT_MD:-$OUTPUT_DIR/secure_static_train_depth_evidence.md}"

if [[ -z "$ACCEPTANCE_JSON" ]]; then
  if [[ -f "$DEFAULT_ACCEPTANCE_JSON" ]]; then
    ACCEPTANCE_JSON="$DEFAULT_ACCEPTANCE_JSON"
  else
    ACCEPTANCE_JSON="$FALLBACK_ACCEPTANCE_JSON"
  fi
fi

ARGS=(
  --bundle-manifest-json "$BUNDLE_DIR/manifest.json"
  --bundle-args-snapshot-json "$BUNDLE_DIR/args_snapshot.json"
  --baseline-eval-json "$SECURE_RUN_DIR/plaintext_baseline_eval.json"
  --modified-eval-json "$SECURE_RUN_DIR/plaintext_modified_eval.json"
  --plaintext-model-compare-json "$SECURE_RUN_DIR/plaintext_model_compare.json"
  --plaintext-secure-compare-json "$SECURE_RUN_DIR/plaintext_vs_secure_score_compare.json"
  --secret-isolated-summary-json "$SECRET_RUN_DIR/secret_isolated_eval_summary.json"
  --output-json "$OUTPUT_JSON"
  --output-md "$OUTPUT_MD"
)

if [[ -f "$ACCEPTANCE_JSON" ]]; then
  ARGS+=(--acceptance-json "$ACCEPTANCE_JSON")
fi

shopt -s nullglob
PAIR_COMPARE_JSONS=(
  "$REPO_ROOT"/results/secure_static_train_depth_evidence/secure_static_depth_pair_*/secure_static_train_depth_pair_compare.json
)
shopt -u nullglob

if [[ ${#PAIR_COMPARE_JSONS[@]} -gt 0 ]]; then
  IFS=$'\n' PAIR_COMPARE_JSONS=($(printf '%s\n' "${PAIR_COMPARE_JSONS[@]}" | sort))
  unset IFS
  for pair_json in "${PAIR_COMPARE_JSONS[@]}"; do
    ARGS+=(--pair-compare-json "$pair_json")
  done
fi

echo "[secure-static-depth-evidence] bundle_dir=$BUNDLE_DIR"
echo "[secure-static-depth-evidence] secure_run_dir=$SECURE_RUN_DIR"
echo "[secure-static-depth-evidence] secret_run_dir=$SECRET_RUN_DIR"
echo "[secure-static-depth-evidence] acceptance_json=$ACCEPTANCE_JSON"
echo "[secure-static-depth-evidence] pair_compare_count=${#PAIR_COMPARE_JSONS[@]}"
echo "[secure-static-depth-evidence] output_json=$OUTPUT_JSON"
echo "[secure-static-depth-evidence] output_md=$OUTPUT_MD"

"$PYTHON_BIN" tools/transshield_secure_static_depth_evidence.py "${ARGS[@]}"

echo "[secure-static-depth-evidence] 完成："
echo "[secure-static-depth-evidence] JSON: $OUTPUT_JSON"
echo "[secure-static-depth-evidence] MD:   $OUTPUT_MD"
