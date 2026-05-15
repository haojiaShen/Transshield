#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="${RUN_NAME:-delivery_line_suite_$(date +%Y%m%d_%H%M%S)}"
SECURE_RUN_DIR="${SECURE_RUN_DIR:-$REPO_ROOT/artifacts/server_pipeline_run/$RUN_NAME}"
SECRET_RUN_DIR="${SECRET_RUN_DIR:-${RUN_ROOT:-$REPO_ROOT/artifacts/server_pipeline_run/secret_depth6_clip0_guarded_eval_20260505_clean}}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/stage_cost_risk_model/$RUN_NAME}"
OUTPUT_JSON="${OUTPUT_JSON:-$OUTPUT_DIR/stage_cost_risk_report.json}"
OUTPUT_MD="${OUTPUT_MD:-$OUTPUT_DIR/stage_cost_risk_report.md}"
ACCEPTANCE_JSON="${ACCEPTANCE_JSON:-$REPO_ROOT/results/delivery_acceptance/$RUN_NAME/delivery_acceptance_report.json}"

ARGS=(
  --run-dir "$SECURE_RUN_DIR"
  --secret-run-dir "$SECRET_RUN_DIR"
  --output-json "$OUTPUT_JSON"
  --output-md "$OUTPUT_MD"
)

if [[ -f "$ACCEPTANCE_JSON" ]]; then
  ARGS+=(--acceptance-json "$ACCEPTANCE_JSON")
fi

echo "[stage-cost-risk] run_dir=$SECURE_RUN_DIR"
echo "[stage-cost-risk] secret_run_dir=$SECRET_RUN_DIR"
echo "[stage-cost-risk] output_json=$OUTPUT_JSON"
echo "[stage-cost-risk] output_md=$OUTPUT_MD"

"$PYTHON_BIN" tools/transshield_stage_cost_risk_report.py "${ARGS[@]}"

echo "[stage-cost-risk] 完成："
echo "[stage-cost-risk] JSON: $OUTPUT_JSON"
echo "[stage-cost-risk] MD:   $OUTPUT_MD"
