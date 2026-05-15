#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
CALIBRATION_ROOT="${CALIBRATION_ROOT:-$REPO_ROOT/results/e2e_static_calibration/e2e_static_fullval_aanone_20260507_1}"
INCLUDE_SPLITS="${INCLUDE_SPLITS:-heldout64 heldout128 heldout238}"
DECISION_LABEL="${DECISION_LABEL:-e2e_spuaware_vs_e2e_smoke32_output_calibration_decision_20260507}"

RUN_DIR="${RUN_DIR:-}"
SPLIT_LABEL="${SPLIT_LABEL:-}"

HELDOUT64_E2E_RUN_DIR="${HELDOUT64_E2E_RUN_DIR:-$REPO_ROOT/artifacts/server_pipeline_run/e2e_aanone_exactln_clip0_heldout64_spuaware_nonisolated_20260507_2/e2e_secure_poc}"
HELDOUT128_E2E_RUN_DIR="${HELDOUT128_E2E_RUN_DIR:-$REPO_ROOT/artifacts/server_pipeline_run/e2e_aanone_exactln_clip0_heldout128_spuaware_nonisolated_20260507_1/e2e_secure_poc}"
HELDOUT238_E2E_RUN_DIR="${HELDOUT238_E2E_RUN_DIR:-$REPO_ROOT/artifacts/server_pipeline_run/e2e_aanone_exactln_clip0_heldout238_spuaware_nonisolated_20260507_1/e2e_secure_poc}"

BRIDGE_OUTPUT_DIR="${BRIDGE_OUTPUT_DIR:-$CALIBRATION_ROOT/e2e_plaintext_bridge_calibration_suite}"
BRIDGE_LABEL="${BRIDGE_LABEL:-e2e_plaintext_bridge_calibration_suite}"

STATIC_BIAS_JSON="${STATIC_BIAS_JSON:-$CALIBRATION_ROOT/e2e_static_output_calibration_public_logit_bias.json}"
SPUAWARE_BIAS_JSON="${SPUAWARE_BIAS_JSON:-$CALIBRATION_ROOT/e2e_static_output_calibration_public_logit_bias_spuaware_smoke32_threshold.json}"
AFFINE_JSON="${AFFINE_JSON:-$CALIBRATION_ROOT/e2e_output_calibration_public_affine_fit_on_spu_smoke32.json}"
TEMPERATURE_JSON="${TEMPERATURE_JSON:-$CALIBRATION_ROOT/e2e_output_calibration_public_temperature_fit_on_spu_smoke32.json}"

mkdir -p "$CALIBRATION_ROOT" "$BRIDGE_OUTPUT_DIR"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "[e2e-cal-suite] missing $label: $path" >&2
    exit 2
  fi
}

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "$path" ]]; then
    echo "[e2e-cal-suite] missing $label: $path" >&2
    exit 2
  fi
}

run_postprocess() {
  CALIBRATION_ROOT="$CALIBRATION_ROOT" \
  INCLUDE_SPLITS="$INCLUDE_SPLITS" \
  DECISION_LABEL="$DECISION_LABEL" \
  PYTHON_BIN="$PYTHON_BIN" \
  RUN_DIR="$RUN_DIR" \
  SPLIT_LABEL="$SPLIT_LABEL" \
    bash "$SCRIPT_DIR/run_e2e_calibration_postprocess.sh"
}

build_bridge_arg() {
  local label="$1"
  local run_dir="$2"
  printf '%s=%s,%s' \
    "$label" \
    "$run_dir/plaintext_same_images_reference.json" \
    "$run_dir/static_whole_forward_reference.json"
}

build_eval_arg() {
  local label="$1"
  local run_dir="$2"
  printf '%s=%s,%s' \
    "$label" \
    "$run_dir/e2e_static_whole_forward_candidate_spu_depth12_partylocal_publiccalibln_uniform_fixed_square_clip0_eval.pt" \
    "$run_dir/plaintext_same_images_reference.json"
}

for path in \
  "$HELDOUT64_E2E_RUN_DIR" \
  "$HELDOUT128_E2E_RUN_DIR" \
  "$HELDOUT238_E2E_RUN_DIR"; do
  require_dir "$path" "heldout e2e run dir"
done

for path in \
  "$STATIC_BIAS_JSON" \
  "$SPUAWARE_BIAS_JSON" \
  "$AFFINE_JSON" \
  "$TEMPERATURE_JSON"; do
  require_file "$path" "calibration json"
done

run_postprocess

"$PYTHON_BIN" tools/transshield_e2e_plaintext_bridge_calibration.py \
  --label "$BRIDGE_LABEL" \
  --bridge "$(build_bridge_arg heldout64 "$HELDOUT64_E2E_RUN_DIR")" \
  --bridge "$(build_bridge_arg heldout128 "$HELDOUT128_E2E_RUN_DIR")" \
  --bridge "$(build_bridge_arg heldout238 "$HELDOUT238_E2E_RUN_DIR")" \
  --eval "$(build_eval_arg heldout64 "$HELDOUT64_E2E_RUN_DIR")" \
  --eval "$(build_eval_arg heldout128 "$HELDOUT128_E2E_RUN_DIR")" \
  --eval "$(build_eval_arg heldout238 "$HELDOUT238_E2E_RUN_DIR")" \
  --compare-calibration "static_bias=$STATIC_BIAS_JSON" \
  --compare-calibration "spuaware_bias=$SPUAWARE_BIAS_JSON" \
  --compare-calibration "e2e_smoke32_affine=$AFFINE_JSON" \
  --compare-calibration "e2e_smoke32_temperature=$TEMPERATURE_JSON" \
  --output-json "$BRIDGE_OUTPUT_DIR/e2e_plaintext_bridge_calibration_report.json" \
  --output-md "$BRIDGE_OUTPUT_DIR/e2e_plaintext_bridge_calibration_report.md" \
  --output-best-calibration-json "$BRIDGE_OUTPUT_DIR/e2e_plaintext_bridge_best_calibration.json" \
  --output-best-bridge-calibration-json "$BRIDGE_OUTPUT_DIR/e2e_plaintext_bridge_best_bridge_calibration.json"

echo "[e2e-cal-suite] wrote $CALIBRATION_ROOT/e2e_output_calibration_decision_report.json"
echo "[e2e-cal-suite] wrote $CALIBRATION_ROOT/e2e_output_calibration_decision_report.md"
echo "[e2e-cal-suite] wrote $BRIDGE_OUTPUT_DIR/e2e_plaintext_bridge_calibration_report.json"
echo "[e2e-cal-suite] wrote $BRIDGE_OUTPUT_DIR/e2e_plaintext_bridge_calibration_report.md"
echo "[e2e-cal-suite] wrote $BRIDGE_OUTPUT_DIR/e2e_plaintext_bridge_best_calibration.json"
echo "[e2e-cal-suite] wrote $BRIDGE_OUTPUT_DIR/e2e_plaintext_bridge_best_bridge_calibration.json"
