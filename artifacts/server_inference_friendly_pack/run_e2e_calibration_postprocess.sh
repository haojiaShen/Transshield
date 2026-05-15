#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
CALIBRATION_ROOT="${CALIBRATION_ROOT:-$REPO_ROOT/results/e2e_static_calibration/e2e_static_fullval_aanone_20260507_1}"
RUN_DIR="${RUN_DIR:-}"
SPLIT_LABEL="${SPLIT_LABEL:-}"
INCLUDE_SPLITS="${INCLUDE_SPLITS:-heldout64 heldout128 heldout238}"
DECISION_LABEL="${DECISION_LABEL:-e2e_spuaware_vs_e2e_smoke32_output_calibration_decision_20260507}"

mkdir -p "$CALIBRATION_ROOT"

if [[ -n "$SPLIT_LABEL" ]]; then
  if [[ -z "$RUN_DIR" ]]; then
    echo "[e2e-cal-postprocess] RUN_DIR must be set when SPLIT_LABEL is set" >&2
    exit 2
  fi
  RUN_DIR="$RUN_DIR" \
  SPLIT_LABEL="$SPLIT_LABEL" \
  CALIBRATION_ROOT="$CALIBRATION_ROOT" \
  PYTHON_BIN="$PYTHON_BIN" \
    bash "$SCRIPT_DIR/run_e2e_calibration_transfer_report.sh"
fi

transfer_args=()
for split in $INCLUDE_SPLITS; do
  report="$CALIBRATION_ROOT/e2e_${split}_spu_smoke32_calibration_transfer_report.json"
  if [[ -f "$report" ]]; then
    transfer_args+=(--transfer-report "$split=$report")
  else
    echo "[e2e-cal-postprocess] skip missing transfer report: $report" >&2
  fi
done

if [[ "${#transfer_args[@]}" -eq 0 ]]; then
  echo "[e2e-cal-postprocess] no transfer reports found under $CALIBRATION_ROOT" >&2
  exit 2
fi

"$PYTHON_BIN" tools/transshield_e2e_calibration_decision_report.py \
  "${transfer_args[@]}" \
  --label "$DECISION_LABEL" \
  --output-json "$CALIBRATION_ROOT/e2e_output_calibration_decision_report.json" \
  --output-md "$CALIBRATION_ROOT/e2e_output_calibration_decision_report.md"

echo "[e2e-cal-postprocess] wrote $CALIBRATION_ROOT/e2e_output_calibration_decision_report.json"
echo "[e2e-cal-postprocess] wrote $CALIBRATION_ROOT/e2e_output_calibration_decision_report.md"
