#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"

MODE="${1:-fullval}"
case "$MODE" in
  fullval)
    DEFAULT_MAX_SAMPLES=0
    ;;
  smoke4)
    DEFAULT_MAX_SAMPLES=4
    ;;
  smoke8)
    DEFAULT_MAX_SAMPLES=8
    ;;
  smoke16)
    DEFAULT_MAX_SAMPLES=16
    ;;
  smoke32)
    DEFAULT_MAX_SAMPLES=32
    ;;
  custom)
    DEFAULT_MAX_SAMPLES="${E2E_GAP_MAX_SAMPLES:-0}"
    ;;
  *)
    echo "Usage: $0 [fullval|smoke4|smoke8|smoke16|smoke32|custom]" >&2
    exit 2
    ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python}"
FULLVAL_GAP_BUNDLE_DIR="${FULLVAL_GAP_BUNDLE_DIR:-$REPO_ROOT/artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_aanone_20260507}"
E2E_GAP_DATASET_DIR="${E2E_GAP_DATASET_DIR:-${DATA_ROOT:-/data/wyb/pneumoniamnist_imagefolder_subset}/val}"
E2E_GAP_MAX_SAMPLES="${E2E_GAP_MAX_SAMPLES:-$DEFAULT_MAX_SAMPLES}"
E2E_GAP_OUTPUT_DIR="${E2E_GAP_OUTPUT_DIR:-$REPO_ROOT/results/e2e_gap_attribution/fullval_plaintext_static_gap_$(date +%Y%m%d_%H%M%S)}"
E2E_GAP_LABEL="${E2E_GAP_LABEL:-$(basename "$E2E_GAP_OUTPUT_DIR")}"
E2E_GAP_DEVICE="${E2E_GAP_DEVICE:-cpu}"

if [[ ! -d "$FULLVAL_GAP_BUNDLE_DIR" ]]; then
  echo "[fullval-gap] missing FULLVAL_GAP_BUNDLE_DIR: $FULLVAL_GAP_BUNDLE_DIR" >&2
  exit 2
fi
if [[ ! -d "$E2E_GAP_DATASET_DIR" ]]; then
  echo "[fullval-gap] missing E2E_GAP_DATASET_DIR: $E2E_GAP_DATASET_DIR" >&2
  exit 2
fi

mkdir -p "$E2E_GAP_OUTPUT_DIR"

INPUT_PT="$E2E_GAP_OUTPUT_DIR/fullval_pixel_values.pt"
INPUT_JSON="$E2E_GAP_OUTPUT_DIR/fullval_pixel_values.json"
PLAINTEXT_JSON="$E2E_GAP_OUTPUT_DIR/fullval_plaintext_reference.json"
STATIC_JSON="$E2E_GAP_OUTPUT_DIR/fullval_static_reference.json"
STATIC_PT="$E2E_GAP_OUTPUT_DIR/fullval_static_reference.pt"
REPORT_JSON="$E2E_GAP_OUTPUT_DIR/fullval_plaintext_static_gap_report.json"
REPORT_MD="$E2E_GAP_OUTPUT_DIR/fullval_plaintext_static_gap_report.md"

PREPROCESS_ARGS=(
  "$PYTHON_BIN" tools/transshield_e2e_secure_infer.py client-preprocess
  --bundle-dir "$FULLVAL_GAP_BUNDLE_DIR"
  --data-path "$E2E_GAP_DATASET_DIR"
  --output-pt "$INPUT_PT"
  --output-json "$INPUT_JSON"
  --include-source-paths
  --include-targets
)
if [[ "$E2E_GAP_MAX_SAMPLES" -gt 0 ]]; then
  PREPROCESS_ARGS+=(--max-samples "$E2E_GAP_MAX_SAMPLES")
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

echo "[fullval-gap] mode=$MODE"
echo "[fullval-gap] bundle_dir=$FULLVAL_GAP_BUNDLE_DIR"
echo "[fullval-gap] dataset_dir=$E2E_GAP_DATASET_DIR"
echo "[fullval-gap] max_samples=$E2E_GAP_MAX_SAMPLES"
echo "[fullval-gap] output_dir=$E2E_GAP_OUTPUT_DIR"
echo "[fullval-gap] label=$E2E_GAP_LABEL"
echo "[fullval-gap] device=$E2E_GAP_DEVICE"

"${PREPROCESS_ARGS[@]}"

"$PYTHON_BIN" tools/transshield_e2e_secure_infer.py plaintext-reference \
  --bundle-dir "$FULLVAL_GAP_BUNDLE_DIR" \
  --input-pt "$INPUT_PT" \
  --device "$E2E_GAP_DEVICE" \
  --output-json "$PLAINTEXT_JSON"

"$PYTHON_BIN" tools/transshield_e2e_secure_infer.py static-whole-forward-reference \
  --bundle-dir "$FULLVAL_GAP_BUNDLE_DIR" \
  --input-pt "$INPUT_PT" \
  --device "$E2E_GAP_DEVICE" \
  --output-json "$STATIC_JSON" \
  --output-pt "$STATIC_PT"

"$PYTHON_BIN" tools/transshield_e2e_plaintext_static_gap_report.py \
  --label "$E2E_GAP_LABEL" \
  --plaintext-json "$PLAINTEXT_JSON" \
  --static-json "$STATIC_JSON" \
  --output-json "$REPORT_JSON" \
  --output-md "$REPORT_MD"

echo "[fullval-gap] wrote $REPORT_JSON"
echo "[fullval-gap] wrote $REPORT_MD"
