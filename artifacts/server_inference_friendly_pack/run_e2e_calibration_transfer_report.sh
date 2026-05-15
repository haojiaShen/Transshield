#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="${RUN_DIR:-}"
SPLIT_LABEL="${SPLIT_LABEL:-}"
CALIBRATION_ROOT="${CALIBRATION_ROOT:-$REPO_ROOT/results/e2e_static_calibration/e2e_static_fullval_aanone_20260507_1}"

if [[ -z "$RUN_DIR" || -z "$SPLIT_LABEL" ]]; then
  echo "Usage: RUN_DIR=<.../e2e_secure_poc> SPLIT_LABEL=<heldoutX> $0" >&2
  exit 2
fi

RUN_DIR="$(cd "$RUN_DIR" && pwd)"
mkdir -p "$CALIBRATION_ROOT"

CANDIDATE_PT="$RUN_DIR/e2e_static_whole_forward_candidate_spu_depth12_partylocal_publiccalibln_uniform_fixed_square_clip0_eval.pt"
REFERENCE_JSON="$RUN_DIR/plaintext_same_images_reference.json"
IMAGE_LIST="$RUN_DIR/e2e_eval_images.txt"

for required in "$CANDIDATE_PT" "$REFERENCE_JSON" "$IMAGE_LIST"; do
  if [[ ! -f "$required" ]]; then
    echo "[e2e-cal-transfer] missing required file: $required" >&2
    exit 2
  fi
done

OUTPUT_JSON="$CALIBRATION_ROOT/e2e_${SPLIT_LABEL}_spu_smoke32_calibration_transfer_report.json"
OUTPUT_CSV="$CALIBRATION_ROOT/e2e_${SPLIT_LABEL}_spu_smoke32_calibration_transfer_report.csv"

"$PYTHON_BIN" tools/transshield_e2e_calibration_drift_report.py \
  --candidate-pt "$CANDIDATE_PT" \
  --plaintext-reference-json "$REFERENCE_JSON" \
  --image-list "$IMAGE_LIST" \
  --calibration static_bias="$CALIBRATION_ROOT/e2e_static_output_calibration_public_logit_bias.json" \
  --calibration spuaware_bias="$CALIBRATION_ROOT/e2e_static_output_calibration_public_logit_bias_spuaware_smoke32_threshold.json" \
  --calibration e2e_smoke32_bias="$CALIBRATION_ROOT/e2e_output_calibration_public_logit_bias_fit_on_spu_smoke32.json" \
  --calibration e2e_smoke32_affine="$CALIBRATION_ROOT/e2e_output_calibration_public_affine_fit_on_spu_smoke32.json" \
  --calibration e2e_smoke32_temperature="$CALIBRATION_ROOT/e2e_output_calibration_public_temperature_fit_on_spu_smoke32.json" \
  --label "e2e_${SPLIT_LABEL}_spu_smoke32_calibration_transfer" \
  --output-json "$OUTPUT_JSON" \
  --output-csv "$OUTPUT_CSV" \
  > "$CALIBRATION_ROOT/e2e_${SPLIT_LABEL}_spu_smoke32_calibration_transfer_report.stdout"

echo "[e2e-cal-transfer] wrote $OUTPUT_JSON"
echo "[e2e-cal-transfer] wrote $OUTPUT_CSV"
"$PYTHON_BIN" - "$OUTPUT_JSON" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(f"[e2e-cal-transfer] sample_count={payload.get('sample_count')}")
for label, item in sorted((payload.get("calibration_results") or {}).items()):
    bce = item.get("binary_cross_entropy")
    wrong = (item.get("score_summary") or {}).get("wrong_count")
    print(f"[e2e-cal-transfer] {label}: acc={item.get('accuracy')} bce={bce} wrong={wrong}")
PY
