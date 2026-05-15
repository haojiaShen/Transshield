#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
PROBE_NAME="${PROBE_NAME:-e2e_aanone_heldout238_selected_policy_probe_$(date +%Y%m%d_%H%M%S)}"
SOURCE_E2E_DIR="${SOURCE_E2E_DIR:-$REPO_ROOT/artifacts/server_pipeline_run/e2e_aanone_exactln_clip0_heldout238_spuaware_nonisolated_20260507_1/e2e_secure_poc}"
SOURCE_SHARE_MANIFEST_JSON="${SOURCE_SHARE_MANIFEST_JSON:-$SOURCE_E2E_DIR/client_pixel_values_debug_share_manifest.json}"
SOURCE_INPUT_PT="${SOURCE_INPUT_PT:-$SOURCE_E2E_DIR/plaintext_same_images_pixel_values.pt}"
SOURCE_EVAL_IMAGE_LIST="${SOURCE_EVAL_IMAGE_LIST:-$SOURCE_E2E_DIR/e2e_eval_images.txt}"
AA_NONE_BUNDLE_DIR="${AA_NONE_BUNDLE_DIR:-$REPO_ROOT/artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_aanone_20260507}"
if [[ "${ALLOW_E2E_POLICY_PROBE_BUNDLE_OVERRIDE:-0}" == "1" ]]; then
  BUNDLE_DIR="${BUNDLE_DIR:-$AA_NONE_BUNDLE_DIR}"
else
  BUNDLE_DIR="$AA_NONE_BUNDLE_DIR"
fi
OUTPUT_CALIBRATION_JSON="${OUTPUT_CALIBRATION_JSON:-$REPO_ROOT/results/e2e_static_calibration/e2e_static_fullval_aanone_20260507_1/e2e_static_output_calibration_public_logit_bias_spuaware_smoke32_threshold.json}"
SELECTED_INDICES="${SELECTED_INDICES:-121,220,167,227,21,49,217,196,23,54}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/artifacts/server_pipeline_run/$PROBE_NAME}"
RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/results/e2e_policy_probe/$PROBE_NAME}"
VARIANT_SPECS="${VARIANT_SPECS:-exact_uniform_clip0:exact:uniform:fixed_square:0:0:0 exact_uniform_clip3:exact:uniform:fixed_square:3:0:0 exact_uniform_clip0_lncmp64:exact:uniform:fixed_square:0:0:64}"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "[e2e-policy-probe] missing $label: $path" >&2
    exit 2
  fi
}

require_file "$SOURCE_SHARE_MANIFEST_JSON" "source share manifest"
require_file "$SOURCE_INPUT_PT" "source plaintext input pt"
require_file "$SOURCE_EVAL_IMAGE_LIST" "source eval image list"
require_file "$OUTPUT_CALIBRATION_JSON" "output calibration JSON"
mkdir -p "$RUN_ROOT" "$RESULT_ROOT"

SLICE_DIR="$RUN_ROOT/selected_slice"
mkdir -p "$SLICE_DIR"
SELECTED_MANIFEST_JSON="$SLICE_DIR/client_pixel_values_debug_share_manifest.json"
SELECTED_PUBLIC_JSON="$SLICE_DIR/client_pixel_values_debug_share_public_manifest.json"
SELECTED_PARTY_DIR="$SLICE_DIR/client_pixel_values_debug_share_party_manifests"
SELECTED_INPUT_PT="$SLICE_DIR/plaintext_same_images_pixel_values.pt"

echo "[e2e-policy-probe] probe_name=$PROBE_NAME"
echo "[e2e-policy-probe] source=$SOURCE_E2E_DIR"
echo "[e2e-policy-probe] indices=$SELECTED_INDICES"
echo "[e2e-policy-probe] variants=$VARIANT_SPECS"

"$PYTHON_BIN" tools/transshield_slice_debug_shares.py \
  --share-manifest-json "$SOURCE_SHARE_MANIFEST_JSON" \
  --indices "$SELECTED_INDICES" \
  --source-paths-file "$SOURCE_EVAL_IMAGE_LIST" \
  --output-prefix "$SLICE_DIR/client_pixel_values_debug_share" \
  --output-json "$SELECTED_MANIFEST_JSON" \
  --output-public-json "$SELECTED_PUBLIC_JSON" \
  --output-party-manifest-dir "$SELECTED_PARTY_DIR" \
  --input-pt "$SOURCE_INPUT_PT" \
  --output-input-pt "$SELECTED_INPUT_PT"

SAMPLE_COUNT="$("$PYTHON_BIN" - "$SELECTED_MANIFEST_JSON" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["sample_count"])
PY
)"

variant_report_args=()
for spec in $VARIANT_SPECS; do
  IFS=':' read -r label ln_policy attention_policy activation_override clip_value block_chunk ln_chunk <<< "$spec"
  if [[ -z "$label" || -z "$ln_policy" || -z "$attention_policy" || -z "$activation_override" || -z "$clip_value" ]]; then
    echo "[e2e-policy-probe] invalid variant spec: $spec" >&2
    exit 2
  fi
  block_chunk="${block_chunk:-0}"
  ln_chunk="${ln_chunk:-0}"
  variant_dir="$RUN_ROOT/$label"
  mkdir -p "$variant_dir"
  candidate_pt="$variant_dir/e2e_candidate_${label}.pt"
  candidate_json="$variant_dir/e2e_candidate_${label}.json"
  calib_json="$variant_dir/e2e_public_layer_norm_calibration_${label}.json"

  if [[ "$ln_policy" == "public_calibrated" && ! -f "$calib_json" ]]; then
    RUN_NAME="${PROBE_NAME}_${label}" \
    E2E_RUN_DIR="$variant_dir" \
    BUNDLE_DIR="$BUNDLE_DIR" \
    E2E_SPU_LAYER_NORM_POLICY="$ln_policy" \
    E2E_SPU_LAYER_NORM_CALIBRATION_JSON="$calib_json" \
    E2E_SPU_ATTENTION_POLICY="$attention_policy" \
    E2E_SPU_ACTIVATION_OVERRIDE="$activation_override" \
    E2E_SPU_ACTIVATION_CLIP_VALUE="$clip_value" \
    E2E_SPU_BLOCK_CHUNK_SIZE="$block_chunk" \
    E2E_SPU_LAYER_NORM_CHUNK_SIZE="$ln_chunk" \
      bash "$SCRIPT_DIR/run_e2e_secure_approx_deploy.sh" make-calib-pixels

    RUN_NAME="${PROBE_NAME}_${label}" \
    E2E_RUN_DIR="$variant_dir" \
    BUNDLE_DIR="$BUNDLE_DIR" \
    E2E_SPU_LAYER_NORM_POLICY="$ln_policy" \
    E2E_SPU_LAYER_NORM_CALIBRATION_JSON="$calib_json" \
    E2E_SPU_ATTENTION_POLICY="$attention_policy" \
    E2E_SPU_ACTIVATION_OVERRIDE="$activation_override" \
    E2E_SPU_ACTIVATION_CLIP_VALUE="$clip_value" \
    E2E_SPU_BLOCK_CHUNK_SIZE="$block_chunk" \
    E2E_SPU_LAYER_NORM_CHUNK_SIZE="$ln_chunk" \
      bash "$SCRIPT_DIR/run_e2e_secure_approx_deploy.sh" calibrate
  fi

  echo "[e2e-policy-probe] run variant=$label ln=$ln_policy attention=$attention_policy clip=$clip_value block_chunk=$block_chunk ln_chunk=$ln_chunk"
  RUN_NAME="${PROBE_NAME}_${label}" \
  E2E_RUN_DIR="$variant_dir" \
  BUNDLE_DIR="$BUNDLE_DIR" \
  E2E_INPUT_SHARE_PUBLIC_MANIFEST_JSON="$SELECTED_PUBLIC_JSON" \
  E2E_INPUT_P1_SHARE_MANIFEST_JSON="$SELECTED_PARTY_DIR/p1_share_manifest.json" \
  E2E_INPUT_P2_SHARE_MANIFEST_JSON="$SELECTED_PARTY_DIR/p2_share_manifest.json" \
  E2E_SPU_LAYER_NORM_POLICY="$ln_policy" \
  E2E_SPU_LAYER_NORM_CALIBRATION_JSON="$calib_json" \
  E2E_SPU_ATTENTION_POLICY="$attention_policy" \
  E2E_SPU_ACTIVATION_OVERRIDE="$activation_override" \
  E2E_SPU_ACTIVATION_CLIP_VALUE="$clip_value" \
  E2E_SPU_BLOCK_CHUNK_SIZE="$block_chunk" \
  E2E_SPU_LAYER_NORM_CHUNK_SIZE="$ln_chunk" \
  E2E_OUTPUT_CALIBRATION_JSON="$OUTPUT_CALIBRATION_JSON" \
  E2E_CANDIDATE_PT="$candidate_pt" \
  E2E_CANDIDATE_JSON="$candidate_json" \
  E2E_RUN_MAX_SAMPLES="$SAMPLE_COUNT" \
  E2E_SPU_BATCH_SIZE=1 \
  E2E_PARTY_LOCAL_SHARE_LOAD=1 \
  E2E_REDACT_PRIVATE_INPUT_PATHS=1 \
  SPU_RUNTIME_REUSE=0 \
    bash "$SCRIPT_DIR/run_e2e_secure_approx_deploy.sh" infer

  variant_report_args+=(--variant "$label=$candidate_pt")
done

"$PYTHON_BIN" tools/transshield_e2e_policy_probe_report.py \
  --label "$PROBE_NAME" \
  --share-manifest-json "$SELECTED_MANIFEST_JSON" \
  "${variant_report_args[@]}" \
  --output-json "$RESULT_ROOT/e2e_policy_probe_report.json" \
  --output-md "$RESULT_ROOT/e2e_policy_probe_report.md"

echo "[e2e-policy-probe] report=$RESULT_ROOT/e2e_policy_probe_report.md"
