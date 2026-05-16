#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"

MODE="${1:-all}"
case "$MODE" in
  make-calib-pixels|calibrate|infer|all)
    shift || true
    ;;
  *)
    echo "Usage: $0 [make-calib-pixels|calibrate|infer|all]" >&2
    exit 1
    ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python}"
BUNDLE_DIR="${BUNDLE_DIR:-$REPO_ROOT/artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_20260430}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/openbumblebee/2pc.json}"

RUN_NAME="${RUN_NAME:-transshield_e2e_approx_deploy}"
E2E_RUN_DIR="${E2E_RUN_DIR:-$REPO_ROOT/artifacts/server_pipeline_run/${RUN_NAME}/e2e_secure_poc}"
mkdir -p "$E2E_RUN_DIR"

PUBLIC_CALIB_DATASET_DIR="${PUBLIC_CALIB_DATASET_DIR:-/data/wyb/pneumoniamnist_imagefolder_subset}"
PUBLIC_CALIB_IMAGE_LIST="${PUBLIC_CALIB_IMAGE_LIST:-$E2E_RUN_DIR/public_calib_images.txt}"
PUBLIC_CALIB_MAX_SAMPLES="${PUBLIC_CALIB_MAX_SAMPLES:-32}"
PUBLIC_CALIB_PT="${PUBLIC_CALIB_PT:-$E2E_RUN_DIR/public_calibration_pixel_values.pt}"
PUBLIC_CALIB_JSON="${PUBLIC_CALIB_JSON:-$E2E_RUN_DIR/public_calibration_pixel_values.json}"

SOURCE_E2E_DIR="${SOURCE_E2E_DIR:-}"
E2E_INPUT_SHARE_PUBLIC_MANIFEST_JSON="${E2E_INPUT_SHARE_PUBLIC_MANIFEST_JSON:-${SOURCE_E2E_DIR:+$SOURCE_E2E_DIR/client_pixel_values_debug_share_public_manifest.json}}"
E2E_INPUT_P1_SHARE_MANIFEST_JSON="${E2E_INPUT_P1_SHARE_MANIFEST_JSON:-${SOURCE_E2E_DIR:+$SOURCE_E2E_DIR/client_pixel_values_debug_share_party_manifests/p1_share_manifest.json}}"
E2E_INPUT_P2_SHARE_MANIFEST_JSON="${E2E_INPUT_P2_SHARE_MANIFEST_JSON:-${SOURCE_E2E_DIR:+$SOURCE_E2E_DIR/client_pixel_values_debug_share_party_manifests/p2_share_manifest.json}}"

E2E_STATIC_DEPTH_LIMIT="${E2E_STATIC_DEPTH_LIMIT:-12}"
E2E_RUN_MAX_SAMPLES="${E2E_RUN_MAX_SAMPLES:-1}"
E2E_SPU_BATCH_SIZE="${E2E_SPU_BATCH_SIZE:-1}"
E2E_SPU_PARAMS_MODE="${E2E_SPU_PARAMS_MODE:-public}"
E2E_PARTY_LOCAL_SHARE_LOAD="${E2E_PARTY_LOCAL_SHARE_LOAD:-1}"
E2E_REDACT_PRIVATE_INPUT_PATHS="${E2E_REDACT_PRIVATE_INPUT_PATHS:-1}"
E2E_SPU_LAYER_NORM_POLICY="${E2E_SPU_LAYER_NORM_POLICY:-public_calibrated}"
E2E_SPU_ATTENTION_POLICY="${E2E_SPU_ATTENTION_POLICY:-uniform}"
E2E_SPU_ACTIVATION_OVERRIDE="${E2E_SPU_ACTIVATION_OVERRIDE:-fixed_square}"
E2E_SPU_ACTIVATION_CLIP_VALUE="${E2E_SPU_ACTIVATION_CLIP_VALUE:-3.0}"
E2E_SPU_ACTIVATION_CLIP_TAG="clip${E2E_SPU_ACTIVATION_CLIP_VALUE//./p}"
E2E_SPU_LAYER_NORM_CALIBRATION_JSON="${E2E_SPU_LAYER_NORM_CALIBRATION_JSON:-$E2E_RUN_DIR/e2e_public_layer_norm_calibration_depth${E2E_STATIC_DEPTH_LIMIT}_${E2E_SPU_ATTENTION_POLICY}_${E2E_SPU_ACTIVATION_OVERRIDE}_${E2E_SPU_ACTIVATION_CLIP_TAG}.json}"
E2E_OUTPUT_CALIBRATION_JSON="${E2E_OUTPUT_CALIBRATION_JSON:-}"
E2E_SPU_BLOCK_CHUNK_SIZE="${E2E_SPU_BLOCK_CHUNK_SIZE:-0}"
E2E_SPU_LAYER_NORM_CHUNK_SIZE="${E2E_SPU_LAYER_NORM_CHUNK_SIZE:-0}"

E2E_CANDIDATE_PT="${E2E_CANDIDATE_PT:-$E2E_RUN_DIR/e2e_static_whole_forward_candidate_spu_depth${E2E_STATIC_DEPTH_LIMIT}_partylocal_publiccalibln_${E2E_SPU_ATTENTION_POLICY}_${E2E_SPU_ACTIVATION_OVERRIDE}_${E2E_SPU_ACTIVATION_CLIP_TAG}.pt}"
E2E_CANDIDATE_JSON="${E2E_CANDIDATE_JSON:-$E2E_RUN_DIR/e2e_static_whole_forward_candidate_spu_depth${E2E_STATIC_DEPTH_LIMIT}_partylocal_publiccalibln_${E2E_SPU_ATTENTION_POLICY}_${E2E_SPU_ACTIVATION_OVERRIDE}_${E2E_SPU_ACTIVATION_CLIP_TAG}.json}"

SPU_RUNTIME_REUSE="${SPU_RUNTIME_REUSE:-0}"
SPU_DISABLE_COLOCATED_OPTIMIZATION="${SPU_DISABLE_COLOCATED_OPTIMIZATION:-1}"
SPU_RUNTIME_STARTUP_TIMEOUT_SEC="${SPU_RUNTIME_STARTUP_TIMEOUT_SEC:-60}"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "[e2e-approx-deploy] missing $label: $path" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "$path" ]]; then
    echo "[e2e-approx-deploy] missing $label: $path" >&2
    exit 1
  fi
}

require_safe_deploy_config() {
  if [[ "$E2E_STATIC_DEPTH_LIMIT" != "12" ]]; then
    echo "[e2e-approx-deploy] refusing non-full-depth deploy config: E2E_STATIC_DEPTH_LIMIT=$E2E_STATIC_DEPTH_LIMIT" >&2
    exit 1
  fi
  if [[ "$E2E_SPU_BATCH_SIZE" != "1" ]]; then
    echo "[e2e-approx-deploy] refusing E2E_SPU_BATCH_SIZE=$E2E_SPU_BATCH_SIZE; validated deploy baseline requires bsz=1." >&2
    exit 1
  fi
  if [[ "$E2E_SPU_LAYER_NORM_POLICY" != "public_calibrated" && "$E2E_SPU_LAYER_NORM_POLICY" != "exact" ]]; then
    echo "[e2e-approx-deploy] refusing layer norm policy $E2E_SPU_LAYER_NORM_POLICY; expected public_calibrated or exact." >&2
    exit 1
  fi
  if [[ "$E2E_SPU_ATTENTION_POLICY" != "uniform" && "$E2E_SPU_ATTENTION_POLICY" != "identity" ]]; then
    echo "[e2e-approx-deploy] refusing attention policy $E2E_SPU_ATTENTION_POLICY; expected uniform or identity." >&2
    exit 1
  fi
  if [[ "$E2E_SPU_ACTIVATION_OVERRIDE" != "fixed_square" && "$E2E_SPU_ACTIVATION_OVERRIDE" != "pade_gelu" && "$E2E_SPU_ACTIVATION_OVERRIDE" != "lut_gelu_4" && "$E2E_SPU_ACTIVATION_OVERRIDE" != "lut_gelu_8" && "$E2E_SPU_ACTIVATION_OVERRIDE" != "lut_gelu_16" && "$E2E_SPU_ACTIVATION_OVERRIDE" != "lut_gelu_32" ]]; then
    echo "[e2e-approx-deploy] refusing activation $E2E_SPU_ACTIVATION_OVERRIDE; expected fixed_square." >&2
    exit 1
  fi
  if [[ "$E2E_PARTY_LOCAL_SHARE_LOAD" != "1" ]]; then
    echo "[e2e-approx-deploy] refusing non-party-local input; set E2E_PARTY_LOCAL_SHARE_LOAD=1." >&2
    exit 1
  fi
}

make_calib_pixels() {
  if [[ ! -f "$PUBLIC_CALIB_IMAGE_LIST" || "${PUBLIC_CALIB_REGENERATE_LIST:-0}" == "1" ]]; then
    require_dir "$PUBLIC_CALIB_DATASET_DIR" "public calibration dataset directory"
    class0_list="${PUBLIC_CALIB_IMAGE_LIST}.class0.tmp"
    class1_list="${PUBLIC_CALIB_IMAGE_LIST}.class1.tmp"
    all_list="${PUBLIC_CALIB_IMAGE_LIST}.all.tmp"
    find "$PUBLIC_CALIB_DATASET_DIR" -type f \( -path "*/0/*" -o -path "*/class_0/*" \) \
      \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) | sort > "$class0_list"
    find "$PUBLIC_CALIB_DATASET_DIR" -type f \( -path "*/1/*" -o -path "*/class_1/*" \) \
      \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) | sort > "$class1_list"
    if [[ -s "$class0_list" && -s "$class1_list" && "$PUBLIC_CALIB_MAX_SAMPLES" -ge 2 ]]; then
      half=$((PUBLIC_CALIB_MAX_SAMPLES / 2))
      remainder=$((PUBLIC_CALIB_MAX_SAMPLES - half))
      head -n "$half" "$class0_list" > "$PUBLIC_CALIB_IMAGE_LIST"
      head -n "$remainder" "$class1_list" >> "$PUBLIC_CALIB_IMAGE_LIST"
      echo "[e2e-approx-deploy] balanced public calibration list: class0=$half class1=$remainder"
    else
      find "$PUBLIC_CALIB_DATASET_DIR" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) | sort > "$all_list"
      head -n "$PUBLIC_CALIB_MAX_SAMPLES" "$all_list" > "$PUBLIC_CALIB_IMAGE_LIST"
      echo "[e2e-approx-deploy] fallback public calibration list from all images"
    fi
    rm -f "$class0_list" "$class1_list" "$all_list"
  fi
  local count
  count="$(wc -l < "$PUBLIC_CALIB_IMAGE_LIST" | tr -d ' ')"
  if [[ "$count" -le 0 ]]; then
    echo "[e2e-approx-deploy] no calibration images found under $PUBLIC_CALIB_DATASET_DIR" >&2
    exit 1
  fi
  echo "[e2e-approx-deploy] public calibration images: $count"
  "$PYTHON_BIN" tools/transshield_e2e_secure_infer.py client-preprocess \
    --bundle-dir "$BUNDLE_DIR" \
    --image-list "$PUBLIC_CALIB_IMAGE_LIST" \
    --max-samples "$PUBLIC_CALIB_MAX_SAMPLES" \
    --output-pt "$PUBLIC_CALIB_PT" \
    --output-json "$PUBLIC_CALIB_JSON"
}

calibrate_ln() {
  require_file "$PUBLIC_CALIB_PT" "public calibration pixel package"
  E2E_INPUT_PT="$PUBLIC_CALIB_PT" \
  E2E_STATIC_DEPTH_LIMIT="$E2E_STATIC_DEPTH_LIMIT" \
  E2E_RUN_MAX_SAMPLES="$PUBLIC_CALIB_MAX_SAMPLES" \
  E2E_SPU_ATTENTION_POLICY="$E2E_SPU_ATTENTION_POLICY" \
  E2E_SPU_ACTIVATION_OVERRIDE="$E2E_SPU_ACTIVATION_OVERRIDE" \
  E2E_SPU_ACTIVATION_CLIP_VALUE="$E2E_SPU_ACTIVATION_CLIP_VALUE" \
  E2E_SPU_LAYER_NORM_CALIBRATION_JSON="$E2E_SPU_LAYER_NORM_CALIBRATION_JSON" \
  PYTHON_BIN="$PYTHON_BIN" \
  REPO_ROOT="$REPO_ROOT" \
  BUNDLE_DIR="$BUNDLE_DIR" \
  RUN_NAME="$RUN_NAME" \
  E2E_RUN_DIR="$E2E_RUN_DIR" \
    bash "$SCRIPT_DIR/run_e2e_secure_whole_forward.sh" calibrate-ln
}

infer_private_shares() {
  require_safe_deploy_config
  if [[ "$E2E_SPU_LAYER_NORM_POLICY" == "public_calibrated" ]]; then
    require_file "$E2E_SPU_LAYER_NORM_CALIBRATION_JSON" "public layer-norm calibration JSON"
  fi
  require_file "$E2E_INPUT_SHARE_PUBLIC_MANIFEST_JSON" "public share manifest"
  require_file "$E2E_INPUT_P1_SHARE_MANIFEST_JSON" "P1 share manifest"
  require_file "$E2E_INPUT_P2_SHARE_MANIFEST_JSON" "P2 share manifest"

  E2E_INPUT_PT="" \
  E2E_INPUT_SHARE_PUBLIC_MANIFEST_JSON="$E2E_INPUT_SHARE_PUBLIC_MANIFEST_JSON" \
  E2E_INPUT_P1_SHARE_MANIFEST_JSON="$E2E_INPUT_P1_SHARE_MANIFEST_JSON" \
  E2E_INPUT_P2_SHARE_MANIFEST_JSON="$E2E_INPUT_P2_SHARE_MANIFEST_JSON" \
  E2E_CANDIDATE_PT="$E2E_CANDIDATE_PT" \
  E2E_CANDIDATE_JSON="$E2E_CANDIDATE_JSON" \
  E2E_RUN_MAX_SAMPLES="$E2E_RUN_MAX_SAMPLES" \
  E2E_STATIC_DEPTH_LIMIT="$E2E_STATIC_DEPTH_LIMIT" \
  E2E_SPU_BATCH_SIZE="$E2E_SPU_BATCH_SIZE" \
  E2E_SPU_PARAMS_MODE="$E2E_SPU_PARAMS_MODE" \
  E2E_PARTY_LOCAL_SHARE_LOAD="$E2E_PARTY_LOCAL_SHARE_LOAD" \
  E2E_REDACT_PRIVATE_INPUT_PATHS="$E2E_REDACT_PRIVATE_INPUT_PATHS" \
  E2E_SPU_LAYER_NORM_POLICY="$E2E_SPU_LAYER_NORM_POLICY" \
  E2E_SPU_LAYER_NORM_CALIBRATION_JSON="$E2E_SPU_LAYER_NORM_CALIBRATION_JSON" \
  E2E_SPU_ATTENTION_POLICY="$E2E_SPU_ATTENTION_POLICY" \
  E2E_SPU_ACTIVATION_OVERRIDE="$E2E_SPU_ACTIVATION_OVERRIDE" \
  E2E_SPU_ACTIVATION_CLIP_VALUE="$E2E_SPU_ACTIVATION_CLIP_VALUE" \
  E2E_SPU_BLOCK_CHUNK_SIZE="$E2E_SPU_BLOCK_CHUNK_SIZE" \
  E2E_SPU_LAYER_NORM_CHUNK_SIZE="$E2E_SPU_LAYER_NORM_CHUNK_SIZE" \
  E2E_OUTPUT_CALIBRATION_JSON="$E2E_OUTPUT_CALIBRATION_JSON" \
  SPU_RUNTIME_REUSE="$SPU_RUNTIME_REUSE" \
  SPU_DISABLE_COLOCATED_OPTIMIZATION="$SPU_DISABLE_COLOCATED_OPTIMIZATION" \
  SPU_RUNTIME_STARTUP_TIMEOUT_SEC="$SPU_RUNTIME_STARTUP_TIMEOUT_SEC" \
  PYTHON_BIN="$PYTHON_BIN" \
  REPO_ROOT="$REPO_ROOT" \
  BUNDLE_DIR="$BUNDLE_DIR" \
  RUN_NAME="$RUN_NAME" \
  E2E_RUN_DIR="$E2E_RUN_DIR" \
  CONFIG_PATH="$CONFIG_PATH" \
    bash "$SCRIPT_DIR/run_e2e_secure_whole_forward.sh" spu "$@"
}

case "$MODE" in
  make-calib-pixels)
    make_calib_pixels
    ;;
  calibrate)
    calibrate_ln
    ;;
  infer)
    infer_private_shares "$@"
    ;;
  all)
    make_calib_pixels
    calibrate_ln
    infer_private_shares "$@"
    ;;
esac

echo "[e2e-approx-deploy] run_dir=$E2E_RUN_DIR"
echo "[e2e-approx-deploy] candidate_json=$E2E_CANDIDATE_JSON"
