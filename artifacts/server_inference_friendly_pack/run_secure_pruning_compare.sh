#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"

RUN_NAME="${RUN_NAME:-secure_pruning_spu_compare_20260511_1}"
LOG_FILE="${LOG_FILE:-$REPO_ROOT/logs/${RUN_NAME}_run.log}"

BUNDLE_DIR="${BUNDLE_DIR:-$REPO_ROOT/artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_aanone_20260507}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/openbumblebee/2pc.json}"
INPUT_BASE_DIR="${INPUT_BASE_DIR:-$REPO_ROOT/artifacts/server_pipeline_run/e2e_aanone_exactln_clip0_smoke16_20260507_1/e2e_secure_poc}"

E2E_INPUT_PT="${E2E_INPUT_PT:-$INPUT_BASE_DIR/client_pixel_values.pt}"
SHARE_PUBLIC_MANIFEST="${SHARE_PUBLIC_MANIFEST:-$INPUT_BASE_DIR/client_pixel_values_debug_share_public_manifest.json}"
P1_SHARE_MANIFEST="${P1_SHARE_MANIFEST:-$INPUT_BASE_DIR/client_pixel_values_debug_share_party_manifests/p1_share_manifest.json}"
P2_SHARE_MANIFEST="${P2_SHARE_MANIFEST:-$INPUT_BASE_DIR/client_pixel_values_debug_share_party_manifests/p2_share_manifest.json}"

MAX_SAMPLES="${MAX_SAMPLES:-8}"
SPU_BATCH_SIZE="${SPU_BATCH_SIZE:-8}"
E2E_STATIC_DEPTH_LIMIT="${E2E_STATIC_DEPTH_LIMIT:--1}"
E2E_SPU_TOKEN_RECYCLE_SCALE="${E2E_SPU_TOKEN_RECYCLE_SCALE:-0}"

RUN_DIR="$REPO_ROOT/artifacts/server_pipeline_run/${RUN_NAME}/e2e_secure_poc"
REFERENCE_PT="$RUN_DIR/runtime_pruning_reference.pt"
REFERENCE_JSON="$RUN_DIR/runtime_pruning_reference.json"
CANDIDATE_PT="$RUN_DIR/e2e_static_whole_forward_candidate_from_server.pt"
CANDIDATE_JSON="$RUN_DIR/e2e_static_whole_forward_candidate_from_server.json"
COMPARE_JSON="$RUN_DIR/secure_pruning_compare_vs_runtime_pruning_reference.json"

mkdir -p "$(dirname "$LOG_FILE")" "$RUN_DIR"

echo "=== Secure Pruning compare wrapper ===" | tee "$LOG_FILE"
echo "Run name: $RUN_NAME" | tee -a "$LOG_FILE"
echo "Bundle: $BUNDLE_DIR" | tee -a "$LOG_FILE"
echo "Input pt: $E2E_INPUT_PT" | tee -a "$LOG_FILE"
echo "Max samples: $MAX_SAMPLES" | tee -a "$LOG_FILE"
echo "SPU batch size: $SPU_BATCH_SIZE" | tee -a "$LOG_FILE"
echo "Static depth limit: $E2E_STATIC_DEPTH_LIMIT" | tee -a "$LOG_FILE"
echo "Token recycle scale: $E2E_SPU_TOKEN_RECYCLE_SCALE" | tee -a "$LOG_FILE"
echo "Start time: $(date)" | tee -a "$LOG_FILE"

if [[ ! -f "$E2E_INPUT_PT" ]]; then
  echo "missing E2E_INPUT_PT: $E2E_INPUT_PT" >&2
  exit 1
fi

CPU_ARGS=(
  integrations/openbumblebee/e2e_secure_vit/transshield_e2e_secure_vit.py
  run
  --runtime cpu
  --bundle-dir "$BUNDLE_DIR"
  --input-pt "$E2E_INPUT_PT"
  --output-pt "$REFERENCE_PT"
  --output-json "$REFERENCE_JSON"
  --device cpu
  --max-samples "$MAX_SAMPLES"
  --cpu-forward-mode runtime_pruning_reference
)
if [[ "$E2E_STATIC_DEPTH_LIMIT" != "-1" ]]; then
  CPU_ARGS+=(--static-depth-limit "$E2E_STATIC_DEPTH_LIMIT")
fi

"$PYTHON_BIN" "${CPU_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"

export RUN_NAME BUNDLE_DIR CONFIG_PATH
export E2E_RUN_DIR="$RUN_DIR"
export E2E_RUN_MAX_SAMPLES="$MAX_SAMPLES"
export E2E_SPU_BATCH_SIZE="$SPU_BATCH_SIZE"
export E2E_STATIC_DEPTH_LIMIT
export E2E_INPUT_PT
export E2E_CANDIDATE_PT="$CANDIDATE_PT"
export E2E_CANDIDATE_JSON="$CANDIDATE_JSON"
export E2E_SPU_LAYER_NORM_POLICY=exact
export E2E_SPU_ACTIVATION_CLIP_VALUE=0
export E2E_SPU_ATTENTION_POLICY=uniform
export E2E_SPU_ACTIVATION_OVERRIDE=fixed_square
export E2E_SPU_PARAMS_MODE=secret
export E2E_SPU_TOKEN_RECYCLE_SCALE
export E2E_PARTY_LOCAL_SHARE_LOAD=1
export E2E_INPUT_SHARE_PUBLIC_MANIFEST_JSON="$SHARE_PUBLIC_MANIFEST"
export E2E_INPUT_P1_SHARE_MANIFEST_JSON="$P1_SHARE_MANIFEST"
export E2E_INPUT_P2_SHARE_MANIFEST_JSON="$P2_SHARE_MANIFEST"
export E2E_REDACT_PRIVATE_INPUT_PATHS=1

bash "$SCRIPT_DIR/run_e2e_secure_whole_forward.sh" spu 2>&1 | tee -a "$LOG_FILE"

"$PYTHON_BIN" integrations/openbumblebee/e2e_secure_vit/transshield_e2e_secure_vit.py verify \
  --reference-pt "$REFERENCE_PT" \
  --candidate-pt "$CANDIDATE_PT" \
  --output-json "$COMPARE_JSON" 2>&1 | tee -a "$LOG_FILE"

echo "=== Compare Summary ===" | tee -a "$LOG_FILE"
"$PYTHON_BIN" - "$CANDIDATE_JSON" "$COMPARE_JSON" <<'PY' 2>&1 | tee -a "$LOG_FILE"
import json
import sys

candidate_json = sys.argv[1]
compare_json = sys.argv[2]

with open(candidate_json, "r", encoding="utf-8") as f:
    candidate = json.load(f)
with open(compare_json, "r", encoding="utf-8") as f:
    compare = json.load(f)

for key in ["sample_count", "finite_logits", "elapsed_sec", "spu_token_recycle_scale"]:
    if key in candidate:
        print(f"candidate_{key}: {candidate[key]}")

prediction_match = compare.get("prediction_match", {})
numeric_match = compare.get("numeric_match", {})
for key in ["argmax_match_ratio", "threshold_match_ratio"]:
    if key in prediction_match:
        print(f"compare_{key}: {prediction_match[key]}")
for key in ["logits_max_abs_error", "probabilities_max_abs_error"]:
    if key in numeric_match:
        print(f"compare_{key}: {numeric_match[key]}")
PY

echo "=== Done ===" | tee -a "$LOG_FILE"
echo "End time: $(date)" | tee -a "$LOG_FILE"
