#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"

# 显式设置，不依赖 load_local_env 的默认值
RUN_NAME="${RUN_NAME:-secure_pruning_spu_smoke8_batch4_partylocal_secret_20260510_1}"
BUNDLE_DIR="${BUNDLE_DIR:-$REPO_ROOT/artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_aanone_20260507}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/openbumblebee/2pc.json}"
LOG_FILE="${LOG_FILE:-$REPO_ROOT/logs/${RUN_NAME}_run.log}"
E2E_SPU_TOKEN_RECYCLE_SCALE="${E2E_SPU_TOKEN_RECYCLE_SCALE:-0}"

SMOKE16_DIR="$REPO_ROOT/artifacts/server_pipeline_run/e2e_aanone_exactln_clip0_smoke16_20260507_1/e2e_secure_poc"
SHARE_PUBLIC_MANIFEST="$SMOKE16_DIR/client_pixel_values_debug_share_public_manifest.json"
P1_SHARE_MANIFEST="$SMOKE16_DIR/client_pixel_values_debug_share_party_manifests/p1_share_manifest.json"
P2_SHARE_MANIFEST="$SMOKE16_DIR/client_pixel_values_debug_share_party_manifests/p2_share_manifest.json"

MAX_SAMPLES="${MAX_SAMPLES:-8}"
SPU_BATCH_SIZE="${SPU_BATCH_SIZE:-4}"

mkdir -p "$(dirname "$LOG_FILE")"

echo "=== Secure Pruning smoke8 batch4 (PredictorLG in-SPU, batch_size=4) ===" | tee "$LOG_FILE"
echo "Run name: $RUN_NAME" | tee -a "$LOG_FILE"
echo "Bundle: $BUNDLE_DIR" | tee -a "$LOG_FILE"
echo "Max samples: $MAX_SAMPLES" | tee -a "$LOG_FILE"
echo "SPU batch size: $SPU_BATCH_SIZE" | tee -a "$LOG_FILE"
echo "Token recycle scale: $E2E_SPU_TOKEN_RECYCLE_SCALE" | tee -a "$LOG_FILE"
echo "Start time: $(date)" | tee -a "$LOG_FILE"

export RUN_NAME BUNDLE_DIR CONFIG_PATH
export E2E_RUN_MAX_SAMPLES="$MAX_SAMPLES"
export E2E_SPU_BATCH_SIZE="$SPU_BATCH_SIZE"
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

echo "=== Done ===" | tee -a "$LOG_FILE"
echo "End time: $(date)" | tee -a "$LOG_FILE"

CANDIDATE_JSON="$REPO_ROOT/artifacts/server_pipeline_run/${RUN_NAME}/e2e_secure_poc/e2e_static_whole_forward_candidate_from_server.json"
if [ -f "$CANDIDATE_JSON" ]; then
    echo "=== Key Metrics ===" | tee -a "$LOG_FILE"
    python3 -c "
import json
with open('$CANDIDATE_JSON') as f:
    d = json.load(f)
for k in ['sample_count','finite_logits','elapsed_sec','backend','forward_scope',
          'host_plaintext_pixel_values_materialized','reveal_policy']:
    if k in d:
        print(f'  {k}: {d[k]}')
if 'sample_count' in d and 'elapsed_sec' in d and d['sample_count']:
    print(f'  sec_per_sample: {d[\"elapsed_sec\"]/d[\"sample_count\"]:.2f}')
" 2>&1 | tee -a "$LOG_FILE"
fi
