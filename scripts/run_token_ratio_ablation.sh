#!/usr/bin/env bash
set -euo pipefail

cd /data/wyb/Transshield_final

BUNDLE_DIR="/data/wyb/Transshield_final/artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_aanone_20260507"
CONFIG_PATH="/data/wyb/Transshield_final/configs/openbumblebee/2pc.json"

SMOKE16_DIR="/data/wyb/Transshield_final/artifacts/server_pipeline_run/e2e_aanone_exactln_clip0_smoke16_20260507_1/e2e_secure_poc"
SHARE_PUBLIC_MANIFEST="$SMOKE16_DIR/client_pixel_values_debug_share_public_manifest.json"
P1_SHARE_MANIFEST="$SMOKE16_DIR/client_pixel_values_debug_share_party_manifests/p1_share_manifest.json"
P2_SHARE_MANIFEST="$SMOKE16_DIR/client_pixel_values_debug_share_party_manifests/p2_share_manifest.json"

TIMESTAMP=$(date +%Y%m%d_%H%M)
MASTER_LOG="/data/wyb/Transshield_final/logs/token_ratio_ablation_${TIMESTAMP}.log"

MAX_SAMPLES=8
SPU_BATCH_SIZE=8

echo "=== Token Ratio Base Rate Ablation ===" | tee "$MASTER_LOG"
echo "Start time: $(date)" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

for BASE_RATE in 0.7 0.6 0.55 0.5; do
  BR_TAG=$(echo "$BASE_RATE" | sed 's/\.//')
  RUN_NAME="token_ratio_ablation_br${BR_TAG}_${TIMESTAMP}"
  LOG_FILE="/data/wyb/Transshield_final/logs/${RUN_NAME}_run.log"

  echo "=== [base_rate=$BASE_RATE] Run: $RUN_NAME ===" | tee -a "$MASTER_LOG"
  echo "  Start: $(date)" | tee -a "$MASTER_LOG"

  export RUN_NAME BUNDLE_DIR CONFIG_PATH
  export E2E_RUN_MAX_SAMPLES="$MAX_SAMPLES"
  export E2E_SPU_BATCH_SIZE="$SPU_BATCH_SIZE"
  export E2E_SPU_LAYER_NORM_POLICY=exact
  export E2E_SPU_ACTIVATION_CLIP_VALUE=0
  export E2E_SPU_ATTENTION_POLICY=uniform
  export E2E_SPU_ACTIVATION_OVERRIDE=fixed_square
  export E2E_SPU_PARAMS_MODE=secret
  export E2E_PARTY_LOCAL_SHARE_LOAD=1
  export E2E_INPUT_SHARE_PUBLIC_MANIFEST_JSON="$SHARE_PUBLIC_MANIFEST"
  export E2E_INPUT_P1_SHARE_MANIFEST_JSON="$P1_SHARE_MANIFEST"
  export E2E_INPUT_P2_SHARE_MANIFEST_JSON="$P2_SHARE_MANIFEST"
  export E2E_REDACT_PRIVATE_INPUT_PATHS=1
  export E2E_SPU_TOKEN_RATIO_BASE_OVERRIDE="$BASE_RATE"

  bash artifacts/server_inference_friendly_pack/run_e2e_secure_whole_forward.sh spu 2>&1 | tee "$LOG_FILE"

  CANDIDATE_JSON="/data/wyb/Transshield_final/artifacts/server_pipeline_run/${RUN_NAME}/e2e_secure_poc/e2e_static_whole_forward_candidate_from_server.json"
  if [ -f "$CANDIDATE_JSON" ]; then
    python3 -c "
import json
with open('$CANDIDATE_JSON') as f:
    d = json.load(f)
meta = d.get('static_forward_metadata', {})
print(f'  sample_count: {d.get(\"sample_count\",\"?\")}')
print(f'  elapsed_sec: {d.get(\"elapsed_sec\",\"?\")}')
print(f'  finite_logits: {d.get(\"finite_logits\",\"?\")}')
print(f'  host_model_params_materialized: {d.get(\"host_model_params_materialized\",\"?\")}')
print(f'  base_rate: {meta.get(\"base_rate\",\"?\")}')
print(f'  token_keep_counts: {meta.get(\"token_keep_counts\",\"?\")}')
print(f'  token_ratio: {meta.get(\"token_ratio\",\"?\")}')
print(f'  argmax_match_ratio: {d.get(\"argmax_match_ratio\",\"?\")}')
print(f'  threshold_match_ratio: {d.get(\"threshold_match_ratio\",\"?\")}')
" 2>&1 | tee -a "$MASTER_LOG"
  else
    echo "  [WARN] candidate JSON not found" | tee -a "$MASTER_LOG"
  fi
  echo "  End: $(date)" | tee -a "$MASTER_LOG"
  echo "" | tee -a "$MASTER_LOG"
done

echo "=== All done: $(date) ===" | tee -a "$MASTER_LOG"
