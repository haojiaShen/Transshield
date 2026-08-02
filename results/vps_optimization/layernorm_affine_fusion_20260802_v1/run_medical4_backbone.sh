#!/usr/bin/env bash
set -euo pipefail

export REPO_ROOT=/opt/transshield-project
export PYTHON_BIN=/opt/transshield-spu/bin/python
export RUN_ROOT="$REPO_ROOT/results/vps_optimization/layernorm_affine_fusion_20260802_v1"
export CONFIG="$RUN_ROOT/configs/fxp16_default.json"
export STATE_JSON="$RUN_ROOT/spu_runtime_state_medical4_backbone.json"
export NODE_LOG_DIR="$RUN_ROOT/node_logs_medical4_backbone"
cd "$REPO_ROOT"

stop_nodes() {
  "$PYTHON_BIN" tools/transshield_spu_runtime_setup.py stop \
    --config "$CONFIG" \
    --template configs/openbumblebee/2pc.template.json \
    --state-json "$STATE_JSON" \
    >"$RUN_ROOT/setup_medical4_backbone_stop.log" 2>&1 || true
}
trap stop_nodes EXIT

"$PYTHON_BIN" tools/transshield_spu_runtime_setup.py start \
  --config "$CONFIG" \
  --template configs/openbumblebee/2pc.template.json \
  --state-json "$STATE_JSON" \
  --log-dir "$NODE_LOG_DIR" \
  --restart \
  | tee "$RUN_ROOT/setup_medical4_backbone_start.log"

cp \
  results/vps_optimization/square_alpha_fusion_20260802_v1/medical4_sample_list.txt \
  "$RUN_ROOT/medical4_sample_list.txt"

"$PYTHON_BIN" tools/report_vps_test.py network-snapshot \
  --interface lo \
  --out "$RUN_ROOT/medical4_backbone.network.before.json"

/usr/bin/time -v -o "$RUN_ROOT/medical4_backbone.time.log" \
  "$PYTHON_BIN" \
  integrations/transshield_runtime/e2e_secure_vit/transshield_e2e_secure_vit.py run \
  --runtime spu \
  --bundle-dir artifacts/frozen_bundle_medical_dynamic_mainline \
  --input-share-public-manifest-json results/vps_report_tests/report_regression_20260801_v1/medical32_public.json \
  --input-p1-share-manifest-json results/vps_report_tests/report_regression_20260801_v1/medical32_party_manifests/p1_share_manifest.json \
  --input-p2-share-manifest-json results/vps_report_tests/report_regression_20260801_v1/medical32_party_manifests/p2_share_manifest.json \
  --party-local-share-load \
  --redact-private-input-paths \
  --output-pt "$RUN_ROOT/medical4_backbone.pt" \
  --output-json "$RUN_ROOT/medical4_backbone.json" \
  --config "$CONFIG" \
  --device cpu \
  --max-samples 4 \
  --static-depth-limit 10 \
  --spu-batch-size 4 \
  --spu-params-mode secret \
  --spu-layer-norm-policy exact \
  --spu-attention-policy uniform \
  --spu-activation-override fixed_square \
  --spu-activation-clip-value 0 \
  --spu-secure-pruning-mode compact \
  --spu-secure-pruning-network unpadded_selection \
  --spu-final-block-cls-only \
  --spu-uniform-attention-value-fusion \
  --spu-compile-cache-dir "$RUN_ROOT/compile_cache_medical4_backbone" \
  --token-ratio-base-override 0.655 \
  --spu-layer-norm-affine-fusion backbone \
  2>"$RUN_ROOT/medical4_backbone.stderr.log" \
  | tee "$RUN_ROOT/medical4_backbone.stdout.log"

"$PYTHON_BIN" tools/report_vps_test.py network-snapshot \
  --interface lo \
  --out "$RUN_ROOT/medical4_backbone.network.after.json"

"$PYTHON_BIN" tools/report_vps_test.py summarize \
  --dataset-key medical_secure_deployment_batch \
  --candidate-pt "$RUN_ROOT/medical4_backbone.pt" \
  --candidate-json "$RUN_ROOT/medical4_backbone.json" \
  --reference-pt "$REPO_ROOT/results/vps_optimization/square_alpha_fusion_20260802_v1/medical4_baseline.pt" \
  --sample-list "$RUN_ROOT/medical4_sample_list.txt" \
  --threshold 0.688152923150007 \
  --network-before "$RUN_ROOT/medical4_backbone.network.before.json" \
  --network-after "$RUN_ROOT/medical4_backbone.network.after.json" \
  --out "$RUN_ROOT/medical4_backbone.summary.json" \
  | tee "$RUN_ROOT/medical4_backbone.summarize.log"
