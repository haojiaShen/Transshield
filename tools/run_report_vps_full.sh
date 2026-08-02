#!/usr/bin/env bash
set -Eeuo pipefail

# Run the report-scope regression only on an explicitly acknowledged VPS.
# The script writes candidate evidence below results/vps_report_tests and never
# mutates the four frozen formal-result directories.
if [[ "${TRANSSHIELD_VPS_EXECUTION_ACK:-}" != "1" ]]; then
  echo "Set TRANSSHIELD_VPS_EXECUTION_ACK=1 on the VPS before running this script." >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-/opt/transshield-project}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT must name a new results/vps_report_tests run directory}"
BASELINE_PY="${BASELINE_PY:-/opt/transshield-spu-mlp-pack-hybrid/bin/python}"
CANDIDATE_PY="${CANDIDATE_PY:-/opt/transshield-spu-pack-sumdiff/bin/python}"
RUNTIME_CONFIG_TEMPLATE="${RUNTIME_CONFIG_TEMPLATE:-/opt/transshield-smoke/configs/transshield_runtime/2pc.runtime.json}"
NETWORK_INTERFACE="${NETWORK_INTERFACE:-lo}"
MEDICAL_THRESHOLD="${MEDICAL_THRESHOLD:-0.6619606018066406}"
SPU_BATCH_SIZE="${SPU_BATCH_SIZE:-16}"

MEDICAL_BUNDLE="$REPO_ROOT/artifacts/frozen_bundle_medical_dynamic_mainline"
FINANCE_BUNDLE="$REPO_ROOT/artifacts/frozen_bundle_finance_boundary_stress"
RUNNER="$REPO_ROOT/integrations/transshield_runtime/e2e_secure_vit/transshield_e2e_secure_vit.py"
SETUP_TOOL="$REPO_ROOT/tools/transshield_spu_runtime_setup.py"

for path in "$BASELINE_PY" "$CANDIDATE_PY" "$RUNTIME_CONFIG_TEMPLATE" "$RUNNER" "$SETUP_TOOL"; do
  if [[ ! -f "$path" ]]; then
    echo "Required file missing: $path" >&2
    exit 2
  fi
done

mkdir -p "$RUN_ROOT" "$RUN_ROOT/lists" "$RUN_ROOT/configs" "$RUN_ROOT/test_logs"
if [[ -e "$RUN_ROOT/report_regression_aggregate.json" ]]; then
  echo "Refusing to overwrite completed aggregate: $RUN_ROOT/report_regression_aggregate.json" >&2
  exit 2
fi

cd "$REPO_ROOT"
exec > >(tee -a "$RUN_ROOT/pipeline.stdout.log") \
  2> >(tee -a "$RUN_ROOT/pipeline.stderr.log" >&2)

ACTIVE_STATE=""
ACTIVE_PY=""
SERVER_PID=""

mark_phase() {
  printf '%s\n' "$1" | tee "$RUN_ROOT/current_phase.txt"
  printf '[%s] %s\n' "$(date -Is)" "$1"
}

stop_spu() {
  if [[ -n "$ACTIVE_STATE" && -n "$ACTIVE_PY" ]]; then
    "$ACTIVE_PY" "$SETUP_TOOL" stop \
      --config "$RUN_ROOT/configs/active.runtime.json" \
      --state-json "$ACTIVE_STATE" || true
    ACTIVE_STATE=""
    ACTIVE_PY=""
  fi
}

stop_server() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
}

cleanup() {
  stop_server
  stop_spu
}
trap cleanup EXIT

start_spu() {
  local python_bin="$1"
  local label="$2"
  local config="$RUN_ROOT/configs/${label}.runtime.json"
  local state="$RUN_ROOT/${label}.runtime_state.json"
  cp "$RUNTIME_CONFIG_TEMPLATE" "$config"
  cp "$config" "$RUN_ROOT/configs/active.runtime.json"
  ACTIVE_STATE="$state"
  ACTIVE_PY="$python_bin"
  "$python_bin" "$SETUP_TOOL" start \
    --config "$RUN_ROOT/configs/active.runtime.json" \
    --state-json "$state" \
    --log-dir "$RUN_ROOT/${label}.node_logs" \
    --restart | tee "$RUN_ROOT/${label}.setup_start.log"
}

snapshot() {
  local python_bin="$1"
  local out="$2"
  "$python_bin" tools/report_vps_test.py network-snapshot \
    --interface "$NETWORK_INTERFACE" --out "$out"
}

run_medical_spu() {
  local python_bin="$1"
  local label="$2"
  local output_stem="$3"

  start_spu "$python_bin" "$label"
  snapshot "$python_bin" "$RUN_ROOT/${output_stem}.network.before.json"
  set +e
  /usr/bin/time -v -o "$RUN_ROOT/${output_stem}.time.log" \
    "$python_bin" "$RUNNER" run \
      --runtime spu \
      --bundle-dir "$MEDICAL_BUNDLE" \
      --input-share-public-manifest-json "$RUN_ROOT/medical32_public.json" \
      --input-p1-share-manifest-json "$RUN_ROOT/medical32_party_manifests/p1_share_manifest.json" \
      --input-p2-share-manifest-json "$RUN_ROOT/medical32_party_manifests/p2_share_manifest.json" \
      --party-local-share-load \
      --redact-private-input-paths \
      --output-pt "$RUN_ROOT/${output_stem}.pt" \
      --output-json "$RUN_ROOT/${output_stem}.json" \
      --config "$RUN_ROOT/configs/active.runtime.json" \
      --device cpu \
      --max-samples 32 \
      --static-depth-limit 10 \
      --spu-batch-size "$SPU_BATCH_SIZE" \
      --spu-params-mode secret \
      --spu-layer-norm-policy exact \
      --spu-attention-policy uniform \
      --spu-activation-override fixed_square \
      --spu-activation-clip-value 0 \
      --spu-secure-pruning-mode compact \
      --spu-secure-pruning-network unpadded_selection \
      --spu-final-block-cls-only \
      --spu-uniform-attention-value-fusion \
      --spu-public-fixed-square-scale \
      --spu-compile-cache-dir "$RUN_ROOT/${output_stem}.compile_cache" \
      >"$RUN_ROOT/${output_stem}.stdout.log" \
      2>"$RUN_ROOT/${output_stem}.stderr.log"
  local rc=$?
  set -e
  printf '%s\n' "$rc" >"$RUN_ROOT/${output_stem}.exit_code"
  snapshot "$python_bin" "$RUN_ROOT/${output_stem}.network.after.json"
  stop_spu
  if [[ "$rc" -ne 0 ]]; then
    echo "$output_stem failed with exit $rc" >&2
    return "$rc"
  fi
}

run_finance_spu() {
  local python_bin="$1"
  local label="finance_candidate"
  local output_stem="finance8_spu_latest"

  start_spu "$python_bin" "$label"
  snapshot "$python_bin" "$RUN_ROOT/${output_stem}.network.before.json"
  set +e
  /usr/bin/time -v -o "$RUN_ROOT/${output_stem}.time.log" \
    "$python_bin" "$RUNNER" run \
      --runtime spu \
      --bundle-dir "$FINANCE_BUNDLE" \
      --input-share-public-manifest-json "$RUN_ROOT/finance8_public.json" \
      --input-p1-share-manifest-json "$RUN_ROOT/finance8_party_manifests/p1_share_manifest.json" \
      --input-p2-share-manifest-json "$RUN_ROOT/finance8_party_manifests/p2_share_manifest.json" \
      --party-local-share-load \
      --redact-private-input-paths \
      --output-pt "$RUN_ROOT/${output_stem}.pt" \
      --output-json "$RUN_ROOT/${output_stem}.json" \
      --config "$RUN_ROOT/configs/active.runtime.json" \
      --device cpu \
      --max-samples 8 \
      --static-depth-limit 12 \
      --spu-batch-size 8 \
      --spu-params-mode secret \
      --spu-layer-norm-policy exact \
      --spu-attention-policy uniform \
      --spu-activation-override fixed_square \
      --spu-activation-clip-value 0 \
      --spu-secure-pruning-mode compact \
      --spu-secure-pruning-network unpadded_selection \
      --spu-final-block-cls-only \
      --spu-uniform-attention-value-fusion \
      --spu-public-fixed-square-scale \
      --spu-compile-cache-dir "$RUN_ROOT/${output_stem}.compile_cache" \
      >"$RUN_ROOT/${output_stem}.stdout.log" \
      2>"$RUN_ROOT/${output_stem}.stderr.log"
  local rc=$?
  set -e
  printf '%s\n' "$rc" >"$RUN_ROOT/${output_stem}.exit_code"
  snapshot "$python_bin" "$RUN_ROOT/${output_stem}.network.after.json"
  stop_spu
  if [[ "$rc" -ne 0 ]]; then
    echo "$output_stem failed with exit $rc" >&2
    return "$rc"
  fi
}

wait_server() {
  local port="$1"
  for _ in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:${port}/api/health" >/dev/null; then
      return 0
    fi
    sleep 0.5
  done
  return 1
}

start_showcase_server() {
  local port="$1"
  local audit_dir="$2"
  local run_dir="$3"
  local rate_limit="$4"
  local log_file="$5"
  mkdir -p "$audit_dir" "$run_dir"
  env \
    TRANSSHIELD_SHOWCASE_RUNTIME_MODE=mock \
    TRANSSHIELD_SHOWCASE_ACCEPTED_SLEEP_SEC=1.5 \
    TRANSSHIELD_SHOWCASE_AUDIT_DIR="$audit_dir" \
    TRANSSHIELD_SHOWCASE_RUN_DIR="$run_dir" \
    TRANSSHIELD_SHOWCASE_PER_IP_WINDOW_LIMIT="$rate_limit" \
    "$CANDIDATE_PY" -m uvicorn showcase_api.app:app \
      --host 127.0.0.1 --port "$port" >"$log_file" 2>&1 &
  SERVER_PID=$!
  wait_server "$port"
}

mark_phase inventory
"$CANDIDATE_PY" tools/report_vps_test.py inventory \
  --medical-data-root "$REPO_ROOT/data/pneumoniamnist_imagefolder_subset/val" \
  --finance-data-root "$REPO_ROOT/data/finance_boundary_stress_imagefolder/val" \
  --runtime-config "$RUNTIME_CONFIG_TEMPLATE" \
  --materialized-list-dir "$RUN_ROOT/lists" \
  --out "$RUN_ROOT/inventory.json"

mark_phase preprocessing
/usr/bin/time -v -o "$RUN_ROOT/medical524_preprocess.time.log" \
  "$CANDIDATE_PY" tools/transshield_e2e_secure_infer.py client-preprocess \
    --bundle-dir "$MEDICAL_BUNDLE" \
    --image-list "$RUN_ROOT/lists/medical_full_validation.txt" \
    --include-source-paths --include-targets \
    --output-pt "$RUN_ROOT/medical524_plain.pt" \
    --output-json "$RUN_ROOT/medical524_plain.json" \
    >"$RUN_ROOT/medical524_preprocess.stdout.log"

"$CANDIDATE_PY" tools/transshield_e2e_secure_infer.py client-preprocess \
  --bundle-dir "$MEDICAL_BUNDLE" \
  --image-list "$RUN_ROOT/lists/medical_secure_deployment_batch.txt" \
  --include-source-paths --include-targets \
  --output-pt "$RUN_ROOT/medical32_plain.pt" \
  --output-json "$RUN_ROOT/medical32_plain.json"
"$CANDIDATE_PY" tools/transshield_e2e_secure_infer.py client-share-preprocess \
  --bundle-dir "$MEDICAL_BUNDLE" \
  --image-list "$RUN_ROOT/lists/medical_secure_deployment_batch.txt" \
  --include-source-paths --include-targets \
  --output-prefix "$RUN_ROOT/medical32_share" \
  --output-json "$RUN_ROOT/medical32_shares.json" \
  --output-public-json "$RUN_ROOT/medical32_public.json" \
  --output-party-manifest-dir "$RUN_ROOT/medical32_party_manifests" \
  >"$RUN_ROOT/medical32_share_preprocess.stdout.log"

"$CANDIDATE_PY" tools/transshield_e2e_secure_infer.py client-preprocess \
  --bundle-dir "$FINANCE_BUNDLE" \
  --image-list "$RUN_ROOT/lists/finance_boundary_stress.txt" \
  --include-source-paths --include-targets \
  --output-pt "$RUN_ROOT/finance8_plain.pt" \
  --output-json "$RUN_ROOT/finance8_plain.json"
"$CANDIDATE_PY" tools/transshield_e2e_secure_infer.py client-share-preprocess \
  --bundle-dir "$FINANCE_BUNDLE" \
  --image-list "$RUN_ROOT/lists/finance_boundary_stress.txt" \
  --include-source-paths --include-targets \
  --output-prefix "$RUN_ROOT/finance8_share" \
  --output-json "$RUN_ROOT/finance8_shares.json" \
  --output-public-json "$RUN_ROOT/finance8_public.json" \
  --output-party-manifest-dir "$RUN_ROOT/finance8_party_manifests" \
  >"$RUN_ROOT/finance8_share_preprocess.stdout.log"

"$CANDIDATE_PY" tools/report_vps_test.py compare-preprocessed \
  --current-pt "$RUN_ROOT/medical32_plain.pt" \
  --frozen-pt "$REPO_ROOT/archive/old_runs/artifacts/server_pipeline_run/medical_dynamic_prepare_final/e2e_secure_poc/plaintext_same_images_pixel_values.pt" \
  --out "$RUN_ROOT/medical32_preprocess_compare.json"
"$CANDIDATE_PY" tools/report_vps_test.py compare-preprocessed \
  --current-pt "$RUN_ROOT/finance8_plain.pt" \
  --frozen-pt "$REPO_ROOT/archive/old_runs/artifacts/server_pipeline_run/finance_dynamic_static_prepare_final/e2e_secure_poc/plaintext_same_images_pixel_values.pt" \
  --out "$RUN_ROOT/finance8_preprocess_compare.json"

mark_phase cpu_references
/usr/bin/time -v -o "$RUN_ROOT/medical524_cpu_depth10.time.log" \
  "$CANDIDATE_PY" "$RUNNER" run \
    --runtime cpu --bundle-dir "$MEDICAL_BUNDLE" \
    --input-pt "$RUN_ROOT/medical524_plain.pt" \
    --output-pt "$RUN_ROOT/medical524_cpu_depth10.pt" \
    --output-json "$RUN_ROOT/medical524_cpu_depth10.json" \
    --device cpu --cpu-forward-mode runtime_pruning_reference \
    --cpu-batch-size 32 --static-depth-limit 10 \
    >"$RUN_ROOT/medical524_cpu_depth10.stdout.log"
"$CANDIDATE_PY" tools/report_vps_test.py summarize \
  --dataset-key medical_full_validation \
  --candidate-pt "$RUN_ROOT/medical524_cpu_depth10.pt" \
  --candidate-json "$RUN_ROOT/medical524_cpu_depth10.json" \
  --sample-list "$RUN_ROOT/lists/medical_full_validation.txt" \
  --threshold "$MEDICAL_THRESHOLD" \
  --out "$RUN_ROOT/medical524_cpu_depth10_summary.json"

"$CANDIDATE_PY" "$RUNNER" run \
  --runtime cpu --bundle-dir "$MEDICAL_BUNDLE" \
  --input-pt "$RUN_ROOT/medical32_plain.pt" \
  --output-pt "$RUN_ROOT/medical32_cpu_depth10.pt" \
  --output-json "$RUN_ROOT/medical32_cpu_depth10.json" \
  --device cpu --cpu-forward-mode runtime_pruning_reference \
  --cpu-batch-size 32 --static-depth-limit 10 \
  >"$RUN_ROOT/medical32_cpu_depth10.stdout.log"
"$CANDIDATE_PY" tools/report_vps_test.py summarize \
  --dataset-key medical_secure_deployment_batch \
  --candidate-pt "$RUN_ROOT/medical32_cpu_depth10.pt" \
  --candidate-json "$RUN_ROOT/medical32_cpu_depth10.json" \
  --sample-list "$RUN_ROOT/lists/medical_secure_deployment_batch.txt" \
  --threshold "$MEDICAL_THRESHOLD" \
  --out "$RUN_ROOT/medical32_cpu_depth10_summary.json"

"$CANDIDATE_PY" "$RUNNER" run \
  --runtime cpu --bundle-dir "$FINANCE_BUNDLE" \
  --input-pt "$RUN_ROOT/finance8_plain.pt" \
  --output-pt "$RUN_ROOT/finance8_cpu_depth12.pt" \
  --output-json "$RUN_ROOT/finance8_cpu_depth12.json" \
  --device cpu --cpu-forward-mode runtime_pruning_reference \
  --cpu-batch-size 8 --static-depth-limit 12 \
  >"$RUN_ROOT/finance8_cpu_depth12.stdout.log"
"$CANDIDATE_PY" tools/report_vps_test.py summarize \
  --dataset-key finance_boundary_stress \
  --candidate-pt "$RUN_ROOT/finance8_cpu_depth12.pt" \
  --candidate-json "$RUN_ROOT/finance8_cpu_depth12.json" \
  --sample-list "$RUN_ROOT/lists/finance_boundary_stress.txt" \
  --threshold 0.5 \
  --out "$RUN_ROOT/finance8_cpu_depth12_summary.json"

mark_phase code_tests
set +e
/usr/bin/time -v -o "$RUN_ROOT/unittest_vps.time.log" \
  "$CANDIDATE_PY" -m unittest discover -s tests -v \
  >"$RUN_ROOT/unittest_vps.stdout.log" \
  2>"$RUN_ROOT/unittest_vps.stderr.log"
UNIT_RC=$?
set -e
printf '%s\n' "$UNIT_RC" >"$RUN_ROOT/unittest_vps.exit_code"
if [[ "$UNIT_RC" -ne 0 ]]; then
  exit "$UNIT_RC"
fi

mark_phase medical32_same_vps_baseline
run_medical_spu "$BASELINE_PY" medical_baseline medical32_spu_baseline
"$BASELINE_PY" tools/report_vps_test.py summarize \
  --dataset-key medical_secure_deployment_batch \
  --candidate-pt "$RUN_ROOT/medical32_spu_baseline.pt" \
  --candidate-json "$RUN_ROOT/medical32_spu_baseline.json" \
  --reference-pt "$RUN_ROOT/medical32_cpu_depth10.pt" \
  --sample-list "$RUN_ROOT/lists/medical_secure_deployment_batch.txt" \
  --threshold "$MEDICAL_THRESHOLD" \
  --network-before "$RUN_ROOT/medical32_spu_baseline.network.before.json" \
  --network-after "$RUN_ROOT/medical32_spu_baseline.network.after.json" \
  --out "$RUN_ROOT/medical32_spu_baseline_summary.json"

mark_phase medical32_candidate
run_medical_spu "$CANDIDATE_PY" medical_candidate medical32_spu_latest
"$CANDIDATE_PY" tools/report_vps_test.py summarize \
  --dataset-key medical_secure_deployment_batch \
  --candidate-pt "$RUN_ROOT/medical32_spu_latest.pt" \
  --candidate-json "$RUN_ROOT/medical32_spu_latest.json" \
  --reference-pt "$RUN_ROOT/medical32_cpu_depth10.pt" \
  --sample-list "$RUN_ROOT/lists/medical_secure_deployment_batch.txt" \
  --threshold "$MEDICAL_THRESHOLD" \
  --network-before "$RUN_ROOT/medical32_spu_latest.network.before.json" \
  --network-after "$RUN_ROOT/medical32_spu_latest.network.after.json" \
  --out "$RUN_ROOT/medical32_spu_latest_summary.json"

mark_phase finance8_candidate
run_finance_spu "$CANDIDATE_PY"
"$CANDIDATE_PY" tools/report_vps_test.py summarize \
  --dataset-key finance_boundary_stress \
  --candidate-pt "$RUN_ROOT/finance8_spu_latest.pt" \
  --candidate-json "$RUN_ROOT/finance8_spu_latest.json" \
  --reference-pt "$RUN_ROOT/finance8_cpu_depth12.pt" \
  --sample-list "$RUN_ROOT/lists/finance_boundary_stress.txt" \
  --threshold 0.5 \
  --network-before "$RUN_ROOT/finance8_spu_latest.network.before.json" \
  --network-after "$RUN_ROOT/finance8_spu_latest.network.after.json" \
  --out "$RUN_ROOT/finance8_spu_latest_summary.json"

mark_phase protocol_fuzz_13
start_showcase_server 17865 "$RUN_ROOT/protocol_audit" "$RUN_ROOT/protocol_runs" 100 "$RUN_ROOT/protocol_server.log"
/usr/bin/time -v -o "$RUN_ROOT/protocol_fuzz_vps.time.log" \
  "$CANDIDATE_PY" tools/showcase_protocol_fuzz.py \
    --base-url http://127.0.0.1:17865 \
    --timeout 12 \
    --server-pid "$SERVER_PID" \
    --audit-rejections-jsonl "$RUN_ROOT/protocol_audit/audit_rejections.jsonl" \
    --out "$RUN_ROOT/protocol_fuzz_vps.json" \
    >"$RUN_ROOT/protocol_fuzz_vps.stdout.log"
stop_server

mark_phase guard_stress_4
start_showcase_server 17866 "$RUN_ROOT/guard_audit" "$RUN_ROOT/guard_runs" 12 "$RUN_ROOT/guard_server.log"
/usr/bin/time -v -o "$RUN_ROOT/guard_stress_vps.time.log" \
  "$CANDIDATE_PY" tools/showcase_guard_stress.py \
    --base-url http://127.0.0.1:17866 \
    --timeout 15 \
    --server-pid "$SERVER_PID" \
    --window-reset-sec 0 \
    --out "$RUN_ROOT/guard_stress_vps.json" \
    >"$RUN_ROOT/guard_stress_vps.stdout.log"
stop_server

mark_phase aggregate
"$CANDIDATE_PY" tools/report_vps_aggregate.py \
  --run-root "$RUN_ROOT" \
  --medical-baseline-summary "$RUN_ROOT/medical32_spu_baseline_summary.json" \
  --change-scope runtime_only \
  --require-report-update-ready \
  --out "$RUN_ROOT/report_regression_aggregate.json" \
  | tee "$RUN_ROOT/report_regression_aggregate.stdout.log"

mark_phase complete
