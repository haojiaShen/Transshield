#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"

MODE="${1:-}"
case "$MODE" in
  prepare|cpu|spu|verify|audit-shares|probe-cpu|probe-spu|probe-compare|block1-smoke|runtime-smoke|calibrate-ln)
    shift
    ;;
  *)
    echo "Usage: $0 [prepare|cpu|spu|verify|audit-shares|probe-cpu|probe-spu|probe-compare|block1-smoke|runtime-smoke|calibrate-ln]" >&2
    exit 1
    ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="${RUN_NAME:-transshield_e2e_secure_poc}"
BUNDLE_DIR="${BUNDLE_DIR:-$REPO_ROOT/artifacts/frozen_bundle_verified_tracka_lr3e5_20260414}"
E2E_RUN_DIR="${E2E_RUN_DIR:-$REPO_ROOT/artifacts/server_pipeline_run/${RUN_NAME}/e2e_secure_poc}"
PACK_DIR="${PACK_DIR:-$E2E_RUN_DIR/whole_forward_pack}"
E2E_INPUT_PT="${E2E_INPUT_PT:-$E2E_RUN_DIR/client_pixel_values.pt}"
E2E_INPUT_SHARE_MANIFEST_JSON="${E2E_INPUT_SHARE_MANIFEST_JSON:-}"
E2E_INPUT_SHARE_PUBLIC_MANIFEST_JSON="${E2E_INPUT_SHARE_PUBLIC_MANIFEST_JSON:-}"
E2E_INPUT_P1_SHARE_MANIFEST_JSON="${E2E_INPUT_P1_SHARE_MANIFEST_JSON:-}"
E2E_INPUT_P2_SHARE_MANIFEST_JSON="${E2E_INPUT_P2_SHARE_MANIFEST_JSON:-}"
E2E_PARTY_LOCAL_SHARE_LOAD="${E2E_PARTY_LOCAL_SHARE_LOAD:-0}"
E2E_REDACT_PRIVATE_INPUT_PATHS="${E2E_REDACT_PRIVATE_INPUT_PATHS:-1}"
E2E_REFERENCE_PT="${E2E_REFERENCE_PT:-$E2E_RUN_DIR/static_whole_forward_reference.pt}"
E2E_CONTRACT_JSON="${E2E_CONTRACT_JSON:-$E2E_RUN_DIR/e2e_secure_contract.json}"
E2E_CANDIDATE_PT="${E2E_CANDIDATE_PT:-$E2E_RUN_DIR/e2e_static_whole_forward_candidate_from_server.pt}"
E2E_CANDIDATE_JSON="${E2E_CANDIDATE_JSON:-$E2E_RUN_DIR/e2e_static_whole_forward_candidate_from_server.json}"
E2E_OUTPUT_CALIBRATION_JSON="${E2E_OUTPUT_CALIBRATION_JSON:-}"
E2E_COMPARE_JSON="${E2E_COMPARE_JSON:-$E2E_RUN_DIR/e2e_static_whole_forward_compare.json}"
E2E_DEVICE="${E2E_DEVICE:-cpu}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/openbumblebee/2pc.json}"
SPU_RUNTIME_REUSE="${SPU_RUNTIME_REUSE:-0}"
SPU_DISABLE_COLOCATED_OPTIMIZATION="${SPU_DISABLE_COLOCATED_OPTIMIZATION:-0}"
SPU_REMOVE_UNSUPPORTED_CHEETAH_FIELDS="${SPU_REMOVE_UNSUPPORTED_CHEETAH_FIELDS:-1}"
SPU_RUNTIME_LOG_DIR="${SPU_RUNTIME_LOG_DIR:-logs/spu_nodes}"
SPU_RUNTIME_STATE_JSON="${SPU_RUNTIME_STATE_JSON:-logs/spu_runtime_ports.json}"
SPU_RUNTIME_STARTUP_TIMEOUT_SEC="${SPU_RUNTIME_STARTUP_TIMEOUT_SEC:-30}"
SPU_RUNTIME_STOP_WAIT_SEC="${SPU_RUNTIME_STOP_WAIT_SEC:-1}"
SPU_RUNTIME_WARMUP_ATTEMPTS="${SPU_RUNTIME_WARMUP_ATTEMPTS:-2}"
E2E_RUN_MAX_SAMPLES="${E2E_RUN_MAX_SAMPLES:-0}"
E2E_STATIC_DEPTH_LIMIT="${E2E_STATIC_DEPTH_LIMIT:--1}"
E2E_SPU_BATCH_SIZE="${E2E_SPU_BATCH_SIZE:-1}"
E2E_SPU_BLOCK_CHUNK_SIZE="${E2E_SPU_BLOCK_CHUNK_SIZE:-0}"
E2E_SPU_LAYER_NORM_CHUNK_SIZE="${E2E_SPU_LAYER_NORM_CHUNK_SIZE:-0}"
E2E_SPU_LAYER_NORM_POLICY="${E2E_SPU_LAYER_NORM_POLICY:-exact}"
E2E_SPU_LAYER_NORM_CALIBRATION_JSON="${E2E_SPU_LAYER_NORM_CALIBRATION_JSON:-$E2E_RUN_DIR/e2e_public_layer_norm_calibration.json}"
E2E_SPU_PARAMS_MODE="${E2E_SPU_PARAMS_MODE:-public}"
E2E_SPU_ATTENTION_POLICY="${E2E_SPU_ATTENTION_POLICY:-smoothed}"
E2E_SPU_ACTIVATION_OVERRIDE="${E2E_SPU_ACTIVATION_OVERRIDE:-bundle}"
E2E_SPU_ACTIVATION_CLIP_VALUE="${E2E_SPU_ACTIVATION_CLIP_VALUE:-0}"
E2E_VERIFY_MAX_SAMPLES="${E2E_VERIFY_MAX_SAMPLES:-0}"
E2E_VERIFY_ALLOW_PREFIX="${E2E_VERIFY_ALLOW_PREFIX:-0}"
E2E_PROBE_BLOCK_INDEX="${E2E_PROBE_BLOCK_INDEX:-0}"
E2E_PROBE_CPU_JSON="${E2E_PROBE_CPU_JSON:-$E2E_RUN_DIR/block$((E2E_PROBE_BLOCK_INDEX + 1))_probe_cpu_depth${E2E_STATIC_DEPTH_LIMIT}.json}"
E2E_PROBE_SPU_JSON="${E2E_PROBE_SPU_JSON:-$E2E_RUN_DIR/block$((E2E_PROBE_BLOCK_INDEX + 1))_probe_spu_depth${E2E_STATIC_DEPTH_LIMIT}.json}"
E2E_PROBE_COMPARE_JSON="${E2E_PROBE_COMPARE_JSON:-$E2E_RUN_DIR/block$((E2E_PROBE_BLOCK_INDEX + 1))_probe_compare_cpu_vs_spu_depth${E2E_STATIC_DEPTH_LIMIT}.json}"
E2E_SHARE_AUDIT_JSON="${E2E_SHARE_AUDIT_JSON:-$E2E_RUN_DIR/e2e_split_share_recomposition_patch_audit.json}"
E2E_BLOCK1_SMOKE_JSON="${E2E_BLOCK1_SMOKE_JSON:-$E2E_RUN_DIR/e2e_block1_subgraph_smoke.json}"
E2E_RUNTIME_SMOKE_JSON="${E2E_RUNTIME_SMOKE_JSON:-$E2E_RUN_DIR/e2e_spu_runtime_primitive_smoke.json}"
E2E_RUNTIME_SMOKE_TOKEN_COUNT="${E2E_RUNTIME_SMOKE_TOKEN_COUNT:-197}"
E2E_RUNTIME_SMOKE_EMBED_DIM="${E2E_RUNTIME_SMOKE_EMBED_DIM:-384}"
E2E_RUNTIME_SMOKE_NUM_HEADS="${E2E_RUNTIME_SMOKE_NUM_HEADS:-6}"
E2E_RUNTIME_SMOKE_MLP_RATIO="${E2E_RUNTIME_SMOKE_MLP_RATIO:-4.0}"
E2E_RUNTIME_SMOKE_LAYER_NORM_CHUNK_SIZE="${E2E_RUNTIME_SMOKE_LAYER_NORM_CHUNK_SIZE:-0}"
E2E_RUNTIME_SMOKE_LAYER_NORM_POLICY="${E2E_RUNTIME_SMOKE_LAYER_NORM_POLICY:-exact}"
E2E_RUNTIME_SMOKE_ATTENTION_POLICY="${E2E_RUNTIME_SMOKE_ATTENTION_POLICY:-standard}"
E2E_RUNTIME_SMOKE_SEED="${E2E_RUNTIME_SMOKE_SEED:-0}"

runtime_state_matches_requested() {
  if [[ ! -f "$SPU_RUNTIME_STATE_JSON" ]]; then
    return 1
  fi
  "$PYTHON_BIN" - "$SPU_RUNTIME_STATE_JSON" "$SPU_DISABLE_COLOCATED_OPTIMIZATION" <<'PY'
import json
import sys
from pathlib import Path

state_path = Path(sys.argv[1])
want_disable = sys.argv[2] == "1"
try:
    payload = json.loads(state_path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)

have_disable = bool(payload.get("disable_colocated_optimization", False))
raise SystemExit(0 if have_disable == want_disable else 1)
PY
}

build_spu_runtime_start_args() {
  local -n target_args="$1"
  target_args=(
    tools/transshield_spu_runtime_setup.py
    start
    --config "$CONFIG_PATH"
    --template configs/openbumblebee/2pc.template.json
    --backup
    --restart
    --log-dir "$SPU_RUNTIME_LOG_DIR"
    --state-json "$SPU_RUNTIME_STATE_JSON"
    --startup-timeout-sec "$SPU_RUNTIME_STARTUP_TIMEOUT_SEC"
    --stop-wait-sec "$SPU_RUNTIME_STOP_WAIT_SEC"
    --warmup-attempts "$SPU_RUNTIME_WARMUP_ATTEMPTS"
  )
  if [[ "$SPU_REMOVE_UNSUPPORTED_CHEETAH_FIELDS" == "1" ]]; then
    target_args+=(--remove-unsupported-cheetah-fields)
  fi
  if [[ "$SPU_DISABLE_COLOCATED_OPTIMIZATION" == "1" ]]; then
    target_args+=(--disable-colocated-optimization)
  fi
}

start_spu_runtime_if_needed() {
  local start_args=()
  build_spu_runtime_start_args start_args
  if [[ "$SPU_RUNTIME_REUSE" == "1" ]]; then
    if ! runtime_state_matches_requested; then
      echo "[e2e-whole-forward-spu] 已有 runtime 的 colocated 设置与当前请求不一致，重新拉起 SPU 节点。"
    elif "$PYTHON_BIN" tools/transshield_spu_runtime_setup.py check --config "$CONFIG_PATH" --startup-timeout-sec 3 >/dev/null 2>&1; then
      echo "[e2e-whole-forward-spu] 检测到可复用的 SPU runtime，跳过重启。"
      return
    else
      echo "[e2e-whole-forward-spu] 未检测到可复用 runtime，重新拉起 SPU 节点。"
    fi
  fi
  "$PYTHON_BIN" "${start_args[@]}"
}

append_positive_int_arg() {
  local -n target_args="$1"
  local name="$2"
  local value="$3"
  if [[ "$value" != "0" ]]; then
    target_args+=("$name" "$value")
  fi
}

case "$MODE" in
  prepare)
    "$PYTHON_BIN" integrations/openbumblebee/e2e_secure_vit/transshield_e2e_secure_vit.py prepare \
      --output-dir "$PACK_DIR" \
      --input-pt "$E2E_INPUT_PT" \
      --reference-pt "$E2E_REFERENCE_PT" \
      --contract-json "$E2E_CONTRACT_JSON" \
      "$@"
    ;;
  cpu|spu)
    RUN_ARGS=()
    INPUT_ARGS=(--input-pt "$E2E_INPUT_PT")
    append_positive_int_arg RUN_ARGS --max-samples "$E2E_RUN_MAX_SAMPLES"
    if [[ "$E2E_STATIC_DEPTH_LIMIT" != "-1" ]]; then
      RUN_ARGS+=(--static-depth-limit "$E2E_STATIC_DEPTH_LIMIT")
    fi
    if [[ "$MODE" == "spu" ]]; then
      start_spu_runtime_if_needed
      RUN_ARGS+=(--spu-batch-size "$E2E_SPU_BATCH_SIZE")
      if [[ "$E2E_SPU_BLOCK_CHUNK_SIZE" != "0" ]]; then
        RUN_ARGS+=(--spu-block-chunk-size "$E2E_SPU_BLOCK_CHUNK_SIZE")
      fi
      if [[ "$E2E_SPU_LAYER_NORM_CHUNK_SIZE" != "0" ]]; then
        RUN_ARGS+=(--spu-layer-norm-chunk-size "$E2E_SPU_LAYER_NORM_CHUNK_SIZE")
      fi
      RUN_ARGS+=(--spu-layer-norm-policy "$E2E_SPU_LAYER_NORM_POLICY")
      if [[ "$E2E_SPU_LAYER_NORM_POLICY" == "public_calibrated" ]]; then
        RUN_ARGS+=(--spu-layer-norm-calibration-json "$E2E_SPU_LAYER_NORM_CALIBRATION_JSON")
      fi
      RUN_ARGS+=(--spu-params-mode "$E2E_SPU_PARAMS_MODE")
      RUN_ARGS+=(--spu-attention-policy "$E2E_SPU_ATTENTION_POLICY")
      RUN_ARGS+=(--spu-activation-override "$E2E_SPU_ACTIVATION_OVERRIDE")
      RUN_ARGS+=(--spu-activation-clip-value "$E2E_SPU_ACTIVATION_CLIP_VALUE")
      if [[ -n "$E2E_INPUT_SHARE_MANIFEST_JSON" ]]; then
        if [[ -n "$E2E_INPUT_SHARE_PUBLIC_MANIFEST_JSON" || -n "$E2E_INPUT_P1_SHARE_MANIFEST_JSON" || -n "$E2E_INPUT_P2_SHARE_MANIFEST_JSON" ]]; then
          echo "[e2e-whole-forward-spu] E2E_INPUT_SHARE_MANIFEST_JSON 不能和 split share manifest 同时设置。" >&2
          exit 1
        fi
        RUN_ARGS+=(--input-share-manifest-json "$E2E_INPUT_SHARE_MANIFEST_JSON")
        INPUT_ARGS=()
      fi
      if [[ -n "$E2E_INPUT_SHARE_PUBLIC_MANIFEST_JSON" || -n "$E2E_INPUT_P1_SHARE_MANIFEST_JSON" || -n "$E2E_INPUT_P2_SHARE_MANIFEST_JSON" ]]; then
        if [[ -z "$E2E_INPUT_SHARE_PUBLIC_MANIFEST_JSON" || -z "$E2E_INPUT_P1_SHARE_MANIFEST_JSON" || -z "$E2E_INPUT_P2_SHARE_MANIFEST_JSON" ]]; then
          echo "[e2e-whole-forward-spu] split share 模式需要同时设置 public/P1/P2 三个 manifest。" >&2
          exit 1
        fi
        INPUT_ARGS=()
        RUN_ARGS+=(--input-share-public-manifest-json "$E2E_INPUT_SHARE_PUBLIC_MANIFEST_JSON")
        RUN_ARGS+=(--input-p1-share-manifest-json "$E2E_INPUT_P1_SHARE_MANIFEST_JSON")
        RUN_ARGS+=(--input-p2-share-manifest-json "$E2E_INPUT_P2_SHARE_MANIFEST_JSON")
        if [[ "$E2E_PARTY_LOCAL_SHARE_LOAD" == "1" ]]; then
          RUN_ARGS+=(--party-local-share-load)
        fi
      elif [[ "$E2E_PARTY_LOCAL_SHARE_LOAD" == "1" ]]; then
        echo "[e2e-whole-forward-spu] E2E_PARTY_LOCAL_SHARE_LOAD=1 需要 split public/P1/P2 manifest。" >&2
        exit 1
      fi
      if [[ "$E2E_REDACT_PRIVATE_INPUT_PATHS" == "1" ]]; then
        RUN_ARGS+=(--redact-private-input-paths)
      fi
    fi
    if [[ -n "$E2E_OUTPUT_CALIBRATION_JSON" ]]; then
      RUN_ARGS+=(--output-calibration-json "$E2E_OUTPUT_CALIBRATION_JSON")
    fi
    "$PYTHON_BIN" integrations/openbumblebee/e2e_secure_vit/transshield_e2e_secure_vit.py run \
      --runtime "$MODE" \
      --bundle-dir "$BUNDLE_DIR" \
      "${INPUT_ARGS[@]}" \
      --output-pt "$E2E_CANDIDATE_PT" \
      --output-json "$E2E_CANDIDATE_JSON" \
      --config "$CONFIG_PATH" \
      --device "$E2E_DEVICE" \
      "${RUN_ARGS[@]}" \
      "$@"
    ;;
  calibrate-ln)
    CALIBRATE_ARGS=()
    append_positive_int_arg CALIBRATE_ARGS --max-samples "$E2E_RUN_MAX_SAMPLES"
    if [[ "$E2E_STATIC_DEPTH_LIMIT" != "-1" ]]; then
      CALIBRATE_ARGS+=(--static-depth-limit "$E2E_STATIC_DEPTH_LIMIT")
    fi
    "$PYTHON_BIN" integrations/openbumblebee/e2e_secure_vit/transshield_e2e_secure_vit.py calibrate-layer-norm \
      --bundle-dir "$BUNDLE_DIR" \
      --input-pt "$E2E_INPUT_PT" \
      --output-json "$E2E_SPU_LAYER_NORM_CALIBRATION_JSON" \
      --spu-attention-policy "$E2E_SPU_ATTENTION_POLICY" \
      --spu-activation-override "$E2E_SPU_ACTIVATION_OVERRIDE" \
      --spu-activation-clip-value "$E2E_SPU_ACTIVATION_CLIP_VALUE" \
      "${CALIBRATE_ARGS[@]}" \
      "$@"
    ;;
  verify)
    VERIFY_ARGS=()
    append_positive_int_arg VERIFY_ARGS --max-samples "$E2E_VERIFY_MAX_SAMPLES"
    if [[ "$E2E_VERIFY_ALLOW_PREFIX" == "1" ]]; then
      VERIFY_ARGS+=(--allow-prefix-candidate)
    fi
    "$PYTHON_BIN" integrations/openbumblebee/e2e_secure_vit/transshield_e2e_secure_vit.py verify \
      --reference-pt "$E2E_REFERENCE_PT" \
      --candidate-pt "$E2E_CANDIDATE_PT" \
      --output-json "$E2E_COMPARE_JSON" \
      "${VERIFY_ARGS[@]}" \
      "$@"
    ;;
  audit-shares)
    start_spu_runtime_if_needed
    AUDIT_ARGS=()
    append_positive_int_arg AUDIT_ARGS --max-samples "$E2E_RUN_MAX_SAMPLES"
    AUDIT_ARGS+=(--spu-params-mode "$E2E_SPU_PARAMS_MODE")
    AUDIT_ARGS+=(--spu-attention-policy "$E2E_SPU_ATTENTION_POLICY")
    AUDIT_ARGS+=(--spu-activation-override "$E2E_SPU_ACTIVATION_OVERRIDE")
    if [[ -n "$E2E_INPUT_SHARE_MANIFEST_JSON" ]]; then
      if [[ -n "$E2E_INPUT_SHARE_PUBLIC_MANIFEST_JSON" || -n "$E2E_INPUT_P1_SHARE_MANIFEST_JSON" || -n "$E2E_INPUT_P2_SHARE_MANIFEST_JSON" ]]; then
        echo "[e2e-whole-forward-audit] E2E_INPUT_SHARE_MANIFEST_JSON 不能和 split share manifest 同时设置。" >&2
        exit 1
      fi
      AUDIT_ARGS+=(--input-share-manifest-json "$E2E_INPUT_SHARE_MANIFEST_JSON")
    fi
    if [[ -n "$E2E_INPUT_SHARE_PUBLIC_MANIFEST_JSON" || -n "$E2E_INPUT_P1_SHARE_MANIFEST_JSON" || -n "$E2E_INPUT_P2_SHARE_MANIFEST_JSON" ]]; then
      if [[ -z "$E2E_INPUT_SHARE_PUBLIC_MANIFEST_JSON" || -z "$E2E_INPUT_P1_SHARE_MANIFEST_JSON" || -z "$E2E_INPUT_P2_SHARE_MANIFEST_JSON" ]]; then
        echo "[e2e-whole-forward-audit] split share 模式需要同时设置 public/P1/P2 三个 manifest。" >&2
        exit 1
      fi
      AUDIT_ARGS+=(--input-share-public-manifest-json "$E2E_INPUT_SHARE_PUBLIC_MANIFEST_JSON")
      AUDIT_ARGS+=(--input-p1-share-manifest-json "$E2E_INPUT_P1_SHARE_MANIFEST_JSON")
      AUDIT_ARGS+=(--input-p2-share-manifest-json "$E2E_INPUT_P2_SHARE_MANIFEST_JSON")
    fi
    "$PYTHON_BIN" integrations/openbumblebee/e2e_secure_vit/transshield_e2e_secure_vit.py audit-input-shares \
      --bundle-dir "$BUNDLE_DIR" \
      --input-pt "$E2E_INPUT_PT" \
      --output-json "$E2E_SHARE_AUDIT_JSON" \
      --config "$CONFIG_PATH" \
      "${AUDIT_ARGS[@]}" \
      "$@"
    ;;
  probe-cpu|probe-spu)
    PROBE_ARGS=()
    append_positive_int_arg PROBE_ARGS --max-samples "$E2E_RUN_MAX_SAMPLES"
    if [[ "$E2E_STATIC_DEPTH_LIMIT" == "-1" ]]; then
      echo "[e2e-whole-forward-probe] E2E_STATIC_DEPTH_LIMIT must be set for probe modes." >&2
      exit 1
    fi
    if [[ "$MODE" == "probe-spu" ]]; then
      start_spu_runtime_if_needed
      PROBE_ARGS+=(--spu-batch-size "$E2E_SPU_BATCH_SIZE")
      if [[ "$E2E_SPU_LAYER_NORM_CHUNK_SIZE" != "0" ]]; then
        PROBE_ARGS+=(--spu-layer-norm-chunk-size "$E2E_SPU_LAYER_NORM_CHUNK_SIZE")
      fi
      PROBE_ARGS+=(--spu-layer-norm-policy "$E2E_SPU_LAYER_NORM_POLICY")
      PROBE_ARGS+=(--spu-params-mode "$E2E_SPU_PARAMS_MODE")
      PROBE_ARGS+=(--spu-attention-policy "$E2E_SPU_ATTENTION_POLICY")
      PROBE_ARGS+=(--spu-activation-override "$E2E_SPU_ACTIVATION_OVERRIDE")
      if [[ -n "$E2E_INPUT_SHARE_MANIFEST_JSON" ]]; then
        if [[ -n "$E2E_INPUT_SHARE_PUBLIC_MANIFEST_JSON" || -n "$E2E_INPUT_P1_SHARE_MANIFEST_JSON" || -n "$E2E_INPUT_P2_SHARE_MANIFEST_JSON" ]]; then
          echo "[e2e-whole-forward-probe] E2E_INPUT_SHARE_MANIFEST_JSON 不能和 split share manifest 同时设置。" >&2
          exit 1
        fi
        PROBE_ARGS+=(--input-share-manifest-json "$E2E_INPUT_SHARE_MANIFEST_JSON")
      fi
      if [[ -n "$E2E_INPUT_SHARE_PUBLIC_MANIFEST_JSON" || -n "$E2E_INPUT_P1_SHARE_MANIFEST_JSON" || -n "$E2E_INPUT_P2_SHARE_MANIFEST_JSON" ]]; then
        if [[ -z "$E2E_INPUT_SHARE_PUBLIC_MANIFEST_JSON" || -z "$E2E_INPUT_P1_SHARE_MANIFEST_JSON" || -z "$E2E_INPUT_P2_SHARE_MANIFEST_JSON" ]]; then
          echo "[e2e-whole-forward-probe] split share 模式需要同时设置 public/P1/P2 三个 manifest。" >&2
          exit 1
        fi
        PROBE_ARGS+=(--input-share-public-manifest-json "$E2E_INPUT_SHARE_PUBLIC_MANIFEST_JSON")
        PROBE_ARGS+=(--input-p1-share-manifest-json "$E2E_INPUT_P1_SHARE_MANIFEST_JSON")
        PROBE_ARGS+=(--input-p2-share-manifest-json "$E2E_INPUT_P2_SHARE_MANIFEST_JSON")
      fi
      PROBE_OUTPUT_JSON="$E2E_PROBE_SPU_JSON"
      PROBE_RUNTIME="spu"
    else
      PROBE_OUTPUT_JSON="$E2E_PROBE_CPU_JSON"
      PROBE_RUNTIME="cpu"
    fi
    "$PYTHON_BIN" integrations/openbumblebee/e2e_secure_vit/transshield_e2e_secure_vit.py probe-block \
      --runtime "$PROBE_RUNTIME" \
      --bundle-dir "$BUNDLE_DIR" \
      --input-pt "$E2E_INPUT_PT" \
      --output-json "$PROBE_OUTPUT_JSON" \
      --config "$CONFIG_PATH" \
      --device "$E2E_DEVICE" \
      --static-depth-limit "$E2E_STATIC_DEPTH_LIMIT" \
      --probe-block-index "$E2E_PROBE_BLOCK_INDEX" \
      "${PROBE_ARGS[@]}" \
      "$@"
    ;;
  probe-compare)
    "$PYTHON_BIN" integrations/openbumblebee/e2e_secure_vit/transshield_e2e_secure_vit.py compare-block-probe \
      --reference-json "$E2E_PROBE_CPU_JSON" \
      --candidate-json "$E2E_PROBE_SPU_JSON" \
      --output-json "$E2E_PROBE_COMPARE_JSON" \
      "$@"
    ;;
  block1-smoke)
    start_spu_runtime_if_needed
    SMOKE_ARGS=()
    append_positive_int_arg SMOKE_ARGS --max-samples "$E2E_RUN_MAX_SAMPLES"
    SMOKE_ARGS+=(--spu-params-mode "$E2E_SPU_PARAMS_MODE")
    if [[ "$E2E_SPU_LAYER_NORM_CHUNK_SIZE" != "0" ]]; then
      SMOKE_ARGS+=(--layer-norm-chunk-size "$E2E_SPU_LAYER_NORM_CHUNK_SIZE")
    fi
    SMOKE_ARGS+=(--layer-norm-policy "$E2E_SPU_LAYER_NORM_POLICY")
    SMOKE_ARGS+=(--spu-attention-policy "$E2E_SPU_ATTENTION_POLICY")
    SMOKE_ARGS+=(--spu-activation-override "$E2E_SPU_ACTIVATION_OVERRIDE")
    "$PYTHON_BIN" integrations/openbumblebee/e2e_secure_vit/transshield_e2e_secure_vit.py block1-subgraph-smoke \
      --bundle-dir "$BUNDLE_DIR" \
      --input-pt "$E2E_INPUT_PT" \
      --output-json "$E2E_BLOCK1_SMOKE_JSON" \
      --config "$CONFIG_PATH" \
      "${SMOKE_ARGS[@]}" \
      "$@"
    ;;
  runtime-smoke)
    start_spu_runtime_if_needed
    "$PYTHON_BIN" integrations/openbumblebee/e2e_secure_vit/transshield_e2e_secure_vit.py runtime-primitive-smoke \
      --output-json "$E2E_RUNTIME_SMOKE_JSON" \
      --config "$CONFIG_PATH" \
      --token-count "$E2E_RUNTIME_SMOKE_TOKEN_COUNT" \
      --embed-dim "$E2E_RUNTIME_SMOKE_EMBED_DIM" \
      --num-heads "$E2E_RUNTIME_SMOKE_NUM_HEADS" \
      --mlp-ratio "$E2E_RUNTIME_SMOKE_MLP_RATIO" \
      --layer-norm-chunk-size "$E2E_RUNTIME_SMOKE_LAYER_NORM_CHUNK_SIZE" \
      --layer-norm-policy "$E2E_RUNTIME_SMOKE_LAYER_NORM_POLICY" \
      --attention-policy "$E2E_RUNTIME_SMOKE_ATTENTION_POLICY" \
      --seed "$E2E_RUNTIME_SMOKE_SEED" \
      "$@"
    ;;
esac
