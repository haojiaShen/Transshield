#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"

MODE="${1:-smoke4}"
case "$MODE" in
  smoke1)
    DEFAULT_MAX_SAMPLES=1
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
    DEFAULT_MAX_SAMPLES="${KEEPMASK_MAX_SAMPLES:-0}"
    ;;
  *)
    echo "Usage: $0 [smoke1|smoke4|smoke8|smoke16|smoke32|custom]" >&2
    exit 2
    ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python}"
KEEPMASK_BUNDLE_DIR="${KEEPMASK_BUNDLE_DIR:-$REPO_ROOT/artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_aanone_20260507}"
KEEPMASK_DATASET_DIR="${KEEPMASK_DATASET_DIR:-${DATA_ROOT:-/data/wyb/pneumoniamnist_imagefolder_subset}/val}"
KEEPMASK_MAX_SAMPLES="${KEEPMASK_MAX_SAMPLES:-$DEFAULT_MAX_SAMPLES}"
KEEPMASK_INPUT_MODE="${KEEPMASK_INPUT_MODE:-party_local}"
KEEPMASK_SPU_PARAMS_MODE="${KEEPMASK_SPU_PARAMS_MODE:-public}"
KEEPMASK_SPU_BATCH_SIZE="${KEEPMASK_SPU_BATCH_SIZE:-1}"
KEEPMASK_SPU_ATTENTION_POLICY="${KEEPMASK_SPU_ATTENTION_POLICY:-uniform}"
KEEPMASK_SHARE_SEED="${KEEPMASK_SHARE_SEED:-0}"
KEEPMASK_OUTPUT_DIR="${KEEPMASK_OUTPUT_DIR:-$REPO_ROOT/results/e2e_gap_attribution/spu_runtime_pruning_keepmask_${MODE}_${KEEPMASK_INPUT_MODE}_${KEEPMASK_SPU_PARAMS_MODE}_$(date +%Y%m%d_%H%M%S)}"
KEEPMASK_DEVICE="${KEEPMASK_DEVICE:-cpu}"
KEEPMASK_CONFIG="${KEEPMASK_CONFIG:-$REPO_ROOT/configs/openbumblebee/2pc.json}"
KEEPMASK_STATIC_DEPTH_LIMIT="${KEEPMASK_STATIC_DEPTH_LIMIT:--1}"
KEEPMASK_REDACT_PRIVATE_INPUT_PATHS="${KEEPMASK_REDACT_PRIVATE_INPUT_PATHS:-1}"

case "$KEEPMASK_INPUT_MODE" in
  plaintext|host_share|party_local)
    ;;
  *)
    echo "[keepmask-bridge] unsupported KEEPMASK_INPUT_MODE: $KEEPMASK_INPUT_MODE" >&2
    exit 2
    ;;
esac

case "$KEEPMASK_SPU_PARAMS_MODE" in
  public|secret)
    ;;
  *)
    echo "[keepmask-bridge] external keep-mask path currently supports only public|secret params mode, got: $KEEPMASK_SPU_PARAMS_MODE" >&2
    exit 2
    ;;
esac

if [[ "$KEEPMASK_SPU_ATTENTION_POLICY" != "uniform" ]]; then
  echo "[keepmask-bridge] external keep-mask path currently requires KEEPMASK_SPU_ATTENTION_POLICY=uniform" >&2
  exit 2
fi
if [[ ! -d "$KEEPMASK_BUNDLE_DIR" ]]; then
  echo "[keepmask-bridge] missing KEEPMASK_BUNDLE_DIR: $KEEPMASK_BUNDLE_DIR" >&2
  exit 2
fi
if [[ ! -d "$KEEPMASK_DATASET_DIR" ]]; then
  echo "[keepmask-bridge] missing KEEPMASK_DATASET_DIR: $KEEPMASK_DATASET_DIR" >&2
  exit 2
fi
if [[ ! -f "$KEEPMASK_CONFIG" ]]; then
  echo "[keepmask-bridge] missing KEEPMASK_CONFIG: $KEEPMASK_CONFIG" >&2
  exit 2
fi

mkdir -p "$KEEPMASK_OUTPUT_DIR" "$KEEPMASK_OUTPUT_DIR/share_party_manifests"

INPUT_PT="$KEEPMASK_OUTPUT_DIR/share_input_pixel_values.pt"
INPUT_JSON="$KEEPMASK_OUTPUT_DIR/share_input_pixel_values.json"
SHARE_PREFIX="$KEEPMASK_OUTPUT_DIR/client_pixel_values_debug_share"
SHARE_MANIFEST_JSON="$KEEPMASK_OUTPUT_DIR/client_pixel_values_debug_share_manifest.json"
SHARE_PUBLIC_JSON="$KEEPMASK_OUTPUT_DIR/client_pixel_values_debug_share_public_manifest.json"
SHARE_PARTY_DIR="$KEEPMASK_OUTPUT_DIR/share_party_manifests"
KEEPMASK_JSON="$KEEPMASK_OUTPUT_DIR/keep_mask_payload.json"
KEEPMASK_PT="$KEEPMASK_OUTPUT_DIR/keep_mask_payload.pt"
REFERENCE_JSON="$KEEPMASK_OUTPUT_DIR/runtime_pruning_reference.json"
REFERENCE_PT="$KEEPMASK_OUTPUT_DIR/runtime_pruning_reference.pt"
CANDIDATE_JSON="$KEEPMASK_OUTPUT_DIR/candidate.json"
CANDIDATE_PT="$KEEPMASK_OUTPUT_DIR/candidate.pt"
VERIFY_JSON="$KEEPMASK_OUTPUT_DIR/verify.json"

PREPROCESS_ARGS=(
  "$PYTHON_BIN" tools/transshield_e2e_secure_infer.py client-preprocess
  --bundle-dir "$KEEPMASK_BUNDLE_DIR"
  --data-path "$KEEPMASK_DATASET_DIR"
  --output-pt "$INPUT_PT"
  --output-json "$INPUT_JSON"
  --include-source-paths
  --include-targets
)
if [[ "$KEEPMASK_MAX_SAMPLES" -gt 0 ]]; then
  PREPROCESS_ARGS+=(--max-samples "$KEEPMASK_MAX_SAMPLES")
fi

SHARE_PREPROCESS_ARGS=(
  "$PYTHON_BIN" tools/transshield_e2e_secure_infer.py client-share-preprocess
  --bundle-dir "$KEEPMASK_BUNDLE_DIR"
  --data-path "$KEEPMASK_DATASET_DIR"
  --output-prefix "$SHARE_PREFIX"
  --output-json "$SHARE_MANIFEST_JSON"
  --output-public-json "$SHARE_PUBLIC_JSON"
  --output-party-manifest-dir "$SHARE_PARTY_DIR"
  --include-source-paths
  --include-targets
  --share-seed "$KEEPMASK_SHARE_SEED"
)
if [[ "$KEEPMASK_MAX_SAMPLES" -gt 0 ]]; then
  SHARE_PREPROCESS_ARGS+=(--max-samples "$KEEPMASK_MAX_SAMPLES")
fi

RUN_ARGS=(
  "$PYTHON_BIN" integrations/openbumblebee/e2e_secure_vit/transshield_e2e_secure_vit.py run
  --runtime spu
  --bundle-dir "$KEEPMASK_BUNDLE_DIR"
  --runtime-pruning-keep-mask-pt "$KEEPMASK_PT"
  --output-pt "$CANDIDATE_PT"
  --output-json "$CANDIDATE_JSON"
  --config "$KEEPMASK_CONFIG"
  --spu-params-mode "$KEEPMASK_SPU_PARAMS_MODE"
  --spu-batch-size "$KEEPMASK_SPU_BATCH_SIZE"
  --spu-attention-policy "$KEEPMASK_SPU_ATTENTION_POLICY"
  --max-samples "$KEEPMASK_MAX_SAMPLES"
  --static-depth-limit "$KEEPMASK_STATIC_DEPTH_LIMIT"
)

if [[ "$KEEPMASK_REDACT_PRIVATE_INPUT_PATHS" == "1" ]]; then
  RUN_ARGS+=(--redact-private-input-paths)
fi

NEED_SHARE_FILES=0
case "$KEEPMASK_INPUT_MODE" in
  plaintext)
    RUN_ARGS+=(--input-pt "$INPUT_PT")
    ;;
  host_share)
    NEED_SHARE_FILES=1
    RUN_ARGS+=(
      --input-share-public-manifest-json "$SHARE_PUBLIC_JSON"
      --input-p1-share-manifest-json "$SHARE_PARTY_DIR/p1_share_manifest.json"
      --input-p2-share-manifest-json "$SHARE_PARTY_DIR/p2_share_manifest.json"
    )
    ;;
  party_local)
    NEED_SHARE_FILES=1
    RUN_ARGS+=(
      --input-share-public-manifest-json "$SHARE_PUBLIC_JSON"
      --input-p1-share-manifest-json "$SHARE_PARTY_DIR/p1_share_manifest.json"
      --input-p2-share-manifest-json "$SHARE_PARTY_DIR/p2_share_manifest.json"
      --party-local-share-load
    )
    ;;
esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

echo "[keepmask-bridge] mode=$MODE"
echo "[keepmask-bridge] bundle_dir=$KEEPMASK_BUNDLE_DIR"
echo "[keepmask-bridge] dataset_dir=$KEEPMASK_DATASET_DIR"
echo "[keepmask-bridge] max_samples=$KEEPMASK_MAX_SAMPLES"
echo "[keepmask-bridge] input_mode=$KEEPMASK_INPUT_MODE"
echo "[keepmask-bridge] spu_params_mode=$KEEPMASK_SPU_PARAMS_MODE"
echo "[keepmask-bridge] output_dir=$KEEPMASK_OUTPUT_DIR"
echo "[keepmask-bridge] device=$KEEPMASK_DEVICE"

"${PREPROCESS_ARGS[@]}"

if [[ "$NEED_SHARE_FILES" == "1" ]]; then
  "${SHARE_PREPROCESS_ARGS[@]}"
fi

"$PYTHON_BIN" tools/transshield_e2e_secure_infer.py export-runtime-pruning-keep-mask-payload \
  --bundle-dir "$KEEPMASK_BUNDLE_DIR" \
  --input-pt "$INPUT_PT" \
  --device "$KEEPMASK_DEVICE" \
  --output-json "$KEEPMASK_JSON" \
  --output-pt "$KEEPMASK_PT" \
  --static-depth-limit "$KEEPMASK_STATIC_DEPTH_LIMIT"

"$PYTHON_BIN" tools/transshield_e2e_secure_infer.py runtime-pruning-whole-forward-reference \
  --bundle-dir "$KEEPMASK_BUNDLE_DIR" \
  --input-pt "$INPUT_PT" \
  --device "$KEEPMASK_DEVICE" \
  --output-json "$REFERENCE_JSON" \
  --output-pt "$REFERENCE_PT" \
  --static-depth-limit "$KEEPMASK_STATIC_DEPTH_LIMIT"

"${RUN_ARGS[@]}"

"$PYTHON_BIN" integrations/openbumblebee/e2e_secure_vit/transshield_e2e_secure_vit.py verify \
  --reference-pt "$REFERENCE_PT" \
  --candidate-pt "$CANDIDATE_PT" \
  --output-json "$VERIFY_JSON"

echo "[keepmask-bridge] wrote $CANDIDATE_JSON"
echo "[keepmask-bridge] wrote $VERIFY_JSON"
