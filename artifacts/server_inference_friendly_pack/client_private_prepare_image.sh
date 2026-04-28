#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
BUNDLE_DIR="${BUNDLE_DIR:-$REPO_ROOT/artifacts/frozen_bundle_verified_tracka_lr3e5_20260414}"
CLIENT_INPUT_IMAGE="${CLIENT_INPUT_IMAGE:-${1:-}}"

if [[ -z "$CLIENT_INPUT_IMAGE" ]]; then
  echo "Usage: CLIENT_INPUT_IMAGE=/path/to/image.png $0" >&2
  echo "   or: $0 /path/to/image.png" >&2
  exit 1
fi
if [[ ! -f "$CLIENT_INPUT_IMAGE" ]]; then
  echo "[client-private-prepare] missing input image: $CLIENT_INPUT_IMAGE" >&2
  exit 1
fi

CLIENT_PREP_RUN_NAME="${CLIENT_PREP_RUN_NAME:-client_private_$(date +%Y%m%d_%H%M%S)}"
CLIENT_PREP_DIR="${CLIENT_PREP_DIR:-$REPO_ROOT/artifacts/client_private_inputs/$CLIENT_PREP_RUN_NAME}"
SHARE_PREFIX="$CLIENT_PREP_DIR/client_pixel_values_debug_share"
SHARE_MANIFEST_JSON="$CLIENT_PREP_DIR/client_pixel_values_debug_share_manifest.json"
SHARE_PUBLIC_JSON="$CLIENT_PREP_DIR/client_pixel_values_debug_share_public_manifest.json"
SHARE_PARTY_DIR="$CLIENT_PREP_DIR/client_pixel_values_debug_share_party_manifests"
SERVER_ENV_SH="$CLIENT_PREP_DIR/server_e2e_infer_env.sh"

mkdir -p "$CLIENT_PREP_DIR" "$SHARE_PARTY_DIR"

"$PYTHON_BIN" tools/transshield_e2e_secure_infer.py client-share-preprocess \
  --bundle-dir "$BUNDLE_DIR" \
  --image "$CLIENT_INPUT_IMAGE" \
  --max-samples 1 \
  --output-prefix "$SHARE_PREFIX" \
  --output-json "$SHARE_MANIFEST_JSON" \
  --output-public-json "$SHARE_PUBLIC_JSON" \
  --output-party-manifest-dir "$SHARE_PARTY_DIR" \
  --share-seed "${CLIENT_SHARE_SEED:-0}"

cat > "$SERVER_ENV_SH" <<EOF
# Source this on the inference server after placing P1/P2 manifests and share files
# at the paths recorded below. The server must not receive the original image.
export SOURCE_E2E_DIR="$CLIENT_PREP_DIR"
export E2E_INPUT_SHARE_PUBLIC_MANIFEST_JSON="$SHARE_PUBLIC_JSON"
export E2E_INPUT_P1_SHARE_MANIFEST_JSON="$SHARE_PARTY_DIR/p1_share_manifest.json"
export E2E_INPUT_P2_SHARE_MANIFEST_JSON="$SHARE_PARTY_DIR/p2_share_manifest.json"
export E2E_RUN_MAX_SAMPLES=1
export E2E_SPU_BATCH_SIZE=1
export E2E_PARTY_LOCAL_SHARE_LOAD=1
export E2E_REDACT_PRIVATE_INPUT_PATHS=1
export E2E_STATIC_DEPTH_LIMIT=12
export E2E_SPU_LAYER_NORM_POLICY=public_calibrated
export E2E_SPU_ATTENTION_POLICY=uniform
export E2E_SPU_ACTIVATION_OVERRIDE=fixed_square
EOF

echo "[client-private-prepare] output_dir=$CLIENT_PREP_DIR"
echo "[client-private-prepare] public_manifest=$SHARE_PUBLIC_JSON"
echo "[client-private-prepare] p1_manifest=$SHARE_PARTY_DIR/p1_share_manifest.json"
echo "[client-private-prepare] p2_manifest=$SHARE_PARTY_DIR/p2_share_manifest.json"
echo "[client-private-prepare] server_env=$SERVER_ENV_SH"
