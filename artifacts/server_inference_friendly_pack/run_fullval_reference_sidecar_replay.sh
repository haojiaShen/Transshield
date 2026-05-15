#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"

MODE="${1:-fullval}"
case "$MODE" in
  fullval)
    DEFAULT_MAX_SAMPLES=0
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
    DEFAULT_MAX_SAMPLES="${REFERENCE_REPLAY_MAX_SAMPLES:-0}"
    ;;
  *)
    echo "Usage: $0 [fullval|smoke4|smoke8|smoke16|smoke32|custom]" >&2
    exit 2
    ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python}"
REFERENCE_REPLAY_BUNDLE_DIR="${REFERENCE_REPLAY_BUNDLE_DIR:-$REPO_ROOT/artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_aanone_20260507}"
REFERENCE_REPLAY_DATASET_DIR="${REFERENCE_REPLAY_DATASET_DIR:-${DATA_ROOT:-/data/wyb/pneumoniamnist_imagefolder_subset}/val}"
REFERENCE_REPLAY_MAX_SAMPLES="${REFERENCE_REPLAY_MAX_SAMPLES:-$DEFAULT_MAX_SAMPLES}"
REFERENCE_REPLAY_OUTPUT_DIR="${REFERENCE_REPLAY_OUTPUT_DIR:-$REPO_ROOT/results/e2e_gap_attribution/fullval_sidecar_replay_$(date +%Y%m%d_%H%M%S)}"
REFERENCE_REPLAY_DEVICE="${REFERENCE_REPLAY_DEVICE:-cpu}"
REFERENCE_REPLAY_EXPORT_BATCH_SIZE="${REFERENCE_REPLAY_EXPORT_BATCH_SIZE:-32}"
REFERENCE_REPLAY_BATCH_SIZE="${REFERENCE_REPLAY_BATCH_SIZE:-32}"
REFERENCE_REPLAY_NUM_WORKERS="${REFERENCE_REPLAY_NUM_WORKERS:-0}"
REFERENCE_REPLAY_THRESHOLD_TOLERANCE="${REFERENCE_REPLAY_THRESHOLD_TOLERANCE:-5e-5}"

if [[ ! -d "$REFERENCE_REPLAY_BUNDLE_DIR" ]]; then
  echo "[reference-replay] missing REFERENCE_REPLAY_BUNDLE_DIR: $REFERENCE_REPLAY_BUNDLE_DIR" >&2
  exit 2
fi
if [[ ! -d "$REFERENCE_REPLAY_DATASET_DIR" ]]; then
  echo "[reference-replay] missing REFERENCE_REPLAY_DATASET_DIR: $REFERENCE_REPLAY_DATASET_DIR" >&2
  exit 2
fi

mkdir -p "$REFERENCE_REPLAY_OUTPUT_DIR"

INPUT_PT="$REFERENCE_REPLAY_OUTPUT_DIR/stage2_secure_network_kth_input_smoke8.pt"
INPUT_JSON="$REFERENCE_REPLAY_OUTPUT_DIR/stage2_secure_network_kth_input_smoke8.json"
KTH_PT="$REFERENCE_REPLAY_OUTPUT_DIR/stage2_secure_network_kth_reference_smoke8.pt"
KTH_JSON="$REFERENCE_REPLAY_OUTPUT_DIR/stage2_secure_network_kth_reference_smoke8.json"
TIE_PT="$REFERENCE_REPLAY_OUTPUT_DIR/stage2_secure_tie_policy_lowest_smoke8.pt"
TIE_JSON="$REFERENCE_REPLAY_OUTPUT_DIR/stage2_secure_tie_policy_lowest_smoke8.json"
MANIFEST_JSON="$REFERENCE_REPLAY_OUTPUT_DIR/stage2_secure_network_kth_manifest.json"
REPLAY_JSON="$REFERENCE_REPLAY_OUTPUT_DIR/pipeline_inference_replay_reference_summary.json"
COMPARE_JSON="$REFERENCE_REPLAY_OUTPUT_DIR/plaintext_vs_reference_replay_score_compare.json"
COMPARE_CSV="$REFERENCE_REPLAY_OUTPUT_DIR/plaintext_vs_reference_replay_score_compare.csv"

EXPORT_ARGS=(
  "$PYTHON_BIN" tools/transshield_secure_sidecar_export_suite.py
  --bundle-dir "$REFERENCE_REPLAY_BUNDLE_DIR"
  --data-path "$REFERENCE_REPLAY_DATASET_DIR"
  --device "$REFERENCE_REPLAY_DEVICE"
  --batch-size "$REFERENCE_REPLAY_EXPORT_BATCH_SIZE"
  --num-workers "$REFERENCE_REPLAY_NUM_WORKERS"
  --max-samples "$REFERENCE_REPLAY_MAX_SAMPLES"
  --input-output-pt "$INPUT_PT"
  --input-output-json "$INPUT_JSON"
  --kth-output-pt "$KTH_PT"
  --kth-output-json "$KTH_JSON"
  --tie-output-pt "$TIE_PT"
  --tie-output-json "$TIE_JSON"
)

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

echo "[reference-replay] mode=$MODE"
echo "[reference-replay] bundle_dir=$REFERENCE_REPLAY_BUNDLE_DIR"
echo "[reference-replay] dataset_dir=$REFERENCE_REPLAY_DATASET_DIR"
echo "[reference-replay] max_samples=$REFERENCE_REPLAY_MAX_SAMPLES"
echo "[reference-replay] output_dir=$REFERENCE_REPLAY_OUTPUT_DIR"
echo "[reference-replay] device=$REFERENCE_REPLAY_DEVICE"

"${EXPORT_ARGS[@]}"

"$PYTHON_BIN" tools/transshield_secure_network_kth.py manifest \
  --bundle-dir "$REFERENCE_REPLAY_BUNDLE_DIR" \
  --output-json "$MANIFEST_JSON"

"$PYTHON_BIN" tools/transshield_openbumblebee_inference_replay.py \
  --bundle-dir "$REFERENCE_REPLAY_BUNDLE_DIR" \
  --input-pt "$INPUT_PT" \
  --kth-payload-pt "$KTH_PT" \
  --tie-payload-pt "$TIE_PT" \
  --enable-model-replay \
  --device "$REFERENCE_REPLAY_DEVICE" \
  --max-samples "$REFERENCE_REPLAY_MAX_SAMPLES" \
  --batch-size "$REFERENCE_REPLAY_BATCH_SIZE" \
  --num-workers "$REFERENCE_REPLAY_NUM_WORKERS" \
  --threshold-tolerance "$REFERENCE_REPLAY_THRESHOLD_TOLERANCE" \
  --output-json "$REPLAY_JSON"

"$PYTHON_BIN" tools/transshield_plaintext_secure_score_compare.py \
  --bundle-dir "$REFERENCE_REPLAY_BUNDLE_DIR" \
  --secure-replay-json "$REPLAY_JSON" \
  --device "$REFERENCE_REPLAY_DEVICE" \
  --batch-size 16 \
  --num-workers 0 \
  --output-json "$COMPARE_JSON" \
  --output-csv "$COMPARE_CSV"

echo "[reference-replay] wrote $REPLAY_JSON"
echo "[reference-replay] wrote $COMPARE_JSON"
echo "[reference-replay] wrote $COMPARE_CSV"
