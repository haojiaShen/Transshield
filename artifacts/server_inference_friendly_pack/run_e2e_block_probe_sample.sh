#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
BUNDLE_DIR="${BUNDLE_DIR:-$REPO_ROOT/artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_aanone_20260507}"
SOURCE_E2E_DIR="${SOURCE_E2E_DIR:-$REPO_ROOT/artifacts/server_pipeline_run/e2e_aanone_exactln_clip0_heldout238_spuaware_nonisolated_20260507_1/e2e_secure_poc}"
SOURCE_SHARE_MANIFEST_JSON="${SOURCE_SHARE_MANIFEST_JSON:-$SOURCE_E2E_DIR/client_pixel_values_debug_share_manifest.json}"
SOURCE_INPUT_PT="${SOURCE_INPUT_PT:-$SOURCE_E2E_DIR/plaintext_same_images_pixel_values.pt}"
SOURCE_EVAL_IMAGE_LIST="${SOURCE_EVAL_IMAGE_LIST:-$SOURCE_E2E_DIR/e2e_eval_images.txt}"
SAMPLE_DIAGNOSIS_CSV="${SAMPLE_DIAGNOSIS_CSV:-$REPO_ROOT/results/e2e_static_calibration/e2e_static_fullval_aanone_20260507_1/e2e_heldout238_spuaware_calibration_drift_report.csv}"
BLOCK_ORDINALS="${BLOCK_ORDINALS:-1 6 9 12}"
SOURCE_INDEX="${SOURCE_INDEX:-}"
PROBE_NAME="${PROBE_NAME:-}"
PROBE_OUTPUT_DIR="${PROBE_OUTPUT_DIR:-}"

if [[ -z "$SOURCE_INDEX" ]]; then
  echo "[e2e-block-probe-sample] SOURCE_INDEX is required." >&2
  exit 2
fi

if [[ -z "$PROBE_NAME" ]]; then
  PROBE_NAME="e2e_aanone_heldout238_idx${SOURCE_INDEX}_blocks_$(date +%Y%m%d_%H%M%S)"
fi
if [[ -z "$PROBE_OUTPUT_DIR" ]]; then
  PROBE_OUTPUT_DIR="$REPO_ROOT/results/e2e_block_probe/$PROBE_NAME"
fi

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "[e2e-block-probe-sample] missing $label: $path" >&2
    exit 2
  fi
}

require_file "$SOURCE_SHARE_MANIFEST_JSON" "source share manifest"
require_file "$SOURCE_INPUT_PT" "source plaintext input pt"
require_file "$SOURCE_EVAL_IMAGE_LIST" "source eval image list"
require_file "$SAMPLE_DIAGNOSIS_CSV" "sample diagnosis csv"

mkdir -p "$PROBE_OUTPUT_DIR"

sample_info_json="$PROBE_OUTPUT_DIR/sample_info.json"
"$PYTHON_BIN" - "$SAMPLE_DIAGNOSIS_CSV" "$SOURCE_INDEX" "$sample_info_json" <<'PY'
import csv
import json
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
source_index = int(sys.argv[2])
output_json = Path(sys.argv[3])
row = None
with csv_path.open("r", encoding="utf-8", newline="") as handle:
    for item in csv.DictReader(handle):
        if int(item["index"]) == source_index:
            row = item
            break
if row is None:
    raise SystemExit(f"source index {source_index} not found in {csv_path}")
payload = {
    "source_index": source_index,
    "image": row["image"],
    "target": int(row["target"]),
    "spuaware_bias_score": float(row["spuaware_bias_score"]),
    "static_bias_score": float(row["static_bias_score"]),
    "raw_score_logit1_minus_logit0": float(row["raw_score_logit1_minus_logit0"]),
}
output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

sample_image="$("$PYTHON_BIN" - "$sample_info_json" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["image"])
PY
)"
sample_target="$("$PYTHON_BIN" - "$sample_info_json" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["target"])
PY
)"
sample_bias_score="$("$PYTHON_BIN" - "$sample_info_json" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["spuaware_bias_score"])
PY
)"

echo "[e2e-block-probe-sample] probe_name=$PROBE_NAME"
echo "[e2e-block-probe-sample] source_index=$SOURCE_INDEX"
echo "[e2e-block-probe-sample] image=$sample_image"
echo "[e2e-block-probe-sample] target=$sample_target"
echo "[e2e-block-probe-sample] spuaware_bias_score=$sample_bias_score"
echo "[e2e-block-probe-sample] blocks=$BLOCK_ORDINALS"

slice_dir="$PROBE_OUTPUT_DIR"
party_dir="$slice_dir/client_pixel_values_debug_share_party_manifests"
mkdir -p "$party_dir"

"$PYTHON_BIN" tools/transshield_slice_debug_shares.py \
  --share-manifest-json "$SOURCE_SHARE_MANIFEST_JSON" \
  --indices "$SOURCE_INDEX" \
  --source-paths-file "$SOURCE_EVAL_IMAGE_LIST" \
  --output-prefix "$slice_dir/client_pixel_values_debug_share" \
  --output-json "$slice_dir/client_pixel_values_debug_share_manifest.json" \
  --output-public-json "$slice_dir/client_pixel_values_debug_share_public_manifest.json" \
  --output-party-manifest-dir "$party_dir" \
  --input-pt "$SOURCE_INPUT_PT" \
  --output-input-pt "$slice_dir/client_pixel_values.pt"

export BUNDLE_DIR
export E2E_RUN_DIR="$PROBE_OUTPUT_DIR"
export E2E_INPUT_PT="$slice_dir/client_pixel_values.pt"
export E2E_INPUT_SHARE_PUBLIC_MANIFEST_JSON="$slice_dir/client_pixel_values_debug_share_public_manifest.json"
export E2E_INPUT_P1_SHARE_MANIFEST_JSON="$party_dir/p1_share_manifest.json"
export E2E_INPUT_P2_SHARE_MANIFEST_JSON="$party_dir/p2_share_manifest.json"
export E2E_RUN_MAX_SAMPLES=1
export E2E_STATIC_DEPTH_LIMIT=12
export E2E_SPU_BATCH_SIZE=1
export E2E_SPU_PARAMS_MODE=public
export E2E_SPU_LAYER_NORM_POLICY="${E2E_SPU_LAYER_NORM_POLICY:-exact}"
export E2E_SPU_ATTENTION_POLICY="${E2E_SPU_ATTENTION_POLICY:-uniform}"
export E2E_SPU_ACTIVATION_OVERRIDE="${E2E_SPU_ACTIVATION_OVERRIDE:-fixed_square}"
export E2E_SPU_ACTIVATION_CLIP_VALUE="${E2E_SPU_ACTIVATION_CLIP_VALUE:-0}"
export SPU_DISABLE_COLOCATED_OPTIMIZATION="${SPU_DISABLE_COLOCATED_OPTIMIZATION:-1}"
export SPU_RUNTIME_REUSE="${SPU_RUNTIME_REUSE:-0}"

for ordinal in $BLOCK_ORDINALS; do
  if ! [[ "$ordinal" =~ ^[0-9]+$ ]] || [[ "$ordinal" -le 0 ]]; then
    echo "[e2e-block-probe-sample] invalid block ordinal: $ordinal" >&2
    exit 2
  fi
  block_index=$((ordinal - 1))
  export E2E_PROBE_BLOCK_INDEX="$block_index"
  export E2E_PROBE_CPU_JSON="$PROBE_OUTPUT_DIR/block${ordinal}_probe_cpu.json"
  export E2E_PROBE_SPU_JSON="$PROBE_OUTPUT_DIR/block${ordinal}_probe_spu.json"
  export E2E_PROBE_COMPARE_JSON="$PROBE_OUTPUT_DIR/block${ordinal}_probe_compare.json"
  echo "[e2e-block-probe-sample] run block=$ordinal"
  bash "$SCRIPT_DIR/run_e2e_secure_whole_forward.sh" probe-cpu > "$PROBE_OUTPUT_DIR/block${ordinal}_cpu.log" 2>&1
  bash "$SCRIPT_DIR/run_e2e_secure_whole_forward.sh" probe-spu > "$PROBE_OUTPUT_DIR/block${ordinal}_spu.log" 2>&1
  bash "$SCRIPT_DIR/run_e2e_secure_whole_forward.sh" probe-compare > "$PROBE_OUTPUT_DIR/block${ordinal}_compare.log" 2>&1
done

"$PYTHON_BIN" tools/transshield_e2e_block_sweep_summary.py \
  --input-dir "$PROBE_OUTPUT_DIR" \
  --pattern 'block*_probe_compare.json' \
  --label "$PROBE_NAME" \
  --sample-label "heldout238_idx${SOURCE_INDEX}_spuaware_high_margin_probe" \
  --sample-image "$sample_image" \
  --sample-target "$sample_target" \
  --sample-bias-score "$sample_bias_score" \
  --output-json "$PROBE_OUTPUT_DIR/block_probe_summary.json" \
  --output-md "$PROBE_OUTPUT_DIR/block_probe_summary.md"

echo "[e2e-block-probe-sample] output_dir=$PROBE_OUTPUT_DIR"
