#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
BUNDLE_DIR="${BUNDLE_DIR:-$REPO_ROOT/artifacts/frozen_bundle_verified_tracka_lr3e5_20260414}"
RUN_NAME="${RUN_NAME:-transshield_e2e_approx_eval_$(date +%Y%m%d_%H%M%S)}"
E2E_EVAL_DATASET_DIR="${E2E_EVAL_DATASET_DIR:-/data/wyb/pneumoniamnist_imagefolder_subset/val}"
E2E_EVAL_MAX_SAMPLES="${E2E_EVAL_MAX_SAMPLES:-8}"
E2E_RUN_DIR="${E2E_RUN_DIR:-$REPO_ROOT/artifacts/server_pipeline_run/$RUN_NAME/e2e_secure_poc}"
E2E_EVAL_IMAGE_LIST="${E2E_EVAL_IMAGE_LIST:-$E2E_RUN_DIR/e2e_eval_images.txt}"
E2E_EVAL_METRICS_JSON="${E2E_EVAL_METRICS_JSON:-$E2E_RUN_DIR/e2e_approx_eval_metrics.json}"
E2E_PREPROCESS_TIMEOUT_SEC="${E2E_PREPROCESS_TIMEOUT_SEC:-180}"
E2E_PLAINTEXT_TIMEOUT_SEC="${E2E_PLAINTEXT_TIMEOUT_SEC:-300}"
E2E_SHARE_PREPROCESS_TIMEOUT_SEC="${E2E_SHARE_PREPROCESS_TIMEOUT_SEC:-180}"
E2E_METRICS_TIMEOUT_SEC="${E2E_METRICS_TIMEOUT_SEC:-180}"
E2E_APPROX_EVAL_ISOLATE_SAMPLES="${E2E_APPROX_EVAL_ISOLATE_SAMPLES:-1}"
E2E_SPU_ATTENTION_POLICY="${E2E_SPU_ATTENTION_POLICY:-uniform}"
E2E_SPU_ACTIVATION_OVERRIDE="${E2E_SPU_ACTIVATION_OVERRIDE:-fixed_square}"
E2E_SPU_ACTIVATION_CLIP_VALUE="${E2E_SPU_ACTIVATION_CLIP_VALUE:-3.0}"
E2E_SPU_ACTIVATION_CLIP_TAG="clip${E2E_SPU_ACTIVATION_CLIP_VALUE//./p}"
E2E_OUTPUT_CALIBRATION_JSON="${E2E_OUTPUT_CALIBRATION_JSON:-}"
E2E_SPU_BLOCK_CHUNK_SIZE="${E2E_SPU_BLOCK_CHUNK_SIZE:-0}"
E2E_SPU_LAYER_NORM_CHUNK_SIZE="${E2E_SPU_LAYER_NORM_CHUNK_SIZE:-0}"
E2E_APPROX_EVAL_SAMPLE_MAX_RETRIES="${E2E_APPROX_EVAL_SAMPLE_MAX_RETRIES:-2}"
E2E_APPROX_EVAL_LOGIT_ABS_GUARD="${E2E_APPROX_EVAL_LOGIT_ABS_GUARD:-1000}"
E2E_ISOLATED_INFER_TIMEOUT_SEC="${E2E_ISOLATED_INFER_TIMEOUT_SEC:-600}"
E2E_ISOLATED_INFER_TIMEOUT_KILL_SEC="${E2E_ISOLATED_INFER_TIMEOUT_KILL_SEC:-60}"

# This eval path is CPU/SPU oriented. Avoid CUDA library probing during torch import on servers
# where the GPU stack may be present but unhealthy.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export CUDA_VISIBLE_DEVICES OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS PYTHONFAULTHANDLER

mkdir -p "$E2E_RUN_DIR"

run_step_with_timeout() {
  local label="$1"
  local timeout_sec="$2"
  shift 2
  echo "[e2e-approx-eval] start $label timeout=${timeout_sec}s"
  if command -v timeout >/dev/null 2>&1; then
    timeout --foreground "$timeout_sec" "$@"
  else
    "$@"
  fi
  echo "[e2e-approx-eval] done $label"
}

validate_candidate_logits() {
  local candidate_pt="$1"
  local candidate_json="$2"
  "$PYTHON_BIN" - "$candidate_pt" "$candidate_json" "$E2E_APPROX_EVAL_LOGIT_ABS_GUARD" <<'PY'
import json
import math
import sys
from pathlib import Path

import torch

candidate_pt = Path(sys.argv[1])
candidate_json = Path(sys.argv[2])
guard = float(sys.argv[3])
payload = torch.load(candidate_pt, map_location="cpu")
summary = json.loads(candidate_json.read_text(encoding="utf-8"))

checks = {}
for name in ("raw_logits_before_output_calibration", "logits", "probabilities"):
    tensor = payload.get(name)
    if tensor is None:
        continue
    tensor = tensor.detach().cpu().float()
    finite = bool(torch.isfinite(tensor).all().item())
    max_abs = float(tensor.abs().max().item()) if tensor.numel() else 0.0
    checks[name] = {"finite": finite, "max_abs": max_abs}

ok = all(item["finite"] and item["max_abs"] <= guard for item in checks.values())
if not bool(summary.get("finite_logits", True)):
    ok = False
print(json.dumps({"ok": ok, "guard": guard, "checks": checks}, sort_keys=True))
raise SystemExit(0 if ok else 1)
PY
}

if [[ ! -d "$E2E_EVAL_DATASET_DIR" ]]; then
  echo "[e2e-approx-eval] missing eval dataset dir: $E2E_EVAL_DATASET_DIR" >&2
  exit 1
fi

generate_eval_image_list() {
  if [[ -d "$E2E_EVAL_DATASET_DIR/0" && -d "$E2E_EVAL_DATASET_DIR/1" && "$E2E_EVAL_MAX_SAMPLES" -ge 2 ]]; then
    local half remainder
    half=$((E2E_EVAL_MAX_SAMPLES / 2))
    remainder=$((E2E_EVAL_MAX_SAMPLES - half))
    local class0_list class1_list
    class0_list="${E2E_EVAL_IMAGE_LIST}.class0.tmp"
    class1_list="${E2E_EVAL_IMAGE_LIST}.class1.tmp"
    find "$E2E_EVAL_DATASET_DIR/0" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) | sort > "$class0_list"
    find "$E2E_EVAL_DATASET_DIR/1" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) | sort > "$class1_list"
    head -n "$half" "$class0_list" > "$E2E_EVAL_IMAGE_LIST"
    head -n "$remainder" "$class1_list" >> "$E2E_EVAL_IMAGE_LIST"
    rm -f "$class0_list" "$class1_list"
  else
    local all_list
    all_list="${E2E_EVAL_IMAGE_LIST}.all.tmp"
    find "$E2E_EVAL_DATASET_DIR" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) | sort > "$all_list"
    head -n "$E2E_EVAL_MAX_SAMPLES" "$all_list" > "$E2E_EVAL_IMAGE_LIST"
    rm -f "$all_list"
  fi
}

if [[ ! -f "$E2E_EVAL_IMAGE_LIST" || "${E2E_EVAL_REGENERATE_LIST:-1}" == "1" ]]; then
  generate_eval_image_list
fi

sample_count="$(wc -l < "$E2E_EVAL_IMAGE_LIST" | tr -d ' ')"
if [[ "$sample_count" -le 0 ]]; then
  echo "[e2e-approx-eval] no images found under $E2E_EVAL_DATASET_DIR" >&2
  exit 1
fi
echo "[e2e-approx-eval] image_list=$E2E_EVAL_IMAGE_LIST"
echo "[e2e-approx-eval] sample_count=$sample_count"
cat "$E2E_EVAL_IMAGE_LIST"

SHARE_PREFIX="$E2E_RUN_DIR/client_pixel_values_debug_share"
SHARE_MANIFEST_JSON="$E2E_RUN_DIR/client_pixel_values_debug_share_manifest.json"
SHARE_PUBLIC_JSON="$E2E_RUN_DIR/client_pixel_values_debug_share_public_manifest.json"
SHARE_PARTY_DIR="$E2E_RUN_DIR/client_pixel_values_debug_share_party_manifests"
PLAINTEXT_INPUT_PT="$E2E_RUN_DIR/plaintext_same_images_pixel_values.pt"
PLAINTEXT_INPUT_JSON="$E2E_RUN_DIR/plaintext_same_images_pixel_values.json"
PLAINTEXT_REFERENCE_JSON="$E2E_RUN_DIR/plaintext_same_images_reference.json"
CALIB_JSON="${E2E_SPU_LAYER_NORM_CALIBRATION_JSON:-$E2E_RUN_DIR/e2e_public_layer_norm_calibration_depth12_${E2E_SPU_ATTENTION_POLICY}_${E2E_SPU_ACTIVATION_OVERRIDE}_${E2E_SPU_ACTIVATION_CLIP_TAG}.json}"
CANDIDATE_PT="${E2E_CANDIDATE_PT:-$E2E_RUN_DIR/e2e_static_whole_forward_candidate_spu_depth12_partylocal_publiccalibln_${E2E_SPU_ATTENTION_POLICY}_${E2E_SPU_ACTIVATION_OVERRIDE}_${E2E_SPU_ACTIVATION_CLIP_TAG}_eval.pt}"
CANDIDATE_JSON="${E2E_CANDIDATE_JSON:-$E2E_RUN_DIR/e2e_static_whole_forward_candidate_spu_depth12_partylocal_publiccalibln_${E2E_SPU_ATTENTION_POLICY}_${E2E_SPU_ACTIVATION_OVERRIDE}_${E2E_SPU_ACTIVATION_CLIP_TAG}_eval.json}"

run_step_with_timeout "client-preprocess plaintext pixel package" "$E2E_PREPROCESS_TIMEOUT_SEC" \
  "$PYTHON_BIN" tools/transshield_e2e_secure_infer.py client-preprocess \
  --bundle-dir "$BUNDLE_DIR" \
  --image-list "$E2E_EVAL_IMAGE_LIST" \
  --max-samples "$E2E_EVAL_MAX_SAMPLES" \
  --output-pt "$PLAINTEXT_INPUT_PT" \
  --output-json "$PLAINTEXT_INPUT_JSON" \
  --include-targets

run_step_with_timeout "original plaintext reference" "$E2E_PLAINTEXT_TIMEOUT_SEC" \
  "$PYTHON_BIN" tools/transshield_e2e_secure_infer.py plaintext-reference \
  --bundle-dir "$BUNDLE_DIR" \
  --input-pt "$PLAINTEXT_INPUT_PT" \
  --device "${PLAINTEXT_EVAL_DEVICE:-cpu}" \
  --output-json "$PLAINTEXT_REFERENCE_JSON"

run_step_with_timeout "client-share-preprocess debug shares" "$E2E_SHARE_PREPROCESS_TIMEOUT_SEC" \
  "$PYTHON_BIN" tools/transshield_e2e_secure_infer.py client-share-preprocess \
  --bundle-dir "$BUNDLE_DIR" \
  --image-list "$E2E_EVAL_IMAGE_LIST" \
  --max-samples "$E2E_EVAL_MAX_SAMPLES" \
  --output-prefix "$SHARE_PREFIX" \
  --output-json "$SHARE_MANIFEST_JSON" \
  --output-public-json "$SHARE_PUBLIC_JSON" \
  --output-party-manifest-dir "$SHARE_PARTY_DIR" \
  --include-targets \
  --share-seed "${E2E_SHARE_SEED:-0}"

if [[ ! -f "$CALIB_JSON" ]]; then
  RUN_NAME="$RUN_NAME" \
  E2E_RUN_DIR="$E2E_RUN_DIR" \
  E2E_SPU_LAYER_NORM_CALIBRATION_JSON="$CALIB_JSON" \
  E2E_SPU_ATTENTION_POLICY="$E2E_SPU_ATTENTION_POLICY" \
  E2E_SPU_ACTIVATION_OVERRIDE="$E2E_SPU_ACTIVATION_OVERRIDE" \
  E2E_SPU_ACTIVATION_CLIP_VALUE="$E2E_SPU_ACTIVATION_CLIP_VALUE" \
  E2E_SPU_BLOCK_CHUNK_SIZE="$E2E_SPU_BLOCK_CHUNK_SIZE" \
  E2E_SPU_LAYER_NORM_CHUNK_SIZE="$E2E_SPU_LAYER_NORM_CHUNK_SIZE" \
  PUBLIC_CALIB_DATASET_DIR="${PUBLIC_CALIB_DATASET_DIR:-/data/wyb/pneumoniamnist_imagefolder_subset}" \
    bash "$SCRIPT_DIR/run_e2e_secure_approx_deploy.sh" make-calib-pixels

  RUN_NAME="$RUN_NAME" \
  E2E_RUN_DIR="$E2E_RUN_DIR" \
  E2E_SPU_LAYER_NORM_CALIBRATION_JSON="$CALIB_JSON" \
  E2E_SPU_ATTENTION_POLICY="$E2E_SPU_ATTENTION_POLICY" \
  E2E_SPU_ACTIVATION_OVERRIDE="$E2E_SPU_ACTIVATION_OVERRIDE" \
  E2E_SPU_ACTIVATION_CLIP_VALUE="$E2E_SPU_ACTIVATION_CLIP_VALUE" \
  E2E_SPU_BLOCK_CHUNK_SIZE="$E2E_SPU_BLOCK_CHUNK_SIZE" \
  E2E_SPU_LAYER_NORM_CHUNK_SIZE="$E2E_SPU_LAYER_NORM_CHUNK_SIZE" \
    bash "$SCRIPT_DIR/run_e2e_secure_approx_deploy.sh" calibrate
fi

if [[ "$E2E_APPROX_EVAL_ISOLATE_SAMPLES" == "1" && "$sample_count" -gt 1 ]]; then
  echo "[e2e-approx-eval] isolated per-sample e2e infer enabled"
  ISOLATED_DIR="$E2E_RUN_DIR/isolated_e2e_candidates"
  mkdir -p "$ISOLATED_DIR"
  ISOLATED_INDEX_JSON="$ISOLATED_DIR/isolated_candidate_index.json"
  : > "$ISOLATED_INDEX_JSON.tmp"
  index=0
  while IFS= read -r image_path; do
    sample_dir="$ISOLATED_DIR/sample_$(printf '%06d' "$index")"
    sample_image_list="$sample_dir/e2e_eval_image.txt"
    sample_share_prefix="$sample_dir/client_pixel_values_debug_share"
    sample_share_manifest="$sample_dir/client_pixel_values_debug_share_manifest.json"
    sample_share_public="$sample_dir/client_pixel_values_debug_share_public_manifest.json"
    sample_share_party_dir="$sample_dir/client_pixel_values_debug_share_party_manifests"
    sample_candidate_pt="$sample_dir/e2e_static_whole_forward_candidate_spu_depth12_partylocal_publiccalibln_${E2E_SPU_ATTENTION_POLICY}_${E2E_SPU_ACTIVATION_OVERRIDE}_${E2E_SPU_ACTIVATION_CLIP_TAG}_eval.pt"
    sample_candidate_json="$sample_dir/e2e_static_whole_forward_candidate_spu_depth12_partylocal_publiccalibln_${E2E_SPU_ATTENTION_POLICY}_${E2E_SPU_ACTIVATION_OVERRIDE}_${E2E_SPU_ACTIVATION_CLIP_TAG}_eval.json"
    mkdir -p "$sample_dir"
    printf '%s\n' "$image_path" > "$sample_image_list"

    run_step_with_timeout "client-share-preprocess isolated sample $index" "$E2E_SHARE_PREPROCESS_TIMEOUT_SEC" \
      "$PYTHON_BIN" tools/transshield_e2e_secure_infer.py client-share-preprocess \
      --bundle-dir "$BUNDLE_DIR" \
      --image-list "$sample_image_list" \
      --max-samples 1 \
      --output-prefix "$sample_share_prefix" \
      --output-json "$sample_share_manifest" \
      --output-public-json "$sample_share_public" \
      --output-party-manifest-dir "$sample_share_party_dir" \
      --include-targets \
      --share-seed "${E2E_SHARE_SEED:-0}"

    attempt=1
    while true; do
      echo "[e2e-approx-eval] isolated sample $index infer attempt $attempt/$E2E_APPROX_EVAL_SAMPLE_MAX_RETRIES"
      sample_infer_cmd=(bash "$SCRIPT_DIR/run_e2e_secure_approx_deploy.sh" infer)
      if command -v timeout >/dev/null 2>&1; then
        sample_infer_cmd=(
          timeout
          -k "$E2E_ISOLATED_INFER_TIMEOUT_KILL_SEC"
          "$E2E_ISOLATED_INFER_TIMEOUT_SEC"
          "${sample_infer_cmd[@]}"
        )
      fi
      if RUN_NAME="${RUN_NAME}_sample_$(printf '%06d' "$index")_attempt_${attempt}" \
        E2E_RUN_DIR="$sample_dir" \
        E2E_INPUT_SHARE_PUBLIC_MANIFEST_JSON="$sample_share_public" \
        E2E_INPUT_P1_SHARE_MANIFEST_JSON="$sample_share_party_dir/p1_share_manifest.json" \
        E2E_INPUT_P2_SHARE_MANIFEST_JSON="$sample_share_party_dir/p2_share_manifest.json" \
        E2E_SPU_LAYER_NORM_CALIBRATION_JSON="$CALIB_JSON" \
        E2E_SPU_ATTENTION_POLICY="$E2E_SPU_ATTENTION_POLICY" \
        E2E_SPU_ACTIVATION_OVERRIDE="$E2E_SPU_ACTIVATION_OVERRIDE" \
        E2E_SPU_ACTIVATION_CLIP_VALUE="$E2E_SPU_ACTIVATION_CLIP_VALUE" \
        E2E_SPU_BLOCK_CHUNK_SIZE="$E2E_SPU_BLOCK_CHUNK_SIZE" \
        E2E_SPU_LAYER_NORM_CHUNK_SIZE="$E2E_SPU_LAYER_NORM_CHUNK_SIZE" \
        E2E_OUTPUT_CALIBRATION_JSON="$E2E_OUTPUT_CALIBRATION_JSON" \
        E2E_CANDIDATE_PT="$sample_candidate_pt" \
        E2E_CANDIDATE_JSON="$sample_candidate_json" \
        E2E_RUN_MAX_SAMPLES=1 \
        E2E_SPU_BATCH_SIZE=1 \
        SPU_RUNTIME_REUSE=0 \
          "${sample_infer_cmd[@]}" && \
          validate_candidate_logits "$sample_candidate_pt" "$sample_candidate_json"; then
        break
      fi
      if [[ "$attempt" -ge "$E2E_APPROX_EVAL_SAMPLE_MAX_RETRIES" ]]; then
        echo "[e2e-approx-eval] isolated sample $index failed logit guard after $attempt attempts" >&2
        exit 1
      fi
      echo "[e2e-approx-eval] retry isolated sample $index after abnormal candidate logits"
      attempt=$((attempt + 1))
    done

    "$PYTHON_BIN" - "$ISOLATED_INDEX_JSON.tmp" "$index" "$image_path" "$sample_candidate_pt" "$sample_candidate_json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
items = []
if path.exists() and path.read_text(encoding="utf-8").strip():
    items = json.loads(path.read_text(encoding="utf-8"))
items.append(
    {
        "index": int(sys.argv[2]),
        "image_path": sys.argv[3],
        "candidate_pt": sys.argv[4],
        "candidate_json": sys.argv[5],
    }
)
path.write_text(json.dumps(items, indent=2, sort_keys=True), encoding="utf-8")
PY
    index=$((index + 1))
  done < "$E2E_EVAL_IMAGE_LIST"
  mv "$ISOLATED_INDEX_JSON.tmp" "$ISOLATED_INDEX_JSON"

  "$PYTHON_BIN" - "$ISOLATED_INDEX_JSON" "$CANDIDATE_PT" "$CANDIDATE_JSON" "$CALIB_JSON" <<'PY'
import json
import sys
import time
from pathlib import Path

import torch

index_json = Path(sys.argv[1])
output_pt = Path(sys.argv[2])
output_json = Path(sys.argv[3])
calib_json = sys.argv[4]
items = json.loads(index_json.read_text(encoding="utf-8"))
payloads = [torch.load(item["candidate_pt"], map_location="cpu") for item in items]
summaries = [json.loads(Path(item["candidate_json"]).read_text(encoding="utf-8")) for item in items]

logits = torch.cat([payload["logits"].detach().cpu().float()[:1] for payload in payloads], dim=0)
probabilities = torch.cat(
    [
        (
            payload.get("probabilities").detach().cpu().float()
            if payload.get("probabilities") is not None
            else torch.softmax(payload["logits"].detach().cpu().float(), dim=-1)
        )[:1]
        for payload in payloads
    ],
    dim=0,
)
raw_logits = None
if all(payload.get("raw_logits_before_output_calibration") is not None for payload in payloads):
    raw_logits = torch.cat(
        [payload["raw_logits_before_output_calibration"].detach().cpu().float()[:1] for payload in payloads],
        dim=0,
    )
threshold = payloads[0].get("threshold")
argmax_predictions = logits.argmax(dim=1)
threshold_predictions = None
if threshold is not None and probabilities.shape[-1] == 2:
    threshold_predictions = (probabilities[:, 1] >= float(threshold)).long()

combined = dict(payloads[0])
combined.update(
    {
        "input_pt": None,
        "input_source": "isolated_per_sample_party_local_candidates",
        "logits": logits,
        "probabilities": probabilities,
        "argmax_predictions": argmax_predictions,
        "threshold_predictions": threshold_predictions,
        "isolated_candidate_index_json": str(index_json),
        "isolated_candidate_count": len(items),
    }
)
if raw_logits is not None:
    combined["raw_logits_before_output_calibration"] = raw_logits
output_pt.parent.mkdir(parents=True, exist_ok=True)
torch.save(combined, output_pt)

elapsed = sum(float(summary.get("elapsed_sec") or 0.0) for summary in summaries)
summary = {
    "manifest_type": "transshield_e2e_static_whole_forward_candidate_summary_v0",
    "runtime": "spu",
    "backend": "jax_spu_static_whole_forward_backend_v0",
    "input_pt": None,
    "input_source": "isolated_per_sample_party_local_candidates",
    "output_pt": str(output_pt),
    "sample_count": int(logits.shape[0]),
    "elapsed_sec": float(elapsed),
    "threshold": None if threshold is None else float(threshold),
    "finite_logits": bool(torch.isfinite(logits).all().item()),
    "max_samples": int(logits.shape[0]),
    "static_depth_limit": 12,
    "effective_static_depth": 12,
    "logits": {
        "shape": list(logits.shape),
        "dtype": str(logits.dtype),
        "min": float(logits.min().item()),
        "max": float(logits.max().item()),
        "mean": float(logits.mean().item()),
        "std": float(logits.std(unbiased=False).item()),
    },
    "probabilities": {
        "shape": list(probabilities.shape),
        "dtype": str(probabilities.dtype),
        "min": float(probabilities.min().item()),
        "max": float(probabilities.max().item()),
        "mean": float(probabilities.mean().item()),
        "std": float(probabilities.std(unbiased=False).item()),
    },
    "raw_logits_before_output_calibration": (
        None
        if raw_logits is None
        else {
            "shape": list(raw_logits.shape),
            "dtype": str(raw_logits.dtype),
            "min": float(raw_logits.min().item()),
            "max": float(raw_logits.max().item()),
            "mean": float(raw_logits.mean().item()),
            "std": float(raw_logits.std(unbiased=False).item()),
        }
    ),
    "prediction_preview": {
        "logits": [[float(value) for value in row] for row in logits.tolist()],
        "probabilities": [[float(value) for value in row] for row in probabilities.tolist()],
        "argmax_predictions": [int(value) for value in argmax_predictions.tolist()],
        "threshold_predictions": (
            None if threshold_predictions is None else [int(value) for value in threshold_predictions.tolist()]
        ),
    },
    "spu": {
        "input_mode": "party_local_debug_share_load",
        "host_plaintext_pixel_values_materialized": False,
        "host_private_share_tensors_loaded": False,
        "private_input_paths_redacted": True,
        "spu_batch_size": 1,
        "spu_layer_norm_policy": "public_calibrated",
        "spu_layer_norm_calibration_json": calib_json,
        "static_forward_metadata": summaries[0].get("spu", {}).get("static_forward_metadata"),
    },
    "privacy_note": (
        "Eval-only aggregate built from isolated per-sample party-local SPU candidates. "
        "Each sample is inferred with the same final-logits-only reveal policy."
    ),
    "isolated_candidate_index_json": str(index_json),
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
}
output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
else
  RUN_NAME="$RUN_NAME" \
  E2E_RUN_DIR="$E2E_RUN_DIR" \
  E2E_INPUT_SHARE_PUBLIC_MANIFEST_JSON="$SHARE_PUBLIC_JSON" \
  E2E_INPUT_P1_SHARE_MANIFEST_JSON="$SHARE_PARTY_DIR/p1_share_manifest.json" \
  E2E_INPUT_P2_SHARE_MANIFEST_JSON="$SHARE_PARTY_DIR/p2_share_manifest.json" \
  E2E_SPU_LAYER_NORM_CALIBRATION_JSON="$CALIB_JSON" \
  E2E_SPU_ATTENTION_POLICY="$E2E_SPU_ATTENTION_POLICY" \
  E2E_SPU_ACTIVATION_OVERRIDE="$E2E_SPU_ACTIVATION_OVERRIDE" \
  E2E_SPU_ACTIVATION_CLIP_VALUE="$E2E_SPU_ACTIVATION_CLIP_VALUE" \
  E2E_SPU_BLOCK_CHUNK_SIZE="$E2E_SPU_BLOCK_CHUNK_SIZE" \
  E2E_SPU_LAYER_NORM_CHUNK_SIZE="$E2E_SPU_LAYER_NORM_CHUNK_SIZE" \
  E2E_OUTPUT_CALIBRATION_JSON="$E2E_OUTPUT_CALIBRATION_JSON" \
  E2E_CANDIDATE_PT="$CANDIDATE_PT" \
  E2E_CANDIDATE_JSON="$CANDIDATE_JSON" \
  E2E_RUN_MAX_SAMPLES="$sample_count" \
  E2E_SPU_BATCH_SIZE=1 \
    bash "$SCRIPT_DIR/run_e2e_secure_approx_deploy.sh" infer
fi

run_step_with_timeout "write e2e vs plaintext metrics" "$E2E_METRICS_TIMEOUT_SEC" \
  "$PYTHON_BIN" - "$SHARE_MANIFEST_JSON" "$PLAINTEXT_REFERENCE_JSON" "$CANDIDATE_PT" "$CANDIDATE_JSON" "$E2E_EVAL_METRICS_JSON" logs/spu_nodes <<'PY'
import json
import re
import sys
from pathlib import Path

import torch

share_manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
plaintext_reference = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
candidate_pt = Path(sys.argv[3])
candidate_json = json.loads(Path(sys.argv[4]).read_text(encoding="utf-8"))
output_json = Path(sys.argv[5])
spu_log_dir = Path(sys.argv[6])

payload = torch.load(candidate_pt, map_location="cpu")
targets = share_manifest.get("targets")
if targets is None:
    raise SystemExit("share manifest does not include targets; rerun with --include-targets")
targets = torch.tensor(targets, dtype=torch.long)
logits = payload["logits"].detach().cpu().float()
probabilities = payload.get("probabilities")
if probabilities is None:
    probabilities = torch.softmax(logits, dim=-1)
else:
    probabilities = probabilities.detach().cpu().float()
count = min(int(targets.numel()), int(logits.shape[0]))
targets = targets[:count]
argmax_predictions = logits.argmax(dim=1)[:count]
threshold = payload.get("threshold")
threshold_predictions = payload.get("threshold_predictions")
if threshold_predictions is None and threshold is not None and probabilities.shape[-1] == 2:
    threshold_predictions = (probabilities[:, 1] >= float(threshold)).long()
if threshold_predictions is not None:
    threshold_predictions = threshold_predictions.detach().cpu()[:count]

argmax_acc = float((argmax_predictions == targets).float().mean().item() * 100.0)
threshold_acc = None
if threshold_predictions is not None:
    threshold_acc = float((threshold_predictions == targets).float().mean().item() * 100.0)

plaintext_rows = plaintext_reference.get("per_sample") or []
plaintext_rows = plaintext_rows[:count]
plaintext_argmax_predictions = torch.tensor([int(row["argmax_prediction"]) for row in plaintext_rows], dtype=torch.long)
plaintext_threshold_values = [
    row.get("threshold_prediction")
    for row in plaintext_rows
]
plaintext_threshold_predictions = None
if all(value is not None for value in plaintext_threshold_values):
    plaintext_threshold_predictions = torch.tensor([int(value) for value in plaintext_threshold_values], dtype=torch.long)

plaintext_argmax_acc = float((plaintext_argmax_predictions == targets).float().mean().item() * 100.0)
plaintext_threshold_acc = None
if plaintext_threshold_predictions is not None:
    plaintext_threshold_acc = float((plaintext_threshold_predictions == targets).float().mean().item() * 100.0)

argmax_match_vs_plaintext = float((argmax_predictions == plaintext_argmax_predictions).float().mean().item())
threshold_match_vs_plaintext = None
if threshold_predictions is not None and plaintext_threshold_predictions is not None:
    threshold_match_vs_plaintext = float((threshold_predictions == plaintext_threshold_predictions).float().mean().item())

LINK_RE = re.compile(
    r"Link details: total send bytes (?P<send>\d+), recv bytes (?P<recv>\d+), "
    r"send actions (?P<send_actions>\d+), recv actions (?P<recv_actions>\d+)"
)

def latest_nonzero_link(path: Path):
    if not path.exists():
        return None
    latest = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = LINK_RE.search(line)
        if not match:
            continue
        item = {
            "log_path": str(path),
            "send_bytes": int(match.group("send")),
            "recv_bytes": int(match.group("recv")),
            "send_actions": int(match.group("send_actions")),
            "recv_actions": int(match.group("recv_actions")),
        }
        if item["send_bytes"] or item["recv_bytes"] or item["send_actions"] or item["recv_actions"]:
            latest = item
    return latest

node_link_details = []
for log_path in sorted(spu_log_dir.glob("node_*.log")):
    item = latest_nonzero_link(log_path)
    if item is not None:
        node_link_details.append(item)

aggregate_send = sum(item["send_bytes"] for item in node_link_details)
aggregate_recv = sum(item["recv_bytes"] for item in node_link_details)
aggregate_total = aggregate_send + aggregate_recv

result = {
    "manifest_type": "transshield_e2e_approx_eval_metrics_v0",
    "image_list": str(Path(share_manifest.get("source_image_list", ""))) if share_manifest.get("source_image_list") else None,
    "plaintext_reference_json": str(Path(sys.argv[2])),
    "candidate_json": str(Path(sys.argv[4])),
    "candidate_pt": str(candidate_pt),
    "sample_count": count,
    "target_count": int(targets.numel()),
    "comparison_scope": "same_image_list_same_targets_original_plaintext_vs_e2e_approx_spu",
    "original_plaintext_same_subset_argmax_accuracy": plaintext_argmax_acc,
    "original_plaintext_same_subset_threshold_accuracy": plaintext_threshold_acc,
    "e2e_argmax_accuracy": argmax_acc,
    "e2e_threshold_accuracy": threshold_acc,
    "argmax_accuracy_gap_e2e_minus_plaintext_pp": argmax_acc - plaintext_argmax_acc,
    "threshold_accuracy_gap_e2e_minus_plaintext_pp": (
        None if threshold_acc is None or plaintext_threshold_acc is None else threshold_acc - plaintext_threshold_acc
    ),
    "prediction_match_vs_original_plaintext": {
        "argmax_match_ratio": argmax_match_vs_plaintext,
        "threshold_match_ratio": threshold_match_vs_plaintext,
    },
    "original_plaintext_full_val_reference": {
        "argmax_accuracy": 93.70229244232178,
        "threshold_accuracy": 94.08397078514099,
        "note": "full-val reference is retained only for context; primary gap fields above use the same eval subset",
    },
    "finite_logits": bool(torch.isfinite(logits).all().item()),
    "e2e_elapsed_sec": candidate_json.get("elapsed_sec"),
    "e2e_communication_from_spu_node_logs": {
        "status": "available" if node_link_details else "missing",
        "node_latest_nonzero_link_details": node_link_details,
        "aggregate_send_bytes": aggregate_send if node_link_details else None,
        "aggregate_recv_bytes": aggregate_recv if node_link_details else None,
        "aggregate_total_bytes": aggregate_total if node_link_details else None,
        "scope_note": "Parsed from latest nonzero Link details in logs/spu_nodes/node_*.log after the e2e run.",
    },
    "original_plaintext_communication": {
        "status": "not_applicable",
        "total_bytes": 0,
        "note": "Plaintext local reference has no 2PC/SPU communication.",
    },
    "privacy_fields": {
        "input_pt": candidate_json.get("input_pt"),
        "input_mode": (candidate_json.get("spu") or {}).get("input_mode"),
        "host_plaintext_pixel_values_materialized": (candidate_json.get("spu") or {}).get("host_plaintext_pixel_values_materialized"),
        "host_private_share_tensors_loaded": (candidate_json.get("spu") or {}).get("host_private_share_tensors_loaded"),
        "private_input_paths_redacted": (candidate_json.get("spu") or {}).get("private_input_paths_redacted"),
    },
    "scope_note": (
        "Primary metrics compare original plaintext and e2e approximate SPU on the same selected images and targets. "
        "Increase E2E_EVAL_MAX_SAMPLES for a more stable estimate; keep E2E_SPU_BATCH_SIZE=1."
    ),
}
output_json.parent.mkdir(parents=True, exist_ok=True)
output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(result, indent=2, sort_keys=True))
PY

echo "[e2e-approx-eval] metrics_json=$E2E_EVAL_METRICS_JSON"
