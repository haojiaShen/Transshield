#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "$REPO_ROOT" || exit 1

SOURCE_RUN="${SOURCE_RUN:-dual_private_vit_blockwise_sample8_20260429_134528}"
SOURCE_SHARE_MANIFEST="${SOURCE_SHARE_MANIFEST:-$REPO_ROOT/artifacts/server_pipeline_run/${SOURCE_RUN}/e2e_secure_poc/client_pixel_values_debug_share_manifest.json}"
CALIB_JSON="${CALIB_JSON:-$REPO_ROOT/artifacts/server_pipeline_run/dual_private_vit_publiccalib32_20260429_001224/e2e_secure_poc/e2e_public_layer_norm_calibration_depth12_uniform_fixed_square_noclip_calib32.json}"
BASE_CONFIG_PATH="${BASE_CONFIG_PATH:-$REPO_ROOT/configs/openbumblebee/2pc.json}"

RUN_NAME="${RUN_NAME:-dual_private_vit_blockwise_sample8_parallel2_no_lsb}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/artifacts/server_pipeline_run/$RUN_NAME}"
WORKER0_INDICES="${WORKER0_INDICES:-0 2 4 6}"
WORKER1_INDICES="${WORKER1_INDICES:-1 3 5 7}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
OUTLIER_ABS_THRESHOLD="${OUTLIER_ABS_THRESHOLD:-10}"
RUN_TIMEOUT_SEC="${RUN_TIMEOUT_SEC:-1500}"

mkdir -p "$RUN_ROOT"

echo "run_root=$RUN_ROOT"
echo "source_share_manifest=$SOURCE_SHARE_MANIFEST"
echo "calib_json=$CALIB_JSON"
ls -lh "$SOURCE_SHARE_MANIFEST" || exit 1
ls -lh "$CALIB_JSON" || exit 1

"$PYTHON_BIN" - "$RUN_ROOT" "$BASE_CONFIG_PATH" <<'PY'
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1]).resolve()
base_config = Path(sys.argv[2]).resolve()

for worker_id in (0, 1):
    worker_dir = run_root / f"worker{worker_id}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    payload = json.load(open(base_config, encoding="utf-8"))
    spu_config = payload["devices"]["SPU"]["config"]
    runtime_config = spu_config.setdefault("runtime_config", {})
    runtime_config.setdefault("cheetah_2pc_config", {})["enable_mul_lsb_error"] = False
    spu_config["experimental_data_folder"] = [
        f"/tmp/transshield_spu_{run_root.name}_worker{worker_id}_0",
        f"/tmp/transshield_spu_{run_root.name}_worker{worker_id}_1",
    ]
    for name in ("2pc.json", "2pc.template.json"):
        path = worker_dir / name
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {path}")
PY

run_worker() {
  local worker_id="$1"
  local indices="$2"

  export CONFIG_PATH="$RUN_ROOT/worker${worker_id}/2pc.json"
  export SPU_RUNTIME_TEMPLATE_PATH="$RUN_ROOT/worker${worker_id}/2pc.template.json"
  export SPU_RUNTIME_LOG_DIR="logs/spu_nodes_${RUN_NAME}_worker${worker_id}"
  export SPU_RUNTIME_STATE_JSON="logs/spu_runtime_ports_${RUN_NAME}_worker${worker_id}.json"

  export E2E_RUN_MAX_SAMPLES=1
  export E2E_STATIC_DEPTH_LIMIT=12
  export E2E_SPU_BATCH_SIZE=1
  export E2E_PARTY_LOCAL_SHARE_LOAD=1
  export E2E_REDACT_PRIVATE_INPUT_PATHS=1
  export E2E_SPU_PARAMS_MODE=secret_blockwise_stage
  export E2E_SPU_LAYER_NORM_POLICY=public_calibrated
  export E2E_SPU_LAYER_NORM_CALIBRATION_JSON="$CALIB_JSON"
  export E2E_SPU_ATTENTION_POLICY=uniform
  export E2E_SPU_ACTIVATION_OVERRIDE=fixed_square
  export E2E_SPU_ACTIVATION_CLIP_VALUE=0
  export SPU_DISABLE_COLOCATED_OPTIMIZATION=1
  export SPU_RUNTIME_REUSE=0
  export SPU_RUNTIME_WARMUP_ATTEMPTS=2
  export SPU_RUNTIME_STOP_WAIT_SEC=3
  export SPU_RUNTIME_STARTUP_TIMEOUT_SEC=60

  for idx in $indices; do
    export IDX="$idx"
    echo "===== worker ${worker_id} idx ${idx} prepare slice ====="

    local slice_dir="$RUN_ROOT/worker${worker_id}/idx${idx}_slice/e2e_secure_poc"
    mkdir -p "$slice_dir"

    if [[ ! -f "$slice_dir/client_pixel_values_debug_share_public_manifest.json" ]]; then
      "$PYTHON_BIN" tools/transshield_slice_debug_shares.py \
        --share-manifest-json "$SOURCE_SHARE_MANIFEST" \
        --start-index "$idx" \
        --end-index "$((idx + 1))" \
        --output-prefix "$slice_dir/client_pixel_values_debug_share" \
        --output-json "$slice_dir/client_pixel_values_debug_share_manifest.json" \
        --output-public-json "$slice_dir/client_pixel_values_debug_share_public_manifest.json" \
        --output-party-manifest-dir "$slice_dir/client_pixel_values_debug_share_party_manifests"
    fi

    export E2E_INPUT_SHARE_PUBLIC_MANIFEST_JSON="$slice_dir/client_pixel_values_debug_share_public_manifest.json"
    export E2E_INPUT_P1_SHARE_MANIFEST_JSON="$slice_dir/client_pixel_values_debug_share_party_manifests/p1_share_manifest.json"
    export E2E_INPUT_P2_SHARE_MANIFEST_JSON="$slice_dir/client_pixel_values_debug_share_party_manifests/p2_share_manifest.json"

    local accepted="$RUN_ROOT/idx${idx}_accepted.json"
    if [[ -s "$accepted" ]]; then
      echo "===== worker ${worker_id} idx ${idx} already accepted ====="
      continue
    fi

    for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
      echo "===== worker ${worker_id} run idx ${idx} attempt ${attempt} ====="

      "$PYTHON_BIN" tools/transshield_spu_runtime_setup.py stop \
        --config "$CONFIG_PATH" \
        --state-json "$SPU_RUNTIME_STATE_JSON" >/dev/null 2>&1 || true

      export RUN_NAME="sample8_parallel2_w${worker_id}_idx${idx}_attempt${attempt}"
      export E2E_RUN_DIR="$RUN_ROOT/worker${worker_id}/idx${idx}_attempt${attempt}/e2e_secure_poc"
      mkdir -p "$E2E_RUN_DIR"

      export RESULT="$E2E_RUN_DIR/e2e_spu_idx${idx}_secret_blockwise_parallel2_attempt${attempt}.json"
      export E2E_CANDIDATE_JSON="$RESULT"
      export E2E_CANDIDATE_PT="$E2E_RUN_DIR/e2e_spu_idx${idx}_secret_blockwise_parallel2_attempt${attempt}.pt"

      if [[ ! -s "$RESULT" ]]; then
        timeout "${RUN_TIMEOUT_SEC}s" bash artifacts/server_inference_friendly_pack/run_e2e_secure_whole_forward.sh spu > "$E2E_RUN_DIR/stdout.log" 2>&1
        echo "run_rc=$?" >> "$E2E_RUN_DIR/stdout.log"
      fi

      "$PYTHON_BIN" tools/transshield_spu_runtime_setup.py stop \
        --config "$CONFIG_PATH" \
        --state-json "$SPU_RUNTIME_STATE_JSON" >/dev/null 2>&1 || true

      if [[ ! -s "$RESULT" ]]; then
        echo "worker ${worker_id} idx ${idx} attempt ${attempt}: missing result"
        grep -nE 'run_rc=|Traceback|RuntimeError|UNAVAILABLE|Socket closed|timeout|Get data timeout|ErrorCode' "$E2E_RUN_DIR/stdout.log" || true
        continue
      fi

      "$PYTHON_BIN" - <<'PY'
import json
import os
import shutil
from pathlib import Path

idx = os.environ["IDX"]
result = Path(os.environ["RESULT"])
accepted = Path(os.environ["RUN_ROOT"]) / f"idx{idx}_accepted.json"
threshold = float(os.environ["OUTLIER_ABS_THRESHOLD"])

d = json.load(open(result, encoding="utf-8"))
p = d["prediction_preview"]
logits = p["logits"][0]
max_abs = max(abs(float(x)) for x in logits)
outlier = max_abs > threshold
print("result=", result)
print("elapsed=", d["elapsed_sec"])
print("logits=", p["logits"])
print("probabilities=", p["probabilities"])
print("argmax=", p["argmax_predictions"])
print("threshold=", p["threshold_predictions"])
print("max_abs_logit=", max_abs)
print("outlier=", outlier)
if not outlier:
    shutil.copy2(result, accepted)
    print("accepted=", accepted)
PY

      if [[ -s "$accepted" ]]; then
        break
      fi
    done
  done
}

export RUN_ROOT OUTLIER_ABS_THRESHOLD
run_worker 0 "$WORKER0_INDICES" > "$RUN_ROOT/worker0.log" 2>&1 &
pid0=$!
run_worker 1 "$WORKER1_INDICES" > "$RUN_ROOT/worker1.log" 2>&1 &
pid1=$!

echo "worker0_pid=$pid0 log=$RUN_ROOT/worker0.log"
echo "worker1_pid=$pid1 log=$RUN_ROOT/worker1.log"

wait "$pid0"
rc0=$?
wait "$pid1"
rc1=$?
echo "worker0_rc=$rc0"
echo "worker1_rc=$rc1"

export SUMMARY_JSON="$RUN_ROOT/sample8_parallel2_no_lsb_summary.json"
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
items = []
missing = []
for idx in range(8):
    path = root / f"idx{idx}_accepted.json"
    if not path.exists():
        missing.append(idx)
        continue
    d = json.load(open(path, encoding="utf-8"))
    p = d["prediction_preview"]
    logits = p["logits"][0]
    max_abs = max(abs(float(x)) for x in logits)
    items.append({
        "idx": idx,
        "path": str(path),
        "elapsed_sec": d["elapsed_sec"],
        "logits": logits,
        "probabilities": p["probabilities"][0],
        "argmax": p["argmax_predictions"][0],
        "threshold": p["threshold_predictions"][0],
        "max_abs_logit": max_abs,
        "outlier_abs_gt_10": max_abs > 10.0,
        "finite_logits": d["finite_logits"],
        "spu_params_mode": d["spu"]["spu_params_mode"],
        "spu_layer_norm_policy": d["spu"]["spu_layer_norm_policy"],
        "config": d["spu"]["config"],
    })

summary = {
    "manifest_type": "transshield_sample8_parallel2_no_lsb_summary_v0",
    "source_share_manifest": os.environ["SOURCE_SHARE_MANIFEST"],
    "calibration_json": os.environ["CALIB_JSON"],
    "worker0_indices": os.environ["WORKER0_INDICES"].split(),
    "worker1_indices": os.environ["WORKER1_INDICES"].split(),
    "sample_count": len(items),
    "missing": missing,
    "outlier_count": sum(x["outlier_abs_gt_10"] for x in items),
    "items": items,
    "argmax_predictions": [x["argmax"] for x in items],
    "threshold_predictions": [x["threshold"] for x in items],
    "sum_elapsed_sec": sum(x["elapsed_sec"] for x in items),
}
out = Path(os.environ["SUMMARY_JSON"])
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
print("summary_json=", out)
PY
