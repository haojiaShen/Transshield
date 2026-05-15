#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "$REPO_ROOT" || exit 1

SOURCE_SHARE_MANIFEST="${SOURCE_SHARE_MANIFEST:?SOURCE_SHARE_MANIFEST is required}"
CALIB_JSON="${CALIB_JSON:?CALIB_JSON is required}"
OUTPUT_CALIB_JSON="${OUTPUT_CALIB_JSON:-${E2E_OUTPUT_CALIBRATION_JSON:-}}"
BASE_CONFIG_PATH="${BASE_CONFIG_PATH:-$REPO_ROOT/configs/openbumblebee/2pc.json}"
STATIC_DEPTH_LIMIT="${STATIC_DEPTH_LIMIT:-${E2E_STATIC_DEPTH_LIMIT:-12}}"

RUN_NAME="${RUN_NAME:-e2e_secret_isolated_eval_$(date +%Y%m%d_%H%M%S)}"
BASE_RUN_NAME="$RUN_NAME"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/artifacts/server_pipeline_run/$RUN_NAME}"
INDEX_LIST="${INDEX_LIST:-}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
OUTLIER_ABS_THRESHOLD="${OUTLIER_ABS_THRESHOLD:-10}"
RUN_TIMEOUT_SEC="${RUN_TIMEOUT_SEC:-900}"
RUN_TIMEOUT_KILL_SEC="${RUN_TIMEOUT_KILL_SEC:-30}"
SUMMARY_JSON="${SUMMARY_JSON:-$RUN_ROOT/secret_isolated_eval_summary.json}"
RUN_COMPLETED=0

mkdir -p "$RUN_ROOT"

echo "run_root=$RUN_ROOT"
echo "source_share_manifest=$SOURCE_SHARE_MANIFEST"
echo "calib_json=$CALIB_JSON"
echo "output_calib_json=$OUTPUT_CALIB_JSON"
echo "static_depth_limit=$STATIC_DEPTH_LIMIT"
echo "max_attempts=$MAX_ATTEMPTS"
echo "run_timeout_sec=$RUN_TIMEOUT_SEC"
echo "outlier_abs_threshold=$OUTLIER_ABS_THRESHOLD"

if [[ ! -f "$SOURCE_SHARE_MANIFEST" ]]; then
  echo "[secret-isolated-eval] missing SOURCE_SHARE_MANIFEST: $SOURCE_SHARE_MANIFEST" >&2
  exit 2
fi
if [[ ! -f "$CALIB_JSON" ]]; then
  echo "[secret-isolated-eval] missing CALIB_JSON: $CALIB_JSON" >&2
  exit 2
fi
if [[ -n "$OUTPUT_CALIB_JSON" && ! -f "$OUTPUT_CALIB_JSON" ]]; then
  echo "[secret-isolated-eval] missing OUTPUT_CALIB_JSON: $OUTPUT_CALIB_JSON" >&2
  exit 2
fi

if [[ -z "$INDEX_LIST" ]]; then
  INDEX_LIST="$("$PYTHON_BIN" - "$SOURCE_SHARE_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
count = int(payload.get("sample_count") or len(payload.get("sample_ids") or []))
print(" ".join(str(i) for i in range(count)))
PY
)"
fi
echo "index_list=$INDEX_LIST"

"$PYTHON_BIN" - "$BASE_CONFIG_PATH" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
value = (
    payload["devices"]["SPU"]["config"]["runtime_config"]
    .get("cheetah_2pc_config", {})
    .get("enable_mul_lsb_error")
)
print(f"enable_mul_lsb_error={value}")
if value is not False:
    raise SystemExit("ERROR: expected cheetah enable_mul_lsb_error=false for stable secret eval")
PY
if [[ "$?" != "0" ]]; then
  exit 2
fi

stop_runtime() {
  "$PYTHON_BIN" tools/transshield_spu_runtime_setup.py stop \
    --config "$BASE_CONFIG_PATH" \
    --state-json logs/spu_runtime_ports.json \
    --stop-wait-sec 1 >/dev/null 2>&1 || true
}

write_summary_snapshot() {
  local complete_flag="${1:-0}"
  "$PYTHON_BIN" - "$RUN_ROOT" "$SOURCE_SHARE_MANIFEST" "$CALIB_JSON" "$OUTPUT_CALIB_JSON" "$SUMMARY_JSON" "$OUTLIER_ABS_THRESHOLD" "$INDEX_LIST" "$complete_flag" <<'PY'
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
share_manifest_path = Path(sys.argv[2])
calib_json = sys.argv[3]
output_calib_json = sys.argv[4] or None
summary_json = Path(sys.argv[5])
guard = float(sys.argv[6])
requested_indices = [int(value) for value in sys.argv[7].split()]
complete_flag = bool(int(sys.argv[8]))

manifest = json.loads(share_manifest_path.read_text(encoding="utf-8"))
targets = manifest.get("targets")
sample_count = int(manifest.get("sample_count") or len(manifest.get("sample_ids") or []))

accepted = []
unstable = []
pending = []
attempts = []

for idx in requested_indices:
    accepted_path = run_root / f"idx{idx}_accepted.json"
    idx_attempts = []
    attempt_dirs = sorted(run_root.glob(f"idx{idx}_attempt*/e2e_secure_poc"))
    slice_dir = run_root / f"idx{idx}_slice/e2e_secure_poc"
    for status_path in sorted(run_root.glob(f"idx{idx}_attempt*/e2e_secure_poc/attempt_status.json")):
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["status_json"] = str(status_path)
        idx_attempts.append(status)
        attempts.append(status)
    if accepted_path.exists():
        payload = json.loads(accepted_path.read_text(encoding="utf-8"))
        preview = payload.get("prediction_preview", {})
        pred = (preview.get("argmax_predictions") or [None])[0]
        target = None if targets is None else int(targets[idx])
        raw = payload.get("raw_logits_before_output_calibration") or payload.get("logits") or {}
        max_abs = None
        if isinstance(raw, dict) and raw.get("min") is not None and raw.get("max") is not None:
            max_abs = max(abs(float(raw["min"])), abs(float(raw["max"])))
        accepted.append(
            {
                "idx": idx,
                "target": target,
                "path": str(accepted_path),
                "elapsed_sec": payload.get("elapsed_sec"),
                "raw_logits_before_output_calibration": payload.get("raw_logits_before_output_calibration"),
                "logits": (preview.get("logits") or [None])[0],
                "probabilities": (preview.get("probabilities") or [None])[0],
                "argmax": pred,
                "threshold": (preview.get("threshold_predictions") or [None])[0],
                "correct_argmax": None if target is None or pred is None else int(pred) == target,
                "max_abs_raw_or_logit": max_abs,
                "spu_params_mode": payload.get("spu", {}).get("spu_params_mode"),
                "spu_layer_norm_policy": payload.get("spu", {}).get("spu_layer_norm_policy"),
            }
        )
    elif idx_attempts:
        unstable.append(
            {
                "idx": idx,
                "target": None if targets is None else int(targets[idx]),
                "attempt_count": len(idx_attempts),
                "attempts": idx_attempts,
            }
        )
    else:
        pending.append(
            {
                "idx": idx,
                "target": None if targets is None else int(targets[idx]),
                "slice_prepared": slice_dir.exists(),
                "attempt_dir_count": len(attempt_dirs),
            }
        )

correct = [item["correct_argmax"] for item in accepted if item["correct_argmax"] is not None]
summary = {
    "manifest_type": "transshield_secret_isolated_eval_guarded_summary_v0",
    "complete": complete_flag and not pending,
    "source_share_manifest": str(share_manifest_path),
    "layer_norm_calibration_json": calib_json,
    "output_calibration_json": output_calib_json,
    "outlier_rule": f"reject if max_abs_raw_or_logit > {guard}",
    "source_sample_count": sample_count,
    "requested_indices": requested_indices,
    "sample_count": len(requested_indices),
    "accepted_count": len(accepted),
    "unstable_count": len(unstable),
    "pending_count": len(pending),
    "accepted_accuracy": None if not correct else sum(bool(value) for value in correct) / len(correct),
    "accepted_items": accepted,
    "unstable_items": unstable,
    "pending_items": pending,
    "attempts": attempts,
    "sum_accepted_elapsed_sec": sum(float(item.get("elapsed_sec") or 0.0) for item in accepted),
}
summary_json.parent.mkdir(parents=True, exist_ok=True)
summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
print("summary_json=", summary_json)
PY
}

cleanup_on_exit() {
  local rc=$?
  trap - EXIT INT TERM
  stop_runtime
  write_summary_snapshot "$RUN_COMPLETED" >/dev/null 2>&1 || true
  return "$rc"
}

trap cleanup_on_exit EXIT
trap 'exit 130' INT TERM

write_attempt_status() {
  local idx="$1"
  local attempt="$2"
  local result_json="$3"
  local run_rc="$4"
  local status_json="$5"

  "$PYTHON_BIN" - "$idx" "$attempt" "$result_json" "$run_rc" "$status_json" "$OUTLIER_ABS_THRESHOLD" <<'PY'
import json
import math
import shutil
import sys
from pathlib import Path

idx = int(sys.argv[1])
attempt = int(sys.argv[2])
result = Path(sys.argv[3])
run_rc = int(sys.argv[4])
status_json = Path(sys.argv[5])
guard = float(sys.argv[6])

status = {
    "idx": idx,
    "attempt": attempt,
    "result_json": str(result),
    "run_rc": run_rc,
    "exists": result.exists() and result.stat().st_size > 0,
    "accepted": False,
    "outlier_abs_gt_threshold": None,
    "max_abs_raw_or_logit": None,
    "error": None,
}

try:
    if run_rc != 0:
        status["error"] = f"run_rc={run_rc}"
    elif not status["exists"]:
        status["error"] = "missing_result_json"
    else:
        payload = json.loads(result.read_text(encoding="utf-8"))
        raw = payload.get("raw_logits_before_output_calibration") or payload.get("logits") or {}
        values = []
        if isinstance(raw, dict):
            for key in ("min", "max"):
                if raw.get(key) is not None:
                    values.append(abs(float(raw[key])))
        if not values:
            preview = payload.get("prediction_preview", {})
            for row in preview.get("logits") or []:
                values.extend(abs(float(value)) for value in row)
        max_abs = max(values) if values else math.inf
        finite = bool(payload.get("finite_logits", True)) and math.isfinite(max_abs)
        outlier = (not finite) or max_abs > guard
        status.update(
            {
                "elapsed_sec": payload.get("elapsed_sec"),
                "finite_logits": finite,
                "max_abs_raw_or_logit": max_abs,
                "outlier_abs_gt_threshold": outlier,
                "logits": (payload.get("prediction_preview", {}).get("logits") or [None])[0],
                "probabilities": (payload.get("prediction_preview", {}).get("probabilities") or [None])[0],
                "argmax": (payload.get("prediction_preview", {}).get("argmax_predictions") or [None])[0],
                "threshold": (payload.get("prediction_preview", {}).get("threshold_predictions") or [None])[0],
                "spu_params_mode": payload.get("spu", {}).get("spu_params_mode"),
                "spu_layer_norm_policy": payload.get("spu", {}).get("spu_layer_norm_policy"),
                "output_calibration": payload.get("output_calibration"),
            }
        )
        if not outlier:
            accepted = result.parents[2] / f"idx{idx}_accepted.json"
            shutil.copy2(result, accepted)
            status["accepted_json"] = str(accepted)
            status["accepted"] = True
except Exception as exc:
    status["error"] = repr(exc)

status_json.parent.mkdir(parents=True, exist_ok=True)
status_json.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(status, indent=2, sort_keys=True))
raise SystemExit(0 if status["accepted"] else 1)
PY
}

for idx in $INDEX_LIST; do
  echo "===== idx ${idx} prepare slice ====="
  slice_dir="$RUN_ROOT/idx${idx}_slice/e2e_secure_poc"
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

  accepted_json="$RUN_ROOT/idx${idx}_accepted.json"
  if [[ -s "$accepted_json" ]]; then
    echo "===== idx ${idx} already accepted: $accepted_json ====="
    continue
  fi

  for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    echo "===== idx ${idx} attempt ${attempt}/${MAX_ATTEMPTS} ====="
    attempt_dir="$RUN_ROOT/idx${idx}_attempt${attempt}/e2e_secure_poc"
    mkdir -p "$attempt_dir"

    stop_runtime

    export CONFIG_PATH="$BASE_CONFIG_PATH"
    export SPU_RUNTIME_TEMPLATE_PATH="${SPU_RUNTIME_TEMPLATE_PATH:-configs/openbumblebee/2pc.template.json}"
    export SPU_RUNTIME_REUSE=0
    export SPU_RUNTIME_WARMUP_ATTEMPTS="${SPU_RUNTIME_WARMUP_ATTEMPTS:-1}"
    export SPU_RUNTIME_STARTUP_TIMEOUT_SEC="${SPU_RUNTIME_STARTUP_TIMEOUT_SEC:-60}"
    export SPU_RUNTIME_STOP_WAIT_SEC="${SPU_RUNTIME_STOP_WAIT_SEC:-1}"

    export RUN_NAME="${BASE_RUN_NAME}_idx${idx}_attempt${attempt}"
    export E2E_RUN_DIR="$attempt_dir"
    export E2E_INPUT_SHARE_PUBLIC_MANIFEST_JSON="$slice_dir/client_pixel_values_debug_share_public_manifest.json"
    export E2E_INPUT_P1_SHARE_MANIFEST_JSON="$slice_dir/client_pixel_values_debug_share_party_manifests/p1_share_manifest.json"
    export E2E_INPUT_P2_SHARE_MANIFEST_JSON="$slice_dir/client_pixel_values_debug_share_party_manifests/p2_share_manifest.json"
    export E2E_PARTY_LOCAL_SHARE_LOAD=1
    export E2E_REDACT_PRIVATE_INPUT_PATHS=1
    export E2E_RUN_MAX_SAMPLES=1
    export E2E_STATIC_DEPTH_LIMIT="$STATIC_DEPTH_LIMIT"
    export E2E_SPU_BATCH_SIZE=1
    export E2E_SPU_BLOCK_CHUNK_SIZE=0
    export E2E_SPU_LAYER_NORM_CHUNK_SIZE=0
    export E2E_SPU_PARAMS_MODE=secret_blockwise_stage
    export E2E_SPU_LAYER_NORM_POLICY=public_calibrated
    export E2E_SPU_LAYER_NORM_CALIBRATION_JSON="$CALIB_JSON"
    export E2E_SPU_ATTENTION_POLICY=uniform
    export E2E_SPU_ACTIVATION_OVERRIDE=fixed_square
    export E2E_SPU_ACTIVATION_CLIP_VALUE=0
    export E2E_OUTPUT_CALIBRATION_JSON="$OUTPUT_CALIB_JSON"
    export E2E_CANDIDATE_JSON="$attempt_dir/e2e_spu_idx${idx}_secret_isolated_attempt${attempt}.json"
    export E2E_CANDIDATE_PT="$attempt_dir/e2e_spu_idx${idx}_secret_isolated_attempt${attempt}.pt"

    if command -v timeout >/dev/null 2>&1; then
      timeout -k "${RUN_TIMEOUT_KILL_SEC}s" "${RUN_TIMEOUT_SEC}s" \
        bash artifacts/server_inference_friendly_pack/run_e2e_secure_whole_forward.sh spu \
        > "$attempt_dir/stdout.log" 2>&1
      run_rc=$?
    else
      bash artifacts/server_inference_friendly_pack/run_e2e_secure_whole_forward.sh spu \
        > "$attempt_dir/stdout.log" 2>&1
      run_rc=$?
    fi
    echo "run_rc=$run_rc" >> "$attempt_dir/stdout.log"

    stop_runtime

    status_json="$attempt_dir/attempt_status.json"
    if write_attempt_status "$idx" "$attempt" "$E2E_CANDIDATE_JSON" "$run_rc" "$status_json"; then
      echo "accepted idx=${idx} attempt=${attempt}"
      break
    fi

    echo "rejected idx=${idx} attempt=${attempt}; continue"
  done

  write_summary_snapshot 0 || true
done

RUN_COMPLETED=1
write_summary_snapshot 1
