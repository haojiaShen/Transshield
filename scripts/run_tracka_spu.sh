#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="${REPO_ROOT:-$DEFAULT_REPO_ROOT}"
TRANSHIELD_TMP_ROOT="${TRANSHIELD_TMP_ROOT:-/data/wyb/bazel_clean/tmp}"

usage() {
  cat <<'EOF' >&2
Usage:
  bash scripts/run_tracka_spu.sh followup
  bash scripts/run_tracka_spu.sh dual-profile
EOF
  exit 1
}

run_python() {
  "$PYTHON_BIN" "$@"
}

require_dir() {
  local dir_path="$1"
  local label="$2"
  if [[ ! -d "$dir_path" ]]; then
    echo "Missing $label: $dir_path" >&2
    exit 1
  fi
}

require_file() {
  local file_path="$1"
  local label="$2"
  if [[ ! -f "$file_path" ]]; then
    echo "Missing $label: $file_path" >&2
    exit 1
  fi
}

choose_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    printf '%s\n' "$PYTHON_BIN"
    return 0
  fi
  local candidates=(
    "$REPO_ROOT/../conda_envs/transshield/bin/python"
    "/data/wyb/conda_envs/transshield/bin/python"
    "/home/yclcg/miniconda3/envs/transshield/bin/python"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi
  echo "No usable python interpreter found." >&2
  return 1
}

default_val_data_path() {
  local candidates=(
    "/data/wyb/pneumoniamnist_imagefolder_subset/val"
    "/home/yclcg/DynamicViT_exp_square/data/pneumoniamnist_imagefolder_subset/val"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -d "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

materialize_threshold_json_if_needed() {
  local threshold_json="$BUNDLE_DIR/threshold_best.json"
  local manifest_json="$BUNDLE_DIR/manifest.json"
  if [[ -f "$threshold_json" ]]; then
    return 0
  fi
  if [[ ! -f "$manifest_json" ]]; then
    echo "Missing threshold_best.json and manifest.json: cannot recover threshold metadata." >&2
    return 1
  fi
  run_python - <<'PY'
import json
import os
from pathlib import Path

bundle_dir = Path(os.environ["BUNDLE_DIR"]).resolve()
manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
threshold_metrics = ((manifest.get("primary") or {}).get("threshold_metrics") or {})
if not threshold_metrics:
    raise SystemExit("manifest.json does not contain primary.threshold_metrics; cannot regenerate threshold_best.json")
output_path = bundle_dir / "threshold_best.json"
if output_path.is_symlink() and not output_path.exists():
    output_path.unlink()
output_path.write_text(json.dumps(threshold_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[repair] regenerated {output_path} from manifest.json")
PY
}

setup_followup_defaults() {
  PYTHON_BIN="$(choose_python)"
  VAL_DATA_PATH="${VAL_DATA_PATH:-$(default_val_data_path || true)}"
  BUNDLE_DIR="${BUNDLE_DIR:-$REPO_ROOT/artifacts/frozen_bundle_verified_tracka_lr3e5_20260414}"
  RUN_NAME="${RUN_NAME:-tracka_lr3e5_timm_spu_followup_$(date +%Y%m%d_%H%M%S)}"
  SECURE_RUN_DIR="${SECURE_RUN_DIR:-$REPO_ROOT/artifacts/server_pipeline_run/$RUN_NAME}"
  CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/openbumblebee/2pc.json}"
  SECURE_EXPORT_DEVICE="${SECURE_EXPORT_DEVICE:-cuda}"
  SECURE_EXPORT_BATCH_SIZE="${SECURE_EXPORT_BATCH_SIZE:-32}"
  SECURE_MAX_SAMPLES="${SECURE_MAX_SAMPLES:-0}"
  SECURE_REPLAY_BATCH_SIZE="${SECURE_REPLAY_BATCH_SIZE:-32}"
  SECURE_REPLAY_NUM_WORKERS="${SECURE_REPLAY_NUM_WORKERS:-0}"
  SPU_DISABLE_COLOCATED_OPTIMIZATION="${SPU_DISABLE_COLOCATED_OPTIMIZATION:-0}"
}

setup_temp_env() {
  mkdir -p "$TRANSHIELD_TMP_ROOT"
  export TMPDIR="$TRANSHIELD_TMP_ROOT"
  export TEMP="$TRANSHIELD_TMP_ROOT"
  export TMP="$TRANSHIELD_TMP_ROOT"
  export TEST_TMPDIR="$TRANSHIELD_TMP_ROOT"
}

validate_followup_prereqs() {
  require_dir "$BUNDLE_DIR" "bundle directory"
  require_file "$CONFIG_PATH" "SPU config"
  require_dir "$VAL_DATA_PATH" "validation dataset directory"

  export BUNDLE_DIR
  materialize_threshold_json_if_needed
  require_file "$BUNDLE_DIR/modified_plaintext_model_state_dict.pth" "model state dict in bundle"

  run_python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("spu") else "Missing Python module: spu")
PY
}

export_runtime_inputs() {
  env \
    PYTHON_BIN="$PYTHON_BIN" \
    REPO_ROOT="$REPO_ROOT" \
    RUN_NAME="$RUN_NAME" \
    BUNDLE_DIR="$BUNDLE_DIR" \
    VAL_DATA_PATH="$VAL_DATA_PATH" \
    SECURE_RUN_DIR="$SECURE_RUN_DIR" \
    SECURE_MAX_SAMPLES="$SECURE_MAX_SAMPLES" \
    SECURE_EXPORT_DEVICE="$SECURE_EXPORT_DEVICE" \
    SECURE_EXPORT_BATCH_SIZE="$SECURE_EXPORT_BATCH_SIZE" \
    SECURE_EXPORT_NUM_WORKERS=0 \
    bash "$REPO_ROOT/artifacts/server_inference_friendly_pack/run_secure_export_inputs.sh"
}

start_spu_runtime() {
  local runtime_args=(
    tools/transshield_spu_runtime_setup.py
    start
    --config "$CONFIG_PATH"
    --template "$REPO_ROOT/configs/openbumblebee/2pc.template.json"
    --backup
    --restart
    --remove-unsupported-cheetah-fields
    --log-dir "$REPO_ROOT/logs/spu_nodes"
    --state-json "$REPO_ROOT/logs/spu_runtime_ports.json"
  )
  if [[ "$SPU_DISABLE_COLOCATED_OPTIMIZATION" == "1" ]]; then
    runtime_args+=(--disable-colocated-optimization)
  fi
  run_python "${runtime_args[@]}"
}

run_pipeline_followup() {
  run_python tools/transshield_openbumblebee_pipeline.py run \
    --runtime spu \
    --bundle-dir "$BUNDLE_DIR" \
    --config "$CONFIG_PATH" \
    --output-dir "$SECURE_RUN_DIR" \
    --eval-replay \
    --eval-max-samples "$SECURE_MAX_SAMPLES" \
    --eval-data-path "$VAL_DATA_PATH"

  run_python tools/transshield_openbumblebee_pipeline.py verify \
    --output-dir "$SECURE_RUN_DIR"

  run_python tools/transshield_openbumblebee_pipeline.py replay \
    --output-dir "$SECURE_RUN_DIR" \
    --bundle-dir "$BUNDLE_DIR" \
    --device cpu \
    --batch-size "$SECURE_REPLAY_BATCH_SIZE" \
    --num-workers "$SECURE_REPLAY_NUM_WORKERS" \
    --max-samples "$SECURE_MAX_SAMPLES" \
    --enable-model-replay

  run_python tools/transshield_plaintext_secure_score_compare.py \
    --bundle-dir "$BUNDLE_DIR" \
    --secure-replay-json "$SECURE_RUN_DIR/pipeline_inference_replay_summary.json" \
    --device cpu \
    --batch-size 16 \
    --num-workers 0 \
    --output-json "$SECURE_RUN_DIR/plaintext_vs_secure_score_compare.json" \
    --output-csv "$SECURE_RUN_DIR/plaintext_vs_secure_score_compare.csv"
}

write_profile_summaries() {
  local fastpath_profile_json="$SECURE_RUN_DIR/fastpath_profile_summary.json"
  local fastpath_profile_md="$SECURE_RUN_DIR/fastpath_profile_summary.md"
  if run_python tools/transshield_fastpath_profile_summary.py \
    "$REPO_ROOT/logs/spu_nodes" \
    "$SECURE_RUN_DIR/step_logs" \
    --output-json "$fastpath_profile_json" \
    --output-md "$fastpath_profile_md"; then
    echo "[profile] wrote fastpath profile: $fastpath_profile_json"
  else
    echo "[profile] fastpath profile scan failed; continuing with C++ LinkDetails-only summary" >&2
  fi

  local secure_profile_args=(
    tools/transshield_secure_profile_summary.py
    --secure-run-dir "$SECURE_RUN_DIR"
    --spu-state-json "$REPO_ROOT/logs/spu_runtime_ports.json"
    --spu-log-dir "$REPO_ROOT/logs/spu_nodes"
    --output-json "$SECURE_RUN_DIR/secure_profile_summary.json"
  )
  if [[ -s "$fastpath_profile_json" ]]; then
    secure_profile_args+=(--fastpath-profile-json "$fastpath_profile_json")
  fi
  run_python "${secure_profile_args[@]}"
}

run_followup() {
  setup_followup_defaults
  setup_temp_env

  cd "$REPO_ROOT"
  validate_followup_prereqs
  export_runtime_inputs
  start_spu_runtime
  run_pipeline_followup
  write_profile_summaries

  echo
  echo "[done] SPU follow-up finished."
  echo "[done] secure run dir: $SECURE_RUN_DIR"
}

run_dual_profile() {
  PYTHON_BIN="$(choose_python)"
  PAIR_ROOT="${PAIR_ROOT:-$REPO_ROOT/artifacts/server_pipeline_run/tracka_lr3e5_timm_spu_dual_profile_20260414}"
  FAST_RUN_DIR="${FAST_RUN_DIR:-$PAIR_ROOT/default_fast_runtime}"
  COMM_RUN_DIR="${COMM_RUN_DIR:-$PAIR_ROOT/diagnostic_comm_runtime}"
  SUMMARY_DIR="${SUMMARY_DIR:-$PAIR_ROOT/summaries}"

  run_profile_mode() {
    local label="$1"
    local run_dir="$2"
    local colocated_flag="$3"
    echo "$label"
    env \
      SECURE_RUN_DIR="$run_dir" \
      SPU_DISABLE_COLOCATED_OPTIMIZATION="$colocated_flag" \
      bash "$REPO_ROOT/scripts/run_tracka_spu.sh" followup
  }

  extract_profile_summary() {
    local run_dir="$1"
    local output_json="$2"
    local output_md="$3"
    local title="$4"
    "$PYTHON_BIN" "$REPO_ROOT/tools/transshield_extract_spu_followup_summary.py" \
      --run-dir "$run_dir" \
      --output-json "$output_json" \
      --output-md "$output_md" \
      --title "$title"
  }

  mkdir -p "$PAIR_ROOT" "$SUMMARY_DIR"

  run_profile_mode "[1/5] running default fast runtime" "$FAST_RUN_DIR" 0

  echo "[2/5] extracting default fast runtime summary"
  extract_profile_summary \
    "$FAST_RUN_DIR" \
    "$SUMMARY_DIR/default_fast_runtime_summary.json" \
    "$SUMMARY_DIR/default_fast_runtime_summary.md" \
    "Default Fast Runtime Summary"

  run_profile_mode "[3/5] running communication-visible diagnostic runtime" "$COMM_RUN_DIR" 1

  echo "[4/5] extracting communication-visible summary"
  extract_profile_summary \
    "$COMM_RUN_DIR" \
    "$SUMMARY_DIR/diagnostic_comm_runtime_summary.json" \
    "$SUMMARY_DIR/diagnostic_comm_runtime_summary.md" \
    "Diagnostic Communication Runtime Summary"

  echo "[5/5] generating compare + merged artifacts"
  "$PYTHON_BIN" "$REPO_ROOT/tools/transshield_runtime_branch_compare.py" \
    --summary-a "$SUMMARY_DIR/default_fast_runtime_summary.json" \
    --summary-b "$SUMMARY_DIR/diagnostic_comm_runtime_summary.json" \
    --label-a default_fast_runtime \
    --label-b diagnostic_comm_runtime \
    --output-json "$SUMMARY_DIR/runtime_branch_compare.json" \
    --output-md "$SUMMARY_DIR/runtime_branch_compare.md"

  "$PYTHON_BIN" "$REPO_ROOT/tools/transshield_merge_aux_comm_profile.py" \
    --primary-profile-json "$SUMMARY_DIR/default_fast_runtime_summary.json" \
    --aux-communication-json "$SUMMARY_DIR/diagnostic_comm_runtime_summary.json" \
    --primary-label default_fast_runtime \
    --aux-label diagnostic_comm_runtime \
    --output-json "$SUMMARY_DIR/merged_secure_profile.json" \
    --output-md "$SUMMARY_DIR/merged_secure_profile.md"

  echo
  echo "[done] dual profile workflow finished"
  echo "[done] pair root: $PAIR_ROOT"
  echo "[done] summaries: $SUMMARY_DIR"
}

unset PYTHONPATH
export PYTHONNOUSERSITE=1

main() {
  local mode="${1:-}"
  [[ -n "$mode" ]] || usage
  shift

  case "$mode" in
    followup)
      run_followup "$@"
      ;;
    dual-profile)
      run_dual_profile "$@"
      ;;
    *)
      usage
      ;;
  esac
}

main "$@"
