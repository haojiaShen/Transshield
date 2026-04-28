#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
BUNDLE_DIR="${BUNDLE_DIR:-artifacts/frozen_bundle_verified_tracka_lr3e5_20260414}"
CONFIG_PATH="${CONFIG_PATH:-configs/openbumblebee/2pc.json}"
PHASE3_SELECTION_MANIFEST="${PHASE3_SELECTION_MANIFEST:-results/blockwise_exact_kth_selection_manifest_default.json}"
DEFAULT_RUNTIME_INPUT_SOURCE_DIR="artifacts/inference_ready_config/selection_mode_runtime_inputs_verified"
RUNTIME_INPUT_SOURCE_DIR="${RUNTIME_INPUT_SOURCE_DIR:-$DEFAULT_RUNTIME_INPUT_SOURCE_DIR}"
MODE_COMPARE_RUN_NAME="${MODE_COMPARE_RUN_NAME:-selection_mode_profile_$(date +%Y%m%d_%H%M%S)}"
MODE_A="${MODE_A:-flat_odd_even}"
MODE_B="${MODE_B:-blockwise_exact_kth}"
LABEL_A="${LABEL_A:-$MODE_A}"
LABEL_B="${LABEL_B:-$MODE_B}"
MODE_A_RUN_DIR="${MODE_A_RUN_DIR:-artifacts/server_pipeline_run/${MODE_COMPARE_RUN_NAME}_${LABEL_A}}"
MODE_B_RUN_DIR="${MODE_B_RUN_DIR:-artifacts/server_pipeline_run/${MODE_COMPARE_RUN_NAME}_${LABEL_B}}"
PROFILE_COMPARE_DIR="${PROFILE_COMPARE_DIR:-artifacts/server_profile_reports/${MODE_COMPARE_RUN_NAME}_selection_mode_compare}"
MODE_A_PAYLOAD_DTYPE="${MODE_A_PAYLOAD_DTYPE:-float32}"
MODE_B_PAYLOAD_DTYPE="${MODE_B_PAYLOAD_DTYPE:-float32}"
MODE_A_PAYLOAD_STAGE_DTYPES="${MODE_A_PAYLOAD_STAGE_DTYPES:-}"
MODE_B_PAYLOAD_STAGE_DTYPES="${MODE_B_PAYLOAD_STAGE_DTYPES:-}"
MODE_A_PAYLOAD_BOUNDARY_WINDOW="${MODE_A_PAYLOAD_BOUNDARY_WINDOW:-0}"
MODE_B_PAYLOAD_BOUNDARY_WINDOW="${MODE_B_PAYLOAD_BOUNDARY_WINDOW:-0}"

REQUIRED_RUNTIME_INPUTS=(
  stage2_secure_network_kth_manifest.json
  stage2_secure_network_kth_input_smoke8.pt
  stage2_secure_network_kth_input_smoke8.json
  stage2_secure_network_kth_reference_smoke8.pt
  stage2_secure_network_kth_reference_smoke8.json
  stage2_secure_tie_policy_lowest_smoke8.pt
  stage2_secure_tie_policy_lowest_smoke8.json
)

has_runtime_inputs() {
  local source_dir="$1"
  [[ -d "$source_dir" ]] || return 1
  local required
  for required in "${REQUIRED_RUNTIME_INPUTS[@]}"; do
    [[ -e "$source_dir/$required" ]] || return 1
  done
}

find_runtime_input_source() {
  local candidate

  if [[ -n "$RUNTIME_INPUT_SOURCE_DIR" ]] && has_runtime_inputs "$RUNTIME_INPUT_SOURCE_DIR"; then
    echo "$RUNTIME_INPUT_SOURCE_DIR"
    return 0
  fi

  if has_runtime_inputs "$BUNDLE_DIR"; then
    echo "$BUNDLE_DIR"
    return 0
  fi

  for candidate in artifacts/server_pipeline_run/*; do
    [[ -e "$candidate" ]] || continue
    if [[ "$candidate" == "$MODE_A_RUN_DIR" || "$candidate" == "$MODE_B_RUN_DIR" ]]; then
      continue
    fi
    if has_runtime_inputs "$candidate"; then
      echo "$candidate"
      return 0
    fi
  done

  return 1
}

prepare_runtime_inputs() {
  local run_dir="$1"
  local source_dir="$2"
  local required

  mkdir -p "$run_dir"
  if has_runtime_inputs "$run_dir"; then
    echo "[selection-profile] 复用已有 runtime inputs: $run_dir"
    return 0
  fi

  if ! has_runtime_inputs "$source_dir"; then
    echo "[selection-profile] 未找到可用 runtime inputs 源目录: $source_dir" >&2
    return 1
  fi

  echo "[selection-profile] 从以下目录复制 runtime inputs:"
  echo "[selection-profile]   $source_dir"
  for required in "${REQUIRED_RUNTIME_INPUTS[@]}"; do
    cp -f "$source_dir/$required" "$run_dir/$required"
  done
}

run_mode() {
  local mode="$1"
  local run_dir="$2"
  local log_dir="$3"
  local state_json="$4"
  local runtime_input_source="$5"
  local payload_dtype="$6"
  local payload_stage_dtypes="$7"
  local payload_boundary_window="$8"
  local extra_args=()

  if [[ "$mode" != "flat_odd_even" ]]; then
    extra_args+=(--phase3-selection-manifest "$PHASE3_SELECTION_MANIFEST")
  fi
  if [[ -n "$payload_stage_dtypes" ]]; then
    extra_args+=(--payload-stage-dtypes "$payload_stage_dtypes")
  fi
  if [[ "$payload_boundary_window" != "0" ]]; then
    extra_args+=(--payload-boundary-window "$payload_boundary_window")
  fi

  echo "[selection-profile] 启动模式: $mode"
  echo "[selection-profile] 输出目录: $run_dir"
  echo "[selection-profile] payload: dtype=$payload_dtype stage_dtypes=${payload_stage_dtypes:-none} boundary_window=$payload_boundary_window"

  prepare_runtime_inputs "$run_dir" "$runtime_input_source"

  "$PYTHON_BIN" tools/transshield_spu_runtime_setup.py start \
    --config "$CONFIG_PATH" \
    --template configs/openbumblebee/2pc.template.json \
    --backup \
    --restart \
    --remove-unsupported-cheetah-fields \
    --log-dir "$log_dir" \
    --state-json "$state_json"

  "$PYTHON_BIN" tools/transshield_openbumblebee_pipeline.py run \
    --runtime spu \
    --bundle-dir "$BUNDLE_DIR" \
    --config "$CONFIG_PATH" \
    --selection-mode "$mode" \
    --payload-dtype "$payload_dtype" \
    "${extra_args[@]}" \
    --output-dir "$run_dir"

  "$PYTHON_BIN" tools/transshield_openbumblebee_pipeline.py verify --output-dir "$run_dir"

  "$PYTHON_BIN" tools/transshield_fastpath_profile_summary.py \
    "$log_dir" \
    --output-json "$run_dir/fastpath_profile_summary.json" \
    --output-md "$run_dir/fastpath_profile_summary.md"

  "$PYTHON_BIN" tools/transshield_secure_profile_summary.py \
    --secure-run-dir "$run_dir" \
    --spu-state-json "$state_json" \
    --spu-log-dir "$log_dir" \
    --fastpath-profile-json "$run_dir/fastpath_profile_summary.json" \
    --output-json "$run_dir/secure_profile_summary.json"
}

LOG_ROOT="logs/selection_mode_profile/${MODE_COMPARE_RUN_NAME}"
RUNTIME_INPUT_SOURCE_DIR_RESOLVED="$(find_runtime_input_source || true)"
if [[ -z "$RUNTIME_INPUT_SOURCE_DIR_RESOLVED" ]]; then
  echo "[selection-profile] 未找到任何可用 runtime inputs。" >&2
  echo "[selection-profile] 请设置 RUNTIME_INPUT_SOURCE_DIR 指向一个包含 stage2_secure_* 输入文件的旧 run 目录。" >&2
  exit 1
fi
echo "[selection-profile] runtime inputs 源目录: $RUNTIME_INPUT_SOURCE_DIR_RESOLVED"

run_mode "$MODE_A" "$MODE_A_RUN_DIR" "$LOG_ROOT/${LABEL_A}" "$LOG_ROOT/${LABEL_A}_ports.json" "$RUNTIME_INPUT_SOURCE_DIR_RESOLVED" "$MODE_A_PAYLOAD_DTYPE" "$MODE_A_PAYLOAD_STAGE_DTYPES" "$MODE_A_PAYLOAD_BOUNDARY_WINDOW"
run_mode "$MODE_B" "$MODE_B_RUN_DIR" "$LOG_ROOT/${LABEL_B}" "$LOG_ROOT/${LABEL_B}_ports.json" "$RUNTIME_INPUT_SOURCE_DIR_RESOLVED" "$MODE_B_PAYLOAD_DTYPE" "$MODE_B_PAYLOAD_STAGE_DTYPES" "$MODE_B_PAYLOAD_BOUNDARY_WINDOW"

"$PYTHON_BIN" tools/transshield_selection_mode_profile_report.py \
  --run-dir-a "$MODE_A_RUN_DIR" \
  --run-dir-b "$MODE_B_RUN_DIR" \
  --label-a "$LABEL_A" \
  --label-b "$LABEL_B" \
  --output-json "$PROFILE_COMPARE_DIR/selection_mode_profile_compare.json" \
  --output-md "$PROFILE_COMPARE_DIR/selection_mode_profile_compare.md"

echo "[selection-profile] 对比完成。"
echo "[selection-profile] JSON: $PROFILE_COMPARE_DIR/selection_mode_profile_compare.json"
echo "[selection-profile] Markdown: $PROFILE_COMPARE_DIR/selection_mode_profile_compare.md"
