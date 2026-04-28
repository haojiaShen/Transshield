#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"

RUNTIME="${1:-cpu}"

REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="${RUN_NAME:-transshield_inference_friendly_deits}"
BUNDLE_DIR="${BUNDLE_DIR:-artifacts/frozen_bundle_verified_tracka_lr3e5_20260414}"
SECURE_RUN_DIR="${SECURE_RUN_DIR:-artifacts/server_pipeline_run/${RUN_NAME}}"
CONFIG_PATH="${CONFIG_PATH:-configs/openbumblebee/2pc.json}"
KTH_SELECTION_MODE="${KTH_SELECTION_MODE:-blockwise_exact_kth}"
PHASE3_SELECTION_MANIFEST="${PHASE3_SELECTION_MANIFEST:-results/blockwise_exact_kth_selection_manifest_default.json}"
SPU_RUNTIME_REUSE="${SPU_RUNTIME_REUSE:-0}"
SPU_DISABLE_COLOCATED_OPTIMIZATION="${SPU_DISABLE_COLOCATED_OPTIMIZATION:-0}"
SPU_REMOVE_UNSUPPORTED_CHEETAH_FIELDS="${SPU_REMOVE_UNSUPPORTED_CHEETAH_FIELDS:-1}"
SPU_RUNTIME_LOG_DIR="${SPU_RUNTIME_LOG_DIR:-logs/spu_nodes}"
SPU_RUNTIME_STATE_JSON="${SPU_RUNTIME_STATE_JSON:-logs/spu_runtime_ports.json}"
SPU_RUNTIME_STARTUP_TIMEOUT_SEC="${SPU_RUNTIME_STARTUP_TIMEOUT_SEC:-30}"
SPU_RUNTIME_STOP_WAIT_SEC="${SPU_RUNTIME_STOP_WAIT_SEC:-1}"
SPU_RUNTIME_WARMUP_ATTEMPTS="${SPU_RUNTIME_WARMUP_ATTEMPTS:-2}"
SKIP_PIPELINE_VERIFY="${SKIP_PIPELINE_VERIFY:-0}"

EXTRA_ARGS=()
if [[ "$KTH_SELECTION_MODE" != "flat_odd_even" ]]; then
  EXTRA_ARGS+=(--phase3-selection-manifest "$PHASE3_SELECTION_MANIFEST")
fi

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

case "$RUNTIME" in
  cpu)
    PREFIX="[secure-cpu]"
    echo "$PREFIX 运行 secure sidecar 的 CPU 参考后端。"
    echo "$PREFIX 这一步是本地明文参考执行，用于调试与链路验证，不是真正 2PC。"
    "$PYTHON_BIN" tools/transshield_openbumblebee_pipeline.py run \
      --runtime cpu \
      --bundle-dir "$BUNDLE_DIR" \
      --selection-mode "$KTH_SELECTION_MODE" \
      "${EXTRA_ARGS[@]}" \
      --output-dir "$SECURE_RUN_DIR"
    ;;
  spu)
    PREFIX="[secure-spu]"
    echo "$PREFIX 运行 secure sidecar 的 SPU 后端。"
    echo "$PREFIX 这一步对应真实 secure 执行，会涉及 secret sharing、协议执行与通信开销。"
    start_args=()
    build_spu_runtime_start_args start_args
    if [[ "$SPU_RUNTIME_REUSE" == "1" ]]; then
      if ! runtime_state_matches_requested; then
        echo "$PREFIX 已有 runtime 的 colocated 设置与当前请求不一致，重新拉起 SPU 节点。"
      elif "$PYTHON_BIN" tools/transshield_spu_runtime_setup.py check --config "$CONFIG_PATH" --startup-timeout-sec 3 >/dev/null 2>&1; then
        echo "$PREFIX 检测到可复用的 SPU runtime，跳过重启。"
      else
        echo "$PREFIX 未检测到可复用 runtime，重新拉起 SPU 节点。"
        "$PYTHON_BIN" "${start_args[@]}"
      fi
    else
      "$PYTHON_BIN" "${start_args[@]}"
    fi
    "$PYTHON_BIN" tools/transshield_openbumblebee_pipeline.py run \
      --runtime spu \
      --bundle-dir "$BUNDLE_DIR" \
      --config "$CONFIG_PATH" \
      --selection-mode "$KTH_SELECTION_MODE" \
      "${EXTRA_ARGS[@]}" \
      --output-dir "$SECURE_RUN_DIR"
    ;;
  *)
    echo "Usage: $0 [cpu|spu]" >&2
    exit 1
    ;;
esac

if [[ "$SKIP_PIPELINE_VERIFY" == "1" ]]; then
  echo "$PREFIX 跳过 pipeline verify（当前用于单图/网页 fast path）。"
else
  "$PYTHON_BIN" tools/transshield_openbumblebee_pipeline.py verify --output-dir "$SECURE_RUN_DIR"
fi
