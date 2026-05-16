#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"

MODE="${1:-smoke16}"
case "$MODE" in
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
  smoke64)
    DEFAULT_MAX_SAMPLES=64
    ;;
  fullval)
    DEFAULT_MAX_SAMPLES=524
    ;;
  custom)
    DEFAULT_MAX_SAMPLES="${E2E_EVAL_MAX_SAMPLES:-16}"
    ;;
  *)
    echo "Usage: $0 [smoke4|smoke8|smoke16|smoke32|smoke64|fullval|custom]" >&2
    exit 2
    ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python}"
LUT_GELU_BUNDLE_DIR="${LUT_GELU_BUNDLE_DIR:-$REPO_ROOT/artifacts/frozen_bundle_secure_static_depth12_uniform_lut_gelu_16_final_20260514}"

E2E_EVAL_DATASET_DIR="${E2E_EVAL_DATASET_DIR:-${DATA_ROOT:-/data/wyb/pneumoniamnist_imagefolder_subset}/val}"
E2E_EVAL_MAX_SAMPLES="${E2E_EVAL_MAX_SAMPLES:-$DEFAULT_MAX_SAMPLES}"
E2E_EVAL_LIST_STRATEGY="${E2E_EVAL_LIST_STRATEGY:-balanced_evenly_spaced}"
E2E_APPROX_EVAL_ISOLATE_SAMPLES="${E2E_APPROX_EVAL_ISOLATE_SAMPLES:-0}"

RUN_NAME="${RUN_NAME:-e2e_lut_gelu_${MODE}_$(date +%Y%m%d_%H%M%S)}"

if [[ ! -d "$LUT_GELU_BUNDLE_DIR" ]]; then
  echo "[e2e-lut-gelu] missing LUT_GELU_BUNDLE_DIR: $LUT_GELU_BUNDLE_DIR" >&2
  exit 2
fi

export PYTHON_BIN
export REPO_ROOT
export RUN_NAME
export BUNDLE_DIR="$LUT_GELU_BUNDLE_DIR"
export E2E_EVAL_DATASET_DIR
export E2E_EVAL_MAX_SAMPLES
export E2E_EVAL_LIST_STRATEGY
export E2E_APPROX_EVAL_ISOLATE_SAMPLES

# LUT GELU specific settings
export E2E_SPU_LAYER_NORM_POLICY="${E2E_SPU_LAYER_NORM_POLICY:-exact}"
export E2E_SPU_ATTENTION_POLICY="${E2E_SPU_ATTENTION_POLICY:-uniform}"
export E2E_SPU_ACTIVATION_OVERRIDE="${E2E_SPU_ACTIVATION_OVERRIDE:-lut_gelu_16}"
export E2E_SPU_ACTIVATION_CLIP_VALUE="${E2E_SPU_ACTIVATION_CLIP_VALUE:-0}"

export E2E_PREPROCESS_TIMEOUT_SEC="${E2E_PREPROCESS_TIMEOUT_SEC:-240}"
export E2E_PLAINTEXT_TIMEOUT_SEC="${E2E_PLAINTEXT_TIMEOUT_SEC:-360}"
export E2E_SHARE_PREPROCESS_TIMEOUT_SEC="${E2E_SHARE_PREPROCESS_TIMEOUT_SEC:-240}"
export E2E_METRICS_TIMEOUT_SEC="${E2E_METRICS_TIMEOUT_SEC:-240}"
export E2E_ISOLATED_INFER_TIMEOUT_SEC="${E2E_ISOLATED_INFER_TIMEOUT_SEC:-900}"
export E2E_ISOLATED_INFER_TIMEOUT_KILL_SEC="${E2E_ISOLATED_INFER_TIMEOUT_KILL_SEC:-90}"
export E2E_AANONE_DRY_RUN="${E2E_AANONE_DRY_RUN:-0}"

# Avoid CUDA probing on servers where CPU/JAX-SPU is the intended runtime.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

echo "[e2e-lut-gelu] mode=$MODE"
echo "[e2e-lut-gelu] run_name=$RUN_NAME"
echo "[e2e-lut-gelu] bundle_dir=$LUT_GELU_BUNDLE_DIR"
echo "[e2e-lut-gelu] max_samples=$E2E_EVAL_MAX_SAMPLES"
echo "[e2e-lut-gelu] list_strategy=$E2E_EVAL_LIST_STRATEGY"
echo "[e2e-lut-gelu] isolate_samples=$E2E_APPROX_EVAL_ISOLATE_SAMPLES"
echo "[e2e-lut-gelu] layer_norm=$E2E_SPU_LAYER_NORM_POLICY activation=$E2E_SPU_ACTIVATION_OVERRIDE clip=$E2E_SPU_ACTIVATION_CLIP_VALUE"

if [[ "$E2E_AANONE_DRY_RUN" == "1" ]]; then
  echo "[e2e-lut-gelu] dry_run=1, skip executing run_e2e_secure_approx_eval.sh"
  exit 0
fi

bash "$SCRIPT_DIR/run_e2e_secure_approx_eval.sh"
