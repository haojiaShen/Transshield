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
  custom)
    DEFAULT_MAX_SAMPLES="${E2E_EVAL_MAX_SAMPLES:-16}"
    ;;
  *)
    echo "Usage: $0 [smoke4|smoke8|smoke16|smoke32|smoke64|custom]" >&2
    exit 2
    ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python}"
AA_NONE_BUNDLE_DIR="${AA_NONE_BUNDLE_DIR:-$REPO_ROOT/artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_aanone_20260507}"
AA_NONE_STATIC_OUTPUT_CALIBRATION_JSON="${AA_NONE_STATIC_OUTPUT_CALIBRATION_JSON:-$REPO_ROOT/results/e2e_static_calibration/e2e_static_fullval_aanone_20260507_1/e2e_static_output_calibration_public_logit_bias.json}"
AA_NONE_SPUAWARE_OUTPUT_CALIBRATION_JSON="${AA_NONE_SPUAWARE_OUTPUT_CALIBRATION_JSON:-$REPO_ROOT/results/e2e_static_calibration/e2e_static_fullval_aanone_20260507_1/e2e_static_output_calibration_public_logit_bias_spuaware_smoke32_threshold.json}"
AA_NONE_AFFINE_OUTPUT_CALIBRATION_JSON="${AA_NONE_AFFINE_OUTPUT_CALIBRATION_JSON:-$REPO_ROOT/results/e2e_static_calibration/e2e_static_fullval_aanone_20260507_1/e2e_output_calibration_public_affine_fit_on_spu_smoke32.json}"
AA_NONE_TEMPERATURE_OUTPUT_CALIBRATION_JSON="${AA_NONE_TEMPERATURE_OUTPUT_CALIBRATION_JSON:-$REPO_ROOT/results/e2e_static_calibration/e2e_static_fullval_aanone_20260507_1/e2e_output_calibration_public_temperature_fit_on_spu_smoke32.json}"
AA_NONE_BRIDGE_OUTPUT_CALIBRATION_JSON="${AA_NONE_BRIDGE_OUTPUT_CALIBRATION_JSON:-$REPO_ROOT/results/e2e_static_calibration/e2e_static_fullval_aanone_20260507_1/e2e_plaintext_bridge_calibration_suite/e2e_plaintext_bridge_best_bridge_calibration.json}"
AA_NONE_OUTPUT_PROFILE="${AA_NONE_OUTPUT_PROFILE:-accuracy_first}"
case "$AA_NONE_OUTPUT_PROFILE" in
  accuracy_first)
    AA_NONE_DEFAULT_OUTPUT_CALIBRATION_JSON="$AA_NONE_SPUAWARE_OUTPUT_CALIBRATION_JSON"
    ;;
  loss_first_affine)
    AA_NONE_DEFAULT_OUTPUT_CALIBRATION_JSON="$AA_NONE_AFFINE_OUTPUT_CALIBRATION_JSON"
    ;;
  loss_first_temperature)
    AA_NONE_DEFAULT_OUTPUT_CALIBRATION_JSON="$AA_NONE_TEMPERATURE_OUTPUT_CALIBRATION_JSON"
    ;;
  static_bias)
    AA_NONE_DEFAULT_OUTPUT_CALIBRATION_JSON="$AA_NONE_STATIC_OUTPUT_CALIBRATION_JSON"
    ;;
  bridge_best)
    AA_NONE_DEFAULT_OUTPUT_CALIBRATION_JSON="$AA_NONE_BRIDGE_OUTPUT_CALIBRATION_JSON"
    ;;
  *)
    echo "[e2e-aanone] unsupported AA_NONE_OUTPUT_PROFILE: $AA_NONE_OUTPUT_PROFILE" >&2
    echo "[e2e-aanone] supported profiles: accuracy_first, loss_first_affine, loss_first_temperature, static_bias, bridge_best" >&2
    exit 2
    ;;
esac
AA_NONE_OUTPUT_CALIBRATION_JSON="${AA_NONE_OUTPUT_CALIBRATION_JSON:-$AA_NONE_DEFAULT_OUTPUT_CALIBRATION_JSON}"
if [[ "${ALLOW_E2E_AANONE_OVERRIDE:-0}" == "1" ]]; then
  BUNDLE_DIR="${BUNDLE_DIR:-$AA_NONE_BUNDLE_DIR}"
  E2E_OUTPUT_CALIBRATION_JSON="${E2E_OUTPUT_CALIBRATION_JSON:-$AA_NONE_OUTPUT_CALIBRATION_JSON}"
else
  BUNDLE_DIR="$AA_NONE_BUNDLE_DIR"
  E2E_OUTPUT_CALIBRATION_JSON="$AA_NONE_OUTPUT_CALIBRATION_JSON"
fi
E2E_EVAL_DATASET_DIR="${E2E_EVAL_DATASET_DIR:-${DATA_ROOT:-/data/wyb/pneumoniamnist_imagefolder_subset}/val}"
E2E_EVAL_MAX_SAMPLES="${E2E_EVAL_MAX_SAMPLES:-$DEFAULT_MAX_SAMPLES}"
E2E_EVAL_LIST_STRATEGY="${E2E_EVAL_LIST_STRATEGY:-balanced_evenly_spaced}"
E2E_APPROX_EVAL_ISOLATE_SAMPLES="${E2E_APPROX_EVAL_ISOLATE_SAMPLES:-0}"

RUN_NAME="${RUN_NAME:-e2e_aanone_exactln_clip0_${MODE}_$(date +%Y%m%d_%H%M%S)}"

if [[ ! -d "$BUNDLE_DIR" ]]; then
  echo "[e2e-aanone] missing BUNDLE_DIR: $BUNDLE_DIR" >&2
  exit 2
fi
if [[ ! -f "$E2E_OUTPUT_CALIBRATION_JSON" ]]; then
  echo "[e2e-aanone] missing E2E_OUTPUT_CALIBRATION_JSON: $E2E_OUTPUT_CALIBRATION_JSON" >&2
  exit 2
fi

export PYTHON_BIN
export REPO_ROOT
export RUN_NAME
export AA_NONE_BUNDLE_DIR
export AA_NONE_STATIC_OUTPUT_CALIBRATION_JSON
export AA_NONE_SPUAWARE_OUTPUT_CALIBRATION_JSON
export AA_NONE_AFFINE_OUTPUT_CALIBRATION_JSON
export AA_NONE_TEMPERATURE_OUTPUT_CALIBRATION_JSON
export AA_NONE_BRIDGE_OUTPUT_CALIBRATION_JSON
export AA_NONE_OUTPUT_PROFILE
export AA_NONE_DEFAULT_OUTPUT_CALIBRATION_JSON
export AA_NONE_OUTPUT_CALIBRATION_JSON
export BUNDLE_DIR
export E2E_OUTPUT_CALIBRATION_JSON
export E2E_EVAL_DATASET_DIR
export E2E_EVAL_MAX_SAMPLES
export E2E_EVAL_LIST_STRATEGY
export E2E_APPROX_EVAL_ISOLATE_SAMPLES

export E2E_SPU_LAYER_NORM_POLICY="${E2E_SPU_LAYER_NORM_POLICY:-exact}"
export E2E_SPU_ATTENTION_POLICY="${E2E_SPU_ATTENTION_POLICY:-uniform}"
export E2E_SPU_ACTIVATION_OVERRIDE="${E2E_SPU_ACTIVATION_OVERRIDE:-fixed_square}"
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

echo "[e2e-aanone] mode=$MODE"
echo "[e2e-aanone] run_name=$RUN_NAME"
echo "[e2e-aanone] bundle_dir=$BUNDLE_DIR"
echo "[e2e-aanone] output_profile=$AA_NONE_OUTPUT_PROFILE"
echo "[e2e-aanone] output_calibration=$E2E_OUTPUT_CALIBRATION_JSON"
echo "[e2e-aanone] max_samples=$E2E_EVAL_MAX_SAMPLES"
echo "[e2e-aanone] list_strategy=$E2E_EVAL_LIST_STRATEGY"
echo "[e2e-aanone] isolate_samples=$E2E_APPROX_EVAL_ISOLATE_SAMPLES"
echo "[e2e-aanone] layer_norm=$E2E_SPU_LAYER_NORM_POLICY activation=$E2E_SPU_ACTIVATION_OVERRIDE clip=$E2E_SPU_ACTIVATION_CLIP_VALUE"

if [[ "$E2E_AANONE_DRY_RUN" == "1" ]]; then
  echo "[e2e-aanone] dry_run=1, skip executing run_e2e_secure_approx_eval.sh"
  exit 0
fi

bash "$SCRIPT_DIR/run_e2e_secure_approx_eval.sh"
