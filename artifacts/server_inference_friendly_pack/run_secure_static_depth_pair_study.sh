#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

MODE="${1:-suite}"
case "$MODE" in
  print|train-baseline|train-candidate|post-baseline|post-candidate|compare|suite)
    shift || true
    ;;
  *)
    echo "Usage: $0 [print|train-baseline|train-candidate|post-baseline|post-candidate|compare|suite]" >&2
    exit 1
    ;;
esac

TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${DATA_ROOT:-/data/wyb/pneumoniamnist_imagefolder_subset}/train}"
VAL_DATA_PATH="${VAL_DATA_PATH:-${DATA_ROOT:-/data/wyb/pneumoniamnist_imagefolder_subset}/val}"
BASE_BUNDLE_DIR="${BASE_BUNDLE_DIR:-$REPO_ROOT/artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_20260430}"
DEFAULT_BASE_CHECKPOINT="$BASE_BUNDLE_DIR/checkpoint-best.pth"
LIGHT_BASE_CHECKPOINT="$BASE_BUNDLE_DIR/modified_plaintext_eval_checkpoint_light.pth"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-}"
if [[ -z "$BASE_CHECKPOINT" ]]; then
  if [[ -f "$DEFAULT_BASE_CHECKPOINT" ]]; then
    BASE_CHECKPOINT="$DEFAULT_BASE_CHECKPOINT"
  else
    BASE_CHECKPOINT="$LIGHT_BASE_CHECKPOINT"
  fi
fi
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-$BASE_CHECKPOINT}"

PAIR_EPOCHS="${PAIR_EPOCHS:-1}"
BASELINE_SECURE_STATIC_DEPTH="${BASELINE_SECURE_STATIC_DEPTH:-0}"
CANDIDATE_SECURE_STATIC_DEPTH="${CANDIDATE_SECURE_STATIC_DEPTH:-12}"
PAIR_SECURE_STATIC_SKIP_PRUNING="${PAIR_SECURE_STATIC_SKIP_PRUNING:-true}"
PAIR_ACCURACY_PROFILE="${PAIR_ACCURACY_PROFILE:-default}"
PAIR_EVAL_DEVICE="${PAIR_EVAL_DEVICE:-cpu}"
PAIR_EVAL_BATCH_SIZE="${PAIR_EVAL_BATCH_SIZE:-32}"
PAIR_EVAL_NUM_WORKERS="${PAIR_EVAL_NUM_WORKERS:-0}"
PAIR_REPO_ROOT="${PAIR_REPO_ROOT:-$REPO_ROOT/training_compat}"
PAIR_CLS_DISTILL_WEIGHT="${PAIR_CLS_DISTILL_WEIGHT:-1.0}"
PAIR_TOKEN_DISTILL_WEIGHT="${PAIR_TOKEN_DISTILL_WEIGHT:-0.02}"

DEFAULT_PAIR_NAME="secure_static_depth_pair_epoch${PAIR_EPOCHS}_depth${CANDIDATE_SECURE_STATIC_DEPTH}_$(date +%Y%m%d_%H%M%S)"
PAIR_NAME="${PAIR_NAME:-$DEFAULT_PAIR_NAME}"
BASELINE_RUN_NAME="${BASELINE_RUN_NAME:-${PAIR_NAME}_depth${BASELINE_SECURE_STATIC_DEPTH}}"
CANDIDATE_RUN_NAME="${CANDIDATE_RUN_NAME:-${PAIR_NAME}_depth${CANDIDATE_SECURE_STATIC_DEPTH}}"
PAIR_OUTPUT_DIR="${PAIR_OUTPUT_DIR:-$REPO_ROOT/results/secure_static_train_depth_evidence/$PAIR_NAME}"
BASELINE_RESULT_DIR="${BASELINE_RESULT_DIR:-$REPO_ROOT/results/secure_static_train_depth_evidence/$BASELINE_RUN_NAME}"
CANDIDATE_RESULT_DIR="${CANDIDATE_RESULT_DIR:-$REPO_ROOT/results/secure_static_train_depth_evidence/$CANDIDATE_RUN_NAME}"
BASELINE_RUN_DIR="${BASELINE_RUN_DIR:-$REPO_ROOT/artifacts/train_runs/$BASELINE_RUN_NAME}"
CANDIDATE_RUN_DIR="${CANDIDATE_RUN_DIR:-$REPO_ROOT/artifacts/train_runs/$CANDIDATE_RUN_NAME}"

BASELINE_THRESHOLD_JSON="$BASELINE_RUN_DIR/threshold_best.json"
CANDIDATE_THRESHOLD_JSON="$CANDIDATE_RUN_DIR/threshold_best.json"
BASELINE_THRESHOLD_EVAL_JSON="$BASELINE_RUN_DIR/threshold_eval.json"
CANDIDATE_THRESHOLD_EVAL_JSON="$CANDIDATE_RUN_DIR/threshold_eval.json"
BASELINE_PLAINTEXT_EVAL_JSON="$BASELINE_RESULT_DIR/plaintext_eval.json"
CANDIDATE_PLAINTEXT_EVAL_JSON="$CANDIDATE_RESULT_DIR/plaintext_eval.json"
PAIR_COMPARE_JSON="$PAIR_OUTPUT_DIR/secure_static_train_depth_pair_compare.json"
PAIR_COMPARE_MD="$PAIR_OUTPUT_DIR/secure_static_train_depth_pair_compare.md"

print_plan() {
  cat <<EOF
[secure-static-depth-pair] pair_name=$PAIR_NAME
[secure-static-depth-pair] baseline_run_name=$BASELINE_RUN_NAME
[secure-static-depth-pair] candidate_run_name=$CANDIDATE_RUN_NAME
[secure-static-depth-pair] pair_epochs=$PAIR_EPOCHS
[secure-static-depth-pair] baseline_secure_static_depth=$BASELINE_SECURE_STATIC_DEPTH
[secure-static-depth-pair] candidate_secure_static_depth=$CANDIDATE_SECURE_STATIC_DEPTH
[secure-static-depth-pair] secure_static_skip_pruning=$PAIR_SECURE_STATIC_SKIP_PRUNING
[secure-static-depth-pair] cls_distill_weight=$PAIR_CLS_DISTILL_WEIGHT
[secure-static-depth-pair] token_distill_weight=$PAIR_TOKEN_DISTILL_WEIGHT
[secure-static-depth-pair] train_data_path=$TRAIN_DATA_PATH
[secure-static-depth-pair] val_data_path=$VAL_DATA_PATH
[secure-static-depth-pair] base_checkpoint=$BASE_CHECKPOINT
[secure-static-depth-pair] teacher_checkpoint=$TEACHER_CHECKPOINT
[secure-static-depth-pair] baseline_run_dir=$BASELINE_RUN_DIR
[secure-static-depth-pair] candidate_run_dir=$CANDIDATE_RUN_DIR
[secure-static-depth-pair] baseline_result_dir=$BASELINE_RESULT_DIR
[secure-static-depth-pair] candidate_result_dir=$CANDIDATE_RESULT_DIR
[secure-static-depth-pair] pair_output_dir=$PAIR_OUTPUT_DIR
EOF
}

ensure_common_inputs() {
  if [[ ! -f "$BASE_CHECKPOINT" ]]; then
    echo "[secure-static-depth-pair] missing BASE_CHECKPOINT: $BASE_CHECKPOINT" >&2
    exit 2
  fi
  if [[ ! -f "$TEACHER_CHECKPOINT" ]]; then
    echo "[secure-static-depth-pair] missing TEACHER_CHECKPOINT: $TEACHER_CHECKPOINT" >&2
    exit 2
  fi
  if [[ ! -d "$TRAIN_DATA_PATH" ]]; then
    echo "[secure-static-depth-pair] missing TRAIN_DATA_PATH: $TRAIN_DATA_PATH" >&2
    exit 2
  fi
  if [[ ! -d "$VAL_DATA_PATH" ]]; then
    echo "[secure-static-depth-pair] missing VAL_DATA_PATH: $VAL_DATA_PATH" >&2
    exit 2
  fi
}

prepare_train_env() {
  unset PROTOCOL_AWARE_PROFILE || true
  unset PROTOCOL_AWARE_FORCE_RECIPE_PRUNING_MARGIN || true
  export TRAIN_DATA_PATH VAL_DATA_PATH BASE_BUNDLE_DIR BASE_CHECKPOINT TEACHER_CHECKPOINT
  export SECURE_STATIC_SKIP_PRUNING="$PAIR_SECURE_STATIC_SKIP_PRUNING"
  export USE_MASK_PRUNING=false
  export ACCURACY_PROFILE="$PAIR_ACCURACY_PROFILE"
  export EPOCHS="$PAIR_EPOCHS"
  export CLS_DISTILL_WEIGHT="$PAIR_CLS_DISTILL_WEIGHT"
  export TOKEN_DISTILL_WEIGHT="$PAIR_TOKEN_DISTILL_WEIGHT"
  export PRUNING_MARGIN_WEIGHT=0.0
  export PRUNING_MARGIN_TARGET="${PRUNING_MARGIN_TARGET:-1e-4}"
  export PRUNING_MARGIN_MODE="${PRUNING_MARGIN_MODE:-hinge}"
  export PRUNING_MARGIN_START_EPOCH=0
  unset PRUNING_MARGIN_STAGE_WEIGHTS || true
}

run_baseline_train() {
  ensure_common_inputs
  prepare_train_env
  export RUN_NAME="$BASELINE_RUN_NAME"
  export RUN_DIR="$BASELINE_RUN_DIR"
  export SECURE_STATIC_DEPTH="$BASELINE_SECURE_STATIC_DEPTH"
  echo "[secure-static-depth-pair] train baseline depth=$BASELINE_SECURE_STATIC_DEPTH -> $BASELINE_RUN_DIR"
  bash "$SCRIPT_DIR/run_secure_static_distill_train.sh" epoch1 "$@"
}

run_candidate_train() {
  ensure_common_inputs
  prepare_train_env
  export RUN_NAME="$CANDIDATE_RUN_NAME"
  export RUN_DIR="$CANDIDATE_RUN_DIR"
  export SECURE_STATIC_DEPTH="$CANDIDATE_SECURE_STATIC_DEPTH"
  echo "[secure-static-depth-pair] train candidate depth=$CANDIDATE_SECURE_STATIC_DEPTH -> $CANDIDATE_RUN_DIR"
  bash "$SCRIPT_DIR/run_secure_static_distill_train.sh" epoch1 "$@"
}

run_post_common() {
  local run_dir="$1"
  local result_dir="$2"
  local threshold_json="$3"
  local threshold_eval_json="$4"
  local plaintext_eval_json="$5"
  local plaintext_eval_csv="$6"
  local label="$7"

  if [[ ! -f "$run_dir/checkpoint-best.pth" ]]; then
    echo "[secure-static-depth-pair] missing checkpoint-best.pth under $run_dir" >&2
    exit 2
  fi

  mkdir -p "$result_dir"
  echo "[secure-static-depth-pair] threshold search -> $threshold_json"
  "$PYTHON_BIN" tools/transshield_binary_threshold_search.py search \
    --repo-root "$PAIR_REPO_ROOT" \
    --checkpoint "$run_dir/checkpoint-best.pth" \
    --data-path "$VAL_DATA_PATH" \
    --device "$PAIR_EVAL_DEVICE" \
    --batch-size "$PAIR_EVAL_BATCH_SIZE" \
    --num-workers "$PAIR_EVAL_NUM_WORKERS" \
    --output-json "$threshold_json"

  echo "[secure-static-depth-pair] threshold eval -> $threshold_eval_json"
  "$PYTHON_BIN" tools/transshield_binary_threshold_search.py eval \
    --repo-root "$PAIR_REPO_ROOT" \
    --checkpoint "$run_dir/checkpoint-best.pth" \
    --threshold-json "$threshold_json" \
    --data-path "$VAL_DATA_PATH" \
    --device "$PAIR_EVAL_DEVICE" \
    --batch-size "$PAIR_EVAL_BATCH_SIZE" \
    --num-workers "$PAIR_EVAL_NUM_WORKERS" \
    --output-json "$threshold_eval_json"

  echo "[secure-static-depth-pair] plaintext eval -> $plaintext_eval_json"
  "$PYTHON_BIN" tools/transshield_plaintext_checkpoint_eval.py \
    --repo-root "$PAIR_REPO_ROOT" \
    --checkpoint "$run_dir/checkpoint-best.pth" \
    --data-path "$VAL_DATA_PATH" \
    --device "$PAIR_EVAL_DEVICE" \
    --batch-size "$PAIR_EVAL_BATCH_SIZE" \
    --num-workers "$PAIR_EVAL_NUM_WORKERS" \
    --threshold-json "$threshold_json" \
    --label "$label" \
    --output-json "$plaintext_eval_json" \
    --output-csv "$plaintext_eval_csv"
}

run_post_baseline() {
  ensure_common_inputs
  run_post_common \
    "$BASELINE_RUN_DIR" \
    "$BASELINE_RESULT_DIR" \
    "$BASELINE_THRESHOLD_JSON" \
    "$BASELINE_THRESHOLD_EVAL_JSON" \
    "$BASELINE_PLAINTEXT_EVAL_JSON" \
    "$BASELINE_RESULT_DIR/plaintext_eval.csv" \
    "$BASELINE_RUN_NAME"
}

run_post_candidate() {
  ensure_common_inputs
  run_post_common \
    "$CANDIDATE_RUN_DIR" \
    "$CANDIDATE_RESULT_DIR" \
    "$CANDIDATE_THRESHOLD_JSON" \
    "$CANDIDATE_THRESHOLD_EVAL_JSON" \
    "$CANDIDATE_PLAINTEXT_EVAL_JSON" \
    "$CANDIDATE_RESULT_DIR/plaintext_eval.csv" \
    "$CANDIDATE_RUN_NAME"
}

run_compare() {
  mkdir -p "$PAIR_OUTPUT_DIR"
  echo "[secure-static-depth-pair] pair compare -> $PAIR_COMPARE_JSON"
  "$PYTHON_BIN" tools/transshield_training_pair_compare.py \
    --study-kind secure_static_train_depth \
    --baseline-run-dir "$BASELINE_RUN_DIR" \
    --candidate-run-dir "$CANDIDATE_RUN_DIR" \
    --baseline-label "$BASELINE_RUN_NAME" \
    --candidate-label "$CANDIDATE_RUN_NAME" \
    --baseline-threshold-search-json "$BASELINE_THRESHOLD_EVAL_JSON" \
    --candidate-threshold-search-json "$CANDIDATE_THRESHOLD_EVAL_JSON" \
    --baseline-plaintext-eval-json "$BASELINE_PLAINTEXT_EVAL_JSON" \
    --candidate-plaintext-eval-json "$CANDIDATE_PLAINTEXT_EVAL_JSON" \
    --output-json "$PAIR_COMPARE_JSON" \
    --output-md "$PAIR_COMPARE_MD"
}

case "$MODE" in
  print)
    print_plan
    ;;
  train-baseline)
    print_plan
    run_baseline_train "$@"
    ;;
  train-candidate)
    print_plan
    run_candidate_train "$@"
    ;;
  post-baseline)
    print_plan
    run_post_baseline
    ;;
  post-candidate)
    print_plan
    run_post_candidate
    ;;
  compare)
    print_plan
    run_compare
    ;;
  suite)
    print_plan
    run_baseline_train "$@"
    run_post_baseline
    run_candidate_train "$@"
    run_post_candidate
    run_compare
    ;;
esac
