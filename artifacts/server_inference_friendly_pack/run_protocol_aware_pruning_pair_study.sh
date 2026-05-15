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

PAIR_EPOCHS="${PAIR_EPOCHS:-3}"
PAIR_SECURE_STATIC_DEPTH="${PAIR_SECURE_STATIC_DEPTH:-12}"
PAIR_ACCURACY_PROFILE="${PAIR_ACCURACY_PROFILE:-default}"
PAIR_CANDIDATE_PROFILE="${PAIR_CANDIDATE_PROFILE:-focused}"
PAIR_EVAL_DEVICE="${PAIR_EVAL_DEVICE:-cpu}"
PAIR_EVAL_BATCH_SIZE="${PAIR_EVAL_BATCH_SIZE:-32}"
PAIR_EVAL_NUM_WORKERS="${PAIR_EVAL_NUM_WORKERS:-0}"
PAIR_REPO_ROOT="${PAIR_REPO_ROOT:-$REPO_ROOT/training_compat}"
STAGE_COST_RISK_JSON="${STAGE_COST_RISK_JSON:-$REPO_ROOT/results/stage_cost_risk_model/stage_cost_risk_20260505_clean/stage_cost_risk_report.json}"

DEFAULT_PAIR_NAME="protocol_aware_pair_epoch${PAIR_EPOCHS}_${PAIR_CANDIDATE_PROFILE}_$(date +%Y%m%d_%H%M%S)"
PAIR_NAME="${PAIR_NAME:-$DEFAULT_PAIR_NAME}"
BASELINE_RUN_NAME="${BASELINE_RUN_NAME:-${PAIR_NAME}_baseline}"
CANDIDATE_RUN_NAME="${CANDIDATE_RUN_NAME:-${PAIR_NAME}_${PAIR_CANDIDATE_PROFILE}}"
PAIR_OUTPUT_DIR="${PAIR_OUTPUT_DIR:-$REPO_ROOT/results/protocol_aware_pruning_objective/$PAIR_NAME}"
BASELINE_RESULT_DIR="${BASELINE_RESULT_DIR:-$REPO_ROOT/results/protocol_aware_pruning_objective/$BASELINE_RUN_NAME}"
CANDIDATE_RESULT_DIR="${CANDIDATE_RESULT_DIR:-$REPO_ROOT/results/protocol_aware_pruning_objective/$CANDIDATE_RUN_NAME}"
BASELINE_RUN_DIR="${BASELINE_RUN_DIR:-$REPO_ROOT/artifacts/train_runs/$BASELINE_RUN_NAME}"
CANDIDATE_RUN_DIR="${CANDIDATE_RUN_DIR:-$REPO_ROOT/artifacts/train_runs/$CANDIDATE_RUN_NAME}"

BASELINE_THRESHOLD_JSON="$BASELINE_RUN_DIR/threshold_best.json"
CANDIDATE_THRESHOLD_JSON="$CANDIDATE_RUN_DIR/threshold_best.json"
BASELINE_THRESHOLD_EVAL_JSON="$BASELINE_RUN_DIR/threshold_eval.json"
CANDIDATE_THRESHOLD_EVAL_JSON="$CANDIDATE_RUN_DIR/threshold_eval.json"
BASELINE_PLAINTEXT_EVAL_JSON="$BASELINE_RESULT_DIR/plaintext_eval.json"
CANDIDATE_PLAINTEXT_EVAL_JSON="$CANDIDATE_RESULT_DIR/plaintext_eval.json"
BASELINE_MARGIN_JSON="$BASELINE_RESULT_DIR/pruning_margin_log_report.json"
CANDIDATE_MARGIN_JSON="$CANDIDATE_RESULT_DIR/pruning_margin_log_report.json"
PAIR_COMPARE_JSON="$PAIR_OUTPUT_DIR/protocol_aware_pair_compare.json"
PAIR_COMPARE_MD="$PAIR_OUTPUT_DIR/protocol_aware_pair_compare.md"

if [[ -n "${PAIR_FOCUS_STAGE_INDEX:-}" ]]; then
  FOCUS_STAGE_INDEX="$PAIR_FOCUS_STAGE_INDEX"
elif [[ -f "$STAGE_COST_RISK_JSON" ]]; then
  FOCUS_STAGE_INDEX="$("$PYTHON_BIN" - "$STAGE_COST_RISK_JSON" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], 'r', encoding='utf-8'))
print((payload.get('basis') or {}).get('dominant_risk_stage_index', 1))
PY
)"
else
  FOCUS_STAGE_INDEX="1"
fi

print_plan() {
  cat <<EOF
[protocol-aware-pair] pair_name=$PAIR_NAME
[protocol-aware-pair] baseline_run_name=$BASELINE_RUN_NAME
[protocol-aware-pair] candidate_run_name=$CANDIDATE_RUN_NAME
[protocol-aware-pair] pair_epochs=$PAIR_EPOCHS
[protocol-aware-pair] candidate_profile=$PAIR_CANDIDATE_PROFILE
[protocol-aware-pair] focus_stage_index=$FOCUS_STAGE_INDEX
[protocol-aware-pair] train_data_path=$TRAIN_DATA_PATH
[protocol-aware-pair] val_data_path=$VAL_DATA_PATH
[protocol-aware-pair] base_checkpoint=$BASE_CHECKPOINT
[protocol-aware-pair] teacher_checkpoint=$TEACHER_CHECKPOINT
[protocol-aware-pair] baseline_run_dir=$BASELINE_RUN_DIR
[protocol-aware-pair] candidate_run_dir=$CANDIDATE_RUN_DIR
[protocol-aware-pair] baseline_result_dir=$BASELINE_RESULT_DIR
[protocol-aware-pair] candidate_result_dir=$CANDIDATE_RESULT_DIR
[protocol-aware-pair] pair_output_dir=$PAIR_OUTPUT_DIR
EOF
}

ensure_common_inputs() {
  if [[ ! -f "$BASE_CHECKPOINT" ]]; then
    echo "[protocol-aware-pair] missing BASE_CHECKPOINT: $BASE_CHECKPOINT" >&2
    exit 2
  fi
  if [[ ! -f "$TEACHER_CHECKPOINT" ]]; then
    echo "[protocol-aware-pair] missing TEACHER_CHECKPOINT: $TEACHER_CHECKPOINT" >&2
    exit 2
  fi
  if [[ ! -d "$TRAIN_DATA_PATH" ]]; then
    echo "[protocol-aware-pair] missing TRAIN_DATA_PATH: $TRAIN_DATA_PATH" >&2
    exit 2
  fi
  if [[ ! -d "$VAL_DATA_PATH" ]]; then
    echo "[protocol-aware-pair] missing VAL_DATA_PATH: $VAL_DATA_PATH" >&2
    exit 2
  fi
}

run_baseline_train() {
  ensure_common_inputs
  unset PROTOCOL_AWARE_FORCE_RECIPE_PRUNING_MARGIN || true
  export TRAIN_DATA_PATH VAL_DATA_PATH BASE_BUNDLE_DIR BASE_CHECKPOINT TEACHER_CHECKPOINT
  export SECURE_STATIC_DEPTH="$PAIR_SECURE_STATIC_DEPTH"
  export SECURE_STATIC_SKIP_PRUNING=false
  export USE_MASK_PRUNING=false
  export ACCURACY_PROFILE="$PAIR_ACCURACY_PROFILE"
  export EPOCHS="$PAIR_EPOCHS"
  export RUN_NAME="$BASELINE_RUN_NAME"
  export RUN_DIR="$BASELINE_RUN_DIR"
  export PRUNING_MARGIN_WEIGHT=0.0
  export PRUNING_MARGIN_TARGET="${PRUNING_MARGIN_TARGET:-1e-4}"
  export PRUNING_MARGIN_MODE="${PRUNING_MARGIN_MODE:-hinge}"
  export PRUNING_MARGIN_START_EPOCH=0
  unset PRUNING_MARGIN_STAGE_WEIGHTS || true
  echo "[protocol-aware-pair] train baseline -> $BASELINE_RUN_DIR"
  bash "$SCRIPT_DIR/run_secure_static_distill_train.sh" epoch1 "$@"
}

run_candidate_train() {
  ensure_common_inputs
  export TRAIN_DATA_PATH VAL_DATA_PATH BASE_BUNDLE_DIR BASE_CHECKPOINT TEACHER_CHECKPOINT
  export SECURE_STATIC_DEPTH="$PAIR_SECURE_STATIC_DEPTH"
  export SECURE_STATIC_SKIP_PRUNING=false
  export USE_MASK_PRUNING=false
  export ACCURACY_PROFILE="$PAIR_ACCURACY_PROFILE"
  export EPOCHS="$PAIR_EPOCHS"
  export PROTOCOL_AWARE_PROFILE="$PAIR_CANDIDATE_PROFILE"
  export RUN_NAME="$CANDIDATE_RUN_NAME"
  export RUN_DIR="$CANDIDATE_RUN_DIR"
  # Let run_protocol_aware_pruning_train.sh hydrate the candidate profile from recipe.
  # Otherwise stale baseline exports (e.g. PRUNING_MARGIN_WEIGHT=0.0) can silently disable the objective.
  unset PRUNING_MARGIN_WEIGHT || true
  unset PRUNING_MARGIN_TARGET || true
  unset PRUNING_MARGIN_MODE || true
  unset PRUNING_MARGIN_STAGE_WEIGHTS || true
  unset PRUNING_MARGIN_START_EPOCH || true
  echo "[protocol-aware-pair] train candidate profile=$PAIR_CANDIDATE_PROFILE -> $CANDIDATE_RUN_DIR"
  PROTOCOL_AWARE_FORCE_RECIPE_PRUNING_MARGIN=1 \
    bash "$SCRIPT_DIR/run_protocol_aware_pruning_train.sh" epoch1 "$@"
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
    echo "[protocol-aware-pair] missing checkpoint-best.pth under $run_dir" >&2
    exit 2
  fi

  mkdir -p "$result_dir"
  echo "[protocol-aware-pair] threshold search -> $threshold_json"
  "$PYTHON_BIN" tools/transshield_binary_threshold_search.py search \
    --repo-root "$PAIR_REPO_ROOT" \
    --checkpoint "$run_dir/checkpoint-best.pth" \
    --data-path "$VAL_DATA_PATH" \
    --device "$PAIR_EVAL_DEVICE" \
    --batch-size "$PAIR_EVAL_BATCH_SIZE" \
    --num-workers "$PAIR_EVAL_NUM_WORKERS" \
    --output-json "$threshold_json"

  echo "[protocol-aware-pair] threshold eval -> $threshold_eval_json"
  "$PYTHON_BIN" tools/transshield_binary_threshold_search.py eval \
    --repo-root "$PAIR_REPO_ROOT" \
    --checkpoint "$run_dir/checkpoint-best.pth" \
    --threshold-json "$threshold_json" \
    --data-path "$VAL_DATA_PATH" \
    --device "$PAIR_EVAL_DEVICE" \
    --batch-size "$PAIR_EVAL_BATCH_SIZE" \
    --num-workers "$PAIR_EVAL_NUM_WORKERS" \
    --output-json "$threshold_eval_json"

  echo "[protocol-aware-pair] plaintext eval -> $plaintext_eval_json"
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

  echo "[protocol-aware-pair] baseline margin report -> $BASELINE_MARGIN_JSON"
  "$PYTHON_BIN" tools/transshield_pruning_margin_log_report.py \
    --train-log "$BASELINE_RUN_DIR/train_stdout.log" \
    --output-json "$BASELINE_MARGIN_JSON" \
    --output-md "$BASELINE_RESULT_DIR/pruning_margin_log_report.md"
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

  local recipe_json="$CANDIDATE_RESULT_DIR/protocol_aware_pruning_recipe.json"
  local margin_md="$CANDIDATE_RESULT_DIR/pruning_margin_log_report.md"
  echo "[protocol-aware-pair] candidate margin report -> $CANDIDATE_MARGIN_JSON"
  if [[ -f "$recipe_json" ]]; then
    "$PYTHON_BIN" tools/transshield_pruning_margin_log_report.py \
      --train-log "$CANDIDATE_RUN_DIR/train_stdout.log" \
      --recipe-json "$recipe_json" \
      --profile "$PAIR_CANDIDATE_PROFILE" \
      --output-json "$CANDIDATE_MARGIN_JSON" \
      --output-md "$margin_md"
  else
    "$PYTHON_BIN" tools/transshield_pruning_margin_log_report.py \
      --train-log "$CANDIDATE_RUN_DIR/train_stdout.log" \
      --profile "$PAIR_CANDIDATE_PROFILE" \
      --output-json "$CANDIDATE_MARGIN_JSON" \
      --output-md "$margin_md"
  fi
}

run_compare() {
  mkdir -p "$PAIR_OUTPUT_DIR"
  echo "[protocol-aware-pair] pair compare -> $PAIR_COMPARE_JSON"
  "$PYTHON_BIN" tools/transshield_training_pair_compare.py \
    --study-kind protocol_aware_pruning \
    --baseline-run-dir "$BASELINE_RUN_DIR" \
    --candidate-run-dir "$CANDIDATE_RUN_DIR" \
    --baseline-label "$BASELINE_RUN_NAME" \
    --candidate-label "$CANDIDATE_RUN_NAME" \
    --baseline-threshold-search-json "$BASELINE_THRESHOLD_EVAL_JSON" \
    --candidate-threshold-search-json "$CANDIDATE_THRESHOLD_EVAL_JSON" \
    --baseline-plaintext-eval-json "$BASELINE_PLAINTEXT_EVAL_JSON" \
    --candidate-plaintext-eval-json "$CANDIDATE_PLAINTEXT_EVAL_JSON" \
    --baseline-margin-report-json "$BASELINE_MARGIN_JSON" \
    --candidate-margin-report-json "$CANDIDATE_MARGIN_JSON" \
    --focus-stage-index "$FOCUS_STAGE_INDEX" \
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
