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
PAIR_SECURE_STATIC_DEPTH="${PAIR_SECURE_STATIC_DEPTH:-12}"
PAIR_SECURE_STATIC_SKIP_PRUNING="${PAIR_SECURE_STATIC_SKIP_PRUNING:-true}"
PAIR_USE_MASK_PRUNING="${PAIR_USE_MASK_PRUNING:-false}"
PAIR_REPO_ROOT="${PAIR_REPO_ROOT:-$REPO_ROOT/training_compat}"
PAIR_EVAL_DEVICE="${PAIR_EVAL_DEVICE:-cpu}"
PAIR_EVAL_BATCH_SIZE="${PAIR_EVAL_BATCH_SIZE:-32}"
PAIR_EVAL_NUM_WORKERS="${PAIR_EVAL_NUM_WORKERS:-0}"
PAIR_CLS_DISTILL_WEIGHT="${PAIR_CLS_DISTILL_WEIGHT:-1.0}"
PAIR_TOKEN_DISTILL_WEIGHT="${PAIR_TOKEN_DISTILL_WEIGHT:-0.02}"
PAIR_SEED="${PAIR_SEED:-0}"

BASELINE_ACCURACY_PROFILE="${BASELINE_ACCURACY_PROFILE:-default}"
BASELINE_SEED="${BASELINE_SEED:-$PAIR_SEED}"
BASELINE_AUGMENTATION_PROFILE="${BASELINE_AUGMENTATION_PROFILE:-timm}"
BASELINE_COLOR_JITTER="${BASELINE_COLOR_JITTER:-0.4}"
BASELINE_AA="${BASELINE_AA:-rand-m9-mstd0.5-inc1}"
BASELINE_REPROB="${BASELINE_REPROB:-0.25}"
BASELINE_BATCH_SIZE="${BASELINE_BATCH_SIZE:-32}"
BASELINE_CLIP_GRAD="${BASELINE_CLIP_GRAD:-1.0}"
BASELINE_CLASS_WEIGHT_MODE="${BASELINE_CLASS_WEIGHT_MODE:-none}"
BASELINE_CLASS_WEIGHT_POWER="${BASELINE_CLASS_WEIGHT_POWER:-1.0}"
BASELINE_TRAIN_SAMPLER_MODE="${BASELINE_TRAIN_SAMPLER_MODE:-distributed}"
BASELINE_MODEL_EMA="${BASELINE_MODEL_EMA:-false}"
BASELINE_SMOOTHING="${BASELINE_SMOOTHING:-0.1}"
BASELINE_WEIGHT_DECAY="${BASELINE_WEIGHT_DECAY:-0.05}"
BASELINE_LR="${BASELINE_LR:-3e-6}"
BASELINE_MIN_LR="${BASELINE_MIN_LR:-1e-7}"
BASELINE_WARMUP_STEPS="${BASELINE_WARMUP_STEPS:-20}"
BASELINE_GROUPA_LR_SCALE="${BASELINE_GROUPA_LR_SCALE:-0.1}"
BASELINE_CLS_TOKEN_FULL_LR="${BASELINE_CLS_TOKEN_FULL_LR:-false}"
BASELINE_TRAIN_POS_EMBED="${BASELINE_TRAIN_POS_EMBED:-false}"
BASELINE_FREEZE_PATCH_EMBED_PROJ="${BASELINE_FREEZE_PATCH_EMBED_PROJ:-false}"
BASELINE_FREEZE_PATCH_EMBED_WEIGHT="${BASELINE_FREEZE_PATCH_EMBED_WEIGHT:-false}"
BASELINE_FREEZE_PATCH_EMBED_BIAS="${BASELINE_FREEZE_PATCH_EMBED_BIAS:-false}"
BASELINE_PATCH_EMBED_BIAS_INIT_MODE="${BASELINE_PATCH_EMBED_BIAS_INIT_MODE:-pretrained}"
BASELINE_SKIP_PATCH_EMBED_BIAS_PRETRAINED="${BASELINE_SKIP_PATCH_EMBED_BIAS_PRETRAINED:-false}"
BASELINE_PRETRAINED_FIX_STEP="${BASELINE_PRETRAINED_FIX_STEP:-0}"

CANDIDATE_ACCURACY_PROFILE="${CANDIDATE_ACCURACY_PROFILE:-default}"
CANDIDATE_SEED="${CANDIDATE_SEED:-$PAIR_SEED}"
CANDIDATE_AUGMENTATION_PROFILE="${CANDIDATE_AUGMENTATION_PROFILE:-timm}"
CANDIDATE_COLOR_JITTER="${CANDIDATE_COLOR_JITTER:-0.4}"
CANDIDATE_AA="${CANDIDATE_AA:-rand-m9-mstd0.5-inc1}"
CANDIDATE_REPROB="${CANDIDATE_REPROB:-0.25}"
CANDIDATE_BATCH_SIZE="${CANDIDATE_BATCH_SIZE:-32}"
CANDIDATE_CLIP_GRAD="${CANDIDATE_CLIP_GRAD:-1.0}"
CANDIDATE_CLASS_WEIGHT_MODE="${CANDIDATE_CLASS_WEIGHT_MODE:-none}"
CANDIDATE_CLASS_WEIGHT_POWER="${CANDIDATE_CLASS_WEIGHT_POWER:-1.0}"
CANDIDATE_TRAIN_SAMPLER_MODE="${CANDIDATE_TRAIN_SAMPLER_MODE:-distributed}"
CANDIDATE_MODEL_EMA="${CANDIDATE_MODEL_EMA:-false}"
CANDIDATE_SMOOTHING="${CANDIDATE_SMOOTHING:-0.1}"
CANDIDATE_WEIGHT_DECAY="${CANDIDATE_WEIGHT_DECAY:-0.05}"
CANDIDATE_LR="${CANDIDATE_LR:-3e-6}"
CANDIDATE_MIN_LR="${CANDIDATE_MIN_LR:-1e-7}"
CANDIDATE_WARMUP_STEPS="${CANDIDATE_WARMUP_STEPS:-20}"
CANDIDATE_GROUPA_LR_SCALE="${CANDIDATE_GROUPA_LR_SCALE:-0.1}"
CANDIDATE_CLS_TOKEN_FULL_LR="${CANDIDATE_CLS_TOKEN_FULL_LR:-false}"
CANDIDATE_TRAIN_POS_EMBED="${CANDIDATE_TRAIN_POS_EMBED:-false}"
CANDIDATE_FREEZE_PATCH_EMBED_PROJ="${CANDIDATE_FREEZE_PATCH_EMBED_PROJ:-false}"
CANDIDATE_FREEZE_PATCH_EMBED_WEIGHT="${CANDIDATE_FREEZE_PATCH_EMBED_WEIGHT:-false}"
CANDIDATE_FREEZE_PATCH_EMBED_BIAS="${CANDIDATE_FREEZE_PATCH_EMBED_BIAS:-false}"
CANDIDATE_PATCH_EMBED_BIAS_INIT_MODE="${CANDIDATE_PATCH_EMBED_BIAS_INIT_MODE:-pretrained}"
CANDIDATE_SKIP_PATCH_EMBED_BIAS_PRETRAINED="${CANDIDATE_SKIP_PATCH_EMBED_BIAS_PRETRAINED:-false}"
CANDIDATE_PRETRAINED_FIX_STEP="${CANDIDATE_PRETRAINED_FIX_STEP:-0}"

DEFAULT_PAIR_NAME="accprof_epoch${PAIR_EPOCHS}_$(date +%Y%m%d_%H%M%S)"
PAIR_NAME="${PAIR_NAME:-$DEFAULT_PAIR_NAME}"
BASELINE_RUN_NAME="${BASELINE_RUN_NAME:-${PAIR_NAME}_baseline}"
CANDIDATE_RUN_NAME="${CANDIDATE_RUN_NAME:-${PAIR_NAME}_candidate}"
PAIR_OUTPUT_DIR="${PAIR_OUTPUT_DIR:-$REPO_ROOT/results/accuracy_profile_imbalance/$PAIR_NAME}"
BASELINE_RESULT_DIR="${BASELINE_RESULT_DIR:-$REPO_ROOT/results/accuracy_profile_imbalance/$BASELINE_RUN_NAME}"
CANDIDATE_RESULT_DIR="${CANDIDATE_RESULT_DIR:-$REPO_ROOT/results/accuracy_profile_imbalance/$CANDIDATE_RUN_NAME}"
BASELINE_RUN_DIR="${BASELINE_RUN_DIR:-$REPO_ROOT/artifacts/train_runs/$BASELINE_RUN_NAME}"
CANDIDATE_RUN_DIR="${CANDIDATE_RUN_DIR:-$REPO_ROOT/artifacts/train_runs/$CANDIDATE_RUN_NAME}"
BASELINE_CHECKPOINT_NAME="${BASELINE_CHECKPOINT_NAME:-checkpoint-best.pth}"
CANDIDATE_CHECKPOINT_NAME="${CANDIDATE_CHECKPOINT_NAME:-checkpoint-best.pth}"
BASELINE_CHECKPOINT_MODEL_KEY="${BASELINE_CHECKPOINT_MODEL_KEY:-model}"
CANDIDATE_CHECKPOINT_MODEL_KEY="${CANDIDATE_CHECKPOINT_MODEL_KEY:-model}"

BASELINE_THRESHOLD_JSON="$BASELINE_RUN_DIR/threshold_best.json"
CANDIDATE_THRESHOLD_JSON="$CANDIDATE_RUN_DIR/threshold_best.json"
BASELINE_THRESHOLD_EVAL_JSON="$BASELINE_RUN_DIR/threshold_eval.json"
CANDIDATE_THRESHOLD_EVAL_JSON="$CANDIDATE_RUN_DIR/threshold_eval.json"
BASELINE_PLAINTEXT_EVAL_JSON="$BASELINE_RESULT_DIR/plaintext_eval.json"
CANDIDATE_PLAINTEXT_EVAL_JSON="$CANDIDATE_RESULT_DIR/plaintext_eval.json"
BASELINE_PLAINTEXT_EVAL_CSV="$BASELINE_RESULT_DIR/plaintext_eval.csv"
CANDIDATE_PLAINTEXT_EVAL_CSV="$CANDIDATE_RESULT_DIR/plaintext_eval.csv"
PAIR_COMPARE_JSON="$PAIR_OUTPUT_DIR/accuracy_profile_compare.json"
PAIR_COMPARE_MD="$PAIR_OUTPUT_DIR/accuracy_profile_compare.md"
PAIR_STUDY_KIND="${PAIR_STUDY_KIND:-accuracy_profile_imbalance_epoch${PAIR_EPOCHS}}"
ENABLE_PUBLIC_LOGIT_BIAS_CALIBRATION="${ENABLE_PUBLIC_LOGIT_BIAS_CALIBRATION:-true}"

print_plan() {
  cat <<EOF
[accprof-pair] pair_name=$PAIR_NAME
[accprof-pair] baseline_run_name=$BASELINE_RUN_NAME
[accprof-pair] candidate_run_name=$CANDIDATE_RUN_NAME
[accprof-pair] pair_epochs=$PAIR_EPOCHS
[accprof-pair] pair_secure_static_depth=$PAIR_SECURE_STATIC_DEPTH
[accprof-pair] pair_secure_static_skip_pruning=$PAIR_SECURE_STATIC_SKIP_PRUNING
[accprof-pair] pair_use_mask_pruning=$PAIR_USE_MASK_PRUNING
[accprof-pair] baseline_accuracy_profile=$BASELINE_ACCURACY_PROFILE
[accprof-pair] baseline_seed=$BASELINE_SEED
[accprof-pair] baseline_augmentation_profile=$BASELINE_AUGMENTATION_PROFILE
[accprof-pair] baseline_color_jitter=$BASELINE_COLOR_JITTER
[accprof-pair] baseline_aa=$BASELINE_AA
[accprof-pair] baseline_reprob=$BASELINE_REPROB
[accprof-pair] baseline_batch_size=$BASELINE_BATCH_SIZE
[accprof-pair] baseline_clip_grad=$BASELINE_CLIP_GRAD
[accprof-pair] baseline_class_weight_mode=$BASELINE_CLASS_WEIGHT_MODE
[accprof-pair] baseline_class_weight_power=$BASELINE_CLASS_WEIGHT_POWER
[accprof-pair] baseline_train_sampler_mode=$BASELINE_TRAIN_SAMPLER_MODE
[accprof-pair] baseline_model_ema=$BASELINE_MODEL_EMA
[accprof-pair] baseline_smoothing=$BASELINE_SMOOTHING
[accprof-pair] baseline_weight_decay=$BASELINE_WEIGHT_DECAY
[accprof-pair] baseline_lr=$BASELINE_LR
[accprof-pair] baseline_min_lr=$BASELINE_MIN_LR
[accprof-pair] baseline_warmup_steps=$BASELINE_WARMUP_STEPS
[accprof-pair] baseline_groupa_lr_scale=$BASELINE_GROUPA_LR_SCALE
[accprof-pair] baseline_cls_token_full_lr=$BASELINE_CLS_TOKEN_FULL_LR
[accprof-pair] baseline_train_pos_embed=$BASELINE_TRAIN_POS_EMBED
[accprof-pair] baseline_freeze_patch_embed_proj=$BASELINE_FREEZE_PATCH_EMBED_PROJ
[accprof-pair] baseline_freeze_patch_embed_weight=$BASELINE_FREEZE_PATCH_EMBED_WEIGHT
[accprof-pair] baseline_freeze_patch_embed_bias=$BASELINE_FREEZE_PATCH_EMBED_BIAS
[accprof-pair] baseline_patch_embed_bias_init_mode=$BASELINE_PATCH_EMBED_BIAS_INIT_MODE
[accprof-pair] baseline_skip_patch_embed_bias_pretrained=$BASELINE_SKIP_PATCH_EMBED_BIAS_PRETRAINED
[accprof-pair] baseline_pretrained_fix_step=$BASELINE_PRETRAINED_FIX_STEP
[accprof-pair] candidate_accuracy_profile=$CANDIDATE_ACCURACY_PROFILE
[accprof-pair] candidate_seed=$CANDIDATE_SEED
[accprof-pair] candidate_augmentation_profile=$CANDIDATE_AUGMENTATION_PROFILE
[accprof-pair] candidate_color_jitter=$CANDIDATE_COLOR_JITTER
[accprof-pair] candidate_aa=$CANDIDATE_AA
[accprof-pair] candidate_reprob=$CANDIDATE_REPROB
[accprof-pair] candidate_batch_size=$CANDIDATE_BATCH_SIZE
[accprof-pair] candidate_clip_grad=$CANDIDATE_CLIP_GRAD
[accprof-pair] candidate_class_weight_mode=$CANDIDATE_CLASS_WEIGHT_MODE
[accprof-pair] candidate_class_weight_power=$CANDIDATE_CLASS_WEIGHT_POWER
[accprof-pair] candidate_train_sampler_mode=$CANDIDATE_TRAIN_SAMPLER_MODE
[accprof-pair] candidate_model_ema=$CANDIDATE_MODEL_EMA
[accprof-pair] candidate_smoothing=$CANDIDATE_SMOOTHING
[accprof-pair] candidate_weight_decay=$CANDIDATE_WEIGHT_DECAY
[accprof-pair] candidate_lr=$CANDIDATE_LR
[accprof-pair] candidate_min_lr=$CANDIDATE_MIN_LR
[accprof-pair] candidate_warmup_steps=$CANDIDATE_WARMUP_STEPS
[accprof-pair] candidate_groupa_lr_scale=$CANDIDATE_GROUPA_LR_SCALE
[accprof-pair] candidate_cls_token_full_lr=$CANDIDATE_CLS_TOKEN_FULL_LR
[accprof-pair] candidate_train_pos_embed=$CANDIDATE_TRAIN_POS_EMBED
[accprof-pair] candidate_freeze_patch_embed_proj=$CANDIDATE_FREEZE_PATCH_EMBED_PROJ
[accprof-pair] candidate_freeze_patch_embed_weight=$CANDIDATE_FREEZE_PATCH_EMBED_WEIGHT
[accprof-pair] candidate_freeze_patch_embed_bias=$CANDIDATE_FREEZE_PATCH_EMBED_BIAS
[accprof-pair] candidate_patch_embed_bias_init_mode=$CANDIDATE_PATCH_EMBED_BIAS_INIT_MODE
[accprof-pair] candidate_skip_patch_embed_bias_pretrained=$CANDIDATE_SKIP_PATCH_EMBED_BIAS_PRETRAINED
[accprof-pair] candidate_pretrained_fix_step=$CANDIDATE_PRETRAINED_FIX_STEP
[accprof-pair] train_data_path=$TRAIN_DATA_PATH
[accprof-pair] val_data_path=$VAL_DATA_PATH
[accprof-pair] base_checkpoint=$BASE_CHECKPOINT
[accprof-pair] teacher_checkpoint=$TEACHER_CHECKPOINT
[accprof-pair] baseline_run_dir=$BASELINE_RUN_DIR
[accprof-pair] candidate_run_dir=$CANDIDATE_RUN_DIR
[accprof-pair] baseline_checkpoint_name=$BASELINE_CHECKPOINT_NAME
[accprof-pair] candidate_checkpoint_name=$CANDIDATE_CHECKPOINT_NAME
[accprof-pair] baseline_checkpoint_model_key=$BASELINE_CHECKPOINT_MODEL_KEY
[accprof-pair] candidate_checkpoint_model_key=$CANDIDATE_CHECKPOINT_MODEL_KEY
[accprof-pair] enable_public_logit_bias_calibration=$ENABLE_PUBLIC_LOGIT_BIAS_CALIBRATION
[accprof-pair] baseline_result_dir=$BASELINE_RESULT_DIR
[accprof-pair] candidate_result_dir=$CANDIDATE_RESULT_DIR
[accprof-pair] pair_output_dir=$PAIR_OUTPUT_DIR
EOF
}

ensure_common_inputs() {
  if [[ ! -f "$BASE_CHECKPOINT" ]]; then
    echo "[accprof-pair] missing BASE_CHECKPOINT: $BASE_CHECKPOINT" >&2
    exit 2
  fi
  if [[ ! -f "$TEACHER_CHECKPOINT" ]]; then
    echo "[accprof-pair] missing TEACHER_CHECKPOINT: $TEACHER_CHECKPOINT" >&2
    exit 2
  fi
  if [[ ! -d "$TRAIN_DATA_PATH" ]]; then
    echo "[accprof-pair] missing TRAIN_DATA_PATH: $TRAIN_DATA_PATH" >&2
    exit 2
  fi
  if [[ ! -d "$VAL_DATA_PATH" ]]; then
    echo "[accprof-pair] missing VAL_DATA_PATH: $VAL_DATA_PATH" >&2
    exit 2
  fi
}

prepare_train_env() {
  unset PROTOCOL_AWARE_PROFILE || true
  unset PROTOCOL_AWARE_FORCE_RECIPE_PRUNING_MARGIN || true
  export TRAIN_DATA_PATH VAL_DATA_PATH BASE_BUNDLE_DIR BASE_CHECKPOINT TEACHER_CHECKPOINT
  export SECURE_STATIC_DEPTH="$PAIR_SECURE_STATIC_DEPTH"
  export SECURE_STATIC_SKIP_PRUNING="$PAIR_SECURE_STATIC_SKIP_PRUNING"
  export USE_MASK_PRUNING="$PAIR_USE_MASK_PRUNING"
  export EPOCHS="$PAIR_EPOCHS"
  export CLS_DISTILL_WEIGHT="$PAIR_CLS_DISTILL_WEIGHT"
  export TOKEN_DISTILL_WEIGHT="$PAIR_TOKEN_DISTILL_WEIGHT"
  export PRUNING_MARGIN_WEIGHT=0.0
  export PRUNING_MARGIN_TARGET="${PRUNING_MARGIN_TARGET:-1e-4}"
  export PRUNING_MARGIN_MODE="${PRUNING_MARGIN_MODE:-hinge}"
  export PRUNING_MARGIN_START_EPOCH=0
  unset PRUNING_MARGIN_STAGE_WEIGHTS || true
}

run_one_train() {
  local run_name="$1"
  local run_dir="$2"
  local accuracy_profile="$3"
  local seed="$4"
  local augmentation_profile="$5"
  local color_jitter="$6"
  local aa="$7"
  local reprob="$8"
  local batch_size="$9"
  local clip_grad="${10}"
  local class_weight_mode="${11}"
  local class_weight_power="${12}"
  local train_sampler_mode="${13}"
  local model_ema="${14}"
  local smoothing="${15}"
  local weight_decay="${16}"
  local lr="${17}"
  local min_lr="${18}"
  local warmup_steps="${19}"
  local groupa_lr_scale="${20}"
  local cls_token_full_lr="${21}"
  local train_pos_embed="${22}"
  local freeze_patch_embed_proj="${23}"
  local freeze_patch_embed_weight="${24}"
  local freeze_patch_embed_bias="${25}"
  local patch_embed_bias_init_mode="${26}"
  local skip_patch_embed_bias_pretrained="${27}"
  local pretrained_fix_step="${28}"
  shift 28

  ensure_common_inputs
  prepare_train_env
  export RUN_NAME="$run_name"
  export RUN_DIR="$run_dir"
  export SEED="$seed"
  export ACCURACY_PROFILE="$accuracy_profile"
  export AUGMENTATION_PROFILE="$augmentation_profile"
  export COLOR_JITTER="$color_jitter"
  export AA="$aa"
  export REPROB="$reprob"
  export BATCH_SIZE="$batch_size"
  export CLIP_GRAD="$clip_grad"
  export CLASS_WEIGHT_MODE="$class_weight_mode"
  export CLASS_WEIGHT_POWER="$class_weight_power"
  export TRAIN_SAMPLER_MODE="$train_sampler_mode"
  export MODEL_EMA="$model_ema"
  export SMOOTHING="$smoothing"
  export WEIGHT_DECAY="$weight_decay"
  export LR="$lr"
  export MIN_LR="$min_lr"
  export WARMUP_STEPS="$warmup_steps"
  export GROUPA_LR_SCALE="$groupa_lr_scale"
  export CLS_TOKEN_FULL_LR="$cls_token_full_lr"
  export TRAIN_POS_EMBED="$train_pos_embed"
  export FREEZE_PATCH_EMBED_PROJ="$freeze_patch_embed_proj"
  export FREEZE_PATCH_EMBED_WEIGHT="$freeze_patch_embed_weight"
  export FREEZE_PATCH_EMBED_BIAS="$freeze_patch_embed_bias"
  export PATCH_EMBED_BIAS_INIT_MODE="$patch_embed_bias_init_mode"
  export SKIP_PATCH_EMBED_BIAS_PRETRAINED="$skip_patch_embed_bias_pretrained"
  export PRETRAINED_FIX_STEP="$pretrained_fix_step"

  echo "[accprof-pair] train $run_name -> $run_dir"
  bash "$SCRIPT_DIR/run_secure_static_distill_train.sh" epoch1 "$@"
}

run_baseline_train() {
  run_one_train \
    "$BASELINE_RUN_NAME" \
    "$BASELINE_RUN_DIR" \
    "$BASELINE_ACCURACY_PROFILE" \
    "$BASELINE_SEED" \
    "$BASELINE_AUGMENTATION_PROFILE" \
    "$BASELINE_COLOR_JITTER" \
    "$BASELINE_AA" \
    "$BASELINE_REPROB" \
    "$BASELINE_BATCH_SIZE" \
    "$BASELINE_CLIP_GRAD" \
    "$BASELINE_CLASS_WEIGHT_MODE" \
    "$BASELINE_CLASS_WEIGHT_POWER" \
    "$BASELINE_TRAIN_SAMPLER_MODE" \
    "$BASELINE_MODEL_EMA" \
    "$BASELINE_SMOOTHING" \
    "$BASELINE_WEIGHT_DECAY" \
    "$BASELINE_LR" \
    "$BASELINE_MIN_LR" \
    "$BASELINE_WARMUP_STEPS" \
    "$BASELINE_GROUPA_LR_SCALE" \
    "$BASELINE_CLS_TOKEN_FULL_LR" \
    "$BASELINE_TRAIN_POS_EMBED" \
    "$BASELINE_FREEZE_PATCH_EMBED_PROJ" \
    "$BASELINE_FREEZE_PATCH_EMBED_WEIGHT" \
    "$BASELINE_FREEZE_PATCH_EMBED_BIAS" \
    "$BASELINE_PATCH_EMBED_BIAS_INIT_MODE" \
    "$BASELINE_SKIP_PATCH_EMBED_BIAS_PRETRAINED" \
    "$BASELINE_PRETRAINED_FIX_STEP" \
    "$@"
}

run_candidate_train() {
  run_one_train \
    "$CANDIDATE_RUN_NAME" \
    "$CANDIDATE_RUN_DIR" \
    "$CANDIDATE_ACCURACY_PROFILE" \
    "$CANDIDATE_SEED" \
    "$CANDIDATE_AUGMENTATION_PROFILE" \
    "$CANDIDATE_COLOR_JITTER" \
    "$CANDIDATE_AA" \
    "$CANDIDATE_REPROB" \
    "$CANDIDATE_BATCH_SIZE" \
    "$CANDIDATE_CLIP_GRAD" \
    "$CANDIDATE_CLASS_WEIGHT_MODE" \
    "$CANDIDATE_CLASS_WEIGHT_POWER" \
    "$CANDIDATE_TRAIN_SAMPLER_MODE" \
    "$CANDIDATE_MODEL_EMA" \
    "$CANDIDATE_SMOOTHING" \
    "$CANDIDATE_WEIGHT_DECAY" \
    "$CANDIDATE_LR" \
    "$CANDIDATE_MIN_LR" \
    "$CANDIDATE_WARMUP_STEPS" \
    "$CANDIDATE_GROUPA_LR_SCALE" \
    "$CANDIDATE_CLS_TOKEN_FULL_LR" \
    "$CANDIDATE_TRAIN_POS_EMBED" \
    "$CANDIDATE_FREEZE_PATCH_EMBED_PROJ" \
    "$CANDIDATE_FREEZE_PATCH_EMBED_WEIGHT" \
    "$CANDIDATE_FREEZE_PATCH_EMBED_BIAS" \
    "$CANDIDATE_PATCH_EMBED_BIAS_INIT_MODE" \
    "$CANDIDATE_SKIP_PATCH_EMBED_BIAS_PRETRAINED" \
    "$CANDIDATE_PRETRAINED_FIX_STEP" \
    "$@"
}

run_post_common() {
  local run_dir="$1"
  local result_dir="$2"
  local threshold_json="$3"
  local threshold_eval_json="$4"
  local plaintext_eval_json="$5"
  local plaintext_eval_csv="$6"
  local label="$7"
  local checkpoint_name="$8"
  local checkpoint_model_key="$9"
  local checkpoint_path="$run_dir/$checkpoint_name"

  if [[ ! -f "$checkpoint_path" ]]; then
    echo "[accprof-pair] missing checkpoint: $checkpoint_path" >&2
    exit 2
  fi

  mkdir -p "$result_dir"
  echo "[accprof-pair] threshold search -> $threshold_json"
  "$PYTHON_BIN" tools/transshield_binary_threshold_search.py search \
    --repo-root "$PAIR_REPO_ROOT" \
    --checkpoint "$checkpoint_path" \
    --checkpoint-model-key "$checkpoint_model_key" \
    --data-path "$VAL_DATA_PATH" \
    --device "$PAIR_EVAL_DEVICE" \
    --batch-size "$PAIR_EVAL_BATCH_SIZE" \
    --num-workers "$PAIR_EVAL_NUM_WORKERS" \
    --output-json "$threshold_json"

  echo "[accprof-pair] threshold eval -> $threshold_eval_json"
  "$PYTHON_BIN" tools/transshield_binary_threshold_search.py eval \
    --repo-root "$PAIR_REPO_ROOT" \
    --checkpoint "$checkpoint_path" \
    --checkpoint-model-key "$checkpoint_model_key" \
    --threshold-json "$threshold_json" \
    --data-path "$VAL_DATA_PATH" \
    --device "$PAIR_EVAL_DEVICE" \
    --batch-size "$PAIR_EVAL_BATCH_SIZE" \
    --num-workers "$PAIR_EVAL_NUM_WORKERS" \
    --output-json "$threshold_eval_json"

  echo "[accprof-pair] plaintext eval -> $plaintext_eval_json"
  "$PYTHON_BIN" tools/transshield_plaintext_checkpoint_eval.py \
    --repo-root "$PAIR_REPO_ROOT" \
    --checkpoint "$checkpoint_path" \
    --checkpoint-model-key "$checkpoint_model_key" \
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
    "$BASELINE_PLAINTEXT_EVAL_CSV" \
    "$BASELINE_RUN_NAME" \
    "$BASELINE_CHECKPOINT_NAME" \
    "$BASELINE_CHECKPOINT_MODEL_KEY"
}

run_post_candidate() {
  ensure_common_inputs
  run_post_common \
    "$CANDIDATE_RUN_DIR" \
    "$CANDIDATE_RESULT_DIR" \
    "$CANDIDATE_THRESHOLD_JSON" \
    "$CANDIDATE_THRESHOLD_EVAL_JSON" \
    "$CANDIDATE_PLAINTEXT_EVAL_JSON" \
    "$CANDIDATE_PLAINTEXT_EVAL_CSV" \
    "$CANDIDATE_RUN_NAME" \
    "$CANDIDATE_CHECKPOINT_NAME" \
    "$CANDIDATE_CHECKPOINT_MODEL_KEY"
}

run_compare() {
  mkdir -p "$PAIR_OUTPUT_DIR"
  echo "[accprof-pair] pair compare -> $PAIR_COMPARE_JSON"
  local calibration_args=()
  if [[ "$ENABLE_PUBLIC_LOGIT_BIAS_CALIBRATION" == "true" ]]; then
    calibration_args=(
      --enable-public-logit-bias-calibration
      --baseline-plaintext-eval-csv "$BASELINE_PLAINTEXT_EVAL_CSV"
      --candidate-plaintext-eval-csv "$CANDIDATE_PLAINTEXT_EVAL_CSV"
    )
  fi
  "$PYTHON_BIN" tools/transshield_training_pair_compare.py \
    --study-kind "$PAIR_STUDY_KIND" \
    --baseline-run-dir "$BASELINE_RUN_DIR" \
    --candidate-run-dir "$CANDIDATE_RUN_DIR" \
    --baseline-label "$BASELINE_RUN_NAME" \
    --candidate-label "$CANDIDATE_RUN_NAME" \
    --baseline-threshold-search-json "$BASELINE_THRESHOLD_EVAL_JSON" \
    --candidate-threshold-search-json "$CANDIDATE_THRESHOLD_EVAL_JSON" \
    --baseline-plaintext-eval-json "$BASELINE_PLAINTEXT_EVAL_JSON" \
    --candidate-plaintext-eval-json "$CANDIDATE_PLAINTEXT_EVAL_JSON" \
    "${calibration_args[@]}" \
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
