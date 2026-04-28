#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_tracka_training_common.sh"

usage() {
  cat <<'EOF' >&2
Usage:
  bash scripts/run_tracka_train.sh source [debug80|epoch3|epoch5|full20] [seed]
  bash scripts/run_tracka_train.sh compat [debug80|epoch1|epoch5|full20] [seed]
EOF
  exit 1
}

setup_source_defaults() {
  MODE="${1:-epoch5}"
  SEED="${2:-0}"

  FINAL_REPO_ROOT="${FINAL_REPO_ROOT:-/data/wyb/Transshield_final}"
  TRAIN_ENTRY="${TRAIN_ENTRY:-$FINAL_REPO_ROOT/training_source_tracka/main.py}"
  PYTHON_BIN="${PYTHON_BIN:-/data/wyb/conda_envs/transshield/bin/python}"
  DATA_ROOT="${DATA_ROOT:-/data/wyb/pneumoniamnist_imagefolder_subset}"
  TMP_ROOT="${TMP_ROOT:-/data/wyb/tmp}"
  NONEMPTY_KEEP_GUARD="${NONEMPTY_KEEP_GUARD:-false}"
  LOSS_GRAD_ATTRIB="${LOSS_GRAD_ATTRIB:-false}"
  LOSS_GRAD_ATTRIB_PARAM="${LOSS_GRAD_ATTRIB_PARAM:-score_predictor.1.out_proj.weight}"
  CLS_DISTILL_WEIGHT="${CLS_DISTILL_WEIGHT:-1.0}"
  TOKEN_DISTILL_WEIGHT="${TOKEN_DISTILL_WEIGHT:-0.02}"
  RATIO_WEIGHT="${RATIO_WEIGHT:-2.0}"
  EPOCHS=20
  PREFIX="tracka-source"

  case "$MODE" in
    debug80)
      DEFAULT_RUN_NAME="tracka_source_debug80_seed${SEED}"
      DEBUG_ARGS=(--debug_max_steps 80)
      STOP_AFTER_EPOCH=0
      SAVE_CKPT_FLAG=false
      ;;
    epoch3)
      DEFAULT_RUN_NAME="tracka_source_epoch3_sched20_seed${SEED}"
      DEBUG_ARGS=()
      STOP_AFTER_EPOCH=3
      SAVE_CKPT_FLAG=true
      ;;
    epoch5)
      DEFAULT_RUN_NAME="tracka_source_epoch5_sched20_seed${SEED}"
      DEBUG_ARGS=()
      STOP_AFTER_EPOCH=5
      SAVE_CKPT_FLAG=true
      ;;
    full20)
      DEFAULT_RUN_NAME="tracka_source_full20_seed${SEED}"
      DEBUG_ARGS=()
      STOP_AFTER_EPOCH=0
      SAVE_CKPT_FLAG=true
      ;;
    *)
      usage
      ;;
  esac

  RUN_NAME="${RUN_NAME:-$DEFAULT_RUN_NAME}"
  OUTPUT_DIR="${OUTPUT_DIR:-$FINAL_REPO_ROOT/artifacts/train_runs/$RUN_NAME}"
  LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/tb}"
}

setup_compat_defaults() {
  MODE="${1:-debug80}"
  SEED="${2:-1}"

  FINAL_REPO_ROOT="${FINAL_REPO_ROOT:-/data/wyb/Transshield_final}"
  TRAIN_ENTRY="${TRAIN_ENTRY:-$FINAL_REPO_ROOT/training_compat/main.py}"
  PYTHON_BIN="${PYTHON_BIN:-/data/wyb/conda_envs/transshield/bin/python}"
  DATA_ROOT="${DATA_ROOT:-/data/wyb/pneumoniamnist_imagefolder_subset}"
  TMP_ROOT="${TMP_ROOT:-/data/wyb/tmp}"
  ACTIVATION_LR_SCALE="${ACTIVATION_LR_SCALE:-10.0}"
  TRAIN_SAMPLER_MODE="${TRAIN_SAMPLER_MODE:-distributed}"
  CROP_PCT="${CROP_PCT:-0.875}"
  WEIGHT_DECAY_END="${WEIGHT_DECAY_END:-0.05}"
  MODEL_EMA="${MODEL_EMA:-false}"
  MODEL_EMA_DECAY="${MODEL_EMA_DECAY:-0.9999}"
  MODEL_EMA_EVAL="${MODEL_EMA_EVAL:-true}"
  NONEMPTY_KEEP_GUARD="${NONEMPTY_KEEP_GUARD:-false}"
  CLS_DISTILL_WEIGHT="${CLS_DISTILL_WEIGHT:-1.0}"
  TOKEN_DISTILL_WEIGHT="${TOKEN_DISTILL_WEIGHT:-0.02}"
  RATIO_WEIGHT="${RATIO_WEIGHT:-2.0}"
  STOP_AFTER_EPOCH="${STOP_AFTER_EPOCH:-}"
  PREFIX="tracka-compat"

  case "$MODE" in
    debug80)
      DEFAULT_RUN_NAME="pneumonia_transshield_tracka_lr3e5_timm_debug80_seed${SEED}"
      DEBUG_ARGS=(--debug_max_steps 80)
      DEFAULT_EPOCHS=20
      DEFAULT_STOP_AFTER_EPOCH=0
      SAVE_CKPT_FLAG=false
      ;;
    epoch1)
      DEFAULT_RUN_NAME="pneumonia_transshield_tracka_lr3e5_timm_epoch1_seed${SEED}"
      DEBUG_ARGS=()
      DEFAULT_EPOCHS=20
      DEFAULT_STOP_AFTER_EPOCH=1
      SAVE_CKPT_FLAG=true
      ;;
    epoch5)
      DEFAULT_RUN_NAME="pneumonia_transshield_tracka_lr3e5_timm_epoch5_seed${SEED}"
      DEBUG_ARGS=()
      DEFAULT_EPOCHS=20
      DEFAULT_STOP_AFTER_EPOCH=5
      SAVE_CKPT_FLAG=true
      ;;
    full20)
      DEFAULT_RUN_NAME="pneumonia_transshield_tracka_lr3e5_timm_seed${SEED}"
      DEBUG_ARGS=()
      DEFAULT_EPOCHS=20
      DEFAULT_STOP_AFTER_EPOCH=0
      SAVE_CKPT_FLAG=true
      ;;
    *)
      usage
      ;;
  esac

  RUN_NAME="${RUN_NAME:-$DEFAULT_RUN_NAME}"
  EPOCHS="${EPOCHS:-$DEFAULT_EPOCHS}"
  if [[ -z "$STOP_AFTER_EPOCH" ]]; then
    STOP_AFTER_EPOCH="$DEFAULT_STOP_AFTER_EPOCH"
  fi

  OUTPUT_DIR="${OUTPUT_DIR:-$FINAL_REPO_ROOT/artifacts/train_runs/$RUN_NAME}"
  LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/tb}"
}

validate_common_prereqs() {
  tracka_require_executable "$PYTHON_BIN" "python interpreter"
  tracka_require_file "$TRAIN_ENTRY" "training entry"
  tracka_require_dataset_layout "$DATA_ROOT"
  tracka_prepare_run_dirs "$TMP_ROOT" "$OUTPUT_DIR" "$LOG_DIR"
  cd "$FINAL_REPO_ROOT"
}

validate_compat_prereqs() {
  tracka_require_file "$FINAL_REPO_ROOT/pretrained/deit_small_patch16_224-cd65a155.pth" "pretrained weight"
}

build_source_args() {
  TRAIN_ARGS=()
  tracka_append_common_args \
    TRAIN_ARGS \
    "$DATA_ROOT" \
    "$OUTPUT_DIR" \
    "$LOG_DIR" \
    "$EPOCHS" \
    "$SEED" \
    "$RATIO_WEIGHT" \
    10.0 \
    "$SAVE_CKPT_FLAG" \
    "$STOP_AFTER_EPOCH" \
    "$CLS_DISTILL_WEIGHT" \
    "$TOKEN_DISTILL_WEIGHT"
  TRAIN_ARGS+=(
    --model_ema false
    --nonempty_keep_guard "$NONEMPTY_KEEP_GUARD"
    --loss_grad_attrib "$LOSS_GRAD_ATTRIB"
    --loss_grad_attrib_param "$LOSS_GRAD_ATTRIB_PARAM"
  )
  TRAIN_ARGS+=("${DEBUG_ARGS[@]}")
}

build_compat_args() {
  TRAIN_ARGS=()
  tracka_append_common_args \
    TRAIN_ARGS \
    "$DATA_ROOT" \
    "$OUTPUT_DIR" \
    "$LOG_DIR" \
    "$EPOCHS" \
    "$SEED" \
    "$RATIO_WEIGHT" \
    "$ACTIVATION_LR_SCALE" \
    "$SAVE_CKPT_FLAG" \
    "$STOP_AFTER_EPOCH" \
    "$CLS_DISTILL_WEIGHT" \
    "$TOKEN_DISTILL_WEIGHT"
  TRAIN_ARGS+=(
    --weight_decay_end "$WEIGHT_DECAY_END"
    --model_ema "$MODEL_EMA"
    --model_ema_decay "$MODEL_EMA_DECAY"
    --model_ema_eval "$MODEL_EMA_EVAL"
    --nonempty_keep_guard "$NONEMPTY_KEEP_GUARD"
    --train_sampler_mode "$TRAIN_SAMPLER_MODE"
    --crop_pct "$CROP_PCT"
  )
  TRAIN_ARGS+=("${DEBUG_ARGS[@]}")
}

print_source_summary() {
  tracka_print_kv "$PREFIX" "mode" "$MODE"
  tracka_print_kv "$PREFIX" "seed" "$SEED"
  tracka_print_kv "$PREFIX" "train_entry" "$TRAIN_ENTRY"
  tracka_print_kv "$PREFIX" "output_dir" "$OUTPUT_DIR"
  tracka_print_kv "$PREFIX" "stop_after_epoch" "$STOP_AFTER_EPOCH"
  tracka_print_kv "$PREFIX" "nonempty_keep_guard" "$NONEMPTY_KEEP_GUARD"
  tracka_print_kv "$PREFIX" "loss_grad_attrib" "$LOSS_GRAD_ATTRIB"
  tracka_print_kv "$PREFIX" "loss_grad_attrib_param" "$LOSS_GRAD_ATTRIB_PARAM"
  tracka_print_kv "$PREFIX" "ratio_weight" "$RATIO_WEIGHT"
  tracka_print_kv "$PREFIX" "cls_distill_weight" "$CLS_DISTILL_WEIGHT"
  tracka_print_kv "$PREFIX" "token_distill_weight" "$TOKEN_DISTILL_WEIGHT"
}

print_compat_summary() {
  tracka_print_kv "$PREFIX" "mode" "$MODE"
  tracka_print_kv "$PREFIX" "seed" "$SEED"
  tracka_print_kv "$PREFIX" "final_repo" "$FINAL_REPO_ROOT"
  tracka_print_kv "$PREFIX" "train_entry" "$TRAIN_ENTRY"
  tracka_print_kv "$PREFIX" "output_dir" "$OUTPUT_DIR"
  tracka_print_kv "$PREFIX" "tmp_root" "$TMP_ROOT"
  tracka_print_kv "$PREFIX" "epochs" "$EPOCHS"
  tracka_print_kv "$PREFIX" "activation_lr_scale" "$ACTIVATION_LR_SCALE"
  tracka_print_kv "$PREFIX" "crop_pct" "$CROP_PCT"
  tracka_print_kv "$PREFIX" "weight_decay_end" "$WEIGHT_DECAY_END"
  tracka_print_kv "$PREFIX" "model_ema" "$MODEL_EMA"
  tracka_print_kv "$PREFIX" "nonempty_keep_guard" "$NONEMPTY_KEEP_GUARD"
  tracka_print_kv "$PREFIX" "stop_after_epoch" "$STOP_AFTER_EPOCH"
  tracka_print_kv "$PREFIX" "train_sampler_mode" "$TRAIN_SAMPLER_MODE"
  tracka_print_kv "$PREFIX" "ratio_weight" "$RATIO_WEIGHT"
  tracka_print_kv "$PREFIX" "cls_distill_weight" "$CLS_DISTILL_WEIGHT"
  tracka_print_kv "$PREFIX" "token_distill_weight" "$TOKEN_DISTILL_WEIGHT"
}

main() {
  local runner="${1:-}"
  [[ -n "$runner" ]] || usage
  shift

  case "$runner" in
    source)
      setup_source_defaults "${1:-}" "${2:-}"
      validate_common_prereqs
      tracka_init_output_log "$OUTPUT_DIR/train_stdout.log"
      print_source_summary
      build_source_args
      ;;
    compat)
      setup_compat_defaults "${1:-}" "${2:-}"
      validate_common_prereqs
      validate_compat_prereqs
      tracka_init_output_log "$OUTPUT_DIR/train_stdout.log"
      print_compat_summary
      build_compat_args
      ;;
    *)
      usage
      ;;
  esac

  tracka_run_training \
    "$PYTHON_BIN" \
    "$TRAIN_ENTRY" \
    "$OUTPUT_DIR/train_stdout.log" \
    "${TRAIN_ARGS[@]}"
}

main "$@"
