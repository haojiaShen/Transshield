#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_ENTRY="${TRAIN_ENTRY:-$REPO_ROOT/main.py}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-}"
VAL_DATA_PATH="${VAL_DATA_PATH:-}"
DEFAULT_TMP_ROOT="$REPO_ROOT/tmp"
if [[ -d /data/wyb ]]; then
  DEFAULT_TMP_ROOT="/data/wyb/tmp"
fi
TMP_ROOT="${TMP_ROOT:-$DEFAULT_TMP_ROOT}"

ABLATION_MODE="${ABLATION_MODE:-debug80}"
ABLATION_RUN_NAME="${ABLATION_RUN_NAME:-margin_aware_pruning_$(date +%Y%m%d_%H%M%S)}"
MARGIN_WEIGHTS="${MARGIN_WEIGHTS:-10 30}"
MARGIN_TARGET="${MARGIN_TARGET:-1e-4}"
MARGIN_MODE="${MARGIN_MODE:-hinge}"
MARGIN_STAGE_WEIGHTS="${MARGIN_STAGE_WEIGHTS:-}"
MARGIN_START_EPOCH="${MARGIN_START_EPOCH:-0}"
MARGIN_SEED="${MARGIN_SEED:-1}"
CLS_DISTILL_WEIGHT="${CLS_DISTILL_WEIGHT:-1.0}"
TOKEN_DISTILL_WEIGHT="${TOKEN_DISTILL_WEIGHT:-0.02}"

TRAIN_RUN_ROOT="${TRAIN_RUN_ROOT:-$REPO_ROOT/artifacts/train_runs}"
CANDIDATE_BUNDLE_ROOT="${CANDIDATE_BUNDLE_ROOT:-$REPO_ROOT/artifacts/frozen_candidates}"
REPORT_ROOT="${REPORT_ROOT:-$REPO_ROOT/results/margin_aware_pruning_ablation/$ABLATION_RUN_NAME}"
BASELINE_RISK_JSON="${BASELINE_RISK_JSON:-$REPO_ROOT/results/stagewise_protocol_risk_tracka_lr3e5_verified_20260415.json}"

TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda}"
REPORT_DEVICE="${REPORT_DEVICE:-cpu}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-4}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-4}"
REPORT_NUM_WORKERS="${REPORT_NUM_WORKERS:-0}"
REPORT_BATCH_SIZE="${REPORT_BATCH_SIZE:-16}"
REPORT_MAX_SAMPLES="${REPORT_MAX_SAMPLES:-0}"

BASE_RATE="${BASE_RATE:-0.7}"
RATIO_WEIGHT="${RATIO_WEIGHT:-2.0}"
TRAIN_LR="${TRAIN_LR:-3e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-50}"
CLIP_GRAD="${CLIP_GRAD:-1.0}"
LR_SCALE="${LR_SCALE:-1.0}"
GROUPA_LR_SCALE="${GROUPA_LR_SCALE:-0.1}"
ACTIVATION_LR_SCALE="${ACTIVATION_LR_SCALE:-10.0}"
TRAIN_SAMPLER_MODE="${TRAIN_SAMPLER_MODE:-distributed}"
USE_AMP="${USE_AMP:-false}"

RUN_SECURE_REPLAY="${RUN_SECURE_REPLAY:-0}"
SECURE_RUNTIME="${SECURE_RUNTIME:-cpu}"
SECURE_MAX_SAMPLES="${SECURE_MAX_SAMPLES:-8}"

slugify() {
  printf '%s' "$1" | sed -e 's/+//g' -e 's/-/m/g' -e 's/[.]/p/g'
}

ensure_prereqs() {
  if [[ -z "$TRAIN_DATA_PATH" || -z "$VAL_DATA_PATH" ]]; then
    echo "请先设置 TRAIN_DATA_PATH 和 VAL_DATA_PATH。" >&2
    exit 1
  fi
  if [[ ! -f "$TRAIN_ENTRY" ]]; then
    echo "找不到训练入口：$TRAIN_ENTRY" >&2
    exit 1
  fi
}

configure_mode() {
  if [[ "$ABLATION_MODE" == "debug80" ]]; then
    EPOCHS="${EPOCHS:-1}"
    DEBUG_ARGS=(--debug_max_steps 80)
    SAVE_CKPT=false
  elif [[ "$ABLATION_MODE" == "full20" ]]; then
    EPOCHS="${EPOCHS:-20}"
    DEBUG_ARGS=()
    SAVE_CKPT=true
  else
    echo "不支持的 ABLATION_MODE=$ABLATION_MODE，可选：debug80 或 full20。" >&2
    exit 1
  fi
}

setup_tmp_env() {
  mkdir -p "$TMP_ROOT" "$TRAIN_RUN_ROOT" "$CANDIDATE_BUNDLE_ROOT" "$REPORT_ROOT"
  export TMPDIR="$TMP_ROOT"
  export TMP="$TMP_ROOT"
  export TEMP="$TMP_ROOT"
}

print_run_config() {
  echo "[margin-ablation] mode=$ABLATION_MODE"
  echo "[margin-ablation] run=$ABLATION_RUN_NAME"
  echo "[margin-ablation] train_entry=$TRAIN_ENTRY"
  echo "[margin-ablation] train_data=$TRAIN_DATA_PATH"
  echo "[margin-ablation] val_data=$VAL_DATA_PATH"
  echo "[margin-ablation] margin_weights=$MARGIN_WEIGHTS"
  echo "[margin-ablation] margin_target=$MARGIN_TARGET"
  echo "[margin-ablation] margin_mode=$MARGIN_MODE"
  echo "[margin-ablation] margin_stage_weights=${MARGIN_STAGE_WEIGHTS:-<all-stages>}"
  echo "[margin-ablation] margin_start_epoch=$MARGIN_START_EPOCH"
  echo "[margin-ablation] cls_distill_weight=$CLS_DISTILL_WEIGHT"
  echo "[margin-ablation] token_distill_weight=$TOKEN_DISTILL_WEIGHT"
  echo "[margin-ablation] train_lr=$TRAIN_LR"
  echo "[margin-ablation] ratio_weight=$RATIO_WEIGHT"
  echo "[margin-ablation] lr_scale=$LR_SCALE"
  echo "[margin-ablation] groupa_lr_scale=$GROUPA_LR_SCALE"
  echo "[margin-ablation] activation_lr_scale=$ACTIVATION_LR_SCALE"
  echo "[margin-ablation] train_sampler_mode=$TRAIN_SAMPLER_MODE"
  echo "[margin-ablation] use_amp=$USE_AMP"
  echo "[margin-ablation] report_root=$REPORT_ROOT"
  echo "[margin-ablation] 注意：本脚本只生成 ablation 候选，不替换当前 Web demo 默认 bundle。"
}

build_margin_args() {
  EXTRA_MARGIN_ARGS=(--pruning_margin_start_epoch "$MARGIN_START_EPOCH")
  if [[ -n "${MARGIN_STAGE_WEIGHTS// }" ]]; then
    EXTRA_MARGIN_ARGS+=(--pruning_margin_stage_weights "$MARGIN_STAGE_WEIGHTS")
  fi
}

run_training_for_weight() {
  local weight="$1"
  local output_dir="$2"
  local log_dir="$3"

  build_margin_args
  echo "[margin-ablation] ===== weight=$weight target=$MARGIN_TARGET ====="

  local train_args=(
    --model deit-s
    --data_set image_folder
    --data_path "$TRAIN_DATA_PATH"
    --eval_data_path "$VAL_DATA_PATH"
    --nb_classes 2
    --output_dir "$output_dir"
    --log_dir "$log_dir"
    --input_size 224
    --batch_size "$TRAIN_BATCH_SIZE"
    --epochs "$EPOCHS"
    --num_workers "$TRAIN_NUM_WORKERS"
    --base_rate "$BASE_RATE"
    --ratio_weight "$RATIO_WEIGHT"
    --lr "$TRAIN_LR"
    --warmup_epochs 0
    --warmup_steps "$WARMUP_STEPS"
    --clip_grad "$CLIP_GRAD"
    --device "$TRAIN_DEVICE"
    --model_ema false
    --save_ckpt "$SAVE_CKPT"
    --save_ckpt_freq 1
    --save_ckpt_num 2
    --auto_resume false
    --use_amp "$USE_AMP"
    --mixup 0
    --cutmix 0
    --seed "$MARGIN_SEED"
    --lr_scale "$LR_SCALE"
    --groupa_lr_scale "$GROUPA_LR_SCALE"
    --activation_lr_scale "$ACTIVATION_LR_SCALE"
    --train_sampler_mode "$TRAIN_SAMPLER_MODE"
    --cls_distill_weight "$CLS_DISTILL_WEIGHT"
    --token_distill_weight "$TOKEN_DISTILL_WEIGHT"
    --use_square_gelu true
    --square_activation_mode learnable_quadratic_gelu_init
    --use_approx_attn false
    --use_mask_pruning true
    --pruning_margin_weight "$weight"
    --pruning_margin_target "$MARGIN_TARGET"
    --pruning_margin_mode "$MARGIN_MODE"
    "${EXTRA_MARGIN_ARGS[@]}"
    --debug_nan true
    "${DEBUG_ARGS[@]}"
    --patch_embed_bias_init_mode zero
    --freeze_patch_embed_proj true
    --pretrained_fix_step 0
  )

  "$PYTHON_BIN" "$TRAIN_ENTRY" "${train_args[@]}" |& tee "$output_dir/train_stdout.log"
}

run_optional_secure_replay() {
  local run_name="$1"
  local bundle_out="$2"
  local secure_run_dir="$3"

  echo "[margin-ablation] 运行候选 bundle 的 secure replay 检查。"
  env \
    RUN_NAME="$run_name" \
    BUNDLE_DIR="$bundle_out" \
    SECURE_RUN_DIR="$secure_run_dir" \
    SECURE_RUNTIME="$SECURE_RUNTIME" \
    SECURE_MAX_SAMPLES="$SECURE_MAX_SAMPLES" \
    VAL_DATA_PATH="$VAL_DATA_PATH" \
    bash "$SCRIPT_DIR/run_selected_image_secure_suite.sh"

  env \
    RUN_NAME="$run_name" \
    BUNDLE_DIR="$bundle_out" \
    SECURE_RUN_DIR="$secure_run_dir" \
    bash "$SCRIPT_DIR/run_secure_score_compare.sh"
}

postprocess_full20_candidate() {
  local run_name="$1"
  local output_dir="$2"
  local bundle_out="$3"
  local stagewise_json="$4"
  local secure_run_dir="$5"

  if [[ ! -f "$output_dir/checkpoint-best.pth" ]]; then
    echo "full20 未生成 checkpoint-best.pth：$output_dir" >&2
    exit 1
  fi

  echo "[margin-ablation] 搜索二分类阈值。"
  "$PYTHON_BIN" tools/transshield_binary_threshold_search.py search \
    --checkpoint "$output_dir/checkpoint-best.pth" \
    --data-path "$VAL_DATA_PATH" \
    --device "$EVAL_DEVICE" \
    --batch-size 32 \
    --num-workers "$EVAL_NUM_WORKERS" \
    --output-json "$output_dir/threshold_best.json"

  echo "[margin-ablation] 冻结候选 bundle。"
  "$PYTHON_BIN" tools/freeze_export_candidate.py \
    --source-dir "$output_dir" \
    --output-dir "$bundle_out" \
    --train-command "bash artifacts/server_inference_friendly_pack/run_margin_aware_pruning_ablation.sh" \
    --threshold-search-command "tools/transshield_binary_threshold_search.py search --checkpoint $output_dir/checkpoint-best.pth --data-path $VAL_DATA_PATH" \
    --eval-command ""

  echo "[margin-ablation] 校验候选 bundle。"
  "$PYTHON_BIN" tools/verify_frozen_candidate.py \
    --bundle-dir "$bundle_out" \
    --device cpu

  echo "[margin-ablation] 生成 stage-wise protocol risk 报告。"
  "$PYTHON_BIN" tools/transshield_stagewise_threshold_report.py \
    --bundle-dir "$bundle_out" \
    --data-path "$VAL_DATA_PATH" \
    --device "$REPORT_DEVICE" \
    --batch-size "$REPORT_BATCH_SIZE" \
    --num-workers "$REPORT_NUM_WORKERS" \
    --max-samples "$REPORT_MAX_SAMPLES" \
    --output-json "$stagewise_json"
  CANDIDATE_REPORTS+=("$stagewise_json")

  if [[ "$RUN_SECURE_REPLAY" == "1" ]]; then
    run_optional_secure_replay "$run_name" "$bundle_out" "$secure_run_dir"
  fi
}

run_weight_loop() {
  for weight in $MARGIN_WEIGHTS; do
    local weight_slug target_slug run_name output_dir log_dir bundle_out stagewise_json secure_run_dir
    weight_slug="$(slugify "$weight")"
    target_slug="$(slugify "$MARGIN_TARGET")"
    run_name="${ABLATION_RUN_NAME}_w${weight_slug}_t${target_slug}"
    output_dir="$TRAIN_RUN_ROOT/$run_name"
    log_dir="$output_dir/tb"
    bundle_out="$CANDIDATE_BUNDLE_ROOT/${run_name}_bundle"
    stagewise_json="$REPORT_ROOT/${run_name}_stagewise_protocol_risk.json"
    secure_run_dir="$REPORT_ROOT/${run_name}_secure_${SECURE_RUNTIME}"

    mkdir -p "$output_dir" "$log_dir"
    run_training_for_weight "$weight" "$output_dir" "$log_dir"

    if [[ "$ABLATION_MODE" != "full20" ]]; then
      echo "[margin-ablation] debug80 完成：$output_dir"
      continue
    fi

    postprocess_full20_candidate "$run_name" "$output_dir" "$bundle_out" "$stagewise_json" "$secure_run_dir"
  done
}

write_compare_report() {
  if [[ "$ABLATION_MODE" == "full20" && "${#CANDIDATE_REPORTS[@]}" -gt 0 ]]; then
    REPORT_ARGS=()
    for report in "${CANDIDATE_REPORTS[@]}"; do
      REPORT_ARGS+=(--candidate-json "$report")
    done
    "$PYTHON_BIN" tools/transshield_margin_ablation_report.py \
      --baseline-json "$BASELINE_RISK_JSON" \
      "${REPORT_ARGS[@]}" \
      --output-json "$REPORT_ROOT/margin_ablation_compare.json" \
      --output-md "$REPORT_ROOT/margin_ablation_compare.md"
    echo "[margin-ablation] 对比报告：$REPORT_ROOT/margin_ablation_compare.md"
  fi
}

main() {
  ensure_prereqs
  configure_mode
  setup_tmp_env
  print_run_config

  CANDIDATE_REPORTS=()
  run_weight_loop
  write_compare_report

  echo "[margin-ablation] 完成。"
}

main "$@"
