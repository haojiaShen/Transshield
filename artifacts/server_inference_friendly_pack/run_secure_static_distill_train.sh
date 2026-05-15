#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi
cd "$REPO_ROOT"

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import numpy  # noqa: F401
import torch  # noqa: F401
import timm  # noqa: F401
PY
then
  echo "[secure-static-train] python env check failed for PYTHON_BIN=$PYTHON_BIN" >&2
  echo "[secure-static-train] expected a Transshield runtime env with at least: numpy, torch, timm" >&2
  echo "[secure-static-train] fix by exporting PYTHON_BIN=${PYTHON_BIN:-python} or installing requirements.txt into the current env" >&2
  exit 2
fi

MODE="${1:-debug80}"
case "$MODE" in
  debug80|epoch1)
    shift || true
    ;;
  *)
    echo "Usage: $0 [debug80|epoch1]" >&2
    exit 1
    ;;
esac

TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${DATA_ROOT:-/data/wyb/pneumoniamnist_imagefolder_subset}/train}"
VAL_DATA_PATH="${VAL_DATA_PATH:-${DATA_ROOT:-/data/wyb/pneumoniamnist_imagefolder_subset}/val}"
if [[ "$TRAIN_DATA_PATH" == "/path/to/pneumoniamnist_imagefolder_subset/train" ]]; then
  TRAIN_DATA_PATH="${DATA_ROOT:-/data/wyb/pneumoniamnist_imagefolder_subset}/train"
fi
if [[ "$VAL_DATA_PATH" == "/path/to/pneumoniamnist_imagefolder_subset/val" ]]; then
  VAL_DATA_PATH="${DATA_ROOT:-/data/wyb/pneumoniamnist_imagefolder_subset}/val"
fi
BASE_BUNDLE_DIR="${BASE_BUNDLE_DIR:-$REPO_ROOT/artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_20260430}"
DEFAULT_BASE_CHECKPOINT="$BASE_BUNDLE_DIR/checkpoint-best.pth"
LIGHT_BASE_CHECKPOINT="$BASE_BUNDLE_DIR/modified_plaintext_eval_checkpoint_light.pth"
if [[ -z "${BASE_CHECKPOINT:-}" ]]; then
  if [[ -f "$DEFAULT_BASE_CHECKPOINT" ]]; then
    BASE_CHECKPOINT="$DEFAULT_BASE_CHECKPOINT"
  else
    BASE_CHECKPOINT="$LIGHT_BASE_CHECKPOINT"
  fi
fi
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-$BASE_CHECKPOINT}"
SECURE_STATIC_DEPTH="${SECURE_STATIC_DEPTH:-6}"
SECURE_STATIC_SKIP_PRUNING="${SECURE_STATIC_SKIP_PRUNING:-true}"
ACCURACY_PROFILE="${ACCURACY_PROFILE:-default}"
AUGMENTATION_PROFILE="${AUGMENTATION_PROFILE:-timm}"
COLOR_JITTER="${COLOR_JITTER:-0.4}"
AA="${AA:-rand-m9-mstd0.5-inc1}"
REPROB="${REPROB:-0.25}"
if [[ "$AA" == "none" ]]; then
  AA=""
fi
TRAIN_SAMPLER_MODE="${TRAIN_SAMPLER_MODE-}"
CLASS_WEIGHT_MODE="${CLASS_WEIGHT_MODE-}"
CLASS_WEIGHT_POWER="${CLASS_WEIGHT_POWER:-1.0}"
MODEL_EMA="${MODEL_EMA:-false}"
SMOOTHING="${SMOOTHING:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
CLS_TOKEN_FULL_LR="${CLS_TOKEN_FULL_LR:-false}"
TRAIN_POS_EMBED="${TRAIN_POS_EMBED:-false}"
FREEZE_PATCH_EMBED_PROJ="${FREEZE_PATCH_EMBED_PROJ:-false}"
FREEZE_PATCH_EMBED_WEIGHT="${FREEZE_PATCH_EMBED_WEIGHT:-false}"
FREEZE_PATCH_EMBED_BIAS="${FREEZE_PATCH_EMBED_BIAS:-false}"
PATCH_EMBED_BIAS_INIT_MODE="${PATCH_EMBED_BIAS_INIT_MODE:-pretrained}"
SKIP_PATCH_EMBED_BIAS_PRETRAINED="${SKIP_PATCH_EMBED_BIAS_PRETRAINED:-false}"
ACTIVATION_LR_SCALE="${ACTIVATION_LR_SCALE:-1.0}"
EVAL_BINARY_THRESHOLD="${EVAL_BINARY_THRESHOLD-}"
USE_MASK_PRUNING="${USE_MASK_PRUNING:-false}"
PRUNING_MARGIN_WEIGHT="${PRUNING_MARGIN_WEIGHT:-0.0}"
PRUNING_MARGIN_TARGET="${PRUNING_MARGIN_TARGET:-1e-4}"
PRUNING_MARGIN_MODE="${PRUNING_MARGIN_MODE:-hinge}"
PRUNING_MARGIN_STAGE_WEIGHTS="${PRUNING_MARGIN_STAGE_WEIGHTS-}"
PRUNING_MARGIN_START_EPOCH="${PRUNING_MARGIN_START_EPOCH:-0}"

case "$ACCURACY_PROFILE" in
  default)
    ;;
  weighted_sqrt_sampler)
    if [[ -z "$TRAIN_SAMPLER_MODE" ]]; then
      TRAIN_SAMPLER_MODE="weighted_sqrt_inverse_freq"
    fi
    ;;
  sqrt_class_weight)
    if [[ -z "$CLASS_WEIGHT_MODE" ]]; then
      CLASS_WEIGHT_MODE="sqrt_inverse_freq"
    fi
    ;;
  *)
    echo "[secure-static-train] unsupported ACCURACY_PROFILE: $ACCURACY_PROFILE" >&2
    exit 2
    ;;
esac

TRAIN_SAMPLER_MODE="${TRAIN_SAMPLER_MODE:-distributed}"
CLASS_WEIGHT_MODE="${CLASS_WEIGHT_MODE:-none}"
COMPARE_PLACEHOLDER_RUN_NAME="transshield_comp_full_compare_YYYYMMDD"
if [[ -z "${RUN_NAME:-}" || "${RUN_NAME:-}" == "$COMPARE_PLACEHOLDER_RUN_NAME" ]]; then
  RUN_NAME="secure_static_depth${SECURE_STATIC_DEPTH}_uniform_fixed_square_${MODE}_${ACCURACY_PROFILE}_$(date +%Y%m%d_%H%M%S)"
fi
DEFAULT_TRAIN_RUN_DIR="$REPO_ROOT/artifacts/train_runs/$RUN_NAME"
if [[ -z "${RUN_DIR:-}" ]]; then
  RUN_DIR="$DEFAULT_TRAIN_RUN_DIR"
elif [[ "$RUN_DIR" == "$REPO_ROOT/artifacts/server_runs/"* ]]; then
  # final_compare_env.template.sh exports RUN_DIR for inference suites; do not reuse it for training runs.
  RUN_DIR="$DEFAULT_TRAIN_RUN_DIR"
fi

mkdir -p "$RUN_DIR"

if [[ ! -f "$BASE_CHECKPOINT" ]]; then
  echo "[secure-static-train] missing BASE_CHECKPOINT: $BASE_CHECKPOINT" >&2
  if [[ -f "$LIGHT_BASE_CHECKPOINT" ]]; then
    echo "[secure-static-train] note: clean deploy bundle usually materializes $LIGHT_BASE_CHECKPOINT instead of checkpoint-best.pth" >&2
  fi
  exit 2
fi
if [[ ! -d "$TRAIN_DATA_PATH" ]]; then
  echo "[secure-static-train] missing TRAIN_DATA_PATH: $TRAIN_DATA_PATH" >&2
  exit 2
fi
if [[ ! -d "$VAL_DATA_PATH" ]]; then
  echo "[secure-static-train] missing VAL_DATA_PATH: $VAL_DATA_PATH" >&2
  exit 2
fi

mkdir -p "${TMPDIR:-/data/wyb/tmp}"
export TMPDIR="${TMPDIR:-/data/wyb/tmp}"
export TMP="${TMP:-$TMPDIR}"
export TEMP="${TEMP:-$TMPDIR}"

COMMON_ARGS=(
  training_compat/main.py
  --model deit-s
  --data_set image_folder
  --augmentation_profile "$AUGMENTATION_PROFILE"
  --data_path "$TRAIN_DATA_PATH"
  --eval_data_path "$VAL_DATA_PATH"
  --nb_classes 2
  --output_dir "$RUN_DIR"
  --log_dir "$RUN_DIR/tb"
  --input_size 224
  --color_jitter "$COLOR_JITTER"
  --aa "$AA"
  --reprob "$REPROB"
  --batch_size "${BATCH_SIZE:-32}"
  --num_workers "${NUM_WORKERS:-4}"
  --base_rate "${BASE_RATE:-0.7}"
  --ratio_weight "${RATIO_WEIGHT:-0.0}"
  --lr "${LR:-3e-6}"
  --min_lr "${MIN_LR:-1e-7}"
  --warmup_epochs 0
  --warmup_steps "${WARMUP_STEPS:-20}"
  --clip_grad "${CLIP_GRAD:-1.0}"
  --weight_decay "$WEIGHT_DECAY"
  --device "${DEVICE:-cuda}"
  --model_ema "$MODEL_EMA"
  --auto_resume false
  --use_amp false
  --mixup 0
  --cutmix 0
  --smoothing "$SMOOTHING"
  --seed "${SEED:-0}"
  --lr_scale "${LR_SCALE:-1.0}"
  --groupa_lr_scale "${GROUPA_LR_SCALE:-0.1}"
  --activation_lr_scale "$ACTIVATION_LR_SCALE"
  --cls_token_full_lr "$CLS_TOKEN_FULL_LR"
  --train_pos_embed "$TRAIN_POS_EMBED"
  --freeze_patch_embed_proj "$FREEZE_PATCH_EMBED_PROJ"
  --freeze_patch_embed_weight "$FREEZE_PATCH_EMBED_WEIGHT"
  --freeze_patch_embed_bias "$FREEZE_PATCH_EMBED_BIAS"
  --patch_embed_bias_init_mode "$PATCH_EMBED_BIAS_INIT_MODE"
  --skip_patch_embed_bias_pretrained "$SKIP_PATCH_EMBED_BIAS_PRETRAINED"
  --pretrained_fix_step "${PRETRAINED_FIX_STEP:-0}"
  --class_weight_mode "$CLASS_WEIGHT_MODE"
  --class_weight_power "$CLASS_WEIGHT_POWER"
  --train_sampler_mode "$TRAIN_SAMPLER_MODE"
  --cls_distill_weight "${CLS_DISTILL_WEIGHT:-1.0}"
  --token_distill_weight "${TOKEN_DISTILL_WEIGHT:-0.02}"
  --use_square_gelu true
  --square_activation_mode fixed_square
  --use_approx_attn true
  --approx_attn_mode uniform
  --use_mask_pruning "$USE_MASK_PRUNING"
  --secure_static_train_depth "$SECURE_STATIC_DEPTH"
  --secure_static_skip_pruning "$SECURE_STATIC_SKIP_PRUNING"
  --pruning_margin_weight "$PRUNING_MARGIN_WEIGHT"
  --pruning_margin_target "$PRUNING_MARGIN_TARGET"
  --pruning_margin_mode "$PRUNING_MARGIN_MODE"
  --pruning_margin_start_epoch "$PRUNING_MARGIN_START_EPOCH"
  --finetune "$BASE_CHECKPOINT"
  --teacher_checkpoint_path "$TEACHER_CHECKPOINT"
  --model_key "${MODEL_KEY:-model|module}"
  --debug_nan true
)

if [[ -n "$PRUNING_MARGIN_STAGE_WEIGHTS" ]]; then
  COMMON_ARGS+=(--pruning_margin_stage_weights "$PRUNING_MARGIN_STAGE_WEIGHTS")
fi

if [[ "$MODE" == "debug80" ]]; then
  EXTRA_ARGS=(
    --epochs 1
    --disable_eval true
    --save_ckpt false
    --debug_max_steps "${DEBUG_MAX_STEPS:-80}"
  )
else
  EXTRA_ARGS=(
    --epochs "${EPOCHS:-1}"
    --disable_eval false
    --save_ckpt true
    --save_ckpt_freq 1
    --save_ckpt_num 2
    --debug_max_steps 0
  )
fi

if [[ -n "$EVAL_BINARY_THRESHOLD" ]]; then
  EXTRA_ARGS+=(--eval_binary_threshold "$EVAL_BINARY_THRESHOLD")
fi

printf '%q ' "$PYTHON_BIN" "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}" "$@" > "$RUN_DIR/command.sh"
printf '\n' >> "$RUN_DIR/command.sh"
chmod +x "$RUN_DIR/command.sh"

echo "run_dir=$RUN_DIR"
echo "base_checkpoint=$BASE_CHECKPOINT"
echo "train_data_path=$TRAIN_DATA_PATH"
echo "val_data_path=$VAL_DATA_PATH"
echo "secure_static_depth=$SECURE_STATIC_DEPTH"
echo "secure_static_skip_pruning=$SECURE_STATIC_SKIP_PRUNING"
echo "use_mask_pruning=$USE_MASK_PRUNING"
echo "mode=$MODE"
echo "accuracy_profile=$ACCURACY_PROFILE"
echo "augmentation_profile=$AUGMENTATION_PROFILE"
echo "color_jitter=$COLOR_JITTER"
echo "aa=$AA"
echo "reprob=$REPROB"
echo "train_sampler_mode=$TRAIN_SAMPLER_MODE"
echo "class_weight_mode=$CLASS_WEIGHT_MODE"
echo "model_ema=$MODEL_EMA"
echo "weight_decay=$WEIGHT_DECAY"
echo "cls_token_full_lr=$CLS_TOKEN_FULL_LR"
echo "train_pos_embed=$TRAIN_POS_EMBED"
echo "freeze_patch_embed_proj=$FREEZE_PATCH_EMBED_PROJ"
echo "freeze_patch_embed_weight=$FREEZE_PATCH_EMBED_WEIGHT"
echo "freeze_patch_embed_bias=$FREEZE_PATCH_EMBED_BIAS"
echo "patch_embed_bias_init_mode=$PATCH_EMBED_BIAS_INIT_MODE"
echo "skip_patch_embed_bias_pretrained=$SKIP_PATCH_EMBED_BIAS_PRETRAINED"
echo "activation_lr_scale=$ACTIVATION_LR_SCALE"
echo "pruning_margin_weight=$PRUNING_MARGIN_WEIGHT"
echo "pruning_margin_target=$PRUNING_MARGIN_TARGET"
echo "pruning_margin_mode=$PRUNING_MARGIN_MODE"
echo "pruning_margin_stage_weights=${PRUNING_MARGIN_STAGE_WEIGHTS:-<default>}"
echo "pruning_margin_start_epoch=$PRUNING_MARGIN_START_EPOCH"

"$PYTHON_BIN" "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}" "$@" 2>&1 | tee "$RUN_DIR/train_stdout.log"
