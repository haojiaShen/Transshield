#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

PYTHON="/data/wyb/conda_envs/transshield/bin/python"
TRAIN_DATA="data/finance_fraud_v3/train"
VAL_DATA="data/finance_fraud_v3/val"
RUN_NAME="finance_predictor_lite_$(date +%Y%m%d_%H%M%S)"
EPOCHS="${1:-30}"
FINETUNE="artifacts/frozen_bundle_finance_lrd_rank192_20260515/modified_plaintext_model_state_dict.pth"

OUTPUT_DIR="artifacts/train_runs/${RUN_NAME}"
mkdir -p "$OUTPUT_DIR"

echo "=== Finance PredictorLite Training ==="
echo "Run: $RUN_NAME"
echo "Epochs: $EPOCHS"
echo "Finetune: $FINETUNE"
echo "Predictor: lite (98K params/stage vs 241K)"
echo "Start: $(date)"

$PYTHON training_compat/main.py \
    --model deit-s \
    --data_set image_folder \
    --data_path "$TRAIN_DATA" \
    --eval_data_path "$VAL_DATA" \
    --nb_classes 2 \
    --input_size 224 \
    --batch_size 16 \
    --epochs "$EPOCHS" \
    --lr 5e-5 \
    --warmup_epochs 0 \
    --warmup_steps 20 \
    --weight_decay 0.05 \
    --drop 0.1 \
    --smoothing 0.1 \
    --reprob 0.0 \
    --aa '' \
    --color_jitter 0.0 \
    --mixup 0 --cutmix 0 \
    --class_weight_mode none \
    --use_square_gelu true \
    --square_activation_mode fixed_square \
    --use_approx_attn true \
    --approx_attn_mode uniform \
    --use_mask_pruning false \
    --model_ema true \
    --auto_resume false \
    --use_amp false \
    --seed 0 \
    --finetune "$FINETUNE" \
    --teacher_checkpoint_path "$FINETUNE" \
    --pretrained_fix_step 0 \
    --output_dir "$OUTPUT_DIR" \
    --predictor_type lite \
    --nonempty_keep_guard true \
    2>&1 | tee "${OUTPUT_DIR}/train_stdout.log"

echo "=== Done: $(date) ==="
