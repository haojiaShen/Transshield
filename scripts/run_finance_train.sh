#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

PYTHON="/home/yclcg/miniconda3/envs/transshield/bin/python"
TRAIN_DATA="data/finance_fraud_detection/train"
VAL_DATA="data/finance_fraud_detection/val"
RUN_NAME="finance_fraud_detection_$(date +%Y%m%d_%H%M%S)"
EPOCHS="${1:-8}"
FINETUNE="artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_20260430/checkpoint-best.pth"

OUTPUT_DIR="artifacts/train_runs/${RUN_NAME}"
mkdir -p "$OUTPUT_DIR"

echo "=== Finance Fraud Detection Training ==="
echo "Run: $RUN_NAME"
echo "Epochs: $EPOCHS"
echo "Finetune: $FINETUNE"
echo "Start: $(date)"

$PYTHON main.py     --model deit-s     --data_set image_folder     --data_path "$TRAIN_DATA"     --eval_data_path "$VAL_DATA"     --nb_classes 2     --input_size 224     --batch_size 32     --epochs "$EPOCHS"     --lr 3e-5     --warmup_epochs 0     --warmup_steps 20     --weight_decay 0.05     --drop 0.1     --smoothing 0.1     --reprob 0.0     --aa ''     --color_jitter 0.0     --mixup 0 --cutmix 0     --class_weight_mode inverse_freq     --use_square_gelu true     --square_activation_mode fixed_square     --use_approx_attn true     --approx_attn_mode uniform     --use_mask_pruning false     --model_ema false     --auto_resume false     --use_amp false     --seed 0     --finetune "$FINETUNE"     --teacher_checkpoint_path "$FINETUNE"     --pretrained_fix_step 0     --output_dir "$OUTPUT_DIR"     2>&1 | tee "${OUTPUT_DIR}/train_stdout.log"

echo "=== Done: $(date) ==="
