#!/usr/bin/env bash
# Freeze PredictorLite training run → frozen bundle
# Usage: bash scripts/run_finance_predictor_lite_freeze.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

PYTHON="/data/wyb/conda_envs/transshield/bin/python"

# Source: PredictorLite training output
TRAIN_DIR="artifacts/train_runs/finance_predictor_lite_20260515_181418"
FINETUNE_SRC="artifacts/frozen_bundle_finance_lrd_rank192_20260515/modified_plaintext_model_state_dict.pth"
BUNDLE_NAME="frozen_bundle_finance_predictor_lite_20260515"
BUNDLE_DIR="artifacts/${BUNDLE_NAME}"

echo "=== Freezing PredictorLite Bundle ==="
echo "Source: $TRAIN_DIR"
echo "Dest: $BUNDLE_DIR"

# Create bundle directory
mkdir -p "$BUNDLE_DIR"

# Copy best checkpoint (non-EMA, for SPU inference)
cp "$TRAIN_DIR/checkpoint-best.pth" "$BUNDLE_DIR/checkpoint-best.pth"
cp "$FINETUNE_SRC" "$BUNDLE_DIR/modified_plaintext_model_state_dict.pth"
# Also copy EMA
if [ -f "$TRAIN_DIR/checkpoint-best-ema.pth" ]; then
    cp "$TRAIN_DIR/checkpoint-best-ema.pth" "$BUNDLE_DIR/checkpoint-best-ema.pth"
fi

# Extract best accuracy from log.txt
BEST_TEST=$($PYTHON -c "
import json
best_acc = 0
best_epoch = 0
with open('$TRAIN_DIR/log.txt') as f:
    for line in f:
        try:
            d = json.loads(line)
            if d.get('test_acc1', 0) > best_acc:
                best_acc = d['test_acc1']
                best_epoch = d['epoch']
        except: pass
print(f'{best_acc:.1f} @ epoch {best_epoch}')
")
BEST_EMA=$($PYTHON -c "
import json
best_acc = 0
best_epoch = 0
with open('$TRAIN_DIR/log.txt') as f:
    for line in f:
        try:
            d = json.loads(line)
            if d.get('test_acc1_ema', 0) > best_acc:
                best_acc = d['test_acc1_ema']
                best_epoch = d['epoch']
        except: pass
print(f'{best_acc:.1f} @ epoch {best_epoch}')
")
TOTAL_PARAMS=$($PYTHON -c "
import json
with open('$TRAIN_DIR/log.txt') as f:
    for line in f:
        try:
            d = json.loads(line)
            if 'n_parameters' in d:
                print(d['n_parameters'])
                break
        except: pass
")

echo "Best test acc: $BEST_TEST"
echo "Best EMA acc: $BEST_EMA"
echo "Total params: $TOTAL_PARAMS"

# Write args_snapshot.json
cat > "$BUNDLE_DIR/args_snapshot.json" << ARGS
{
  "model": "deit-s",
  "nb_classes": 2,
  "input_size": 224,
  "use_square_gelu": true,
  "square_activation_mode": "fixed_square",
  "use_approx_attn": true,
  "approx_attn_mode": "uniform",
  "use_mask_pruning": false,
  "secure_static_train_depth": 0,
  "secure_static_skip_pruning": true,
  "eval_pruning_mode": "topk_argsort",
  "eval_tie_policy": "lowest_index",
  "domain": "finance_fraud_detection",
  "dataset": "finance_fraud_v3",
  "predictor_type": "lite",
  "predictor_params_per_stage": 97970,
  "predictor_total_params": 293910,
  "predictor_reduction_vs_lg": "59.3%",
  "lrd_rank": 192,
  "lrd_merged": true,
  "innovation_lrd": true,
  "innovation_predictor_lite": true,
  "train_accuracy": 100.0,
  "best_test_accuracy": "$BEST_TEST",
  "best_ema_accuracy": "$BEST_EMA",
  "total_parameters": $TOTAL_PARAMS,
  "depth": 12,
  "embed_dim": 384,
  "num_heads": 6,
  "patch_size": 16,
  "epochs_trained": 30,
  "lr": 5e-05,
  "finetune_source": "$FINETUNE_SRC",
  "frozen_date": "$(date -Iseconds)",
  "frozen_source": "$TRAIN_DIR"
}
ARGS

echo "=== Bundle frozen at: $BUNDLE_DIR ==="
echo "=== Done: $(date) ==="
