#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/wyb/Transshield_final}"
PYTHON_BIN="${PYTHON_BIN:-/data/wyb/conda_envs/transshield/bin/python}"
DATA_ROOT="${DATA_ROOT:-/data/wyb/pneumoniamnist_imagefolder_subset}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/artifacts/train_runs/pneumonia_transshield_tracka_lr3e5_timm}"
TMP_ROOT="${TMP_ROOT:-/data/wyb/tmp}"

mkdir -p "$TMP_ROOT" "$RUN_DIR"
export TMPDIR="$TMP_ROOT"
export TMP="$TMP_ROOT"
export TEMP="$TMP_ROOT"
cd "$REPO_ROOT"

# Training command
"$PYTHON_BIN" "$REPO_ROOT/training_compat/main.py" --model deit-s --data_set image_folder --data_path "$DATA_ROOT/train" --eval_data_path "$DATA_ROOT/val" --nb_classes 2 --output_dir "$RUN_DIR" --log_dir "$RUN_DIR/tb" --input_size 224 --batch_size 32 --epochs 20 --num_workers 4 --base_rate 0.7 --ratio_weight 2.0 --lr 3e-5 --warmup_epochs 0 --warmup_steps 50 --clip_grad 1.0 --device cuda --model_ema false --save_ckpt true --save_ckpt_freq 1 --save_ckpt_num 2 --auto_resume false --use_amp false --mixup 0 --cutmix 0 --seed 0 --lr_scale 1.0 --groupa_lr_scale 0.1 --activation_lr_scale 10.0 --cls_distill_weight 1.0 --token_distill_weight 0.02 --use_square_gelu true --square_activation_mode learnable_quadratic_gelu_init --use_approx_attn false --use_mask_pruning true --debug_nan true --patch_embed_bias_init_mode zero --freeze_patch_embed_proj true --pretrained_fix_step 0

# Threshold search command
"$PYTHON_BIN" tools/transshield_binary_threshold_search.py search --checkpoint "$RUN_DIR/checkpoint-best.pth" --data-path "$DATA_ROOT/val" --device cuda --batch-size 32 --num-workers 4 --output-json "$RUN_DIR/threshold_best.json"

# Thresholded eval command
"$PYTHON_BIN" tools/transshield_binary_threshold_search.py eval --checkpoint "$RUN_DIR/checkpoint-best.pth" --threshold-json "$RUN_DIR/threshold_best.json" --data-path "$DATA_ROOT/val" --device cuda --batch-size 32 --num-workers 4 --output-json "$RUN_DIR/threshold_eval.json"
