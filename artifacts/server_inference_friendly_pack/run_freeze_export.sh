#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="${RUN_NAME:-transshield_inference_friendly_deits}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-}"
VAL_DATA_PATH="${VAL_DATA_PATH:-}"
RUN_DIR="${RUN_DIR:-artifacts/server_runs/${RUN_NAME}}"
BUNDLE_DIR="${BUNDLE_DIR:-artifacts/server_bundles/${RUN_NAME}_bundle}"
SECURE_RUN_DIR="${SECURE_RUN_DIR:-artifacts/server_pipeline_run/${RUN_NAME}}"
CONFIG_PATH="${CONFIG_PATH:-configs/openbumblebee/2pc.json}"

echo "[archive] 执行 modified 模型 bundle 冻结导出。"
echo "[archive] 该脚本属于“从训练重建 bundle”的全流程，不是默认比赛展示主入口。"

"$PYTHON_BIN" tools/freeze_export_bundle.py --source-dir "$RUN_DIR" --output-dir "$BUNDLE_DIR" --train-command "\"$PYTHON_BIN\" main.py --model deit-s --data_set image_folder --data_path \"$TRAIN_DATA_PATH\" --eval_data_path \"$VAL_DATA_PATH\" --nb_classes 2 --output_dir \"$RUN_DIR\" --log_dir \"$RUN_DIR/tb\" --input_size 224 --batch_size 32 --epochs 8 --num_workers 4 --base_rate 0.7 --ratio_weight 2.0 --lr 1e-05 --warmup_epochs 0 --warmup_steps 50 --clip_grad 1.0 --device cuda --model_ema false --save_ckpt true --save_ckpt_freq 1 --save_ckpt_num 2 --auto_resume false --use_amp false --mixup 0 --cutmix 0 --seed 0 --lr_scale 1.0 --groupa_lr_scale 0.1 --activation_lr_scale 10.0 --cls_distill_weight 1.0 --token_distill_weight 0.02 --square_activation_mode learnable_quadratic_gelu_init --approx_attn_mode relu --eval_tie_policy lowest_index --patch_embed_bias_init_mode zero --freeze_patch_embed_proj true --pretrained_fix_step 0 --inference_friendly_ops true" --threshold-search-command "\"$PYTHON_BIN\" tools/transshield_binary_threshold_search.py search --checkpoint \"$RUN_DIR/checkpoint-best.pth\" --data-path \"$VAL_DATA_PATH\" --device cuda --batch-size 32 --num-workers 4 --output-json \"$RUN_DIR/threshold_best.json\"" --eval-command "\"$PYTHON_BIN\" tools/transshield_binary_threshold_search.py eval --checkpoint \"$RUN_DIR/checkpoint-best.pth\" --threshold-json \"$RUN_DIR/threshold_best.json\" --data-path \"$VAL_DATA_PATH\" --device cuda --batch-size 32 --num-workers 4 --output-json \"$RUN_DIR/threshold_eval.json\""
