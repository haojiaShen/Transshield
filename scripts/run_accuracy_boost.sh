#!/usr/bin/env bash
set -euo pipefail

# 精度提升训练脚本
# 目标：将医疗模型精度从91.98%提升到93%+

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINAL_REPO_ROOT="/data/wyb/Transshield_final"
PYTHON_BIN="/data/wyb/conda_envs/transshield/bin/python"
DATA_ROOT="/data/wyb/pneumoniamnist_imagefolder_subset"
OUTPUT_ROOT="$FINAL_REPO_ROOT/artifacts/accuracy_boost_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTPUT_ROOT"

echo "=== 精度提升训练 ==="
echo "输出目录: $OUTPUT_ROOT"

# 基础配置（继承当前最优配置）
BASE_ARGS=(
    --model deit_small_patch16_224
    --data-set image_folder
    --data-path "$DATA_ROOT/train"
    --eval-data-path "$DATA_ROOT/val"
    --batch-size 32
    --epochs 20
    --lr 5e-06
    --min_lr 1e-07
    --warmup_epochs 2
    --smoothing 0.1
    --mixup 0.2
    --mixup_mode batch
    --cutmix 0.2
    --cutmix_minmax "0.2 0.8"
    --num_workers 4
    --pin-mem
    --finetune "$FINAL_REPO_ROOT/artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_aanone_20260507/modified_plaintext_eval_checkpoint_light.pth"
    --teacher_checkpoint_path "$FINAL_REPO_ROOT/artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_aanone_20260507/modified_plaintext_eval_checkpoint_light.pth"
    --cls_distill_weight 1.0
    --token_distill_weight 0.02
    --approx_attn_mode uniform
    --act_layer square
    --layer_norm_type exact
    --clip_grad 1.0
    --secure_static_train_depth 12
    --output_dir "$OUTPUT_ROOT"
)

# 运行训练
echo "开始训练..."
$PYTHON_BIN main.py "${BASE_ARGS[@]}" 2>&1 | tee "$OUTPUT_ROOT/train.log"

echo "训练完成！"
echo "结果保存在: $OUTPUT_ROOT"
