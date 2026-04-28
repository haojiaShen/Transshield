# `Transshield_final` 比赛展示版环境模板。
# 推荐直接 `source` 本文件，然后按需覆盖少量变量：
#   1. 设置 `REPO_ROOT`
#   2. 设置 `TRAIN_DATA_PATH` 与 `VAL_DATA_PATH`
#   3. 选择 `SECURE_RUNTIME=cpu` 或 `SECURE_RUNTIME=spu`
#
# 说明：
# - `cpu` 表示 secure sidecar 的本地明文参考执行，用于调试和快速链路验证，不是真正 2PC
# - `spu` 表示基于 SPU / OpenBumbleBee 的真实 secure 执行
#
# 可选 conda 激活示例：
# source /path/to/miniconda3/etc/profile.d/conda.sh
# conda activate your_transshield_env

export REPO_ROOT="${REPO_ROOT:-$(pwd)}"
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$REPO_ROOT"

export RUN_NAME="${RUN_NAME:-transshield_comp_full_compare_YYYYMMDD}"
export TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/path/to/pneumoniamnist_imagefolder_subset/train}"
export VAL_DATA_PATH="${VAL_DATA_PATH:-/path/to/pneumoniamnist_imagefolder_subset/val}"
export RUN_DIR="${RUN_DIR:-$REPO_ROOT/artifacts/server_runs/${RUN_NAME}}"
export BUNDLE_DIR="${BUNDLE_DIR:-$REPO_ROOT/artifacts/frozen_bundle_verified_tracka_lr3e5_20260414}"
export SECURE_RUN_DIR="${SECURE_RUN_DIR:-$REPO_ROOT/artifacts/server_pipeline_run/${RUN_NAME}}"
export CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/openbumblebee/2pc.json}"
export KTH_SELECTION_MODE="${KTH_SELECTION_MODE:-blockwise_exact_kth}"
export PHASE3_SELECTION_MANIFEST="${PHASE3_SELECTION_MANIFEST:-$REPO_ROOT/results/blockwise_exact_kth_selection_manifest_default.json}"
export SECURE_MAX_SAMPLES="${SECURE_MAX_SAMPLES:-0}"
export SECURE_EXPORT_DEVICE="${SECURE_EXPORT_DEVICE:-cpu}"
export SECURE_RUNTIME="${SECURE_RUNTIME:-spu}"
export PLAINTEXT_EVAL_DEVICE="${PLAINTEXT_EVAL_DEVICE:-cpu}"
export PLAINTEXT_EVAL_BATCH_SIZE="${PLAINTEXT_EVAL_BATCH_SIZE:-32}"
export PLAINTEXT_EVAL_NUM_WORKERS="${PLAINTEXT_EVAL_NUM_WORKERS:-0}"
export PLAINTEXT_MAX_SAMPLES="${PLAINTEXT_MAX_SAMPLES:-0}"

# 可选：指定图片推理输入。
# 适合“只处理我给定的图片/图片列表/图片目录”的演示模式。
export INPUT_IMAGE="${INPUT_IMAGE:-}"
export INPUT_IMAGE_LIST="${INPUT_IMAGE_LIST:-}"
export INPUT_IMAGE_DIR="${INPUT_IMAGE_DIR:-}"
export INPUT_GLOB_PATTERN="${INPUT_GLOB_PATTERN:-*}"
export CLASS_NAMES="${CLASS_NAMES:-class_0,class_1}"

# Group 1：baseline plaintext
# 仓库默认内置 baseline 运行时代码、轻量评估权重与阈值 JSON。
# 仅在你明确需要替换 baseline 资产时再覆盖。
export BASELINE_REPO_ROOT="${BASELINE_REPO_ROOT:-$REPO_ROOT/references/original_plaintext_runtime}"
export BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-$REPO_ROOT/artifacts/baselines/baseline_plaintext_eval_checkpoint_light.pth}"
export BASELINE_THRESHOLD_JSON="${BASELINE_THRESHOLD_JSON:-$REPO_ROOT/artifacts/baselines/original_plaintext_threshold_best_fix3.json}"
export BASELINE_LABEL=baseline_plaintext

# Group 2：modified plaintext
export MODIFIED_CHECKPOINT="${MODIFIED_CHECKPOINT:-$BUNDLE_DIR/modified_plaintext_eval_checkpoint_light.pth}"
export MODIFIED_THRESHOLD_JSON="${MODIFIED_THRESHOLD_JSON:-$BUNDLE_DIR/threshold_best.json}"
export MODIFIED_LABEL=modified_plaintext

# Group 3：可选 secure profile 对比基线
# 不是默认比赛主流程的必填项；仅在做 profile 对照时使用。
export SECURE_BASELINE_PROFILE_JSON=
export SECURE_BASELINE_LABEL=original_secure
