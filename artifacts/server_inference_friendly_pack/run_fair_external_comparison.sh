#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="${RUN_NAME:-fair_external_$(date +%Y%m%d_%H%M%S)}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-}"
VAL_DATA_PATH="${VAL_DATA_PATH:-}"
BUNDLE_DIR="${BUNDLE_DIR:-$REPO_ROOT/artifacts/frozen_bundle_verified_tracka_lr3e5_20260414}"
SECURE_RUN_DIR="${SECURE_RUN_DIR:-$REPO_ROOT/artifacts/server_pipeline_run/${RUN_NAME}_transshield}"
FAIR_OUTPUT_DIR="${FAIR_OUTPUT_DIR:-$REPO_ROOT/results/fair_external_comparison/${RUN_NAME}}"

EXTERNAL_BASELINES_ROOT="${EXTERNAL_BASELINES_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/external_baselines}"
MPCVIT_ROOT="${MPCVIT_ROOT:-$EXTERNAL_BASELINES_ROOT/mpcvit}"
MPCVIT_OUTPUT_ROOT="${MPCVIT_OUTPUT_ROOT:-$FAIR_OUTPUT_DIR/mpcvit}"
MPCVIT_SEEDS="${MPCVIT_SEEDS:-0}"
MPCVIT_EPOCHS="${MPCVIT_EPOCHS:-20}"
MPCVIT_BATCH_SIZE="${MPCVIT_BATCH_SIZE:-128}"
MPCVIT_NUM_WORKERS="${MPCVIT_NUM_WORKERS:-4}"
MPCVIT_DEVICE="${MPCVIT_DEVICE:-cuda}"
MPCVIT_MODEL_NAME="${MPCVIT_MODEL_NAME:-vit_7_4_32}"
MPCVIT_IMG_SIZE="${MPCVIT_IMG_SIZE:-32}"
MPCVIT_CLASS_BALANCED_LOSS="${MPCVIT_CLASS_BALANCED_LOSS:-1}"
MPCVIT_NO_HFLIP="${MPCVIT_NO_HFLIP:-1}"

SECURE_RUNTIME="${SECURE_RUNTIME:-spu}"
SECURE_MAX_SAMPLES="${SECURE_MAX_SAMPLES:-0}"
PLAINTEXT_MAX_SAMPLES="${PLAINTEXT_MAX_SAMPLES:-0}"
PLAINTEXT_EVAL_DEVICE="${PLAINTEXT_EVAL_DEVICE:-cpu}"
SECURE_EXPORT_DEVICE="${SECURE_EXPORT_DEVICE:-cpu}"

RUN_TRANSSHIELD="${RUN_TRANSSHIELD:-1}"
RUN_MPCVIT="${RUN_MPCVIT:-1}"

if [[ -z "$TRAIN_DATA_PATH" || -z "$VAL_DATA_PATH" ]]; then
  echo "请先设置 TRAIN_DATA_PATH 和 VAL_DATA_PATH。" >&2
  exit 1
fi

mkdir -p "$FAIR_OUTPUT_DIR" "$SECURE_RUN_DIR" "$MPCVIT_OUTPUT_ROOT"

echo "[fair] RUN_NAME=$RUN_NAME"
echo "[fair] TRAIN_DATA_PATH=$TRAIN_DATA_PATH"
echo "[fair] VAL_DATA_PATH=$VAL_DATA_PATH"
echo "[fair] BUNDLE_DIR=$BUNDLE_DIR"
echo "[fair] SECURE_RUNTIME=$SECURE_RUNTIME"
echo "[fair] FAIR_OUTPUT_DIR=$FAIR_OUTPUT_DIR"
echo "[fair] MPCVIT_MODEL_NAME=$MPCVIT_MODEL_NAME"
echo "[fair] MPCVIT_IMG_SIZE=$MPCVIT_IMG_SIZE"

if [[ "$RUN_TRANSSHIELD" == "1" ]]; then
  echo "[fair] Step 1：运行 Transshield modified 明文评估。"
  RUN_NAME="$RUN_NAME" \
  TRAIN_DATA_PATH="$TRAIN_DATA_PATH" \
  VAL_DATA_PATH="$VAL_DATA_PATH" \
  BUNDLE_DIR="$BUNDLE_DIR" \
  SECURE_RUN_DIR="$SECURE_RUN_DIR" \
  PLAINTEXT_EVAL_DEVICE="$PLAINTEXT_EVAL_DEVICE" \
  PLAINTEXT_MAX_SAMPLES="$PLAINTEXT_MAX_SAMPLES" \
  "$SCRIPT_DIR/run_plaintext_eval.sh" modified

  echo "[fair] Step 2：运行 Transshield secure export / SPU / replay / compare。"
  RUN_NAME="$RUN_NAME" \
  TRAIN_DATA_PATH="$TRAIN_DATA_PATH" \
  VAL_DATA_PATH="$VAL_DATA_PATH" \
  BUNDLE_DIR="$BUNDLE_DIR" \
  SECURE_RUN_DIR="$SECURE_RUN_DIR" \
  SECURE_RUNTIME="$SECURE_RUNTIME" \
  SECURE_MAX_SAMPLES="$SECURE_MAX_SAMPLES" \
  SECURE_EXPORT_DEVICE="$SECURE_EXPORT_DEVICE" \
  "$SCRIPT_DIR/run_secure_export_inputs.sh"

  if [[ "$SECURE_RUNTIME" == "cpu" ]]; then
    RUN_NAME="$RUN_NAME" BUNDLE_DIR="$BUNDLE_DIR" SECURE_RUN_DIR="$SECURE_RUN_DIR" "$SCRIPT_DIR/run_secure_pipeline.sh" cpu
  else
    RUN_NAME="$RUN_NAME" BUNDLE_DIR="$BUNDLE_DIR" SECURE_RUN_DIR="$SECURE_RUN_DIR" "$SCRIPT_DIR/run_secure_pipeline.sh" spu
  fi

  RUN_NAME="$RUN_NAME" BUNDLE_DIR="$BUNDLE_DIR" SECURE_RUN_DIR="$SECURE_RUN_DIR" "$SCRIPT_DIR/run_secure_replay.sh"
  RUN_NAME="$RUN_NAME" BUNDLE_DIR="$BUNDLE_DIR" SECURE_RUN_DIR="$SECURE_RUN_DIR" "$SCRIPT_DIR/run_secure_score_compare.sh"
  RUN_NAME="$RUN_NAME" BUNDLE_DIR="$BUNDLE_DIR" SECURE_RUN_DIR="$SECURE_RUN_DIR" "$SCRIPT_DIR/run_secure_profile_summary.sh"
else
  echo "[fair] 跳过 Transshield 运行，复用 SECURE_RUN_DIR=$SECURE_RUN_DIR"
fi

MPCVIT_SUMMARY_ARG=()
if [[ "$RUN_MPCVIT" == "1" ]]; then
  if [[ ! -d "$MPCVIT_ROOT" ]]; then
    echo "找不到 MPCViT 仓库：$MPCVIT_ROOT" >&2
    exit 1
  fi

  echo "[fair] Step 3：运行 MPCViT 同数据集训练 / 验证。"
  env \
    PYTHON_BIN="$PYTHON_BIN" \
    TRAIN_DIR="$TRAIN_DATA_PATH" \
    VAL_DIR="$VAL_DATA_PATH" \
    OUTPUT_ROOT="$MPCVIT_OUTPUT_ROOT" \
    SEEDS="$MPCVIT_SEEDS" \
    EPOCHS="$MPCVIT_EPOCHS" \
    BATCH_SIZE="$MPCVIT_BATCH_SIZE" \
    NUM_WORKERS="$MPCVIT_NUM_WORKERS" \
    DEVICE="$MPCVIT_DEVICE" \
    MODEL_NAME="$MPCVIT_MODEL_NAME" \
    IMG_SIZE="$MPCVIT_IMG_SIZE" \
    CLASS_BALANCED_LOSS="$MPCVIT_CLASS_BALANCED_LOSS" \
    NO_HFLIP="$MPCVIT_NO_HFLIP" \
    bash "$MPCVIT_ROOT/tools/run_pneumoniamnist_multiseed_server.sh"
  MPCVIT_SUMMARY_ARG=(--mpcvit-multiseed-json "$MPCVIT_OUTPUT_ROOT/summary_multiseed.json")
else
  if [[ -n "${MPCVIT_SUMMARY_JSON:-}" ]]; then
    MPCVIT_SUMMARY_ARG=(--mpcvit-summary-json "$MPCVIT_SUMMARY_JSON")
  elif [[ -n "${MPCVIT_MULTI_SEED_JSON:-}" ]]; then
    MPCVIT_SUMMARY_ARG=(--mpcvit-multiseed-json "$MPCVIT_MULTI_SEED_JSON")
  else
    echo "[fair] 跳过 MPCViT 运行，且未提供 MPCVIT_SUMMARY_JSON / MPCVIT_MULTI_SEED_JSON；报告里不会有外部结果。"
  fi
fi

echo "[fair] Step 4：生成公平对比报告。"
"$PYTHON_BIN" tools/transshield_fair_external_comparison.py \
  --transshield-run-dir "$SECURE_RUN_DIR" \
  "${MPCVIT_SUMMARY_ARG[@]}" \
  --train-data-path "$TRAIN_DATA_PATH" \
  --val-data-path "$VAL_DATA_PATH" \
  --output-json "$FAIR_OUTPUT_DIR/fair_external_comparison.json" \
  --output-md "$FAIR_OUTPUT_DIR/fair_external_comparison.md"

echo "[fair] 完成："
echo "[fair] JSON: $FAIR_OUTPUT_DIR/fair_external_comparison.json"
echo "[fair] MD:   $FAIR_OUTPUT_DIR/fair_external_comparison.md"
