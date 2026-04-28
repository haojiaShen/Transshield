#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="${RUN_NAME:-standardized_secure_benchmark_$(date +%Y%m%d_%H%M%S)}"
EXTERNAL_BASELINES_ROOT="${EXTERNAL_BASELINES_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/external_baselines}"
MPCFORMER_ROOT="${MPCFORMER_ROOT:-$EXTERNAL_BASELINES_ROOT/MPCFormer}"
BENCH_OUTPUT_ROOT="${BENCH_OUTPUT_ROOT:-$REPO_ROOT/results/standardized_secure_benchmark/$RUN_NAME}"

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29600}"
WORLD_SIZE="${WORLD_SIZE:-2}"
MPCFORMER_BATCH_SIZE="${MPCFORMER_BATCH_SIZE:-1}"
MPCFORMER_WARMUP="${MPCFORMER_WARMUP:-1}"
MPCFORMER_REPEATS="${MPCFORMER_REPEATS:-3}"
MPCFORMER_SEED="${MPCFORMER_SEED:-0}"

RUN_ARCHITECTURE_PROXY="${RUN_ARCHITECTURE_PROXY:-1}"
RUN_SAME_SHAPE_OPERATOR_PROXY="${RUN_SAME_SHAPE_OPERATOR_PROXY:-0}"

if [[ ! -x "$MPCFORMER_ROOT/tools/run_transformer_local2pc_server.sh" ]]; then
  echo "找不到 MPCFormer benchmark 入口：$MPCFORMER_ROOT/tools/run_transformer_local2pc_server.sh" >&2
  exit 1
fi

mkdir -p "$BENCH_OUTPUT_ROOT"

echo "[std-bench] RUN_NAME=$RUN_NAME"
echo "[std-bench] MPCFORMER_ROOT=$MPCFORMER_ROOT"
echo "[std-bench] BENCH_OUTPUT_ROOT=$BENCH_OUTPUT_ROOT"
echo "[std-bench] 注意：这是统一 secure transformer benchmark，不是 full-val 医学图像 pipeline。"

PROFILE_INDEX=0

run_profile() {
  local profile_id="$1"
  local display_name="$2"
  local role="$3"
  local comparison_group="$4"
  local model_source="$5"
  local scope_note="$6"
  local num_hidden_layers="$7"
  local hidden_size="$8"
  local intermediate_size="$9"
  local sequence_length="${10}"
  local num_attention_heads="${11}"
  local hidden_act="${12}"
  local softmax_act="${13}"

  local output_dir="$BENCH_OUTPUT_ROOT/$profile_id"
  local master_port=$((MASTER_PORT_BASE + PROFILE_INDEX))
  mkdir -p "$output_dir"

  cat > "$output_dir/profile_meta.json" <<JSON
{
  "profile_id": "$profile_id",
  "display_name": "$display_name",
  "role": "$role",
  "comparison_group": "$comparison_group",
  "model_source": "$model_source",
  "scope_note": "$scope_note",
  "batch_size": $MPCFORMER_BATCH_SIZE,
  "num_hidden_layers": $num_hidden_layers,
  "hidden_size": $hidden_size,
  "intermediate_size": $intermediate_size,
  "sequence_length": $sequence_length,
  "num_attention_heads": $num_attention_heads,
  "hidden_act": "$hidden_act",
  "softmax_act": "$softmax_act",
  "warmup": $MPCFORMER_WARMUP,
  "repeats": $MPCFORMER_REPEATS,
  "world_size": $WORLD_SIZE
}
JSON

  echo "[std-bench] running $profile_id on port $master_port"
  env \
    OUTPUT_DIR="$output_dir" \
    PYTHON_BIN="$PYTHON_BIN" \
    MPCFORMER_ROOT="$MPCFORMER_ROOT" \
    MASTER_ADDR="$MASTER_ADDR" \
    MASTER_PORT="$master_port" \
    WORLD_SIZE="$WORLD_SIZE" \
    MPCFORMER_BATCH_SIZE="$MPCFORMER_BATCH_SIZE" \
    MPCFORMER_NUM_HIDDEN_LAYERS="$num_hidden_layers" \
    MPCFORMER_HIDDEN_SIZE="$hidden_size" \
    MPCFORMER_INTERMEDIATE_SIZE="$intermediate_size" \
    MPCFORMER_SEQUENCE_LENGTH="$sequence_length" \
    MPCFORMER_MAX_POSITION_EMBEDDINGS="$sequence_length" \
    MPCFORMER_NUM_ATTENTION_HEADS="$num_attention_heads" \
    MPCFORMER_HIDDEN_ACT="$hidden_act" \
    MPCFORMER_SOFTMAX_ACT="$softmax_act" \
    MPCFORMER_WARMUP="$MPCFORMER_WARMUP" \
    MPCFORMER_REPEATS="$MPCFORMER_REPEATS" \
    MPCFORMER_SEED="$MPCFORMER_SEED" \
    bash "$MPCFORMER_ROOT/tools/run_transformer_local2pc_server.sh"

  PROFILE_INDEX=$((PROFILE_INDEX + 1))
}

if [[ "$RUN_ARCHITECTURE_PROXY" == "1" ]]; then
  run_profile \
    "transshield_final_arch_proxy" \
    "Transshield 当前最终模型 proxy" \
    "current_project" \
    "architecture_proxy" \
    "artifacts/frozen_bundle_verified_tracka_lr3e5_20260414 · DeiT-S / 224px / dynamic token pruning" \
    "同一 MPCFormer local 2PC benchmark harness；使用当前最终模型的 transformer 结构参数近似。" \
    12 384 1536 197 6 "quad" "softmax"

  run_profile \
    "mpcvit_vit_7_4_32_arch_proxy" \
    "MPCViT vit_7_4_32 proxy" \
    "external_baseline" \
    "architecture_proxy" \
    "external_baselines/mpcvit · vit_7_4_32 / 32px" \
    "同一 MPCFormer local 2PC benchmark harness；使用 MPCViT 当前同数据集 baseline 的结构参数近似。" \
    7 256 512 65 4 "relu" "softmax_2RELU"
fi

if [[ "$RUN_SAME_SHAPE_OPERATOR_PROXY" == "1" ]]; then
  run_profile \
    "transshield_ops_same_shape_proxy" \
    "Transshield secure-friendly ops same-shape proxy" \
    "current_project" \
    "same_shape_operator_proxy" \
    "Transshield operator replacement proxy" \
    "固定 DeiT-S 形状，仅替换 secure-friendly activation / softmax proxy，用于观察算子配置开销。" \
    12 384 1536 197 6 "quad" "softmax_2QUAD"

  run_profile \
    "baseline_ops_same_shape_proxy" \
    "External baseline ops same-shape proxy" \
    "external_baseline" \
    "same_shape_operator_proxy" \
    "MPCFormer baseline transformer ops proxy" \
    "固定 DeiT-S 形状，使用外部 secure transformer benchmark 中的 baseline 非线性 / softmax 配置。" \
    12 384 1536 197 6 "relu" "softmax"
fi

"$PYTHON_BIN" tools/transshield_standardized_secure_benchmark_report.py \
  --output-root "$BENCH_OUTPUT_ROOT" \
  --output-json "$BENCH_OUTPUT_ROOT/standardized_secure_benchmark.json" \
  --output-md "$BENCH_OUTPUT_ROOT/standardized_secure_benchmark.md"

echo "[std-bench] 完成："
echo "[std-bench] JSON: $BENCH_OUTPUT_ROOT/standardized_secure_benchmark.json"
echo "[std-bench] MD:   $BENCH_OUTPUT_ROOT/standardized_secure_benchmark.md"
