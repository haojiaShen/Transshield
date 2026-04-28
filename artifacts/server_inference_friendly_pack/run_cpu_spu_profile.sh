#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="${RUN_NAME:-transshield_profile_pair}"
BUNDLE_DIR="${BUNDLE_DIR:-artifacts/frozen_bundle_verified_tracka_lr3e5_20260414}"
CONFIG_PATH="${CONFIG_PATH:-configs/openbumblebee/2pc.json}"
PROFILE_REUSE_EXISTING="${PROFILE_REUSE_EXISTING:-0}"

CPU_PROFILE_RUN_NAME="${CPU_PROFILE_RUN_NAME:-${RUN_NAME}_cpu}"
SPU_PROFILE_RUN_NAME="${SPU_PROFILE_RUN_NAME:-${RUN_NAME}_spu}"
CPU_SECURE_RUN_DIR="${CPU_SECURE_RUN_DIR:-artifacts/server_pipeline_run/${CPU_PROFILE_RUN_NAME}}"
SPU_SECURE_RUN_DIR="${SPU_SECURE_RUN_DIR:-artifacts/server_pipeline_run/${SPU_PROFILE_RUN_NAME}}"
PROFILE_REPORT_DIR="${PROFILE_REPORT_DIR:-artifacts/server_profile_reports/${RUN_NAME}}"

echo "[profile-pair] 生成 CPU vs SPU secure profiling。"
echo "[profile-pair] 输出会写入各自 secure 运行目录与 $PROFILE_REPORT_DIR。"

if [[ "$PROFILE_REUSE_EXISTING" != "1" ]]; then
  echo "[profile-pair] 为 CPU profile 准备 secure 导出输入。"
  env \
    RUN_NAME="$CPU_PROFILE_RUN_NAME" \
    BUNDLE_DIR="$BUNDLE_DIR" \
    CONFIG_PATH="$CONFIG_PATH" \
    SECURE_RUN_DIR="$CPU_SECURE_RUN_DIR" \
    bash artifacts/server_inference_friendly_pack/run_secure_export_inputs.sh

  echo "[profile-pair] 运行 CPU 参考 secure pipeline。"
  env \
    RUN_NAME="$CPU_PROFILE_RUN_NAME" \
    BUNDLE_DIR="$BUNDLE_DIR" \
    CONFIG_PATH="$CONFIG_PATH" \
    SECURE_RUN_DIR="$CPU_SECURE_RUN_DIR" \
    bash artifacts/server_inference_friendly_pack/run_secure_pipeline.sh cpu

  echo "[profile-pair] 为 SPU profile 准备 secure 导出输入。"
  env \
    RUN_NAME="$SPU_PROFILE_RUN_NAME" \
    BUNDLE_DIR="$BUNDLE_DIR" \
    CONFIG_PATH="$CONFIG_PATH" \
    SECURE_RUN_DIR="$SPU_SECURE_RUN_DIR" \
    bash artifacts/server_inference_friendly_pack/run_secure_export_inputs.sh

  echo "[profile-pair] 运行 SPU secure pipeline。"
  env \
    RUN_NAME="$SPU_PROFILE_RUN_NAME" \
    BUNDLE_DIR="$BUNDLE_DIR" \
    CONFIG_PATH="$CONFIG_PATH" \
    SECURE_RUN_DIR="$SPU_SECURE_RUN_DIR" \
    bash artifacts/server_inference_friendly_pack/run_secure_pipeline.sh spu
else
  echo "[profile-pair] 复用已有 CPU / SPU 运行目录，不重新执行 pipeline。"
fi

echo "[profile-pair] 汇总 CPU profile。"
"$PYTHON_BIN" tools/transshield_secure_profile_summary.py \
  --secure-run-dir "$CPU_SECURE_RUN_DIR" \
  --spu-state-json logs/spu_runtime_ports.json \
  --spu-log-dir logs/spu_nodes \
  --output-json "$CPU_SECURE_RUN_DIR/secure_profile_summary.json"

echo "[profile-pair] 汇总 SPU profile。"
"$PYTHON_BIN" tools/transshield_secure_profile_summary.py \
  --secure-run-dir "$SPU_SECURE_RUN_DIR" \
  --spu-state-json logs/spu_runtime_ports.json \
  --spu-log-dir logs/spu_nodes \
  --output-json "$SPU_SECURE_RUN_DIR/secure_profile_summary.json"

mkdir -p "$PROFILE_REPORT_DIR"

echo "[profile-pair] 生成 CPU vs SPU 对比报告。"
"$PYTHON_BIN" tools/transshield_cpu_spu_profile_report.py \
  --cpu-summary "$CPU_SECURE_RUN_DIR/secure_profile_summary.json" \
  --spu-summary "$SPU_SECURE_RUN_DIR/secure_profile_summary.json" \
  --output-json "$PROFILE_REPORT_DIR/cpu_vs_spu_profile_report.json" \
  --output-md "$PROFILE_REPORT_DIR/cpu_vs_spu_profile_report.md"

echo "[profile-pair] 完成。"
echo "[profile-pair] JSON: $PROFILE_REPORT_DIR/cpu_vs_spu_profile_report.json"
echo "[profile-pair] Markdown: $PROFILE_REPORT_DIR/cpu_vs_spu_profile_report.md"
