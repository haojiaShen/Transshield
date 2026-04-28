#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="${RUN_NAME:-transshield_e2e_secure_poc}"
BUNDLE_DIR="${BUNDLE_DIR:-$REPO_ROOT/artifacts/frozen_bundle_verified_tracka_lr3e5_20260414}"
E2E_RUN_DIR="${E2E_RUN_DIR:-$REPO_ROOT/artifacts/server_pipeline_run/${RUN_NAME}/e2e_secure_poc}"
E2E_INPUT_PT="${E2E_INPUT_PT:-$E2E_RUN_DIR/client_pixel_values.pt}"
E2E_INPUT_JSON="${E2E_INPUT_JSON:-$E2E_RUN_DIR/client_pixel_values.json}"
E2E_CONTRACT_JSON="${E2E_CONTRACT_JSON:-$E2E_RUN_DIR/e2e_secure_contract.json}"
E2E_REFERENCE_JSON="${E2E_REFERENCE_JSON:-$E2E_RUN_DIR/plaintext_reference.json}"
E2E_STATIC_REFERENCE_JSON="${E2E_STATIC_REFERENCE_JSON:-$E2E_RUN_DIR/static_whole_forward_reference.json}"
E2E_STATIC_REFERENCE_PT="${E2E_STATIC_REFERENCE_PT:-$E2E_RUN_DIR/static_whole_forward_reference.pt}"
E2E_SPU_PLAN_JSON="${E2E_SPU_PLAN_JSON:-$E2E_RUN_DIR/spu_plan.json}"
E2E_DEVICE="${E2E_DEVICE:-cpu}"
E2E_MAX_SAMPLES="${E2E_MAX_SAMPLES:-0}"
E2E_INCLUDE_TARGETS="${E2E_INCLUDE_TARGETS:-0}"
E2E_INCLUDE_SOURCE_PATHS="${E2E_INCLUDE_SOURCE_PATHS:-0}"
E2E_GENERATE_DEBUG_SHARES="${E2E_GENERATE_DEBUG_SHARES:-0}"
E2E_DEBUG_SHARE_PREFIX="${E2E_DEBUG_SHARE_PREFIX:-$E2E_RUN_DIR/client_pixel_values_debug_share}"
E2E_SHARE_ONLY="${E2E_SHARE_ONLY:-0}"
E2E_SHARE_MANIFEST_JSON="${E2E_SHARE_MANIFEST_JSON:-$E2E_RUN_DIR/client_pixel_values_debug_share_manifest.json}"
E2E_SHARE_PUBLIC_MANIFEST_JSON="${E2E_SHARE_PUBLIC_MANIFEST_JSON:-$E2E_RUN_DIR/client_pixel_values_debug_share_public_manifest.json}"
E2E_SHARE_PARTY_MANIFEST_DIR="${E2E_SHARE_PARTY_MANIFEST_DIR:-$E2E_RUN_DIR/client_pixel_values_debug_share_party_manifests}"
E2E_RECONSTRUCT_FROM_SHARES="${E2E_RECONSTRUCT_FROM_SHARES:-0}"
E2E_RECONSTRUCT_JSON="${E2E_RECONSTRUCT_JSON:-$E2E_RUN_DIR/client_pixel_values_debug_reconstruct.json}"
VAL_DATA_PATH="${VAL_DATA_PATH:-}"
INPUT_IMAGE="${INPUT_IMAGE:-}"
INPUT_IMAGE_LIST="${INPUT_IMAGE_LIST:-}"
INPUT_IMAGE_DIR="${INPUT_IMAGE_DIR:-}"
INPUT_GLOB_PATTERN="${INPUT_GLOB_PATTERN:-*}"

mkdir -p "$E2E_RUN_DIR"

echo "[e2e-poc] 生成 e2e secure inference 边界合同。"
"$PYTHON_BIN" tools/transshield_e2e_secure_infer.py contract \
  --bundle-dir "$BUNDLE_DIR" \
  --output-json "$E2E_CONTRACT_JSON"

if [[ "$E2E_SHARE_ONLY" == "1" && ! -f "$E2E_SHARE_MANIFEST_JSON" ]]; then
  if [[ -z "$VAL_DATA_PATH" && -z "$INPUT_IMAGE" && -z "$INPUT_IMAGE_LIST" && -z "$INPUT_IMAGE_DIR" ]]; then
    echo "[e2e-poc] E2E_SHARE_ONLY=1 时，请先设置 VAL_DATA_PATH 或 INPUT_IMAGE / INPUT_IMAGE_LIST / INPUT_IMAGE_DIR。" >&2
    exit 1
  fi

  echo "[e2e-poc] 生成客户端 debug additive share manifest，不写 plaintext pixel package。"
  SHARE_ARGS=(
    tools/transshield_e2e_secure_infer.py client-share-preprocess
    --bundle-dir "$BUNDLE_DIR"
    --output-prefix "$E2E_DEBUG_SHARE_PREFIX"
    --output-json "$E2E_SHARE_MANIFEST_JSON"
    --output-public-json "$E2E_SHARE_PUBLIC_MANIFEST_JSON"
    --output-party-manifest-dir "$E2E_SHARE_PARTY_MANIFEST_DIR"
    --max-samples "$E2E_MAX_SAMPLES"
  )
  if [[ -n "$VAL_DATA_PATH" ]]; then
    SHARE_ARGS+=(--data-path "$VAL_DATA_PATH")
  fi
  if [[ -n "$INPUT_IMAGE" ]]; then
    SHARE_ARGS+=(--image "$INPUT_IMAGE")
  fi
  if [[ -n "$INPUT_IMAGE_LIST" ]]; then
    SHARE_ARGS+=(--image-list "$INPUT_IMAGE_LIST")
  fi
  if [[ -n "$INPUT_IMAGE_DIR" ]]; then
    SHARE_ARGS+=(--input-dir "$INPUT_IMAGE_DIR" --glob-pattern "$INPUT_GLOB_PATTERN")
  fi
  if [[ "$E2E_INCLUDE_TARGETS" == "1" ]]; then
    SHARE_ARGS+=(--include-targets)
  fi
  if [[ "$E2E_INCLUDE_SOURCE_PATHS" == "1" ]]; then
    SHARE_ARGS+=(--include-source-paths)
  fi
  "$PYTHON_BIN" "${SHARE_ARGS[@]}"
fi

if [[ "$E2E_SHARE_ONLY" == "1" && -f "$E2E_SHARE_MANIFEST_JSON" ]]; then
  if [[ ! -f "$E2E_SHARE_PUBLIC_MANIFEST_JSON" || ! -f "$E2E_SHARE_PARTY_MANIFEST_DIR/p1_share_manifest.json" || ! -f "$E2E_SHARE_PARTY_MANIFEST_DIR/p2_share_manifest.json" ]]; then
    echo "[e2e-poc] 从既有 debug share manifest 写出 public/P1/P2 split manifests。"
    "$PYTHON_BIN" tools/transshield_e2e_secure_infer.py split-debug-share-manifest \
      --share-manifest-json "$E2E_SHARE_MANIFEST_JSON" \
      --output-public-json "$E2E_SHARE_PUBLIC_MANIFEST_JSON" \
      --output-party-manifest-dir "$E2E_SHARE_PARTY_MANIFEST_DIR"
  fi
fi

if [[ "$E2E_RECONSTRUCT_FROM_SHARES" == "1" && ! -f "$E2E_INPUT_PT" ]]; then
  echo "[e2e-poc] 从 debug additive shares 重构 plaintext pixel package 以兼容当前 backend。"
  echo "[e2e-poc] 注意：这是调试桥接，不是最终全隐私服务器路径。"
  "$PYTHON_BIN" tools/transshield_e2e_secure_infer.py reconstruct-debug-shares \
    --share-manifest-json "$E2E_SHARE_MANIFEST_JSON" \
    --output-pt "$E2E_INPUT_PT" \
    --output-json "$E2E_RECONSTRUCT_JSON"
fi

if [[ "$E2E_SHARE_ONLY" == "1" && "$E2E_RECONSTRUCT_FROM_SHARES" != "1" ]]; then
  echo "[e2e-poc] E2E_SHARE_ONLY=1 且未启用重构，已在 share manifest 阶段停止。"
  echo "[e2e-poc] 输出：$E2E_SHARE_MANIFEST_JSON"
  echo "[e2e-poc] public manifest：$E2E_SHARE_PUBLIC_MANIFEST_JSON"
  echo "[e2e-poc] party manifests：$E2E_SHARE_PARTY_MANIFEST_DIR"
  exit 0
fi

if [[ ! -f "$E2E_INPUT_PT" ]]; then
  if [[ -z "$VAL_DATA_PATH" && -z "$INPUT_IMAGE" && -z "$INPUT_IMAGE_LIST" && -z "$INPUT_IMAGE_DIR" ]]; then
    echo "[e2e-poc] 若未提供现成 E2E_INPUT_PT，请先设置 VAL_DATA_PATH 或 INPUT_IMAGE / INPUT_IMAGE_LIST / INPUT_IMAGE_DIR。" >&2
    exit 1
  fi

  echo "[e2e-poc] 生成客户端预处理像素包。"
  echo "[e2e-poc] 注意：这一步仍是 plaintext client tensor，不是正式 MPC share。"

  PREPROCESS_ARGS=(
    tools/transshield_e2e_secure_infer.py client-preprocess
    --bundle-dir "$BUNDLE_DIR"
    --output-pt "$E2E_INPUT_PT"
    --output-json "$E2E_INPUT_JSON"
    --max-samples "$E2E_MAX_SAMPLES"
  )
  if [[ -n "$VAL_DATA_PATH" ]]; then
    PREPROCESS_ARGS+=(--data-path "$VAL_DATA_PATH")
  fi
  if [[ -n "$INPUT_IMAGE" ]]; then
    PREPROCESS_ARGS+=(--image "$INPUT_IMAGE")
  fi
  if [[ -n "$INPUT_IMAGE_LIST" ]]; then
    PREPROCESS_ARGS+=(--image-list "$INPUT_IMAGE_LIST")
  fi
  if [[ -n "$INPUT_IMAGE_DIR" ]]; then
    PREPROCESS_ARGS+=(--input-dir "$INPUT_IMAGE_DIR" --glob-pattern "$INPUT_GLOB_PATTERN")
  fi
  if [[ "$E2E_INCLUDE_TARGETS" == "1" ]]; then
    PREPROCESS_ARGS+=(--include-targets)
  fi
  if [[ "$E2E_INCLUDE_SOURCE_PATHS" == "1" ]]; then
    PREPROCESS_ARGS+=(--include-source-paths)
  fi
  if [[ "$E2E_GENERATE_DEBUG_SHARES" == "1" ]]; then
    PREPROCESS_ARGS+=(--debug-share-prefix "$E2E_DEBUG_SHARE_PREFIX")
  fi
  "$PYTHON_BIN" "${PREPROCESS_ARGS[@]}"
else
  echo "[e2e-poc] 复用现有客户端像素包：$E2E_INPUT_PT"
fi

echo "[e2e-poc] 运行 plaintext reference。"
"$PYTHON_BIN" tools/transshield_e2e_secure_infer.py plaintext-reference \
  --bundle-dir "$BUNDLE_DIR" \
  --input-pt "$E2E_INPUT_PT" \
  --device "$E2E_DEVICE" \
  --output-json "$E2E_REFERENCE_JSON"

echo "[e2e-poc] 运行 static whole-forward plaintext reference。"
"$PYTHON_BIN" tools/transshield_e2e_secure_infer.py static-whole-forward-reference \
  --bundle-dir "$BUNDLE_DIR" \
  --input-pt "$E2E_INPUT_PT" \
  --device "$E2E_DEVICE" \
  --output-json "$E2E_STATIC_REFERENCE_JSON" \
  --output-pt "$E2E_STATIC_REFERENCE_PT"

echo "[e2e-poc] 写出后续 SPU 整网实现计划。"
"$PYTHON_BIN" tools/transshield_e2e_secure_infer.py spu-plan \
  --bundle-dir "$BUNDLE_DIR" \
  --output-json "$E2E_SPU_PLAN_JSON"

echo "[e2e-poc] 完成。输出目录：$E2E_RUN_DIR"
