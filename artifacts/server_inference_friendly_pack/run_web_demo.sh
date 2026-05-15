#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
# Web live run 当前默认跟随 secure-static bundle；
# 前端静态成绩板若未单独更新，仍可能显示 verified_tracka 的旧正式成绩。
BUNDLE_DIR="${BUNDLE_DIR:-artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_20260430}"
PLAINTEXT_EVAL_DEVICE="${PLAINTEXT_EVAL_DEVICE:-cpu}"
CLASS_NAMES="${CLASS_NAMES:-class_0,class_1}"
WEB_DEMO_HOST="${WEB_DEMO_HOST:-127.0.0.1}"
WEB_DEMO_PORT="${WEB_DEMO_PORT:-7860}"
WEB_DEMO_UPLOAD_DIR="${WEB_DEMO_UPLOAD_DIR:-artifacts/web_demo_uploads}"
WEB_DEMO_SECURE_TIMEOUT_SEC="${WEB_DEMO_SECURE_TIMEOUT_SEC:-600}"
WEB_DEMO_MAX_UPLOAD_MB="${WEB_DEMO_MAX_UPLOAD_MB:-10}"
WEB_DEMO_REUSE_SPU_RUNTIME="${WEB_DEMO_REUSE_SPU_RUNTIME:-0}"
WEB_DEMO_SKIP_PIPELINE_VERIFY="${WEB_DEMO_SKIP_PIPELINE_VERIFY:-1}"
# 当前默认展示 profile 是 secret depth6 clip0 showcase，不是旧 public depth12 clip3 线。
WEB_DEMO_E2E_PROFILE="${WEB_DEMO_E2E_PROFILE:-secret_depth6_clip0_showcase}"
WEB_DEMO_E2E_EXECUTION_MODE="${WEB_DEMO_E2E_EXECUTION_MODE:-local}"
WEB_DEMO_REMOTE_SSH_TARGET="${WEB_DEMO_REMOTE_SSH_TARGET:-}"
WEB_DEMO_REMOTE_SSH_PORT="${WEB_DEMO_REMOTE_SSH_PORT:-9001}"
WEB_DEMO_REMOTE_REPO_ROOT="${WEB_DEMO_REMOTE_REPO_ROOT:-${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}}"
WEB_DEMO_REMOTE_PYTHON_BIN="${WEB_DEMO_REMOTE_PYTHON_BIN:-${PYTHON_BIN:-python}}"
export WEB_DEMO_SECURE_TIMEOUT_SEC WEB_DEMO_MAX_UPLOAD_MB WEB_DEMO_REUSE_SPU_RUNTIME WEB_DEMO_SKIP_PIPELINE_VERIFY WEB_DEMO_E2E_PROFILE
export WEB_DEMO_E2E_EXECUTION_MODE WEB_DEMO_REMOTE_SSH_TARGET WEB_DEMO_REMOTE_SSH_PORT WEB_DEMO_REMOTE_REPO_ROOT WEB_DEMO_REMOTE_PYTHON_BIN

echo "[web-demo] 启动前后端一体化图像处理流程 Demo。"
echo "[web-demo] 当前版本不启用 GPT/聊天接口，右侧展示模型处理流程与 secure profiling。"
echo "[web-demo] 默认 bundle：${BUNDLE_DIR}"
echo "[web-demo] E2E profile：${WEB_DEMO_E2E_PROFILE}"
echo "[web-demo] E2E execution mode：${WEB_DEMO_E2E_EXECUTION_MODE}（local=本机 SPU，ssh=服务器 SPU 远程执行）。"
if [[ "$WEB_DEMO_E2E_EXECUTION_MODE" == "ssh" ]]; then
  echo "[web-demo] remote target：${WEB_DEMO_REMOTE_SSH_TARGET:-<unset>}；remote repo：${WEB_DEMO_REMOTE_REPO_ROOT}；remote python：${WEB_DEMO_REMOTE_PYTHON_BIN}"
fi
echo "[web-demo] secure pipeline 超时保护：${WEB_DEMO_SECURE_TIMEOUT_SEC}s；上传图片大小限制：${WEB_DEMO_MAX_UPLOAD_MB} MB。"
echo "[web-demo] SPU runtime 复用：${WEB_DEMO_REUSE_SPU_RUNTIME}（1=优先复用，0=每次强制重启）。"
echo "[web-demo] pipeline verify：${WEB_DEMO_SKIP_PIPELINE_VERIFY}（1=单图 fast path 跳过逐次 verify）。"
if [[ "$WEB_DEMO_HOST" == "0.0.0.0" ]]; then
  echo "[web-demo] 当前监听 0.0.0.0:${WEB_DEMO_PORT}；请在浏览器访问服务器 IP，例如 http://<server-ip>:${WEB_DEMO_PORT}"
else
  echo "[web-demo] 打开浏览器访问 http://${WEB_DEMO_HOST}:${WEB_DEMO_PORT}"
fi

"$PYTHON_BIN" tools/transshield_chat_demo.py \
  --bundle-dir "$BUNDLE_DIR" \
  --device "$PLAINTEXT_EVAL_DEVICE" \
  --class-names "$CLASS_NAMES" \
  --host "$WEB_DEMO_HOST" \
  --port "$WEB_DEMO_PORT" \
  --upload-dir "$WEB_DEMO_UPLOAD_DIR"
