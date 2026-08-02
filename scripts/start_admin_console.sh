#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$REPO_ROOT/artifacts/showcase_server_logs"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  for candidate in \
    "$REPO_ROOT/.venv/bin/python" \
    "/home/yclcg/miniconda3/envs/transshield/bin/python" \
    "/data/wyb/conda_envs/transshield/bin/python" \
    "$(command -v python3 2>/dev/null || true)"; do
    if [[ -x "$candidate" ]] && "$candidate" -c 'import fastapi, uvicorn' >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi
: "${PYTHON_BIN:?No Python with fastapi and uvicorn was found. Set PYTHON_BIN explicitly.}"
ADMIN_API_HOST="${TRANSSHIELD_ADMIN_API_HOST:-0.0.0.0}"
ADMIN_API_PORT="${TRANSSHIELD_ADMIN_API_PORT:-7863}"
ADMIN_UI_HOST="${TRANSSHIELD_ADMIN_UI_HOST:-0.0.0.0}"
ADMIN_UI_PORT="${TRANSSHIELD_ADMIN_UI_PORT:-4174}"
ADMIN_API_SESSION="${TRANSSHIELD_ADMIN_API_SESSION:-transshield_admin_api_${ADMIN_API_PORT}}"
ADMIN_UI_SESSION="${TRANSSHIELD_ADMIN_UI_SESSION:-transshield_admin_ui_${ADMIN_UI_PORT}}"
TRAIN_DATA_PATH="${TRANSSHIELD_SHOWCASE_DEFAULT_TRAIN_DATA_PATH:-$REPO_ROOT/data/pneumoniamnist_imagefolder_subset/train}"
EVAL_DATA_PATH="${TRANSSHIELD_SHOWCASE_DEFAULT_EVAL_DATA_PATH:-$REPO_ROOT/data/pneumoniamnist_imagefolder_subset/val}"
ADMIN_PASSWORD="${TRANSSHIELD_ADMIN_PASSWORD:-admin123}"
CORS_ALLOWED_ORIGINS="${TRANSSHIELD_CORS_ALLOWED_ORIGINS:-http://127.0.0.1:4174,http://localhost:4174,http://127.0.0.1:7863,http://localhost:7863,http://127.0.0.1:7862,http://localhost:7862,http://127.0.0.1:17862,http://localhost:17862}"
ADMIN_API_LOG="${TRANSSHIELD_ADMIN_API_LOG:-$LOG_DIR/uvicorn_${ADMIN_API_PORT}_admin.log}"
ADMIN_UI_LOG="${TRANSSHIELD_ADMIN_UI_LOG:-$LOG_DIR/showcase_admin_${ADMIN_UI_PORT}.log}"

quote() {
  printf "'%s'" "${1//\'/\'\"\'\"\'}"
}

wait_for_url() {
  local url="$1"
  local timeout="${2:-30}"
  local start
  start="$(date +%s)"
  while true; do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    if (( "$(date +%s)" - start >= timeout )); then
      echo "timeout waiting for $url" >&2
      return 1
    fi
    sleep 1
  done
}

command -v tmux >/dev/null
[[ -x "$PYTHON_BIN" ]]
[[ -f "$REPO_ROOT/showcase_admin/dist/index.html" ]]
[[ -x "$REPO_ROOT/showcase_admin/node_modules/.bin/vite" ]]

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

export TRANSSHIELD_SHOWCASE_PYTHON_BIN="$PYTHON_BIN"
export TRANSSHIELD_SHOWCASE_DEFAULT_TRAIN_DATA_PATH="$TRAIN_DATA_PATH"
export TRANSSHIELD_SHOWCASE_DEFAULT_EVAL_DATA_PATH="$EVAL_DATA_PATH"
export TRANSSHIELD_ADMIN_PASSWORD="$ADMIN_PASSWORD"
export TRANSSHIELD_CORS_ALLOWED_ORIGINS="$CORS_ALLOWED_ORIGINS"

tmux kill-session -t "$ADMIN_API_SESSION" 2>/dev/null || true
tmux kill-session -t "$ADMIN_UI_SESSION" 2>/dev/null || true

api_command="cd $(quote "$REPO_ROOT") && export TRANSSHIELD_SHOWCASE_PYTHON_BIN=$(quote "$TRANSSHIELD_SHOWCASE_PYTHON_BIN") TRANSSHIELD_SHOWCASE_DEFAULT_TRAIN_DATA_PATH=$(quote "$TRANSSHIELD_SHOWCASE_DEFAULT_TRAIN_DATA_PATH") TRANSSHIELD_SHOWCASE_DEFAULT_EVAL_DATA_PATH=$(quote "$TRANSSHIELD_SHOWCASE_DEFAULT_EVAL_DATA_PATH") TRANSSHIELD_ADMIN_PASSWORD=$(quote "$TRANSSHIELD_ADMIN_PASSWORD") TRANSSHIELD_CORS_ALLOWED_ORIGINS=$(quote "$TRANSSHIELD_CORS_ALLOWED_ORIGINS") && $(quote "$PYTHON_BIN") -m uvicorn showcase_api.app:app --host $(quote "$ADMIN_API_HOST") --port $(quote "$ADMIN_API_PORT") > $(quote "$ADMIN_API_LOG") 2>&1"
ui_command="cd $(quote "$REPO_ROOT/showcase_admin") && ./node_modules/.bin/vite preview --host $(quote "$ADMIN_UI_HOST") --port $(quote "$ADMIN_UI_PORT") > $(quote "$ADMIN_UI_LOG") 2>&1"

tmux new-session -d -s "$ADMIN_API_SESSION" "$api_command"
tmux new-session -d -s "$ADMIN_UI_SESSION" "$ui_command"

wait_for_url "http://127.0.0.1:${ADMIN_API_PORT}/api/health" 30
wait_for_url "http://127.0.0.1:${ADMIN_UI_PORT}/" 30

echo "admin api: http://127.0.0.1:${ADMIN_API_PORT}/api/health"
echo "admin ui: http://127.0.0.1:${ADMIN_UI_PORT}/"
