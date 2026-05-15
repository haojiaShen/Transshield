#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${1:-$REPO_ROOT/../Transshield_final_server_clean}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "[clean-deploy] missing python interpreter; set PYTHON_BIN explicitly" >&2
  exit 127
fi

cd "$REPO_ROOT"

"$PYTHON_BIN" tools/transshield_build_clean_deploy_repo.py \
  --source-root "$REPO_ROOT" \
  --output-dir "$OUTPUT_DIR"

echo "[clean-deploy] output: $OUTPUT_DIR"
echo "[clean-deploy] manifest: $OUTPUT_DIR/clean_deploy_manifest.json"
