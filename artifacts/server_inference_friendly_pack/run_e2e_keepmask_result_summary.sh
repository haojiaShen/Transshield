#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"

RUN_DIR="${1:-${RUN_DIR:-}}"
if [[ -z "$RUN_DIR" ]]; then
  echo "Usage: $0 <run_dir>" >&2
  exit 2
fi

OUTPUT_JSON="${OUTPUT_JSON:-$RUN_DIR/keepmask_result_summary.json}"

"${PYTHON_BIN:-python}" tools/transshield_e2e_keepmask_result_summary.py \
  --run-dir "$RUN_DIR" \
  --output-json "$OUTPUT_JSON"
