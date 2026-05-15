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

SUMMARY_JSON="${SUMMARY_JSON:-$RUN_DIR/keepmask_result_summary.json}"
OUTPUT_MD="${OUTPUT_MD:-$RUN_DIR/keepmask_result_snippet.md}"

"${PYTHON_BIN:-python}" tools/transshield_e2e_keepmask_doc_snippet.py \
  --summary-json "$SUMMARY_JSON" \
  --output-md "$OUTPUT_MD"
