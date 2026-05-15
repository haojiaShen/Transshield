#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"

if [[ "$#" -lt 2 ]]; then
  echo "Usage: $0 <output_md> <summary_json1> [summary_json2 ...]" >&2
  exit 2
fi

OUTPUT_MD="$1"
shift

ARGS=()
for summary_json in "$@"; do
  ARGS+=(--summary-json "$summary_json")
done

"${PYTHON_BIN:-python}" tools/transshield_e2e_keepmask_doc_block.py \
  "${ARGS[@]}" \
  --output-md "$OUTPUT_MD"
