#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 <output_dir> [summary_json1 ...]" >&2
  exit 2
fi

OUTPUT_DIR="$1"
shift

OUTPUT_JSON="${OUTPUT_JSON:-$OUTPUT_DIR/keepmask_scaling_report.json}"
OUTPUT_MD="${OUTPUT_MD:-$OUTPUT_DIR/keepmask_scaling_report.md}"

if [[ "$#" -eq 0 ]]; then
  mapfile -t SUMMARY_JSONS < <(
    find results/e2e_gap_attribution -path '*/keepmask_result_summary.json' | \
      grep 'keepmask_wholeforward_wrapper_spu_smoke.*partylocal_secret' | \
      sort
  )
else
  SUMMARY_JSONS=("$@")
fi

if [[ "${#SUMMARY_JSONS[@]}" -eq 0 ]]; then
  echo "No keep-mask summary JSONs found." >&2
  exit 2
fi

ARGS=()
for summary_json in "${SUMMARY_JSONS[@]}"; do
  ARGS+=(--summary-json "$summary_json")
done

"${PYTHON_BIN:-python}" tools/transshield_e2e_keepmask_scaling_report.py \
  "${ARGS[@]}" \
  --output-json "$OUTPUT_JSON" \
  --output-md "$OUTPUT_MD"
