#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

VARIANT="${1:-}"
case "$VARIANT" in
  baseline|modified)
    shift
    ;;
  *)
    echo "Usage: $0 [baseline|modified]" >&2
    exit 1
    ;;
esac

exec bash "$SCRIPT_DIR/_run_plaintext_eval_variant.sh" "$VARIANT" "$@"
