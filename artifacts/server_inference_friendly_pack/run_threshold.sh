#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODE="${1:-}"
case "$MODE" in
  search|eval)
    shift
    ;;
  *)
    echo "Usage: $0 [search|eval]" >&2
    exit 1
    ;;
esac

exec bash "$SCRIPT_DIR/_run_threshold_step.sh" "$MODE" "$@"
