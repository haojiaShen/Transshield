#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

RUNTIME="${1:-}"
case "$RUNTIME" in
  cpu|spu)
    shift
    ;;
  *)
    echo "Usage: $0 [cpu|spu]" >&2
    exit 1
    ;;
esac

exec bash "$SCRIPT_DIR/_run_secure_pipeline_by_runtime.sh" "$RUNTIME" "$@"
