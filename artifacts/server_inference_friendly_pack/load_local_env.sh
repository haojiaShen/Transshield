#!/usr/bin/env bash
set -euo pipefail

if [[ "${TRANSSHIELD_LOCAL_ENV_LOADED:-0}" == "1" ]]; then
  return 0 2>/dev/null || exit 0
fi
export TRANSSHIELD_LOCAL_ENV_LOADED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export REPO_ROOT="${REPO_ROOT:-$DEFAULT_REPO_ROOT}"

TEMPLATE_ENV="$SCRIPT_DIR/final_compare_env.template.sh"
LOCAL_ENV="$SCRIPT_DIR/final_compare_env.local.sh"

if [[ "${TRANSSHIELD_USE_LOCAL_ENV:-0}" == "1" && -f "$LOCAL_ENV" ]]; then
  # shellcheck source=/dev/null
  source "$LOCAL_ENV"
fi

if [[ -f "$TEMPLATE_ENV" ]]; then
  # shellcheck source=/dev/null
  source "$TEMPLATE_ENV"
fi
