#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi
MODE="${1:-debug80}"
case "$MODE" in
  recipe|print-env|debug80|epoch1)
    shift || true
    ;;
  *)
    echo "Usage: $0 [recipe|print-env|debug80|epoch1]" >&2
    exit 1
    ;;
esac

PROTOCOL_AWARE_PROFILE="${PROTOCOL_AWARE_PROFILE:-conservative}"
STAGE_COST_RISK_JSON="${STAGE_COST_RISK_JSON:-$REPO_ROOT/results/stage_cost_risk_model/stage_cost_risk_20260505_clean/stage_cost_risk_report.json}"
FALLBACK_RECIPE_JSON="${PROTOCOL_AWARE_BASE_RECIPE_JSON:-$REPO_ROOT/results/protocol_aware_pruning_objective/protocol_aware_recipe_20260505_clean/protocol_aware_pruning_recipe.json}"
FALLBACK_RECIPE_MD="${PROTOCOL_AWARE_BASE_RECIPE_MD:-$REPO_ROOT/results/protocol_aware_pruning_objective/protocol_aware_recipe_20260505_clean/protocol_aware_pruning_recipe.md}"
COMPARE_PLACEHOLDER_RUN_NAME="transshield_comp_full_compare_YYYYMMDD"

if [[ "$MODE" == "recipe" ]]; then
  RECIPE_NAME="${RECIPE_NAME:-protocol_aware_recipe_$(date +%Y%m%d_%H%M%S)}"
  OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/protocol_aware_pruning_objective/$RECIPE_NAME}"
else
  if [[ -z "${RUN_NAME:-}" || "${RUN_NAME:-}" == "$COMPARE_PLACEHOLDER_RUN_NAME" ]]; then
    RUN_NAME="protocol_aware_pruning_${MODE}_${PROTOCOL_AWARE_PROFILE}_$(date +%Y%m%d_%H%M%S)"
  fi
  export RUN_NAME
  OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/protocol_aware_pruning_objective/$RUN_NAME}"
fi

RECIPE_JSON="${RECIPE_JSON:-$OUTPUT_DIR/protocol_aware_pruning_recipe.json}"
RECIPE_MD="${RECIPE_MD:-$OUTPUT_DIR/protocol_aware_pruning_recipe.md}"

echo "[protocol-aware-pruning] stage_cost_risk_json=$STAGE_COST_RISK_JSON"
echo "[protocol-aware-pruning] output_dir=$OUTPUT_DIR"
echo "[protocol-aware-pruning] recipe_json=$RECIPE_JSON"
echo "[protocol-aware-pruning] recipe_md=$RECIPE_MD"
mkdir -p "$OUTPUT_DIR"

if [[ -f "$STAGE_COST_RISK_JSON" ]]; then
  echo "[protocol-aware-pruning] stage_cost_risk_json=$STAGE_COST_RISK_JSON"
  "$PYTHON_BIN" tools/transshield_protocol_aware_pruning_recipe.py \
    --stage-cost-risk-json "$STAGE_COST_RISK_JSON" \
    --output-json "$RECIPE_JSON" \
    --output-md "$RECIPE_MD"
elif [[ -f "$FALLBACK_RECIPE_JSON" ]]; then
  echo "[protocol-aware-pruning] stage_cost_risk_json missing, reuse shipped fallback recipe" >&2
  cp "$FALLBACK_RECIPE_JSON" "$RECIPE_JSON"
  if [[ -f "$FALLBACK_RECIPE_MD" ]]; then
    cp "$FALLBACK_RECIPE_MD" "$RECIPE_MD"
  fi
else
  echo "[protocol-aware-pruning] missing STAGE_COST_RISK_JSON: $STAGE_COST_RISK_JSON" >&2
  echo "[protocol-aware-pruning] missing fallback recipe: $FALLBACK_RECIPE_JSON" >&2
  exit 2
fi

if [[ "$MODE" == "recipe" ]]; then
  echo "[protocol-aware-pruning] recipe only mode complete"
  exit 0
fi

SECURE_STATIC_DEPTH="${SECURE_STATIC_DEPTH:-12}"
SECURE_STATIC_SKIP_PRUNING="${SECURE_STATIC_SKIP_PRUNING:-false}"
USE_MASK_PRUNING="${USE_MASK_PRUNING:-false}"
DEFAULT_TRAIN_RUN_DIR="$REPO_ROOT/artifacts/train_runs/$RUN_NAME"
if [[ -z "${RUN_DIR:-}" || "$RUN_DIR" == "$REPO_ROOT/artifacts/server_runs/"* ]]; then
  RUN_DIR="$DEFAULT_TRAIN_RUN_DIR"
fi
export SECURE_STATIC_DEPTH
export SECURE_STATIC_SKIP_PRUNING
export USE_MASK_PRUNING
export RUN_DIR

PROFILE_EXPORTS="$("$PYTHON_BIN" - "$RECIPE_JSON" "$PROTOCOL_AWARE_PROFILE" <<'PY'
import json
import sys

recipe = json.load(open(sys.argv[1], 'r', encoding='utf-8'))
profile_name = sys.argv[2]
profiles = recipe.get('profiles') or {}
if profile_name not in profiles:
    raise SystemExit(f'unknown profile: {profile_name}')
profile = profiles[profile_name]
print(f"PRUNING_MARGIN_WEIGHT={profile['pruning_margin_weight']}")
print(f"PRUNING_MARGIN_TARGET={profile['pruning_margin_target']}")
print(f"PRUNING_MARGIN_MODE={profile['pruning_margin_mode']}")
print(f"PRUNING_MARGIN_STAGE_WEIGHTS={profile['pruning_margin_stage_weights_csv']}")
print(f"PRUNING_MARGIN_START_EPOCH={profile['pruning_margin_start_epoch']}")
PY
)"

FORCE_RECIPE_PRUNING_MARGIN="${PROTOCOL_AWARE_FORCE_RECIPE_PRUNING_MARGIN:-0}"
case "$FORCE_RECIPE_PRUNING_MARGIN" in
  1|true|TRUE|yes|YES|on|ON)
    FORCE_RECIPE_PRUNING_MARGIN=1
    ;;
  *)
    FORCE_RECIPE_PRUNING_MARGIN=0
    ;;
esac

while IFS='=' read -r key value; do
  [[ -z "$key" ]] && continue
  current_value=""
  has_current_value=0
  if [[ -n "${!key+x}" ]]; then
    current_value="${!key}"
    has_current_value=1
  fi
  if [[ "$FORCE_RECIPE_PRUNING_MARGIN" == "1" ]]; then
    if [[ "$has_current_value" == "1" && "$current_value" != "$value" ]]; then
      echo "[protocol-aware-pruning] override $key: $current_value -> $value" >&2
    fi
    export "$key=$value"
  elif [[ "$has_current_value" == "0" ]]; then
    export "$key=$value"
  elif [[ "$current_value" != "$value" ]]; then
    echo "[protocol-aware-pruning] keep existing $key=$current_value (recipe suggests $value)" >&2
  fi
done <<< "$PROFILE_EXPORTS"

echo "[protocol-aware-pruning] profile=$PROTOCOL_AWARE_PROFILE"
echo "[protocol-aware-pruning] force_recipe_pruning_margin=$FORCE_RECIPE_PRUNING_MARGIN"
echo "[protocol-aware-pruning] pruning_margin_weight=$PRUNING_MARGIN_WEIGHT"
echo "[protocol-aware-pruning] pruning_margin_target=$PRUNING_MARGIN_TARGET"
echo "[protocol-aware-pruning] pruning_margin_mode=$PRUNING_MARGIN_MODE"
echo "[protocol-aware-pruning] pruning_margin_stage_weights=$PRUNING_MARGIN_STAGE_WEIGHTS"
echo "[protocol-aware-pruning] pruning_margin_start_epoch=$PRUNING_MARGIN_START_EPOCH"
echo "[protocol-aware-pruning] secure_static_depth=$SECURE_STATIC_DEPTH"
echo "[protocol-aware-pruning] secure_static_skip_pruning=$SECURE_STATIC_SKIP_PRUNING"
echo "[protocol-aware-pruning] run_dir=$RUN_DIR"

if [[ "$MODE" == "print-env" ]]; then
  echo "[protocol-aware-pruning] print-env mode complete"
  exit 0
fi

bash "$SCRIPT_DIR/run_secure_static_distill_train.sh" "$MODE" "$@"
