#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

cd "$REPO_ROOT"

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import torch  # noqa: F401
import torchvision  # noqa: F401
from PIL import Image  # noqa: F401
PY
then
  echo "[cnn-baseline] python env check failed for PYTHON_BIN=$PYTHON_BIN" >&2
  echo "[cnn-baseline] expected packages: torch, torchvision, pillow" >&2
  exit 2
fi

export TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${DATA_ROOT:-/data/wyb/pneumoniamnist_imagefolder_subset}/train}"
export VAL_DATA_PATH="${VAL_DATA_PATH:-${DATA_ROOT:-/data/wyb/pneumoniamnist_imagefolder_subset}/val}"
if [[ ! -d "$TRAIN_DATA_PATH" ]]; then
  echo "[cnn-baseline] missing TRAIN_DATA_PATH: $TRAIN_DATA_PATH" >&2
  exit 2
fi
if [[ ! -d "$VAL_DATA_PATH" ]]; then
  echo "[cnn-baseline] missing VAL_DATA_PATH: $VAL_DATA_PATH" >&2
  exit 2
fi

export TRAIN_RUN_ROOT="${TRAIN_RUN_ROOT:-$REPO_ROOT/artifacts/train_runs}"
mkdir -p "$TRAIN_RUN_ROOT"

run_cnn="${RUN_NAME:-cnn_plaintext_resnet18_$(date +%Y%m%d_%H%M%S)}"
export RUN_NAME="$run_cnn"
export RUN_DIR="$TRAIN_RUN_ROOT/$RUN_NAME"
mkdir -p "$RUN_DIR"

export CNN_ARCH="${CNN_ARCH:-resnet18}"
export EPOCHS="${EPOCHS:-8}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export LR="${LR:-1e-4}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
export DEVICE="${DEVICE:-cuda}"
export SEED="${SEED:-0}"
export CNN_CLASS_WEIGHT_MODE="${CNN_CLASS_WEIGHT_MODE:-inverse_freq}"
export CNN_PRETRAINED="${CNN_PRETRAINED:-true}"
export CNN_FREEZE_BACKBONE="${CNN_FREEZE_BACKBONE:-false}"
export CNN_AMP="${CNN_AMP:-false}"

mkdir -p "${TMPDIR:-/data/wyb/tmp}"
export TMPDIR="${TMPDIR:-/data/wyb/tmp}"
export TMP="${TMP:-$TMPDIR}"
export TEMP="${TEMP:-$TMPDIR}"

ARGS=(
  tools/train_cnn_baseline.py
  --train-data "$TRAIN_DATA_PATH"
  --val-data "$VAL_DATA_PATH"
  --output-dir "$RUN_DIR"
  --arch "$CNN_ARCH"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --lr "$LR"
  --weight-decay "$WEIGHT_DECAY"
  --device "$DEVICE"
  --seed "$SEED"
  --class-weight-mode "$CNN_CLASS_WEIGHT_MODE"
)

if [[ "$CNN_PRETRAINED" == "true" ]]; then
  ARGS+=(--pretrained)
fi
if [[ "$CNN_FREEZE_BACKBONE" == "true" ]]; then
  ARGS+=(--freeze-backbone)
fi
if [[ "$CNN_AMP" == "true" ]]; then
  ARGS+=(--amp)
fi

printf '%q ' "$PYTHON_BIN" "${ARGS[@]}" > "$RUN_DIR/command.sh"
printf '\n' >> "$RUN_DIR/command.sh"
chmod +x "$RUN_DIR/command.sh"

LOG="$RUN_DIR/train_stdout.log"
echo "run_name=$RUN_NAME"
echo "run_dir=$RUN_DIR"
echo "log=$LOG"
echo "summary_json=$RUN_DIR/cnn_baseline_summary.json"

"$PYTHON_BIN" "${ARGS[@]}" 2>&1 | tee "$LOG"
