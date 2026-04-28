#!/usr/bin/env bash
set -euo pipefail

# Training command
bash artifacts/server_inference_friendly_pack/run_margin_aware_pruning_ablation.sh

# Threshold search command
tools/transshield_binary_threshold_search.py search --checkpoint /data/wyb/Transshield_final/artifacts/train_runs/margin_formal_hparams_soft_stage2_20260417_231946_w3_t1em4/checkpoint-best.pth --data-path /data/wyb/pneumoniamnist_imagefolder_subset/val

