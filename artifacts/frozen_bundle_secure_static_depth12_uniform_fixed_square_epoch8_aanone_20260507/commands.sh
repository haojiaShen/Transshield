#!/usr/bin/env bash
set -euo pipefail

# Training command
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh train-candidate # PAIR_NAME=accprof_epoch8_20260507_default_vs_aanone_1 CANDIDATE_AA=none

# Threshold search command
/data/wyb/conda_envs/transshield/bin/python tools/transshield_binary_threshold_search.py search --checkpoint /data/wyb/Transshield_final/artifacts/train_runs/secure_static_accprof_epoch8_20260507_aanone_1/checkpoint-best.pth --data-path /data/wyb/pneumoniamnist_imagefolder_subset/val --device cuda --batch-size 32 --num-workers 4 --output-json /data/wyb/Transshield_final/artifacts/train_runs/secure_static_accprof_epoch8_20260507_aanone_1/threshold_best.json

# Thresholded eval command
/data/wyb/conda_envs/transshield/bin/python tools/transshield_binary_threshold_search.py eval --checkpoint /data/wyb/Transshield_final/artifacts/train_runs/secure_static_accprof_epoch8_20260507_aanone_1/checkpoint-best.pth --threshold-json /data/wyb/Transshield_final/artifacts/train_runs/secure_static_accprof_epoch8_20260507_aanone_1/threshold_best.json --data-path /data/wyb/pneumoniamnist_imagefolder_subset/val --device cuda --batch-size 32 --num-workers 4 --output-json /data/wyb/Transshield_final/artifacts/train_runs/secure_static_accprof_epoch8_20260507_aanone_1/threshold_eval.json

