# E2E Policy Probe Report

- label: `e2e_aanone_heldout238_selected_policy_probe_20260508_1`
- sample_count: `10`
- baseline_variant: `exact_uniform_clip0`
- status: `policy_variant_recovers_baseline_wrong_samples`
- recovered_by_any_nonbaseline_variant: `4`
- regressed_by_any_nonbaseline_variant: `4`

## Variant Summary

| variant | accuracy | correct | wrong | mean_abs_score | finite |
|---|---:|---:|---:|---:|---|
| exact_uniform_clip0 | 40.0000 | 4 | 6 | 0.39426 | True |
| exact_uniform_clip3 | 40.0000 | 4 | 6 | 0.626538 | True |
| exact_uniform_clip0_lncmp64 | 40.0000 | 4 | 6 | 0.394687 | True |

## Per-Sample Scores

| source_index | target | image | exact_uniform_clip0 pred/score | exact_uniform_clip3 pred/score | exact_uniform_clip0_lncmp64 pred/score |
|---:|---:|---|---:|---:|---:|
| 121 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00009.png | 0/-0.213379 wrong | 0/-0.700928 wrong | 0/-0.213318 wrong |
| 220 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00445.png | 0/-0.201675 wrong | 0/-0.685089 wrong | 0/-0.200516 wrong |
| 167 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00216.png | 1/0.0102539 ok | 0/-0.728302 wrong | 1/0.0127258 ok |
| 227 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00474.png | 1/0.00762936 ok | 0/-0.662155 wrong | 1/0.00686643 ok |
| 21 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00093.png | 1/0.977539 wrong | 0/-0.558365 ok | 1/0.979019 wrong |
| 49 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00210.png | 1/0.379456 wrong | 0/-0.549957 ok | 1/0.380081 wrong |
| 217 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00434.png | 1/0.383591 ok | 0/-0.623459 wrong | 1/0.383972 ok |
| 196 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00342.png | 1/0.620621 ok | 0/-0.65004 wrong | 1/0.619781 ok |
| 23 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00114.png | 1/0.747269 wrong | 0/-0.606415 ok | 1/0.748718 wrong |
| 54 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00223.png | 1/0.401184 wrong | 0/-0.500671 ok | 1/0.401871 wrong |

## Interpretation

- At least one secure-graph policy variant flips a baseline-wrong selected sample.
- Compare mean_abs_score and raw_score shifts here with public affine/temperature calibration reports; post-reveal calibration can improve BCE/confidence without changing the secret SPU graph.
