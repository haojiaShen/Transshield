# E2E Policy Probe Report

- label: `e2e_aanone_heldout238_mixed_window_exact_probe_20260508_1_anchored`
- sample_count: `12`
- baseline_variant: `source_heldout238_exact_uniform_clip0`
- status: `policy_variant_recovers_baseline_wrong_samples`
- recovered_by_any_nonbaseline_variant: `2`
- regressed_by_any_nonbaseline_variant: `0`

## Variant Summary

| variant | accuracy | correct | wrong | mean_abs_score | finite |
|---|---:|---:|---:|---:|---|
| source_heldout238_exact_uniform_clip0 | 50.0000 | 6 | 6 | 0.807346 | True |
| rerun_window_exact_uniform_clip0 | 66.6667 | 8 | 4 | 0.461951 | True |

## Per-Sample Scores

| source_index | target | image | source_heldout238_exact_uniform_clip0 pred/score | rerun_window_exact_uniform_clip0 pred/score |
|---:|---:|---|---:|---:|
| 121 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00009.png | 0/-0.692276 wrong | 0/-0.211227 wrong |
| 220 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00445.png | 0/-0.64122 wrong | 0/-0.200714 wrong |
| 167 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00216.png | 0/-0.553375 wrong | 1/0.0103759 ok |
| 227 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00474.png | 0/-0.48085 wrong | 1/0.00999448 ok |
| 21 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00093.png | 1/0.32608 wrong | 1/0.978104 wrong |
| 49 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00210.png | 1/0.26976 wrong | 1/0.379517 wrong |
| 7 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00033.png | 0/-0.910385 ok | 0/-0.439957 ok |
| 12 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00046.png | 0/-0.89682 ok | 0/-0.440216 ok |
| 105 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00471.png | 0/-0.861267 ok | 0/-0.419022 ok |
| 95 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00420.png | 0/-0.83139 ok | 0/-0.308182 ok |
| 192 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00323.png | 1/1.65446 ok | 1/1.06972 ok |
| 140 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00094.png | 1/1.57027 ok | 1/1.07639 ok |

## Interpretation

- At least one secure-graph policy variant flips a baseline-wrong selected sample.
- Compare mean_abs_score and raw_score shifts here with public affine/temperature calibration reports; post-reveal calibration can improve BCE/confidence without changing the secret SPU graph.
