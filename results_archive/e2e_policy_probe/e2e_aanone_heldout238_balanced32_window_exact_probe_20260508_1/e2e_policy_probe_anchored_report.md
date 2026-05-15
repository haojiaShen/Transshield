# E2E Policy Probe Report

- label: `e2e_aanone_heldout238_balanced32_window_exact_probe_20260508_1_anchored`
- sample_count: `32`
- baseline_variant: `source_heldout238_exact_uniform_clip0`
- status: `policy_variant_recovers_baseline_wrong_samples`
- recovered_by_any_nonbaseline_variant: `10`
- regressed_by_any_nonbaseline_variant: `0`

## Variant Summary

| variant | accuracy | correct | wrong | mean_abs_score | finite |
|---|---:|---:|---:|---:|---|
| source_heldout238_exact_uniform_clip0 | 50.0000 | 16 | 16 | 0.659814 | True |
| rerun_window_exact_uniform_clip0 | 81.2500 | 26 | 6 | 0.432369 | True |

## Per-Sample Scores

| source_index | target | image | source_heldout238_exact_uniform_clip0 pred/score | rerun_window_exact_uniform_clip0 pred/score |
|---:|---:|---|---:|---:|
| 21 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00093.png | 1/0.32608 wrong | 1/0.978683 wrong |
| 49 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00210.png | 1/0.26976 wrong | 1/0.380096 wrong |
| 71 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00311.png | 1/0.263 wrong | 1/0.53447 wrong |
| 54 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00223.png | 1/0.214874 wrong | 1/0.402283 wrong |
| 121 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00009.png | 0/-0.692276 wrong | 0/-0.213486 wrong |
| 220 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00445.png | 0/-0.64122 wrong | 0/-0.200012 wrong |
| 167 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00216.png | 0/-0.553375 wrong | 1/0.00985715 ok |
| 227 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00474.png | 0/-0.48085 wrong | 1/0.00859067 ok |
| 206 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00385.png | 0/-0.287979 wrong | 1/0.0457916 ok |
| 168 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00222.png | 0/-0.256928 wrong | 1/0.216125 ok |
| 233 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00506.png | 0/-0.247223 wrong | 1/0.198975 ok |
| 223 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00460.png | 0/-0.218674 wrong | 1/0.275146 ok |
| 129 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00050.png | 0/-0.183716 wrong | 1/0.335541 ok |
| 119 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00001.png | 0/-0.13411 wrong | 1/0.277206 ok |
| 216 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00430.png | 0/-0.0655365 wrong | 1/0.373886 ok |
| 196 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00342.png | 0/-0.0606842 wrong | 1/0.620026 ok |
| 7 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00033.png | 0/-0.910385 ok | 0/-0.441467 ok |
| 12 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00046.png | 0/-0.89682 ok | 0/-0.441269 ok |
| 105 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00471.png | 0/-0.861267 ok | 0/-0.4189 ok |
| 95 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00420.png | 0/-0.83139 ok | 0/-0.306824 ok |
| 38 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00177.png | 0/-0.831039 ok | 0/-0.365921 ok |
| 93 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00411.png | 0/-0.827835 ok | 0/-0.317184 ok |
| 43 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00198.png | 0/-0.818237 ok | 0/-0.310898 ok |
| 4 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00028.png | 0/-0.80217 ok | 0/-0.237167 ok |
| 53 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00220.png | 0/-0.801773 ok | 0/-0.298157 ok |
| 14 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00052.png | 0/-0.799713 ok | 0/-0.323639 ok |
| 3 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00022.png | 0/-0.773209 ok | 0/-0.313553 ok |
| 81 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00358.png | 0/-0.768982 ok | 0/-0.272781 ok |
| 192 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00323.png | 1/1.65446 ok | 1/1.07068 ok |
| 140 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00094.png | 1/1.57027 ok | 1/1.07739 ok |
| 231 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00495.png | 1/1.53764 ok | 1/1.29625 ok |
| 194 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00331.png | 1/1.53256 ok | 1/1.27356 ok |

## Interpretation

- At least one secure-graph policy variant flips a baseline-wrong selected sample.
- Compare mean_abs_score and raw_score shifts here with public affine/temperature calibration reports; post-reveal calibration can improve BCE/confidence without changing the secret SPU graph.
