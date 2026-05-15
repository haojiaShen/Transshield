# E2E Policy Probe Report

- label: `e2e_aanone_heldout238_even32_clip3_regression_probe_20260508_1_anchored`
- sample_count: `32`
- baseline_variant: `source_heldout238_exact_uniform_clip0`
- status: `policy_variant_regression_dominates_recovery`
- recovered_by_any_nonbaseline_variant: `1`
- regressed_by_any_nonbaseline_variant: `14`

## Variant Summary

| variant | accuracy | correct | wrong | mean_abs_score | finite |
|---|---:|---:|---:|---:|---|
| source_heldout238_exact_uniform_clip0 | 90.6250 | 29 | 3 | 0.546865 | True |
| rerun_window_exact_uniform_clip3 | 50.0000 | 16 | 16 | 0.576767 | True |

## Per-Sample Scores

| source_index | target | image | source_heldout238_exact_uniform_clip0 pred/score | rerun_window_exact_uniform_clip3 pred/score |
|---:|---:|---|---:|---:|
| 0 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00007.png | 0/-0.184113 ok | 0/-0.630524 ok |
| 8 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00037.png | 0/-0.473007 ok | 0/-0.650528 ok |
| 16 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00072.png | 0/-0.646835 ok | 0/-0.707336 ok |
| 24 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00124.png | 0/-0.30632 ok | 0/-0.606522 ok |
| 31 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00155.png | 0/-0.331345 ok | 0/-0.688583 ok |
| 39 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00185.png | 0/-0.514832 ok | 0/-0.688141 ok |
| 47 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00203.png | 0/-0.0175019 ok | 0/-0.591309 ok |
| 55 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00226.png | 0/-0.323242 ok | 0/-0.682281 ok |
| 63 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00281.png | 0/-0.487564 ok | 0/-0.699051 ok |
| 71 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00311.png | 1/0.263 wrong | 0/-0.568802 ok |
| 79 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00353.png | 0/-0.272736 ok | 0/-0.645584 ok |
| 87 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00388.png | 0/-0.640457 ok | 0/-0.709656 ok |
| 94 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00416.png | 0/-0.310196 ok | 0/-0.709686 ok |
| 102 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00455.png | 0/-0.535217 ok | 0/-0.704605 ok |
| 110 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00484.png | 0/-0.317886 ok | 0/-0.603897 ok |
| 118 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00519.png | 0/-0.467178 ok | 0/-0.68132 ok |
| 119 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00001.png | 0/-0.13411 wrong | 0/-0.559982 wrong |
| 127 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00039.png | 1/0.685059 ok | 0/-0.413666 wrong |
| 135 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00073.png | 1/0.29303 ok | 0/-0.575424 wrong |
| 143 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00105.png | 1/1.17683 ok | 0/-0.377625 wrong |
| 150 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00134.png | 1/0.386398 ok | 0/-0.545197 wrong |
| 158 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00167.png | 1/0.714859 ok | 0/-0.548141 wrong |
| 166 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00211.png | 1/0.980453 ok | 0/-0.516296 wrong |
| 174 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00247.png | 1/1.24106 ok | 0/-0.389435 wrong |
| 182 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00280.png | 1/1.45564 ok | 0/-0.0492554 wrong |
| 190 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00317.png | 1/0.552582 ok | 0/-0.58696 wrong |
| 198 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00350.png | 1/0.24556 ok | 0/-0.626022 wrong |
| 206 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00385.png | 0/-0.287979 wrong | 0/-0.634964 wrong |
| 213 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00414.png | 1/0.678238 ok | 0/-0.561081 wrong |
| 221 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00449.png | 1/0.846786 ok | 0/-0.472137 wrong |
| 229 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00486.png | 1/1.25659 ok | 0/-0.486755 wrong |
| 237 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00521.png | 1/0.473068 ok | 0/-0.545792 wrong |

## Interpretation

- At least one secure-graph policy variant flips a baseline-wrong selected sample.
- Compare mean_abs_score and raw_score shifts here with public affine/temperature calibration reports; post-reveal calibration can improve BCE/confidence without changing the secret SPU graph.
