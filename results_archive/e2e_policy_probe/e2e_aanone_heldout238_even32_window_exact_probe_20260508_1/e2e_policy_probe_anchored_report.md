# E2E Policy Probe Report

- label: `e2e_aanone_heldout238_even32_window_exact_probe_20260508_1_anchored`
- sample_count: `32`
- baseline_variant: `source_heldout238_exact_uniform_clip0`
- status: `no_policy_variant_recovery_on_selected_samples`
- recovered_by_any_nonbaseline_variant: `0`
- regressed_by_any_nonbaseline_variant: `0`

## Variant Summary

| variant | accuracy | correct | wrong | mean_abs_score | finite |
|---|---:|---:|---:|---:|---|
| source_heldout238_exact_uniform_clip0 | 90.6250 | 29 | 3 | 0.546865 | True |
| rerun_window_exact_uniform_clip0 | 90.6250 | 29 | 3 | 0.547298 | True |

## Per-Sample Scores

| source_index | target | image | source_heldout238_exact_uniform_clip0 pred/score | rerun_window_exact_uniform_clip0 pred/score |
|---:|---:|---|---:|---:|
| 0 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00007.png | 0/-0.184113 ok | 0/-0.185043 ok |
| 8 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00037.png | 0/-0.473007 ok | 0/-0.473877 ok |
| 16 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00072.png | 0/-0.646835 ok | 0/-0.646301 ok |
| 24 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00124.png | 0/-0.30632 ok | 0/-0.307022 ok |
| 31 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00155.png | 0/-0.331345 ok | 0/-0.330978 ok |
| 39 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00185.png | 0/-0.514832 ok | 0/-0.514496 ok |
| 47 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00203.png | 0/-0.0175019 ok | 0/-0.0163575 ok |
| 55 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00226.png | 0/-0.323242 ok | 0/-0.323151 ok |
| 63 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00281.png | 0/-0.487564 ok | 0/-0.489487 ok |
| 71 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00311.png | 1/0.263 wrong | 1/0.263916 wrong |
| 79 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00353.png | 0/-0.272736 ok | 0/-0.272339 ok |
| 87 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00388.png | 0/-0.640457 ok | 0/-0.639877 ok |
| 94 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00416.png | 0/-0.310196 ok | 0/-0.310501 ok |
| 102 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00455.png | 0/-0.535217 ok | 0/-0.535599 ok |
| 110 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00484.png | 0/-0.317886 ok | 0/-0.318481 ok |
| 118 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00519.png | 0/-0.467178 ok | 0/-0.467621 ok |
| 119 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00001.png | 0/-0.13411 wrong | 0/-0.131561 wrong |
| 127 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00039.png | 1/0.685059 ok | 1/0.686966 ok |
| 135 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00073.png | 1/0.29303 ok | 1/0.293579 ok |
| 143 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00105.png | 1/1.17683 ok | 1/1.17776 ok |
| 150 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00134.png | 1/0.386398 ok | 1/0.38765 ok |
| 158 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00167.png | 1/0.714859 ok | 1/0.716812 ok |
| 166 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00211.png | 1/0.980453 ok | 1/0.981644 ok |
| 174 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00247.png | 1/1.24106 ok | 1/1.24326 ok |
| 182 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00280.png | 1/1.45564 ok | 1/1.45567 ok |
| 190 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00317.png | 1/0.552582 ok | 1/0.552536 ok |
| 198 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00350.png | 1/0.24556 ok | 1/0.245346 ok |
| 206 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00385.png | 0/-0.287979 wrong | 0/-0.288788 wrong |
| 213 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00414.png | 1/0.678238 ok | 1/0.679276 ok |
| 221 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00449.png | 1/0.846786 ok | 1/0.846329 ok |
| 229 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00486.png | 1/1.25659 ok | 1/1.25893 ok |
| 237 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00521.png | 1/0.473068 ok | 1/0.472382 ok |

## Interpretation

- The tested secure-graph policy variants did not flip the selected baseline-wrong samples.
- Compare mean_abs_score and raw_score shifts here with public affine/temperature calibration reports; post-reveal calibration can improve BCE/confidence without changing the secret SPU graph.
