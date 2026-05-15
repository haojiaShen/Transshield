# E2E Policy Probe Report

- label: `e2e_aanone_heldout238_even32_publiccalib_clip0_probe_20260508_1`
- sample_count: `32`
- baseline_variant: `exact_uniform_clip0`
- status: `policy_variant_regression_dominates_recovery`
- recovered_by_any_nonbaseline_variant: `1`
- regressed_by_any_nonbaseline_variant: `15`

## Variant Summary

| variant | accuracy | correct | wrong | mean_abs_score | finite |
|---|---:|---:|---:|---:|---|
| exact_uniform_clip0 | 90.6250 | 29 | 3 | 0.547371 | True |
| publiccalib_uniform_clip0 | 46.8750 | 15 | 17 | 2719.39 | True |

## Per-Sample Scores

| source_index | target | image | exact_uniform_clip0 pred/score | publiccalib_uniform_clip0 pred/score |
|---:|---:|---|---:|---:|
| 0 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00007.png | 0/-0.186401 ok | 1/0.393814 wrong |
| 8 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00037.png | 0/-0.472427 ok | 1/0.39798 wrong |
| 16 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00072.png | 0/-0.643692 ok | 1/0.397064 wrong |
| 24 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00124.png | 0/-0.306763 ok | 1/0.385254 wrong |
| 31 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00155.png | 0/-0.333694 ok | 1/0.390259 wrong |
| 39 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00185.png | 0/-0.514221 ok | 1/0.393249 wrong |
| 47 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00203.png | 0/-0.0168457 ok | 1/0.398209 wrong |
| 55 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00226.png | 0/-0.32193 ok | 1/0.388748 wrong |
| 63 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00281.png | 0/-0.488342 ok | 1/0.391312 wrong |
| 71 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00311.png | 1/0.266327 wrong | 1/0.390335 wrong |
| 79 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00353.png | 0/-0.272507 ok | 1/0.397125 wrong |
| 87 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00388.png | 0/-0.641418 ok | 1/0.397278 wrong |
| 94 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00416.png | 0/-0.310547 ok | 1/0.440094 wrong |
| 102 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00455.png | 0/-0.536789 ok | 1/0.389816 wrong |
| 110 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00484.png | 0/-0.318588 ok | 1/0.387665 wrong |
| 118 | 0 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00519.png | 0/-0.468216 ok | 1/0.396637 wrong |
| 119 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00001.png | 0/-0.130142 wrong | 0/-87006.5 wrong |
| 127 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00039.png | 1/0.685074 ok | 1/0.395554 ok |
| 135 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00073.png | 1/0.295197 ok | 1/0.395401 ok |
| 143 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00105.png | 1/1.1757 ok | 1/0.391327 ok |
| 150 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00134.png | 1/0.387848 ok | 1/0.391815 ok |
| 158 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00167.png | 1/0.718048 ok | 1/0.399673 ok |
| 166 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00211.png | 1/0.982468 ok | 1/0.620987 ok |
| 174 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00247.png | 1/1.24117 ok | 1/0.391083 ok |
| 182 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00280.png | 1/1.45564 ok | 1/0.332977 ok |
| 190 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00317.png | 1/0.551514 ok | 1/0.555862 ok |
| 198 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00350.png | 1/0.245941 ok | 1/0.517685 ok |
| 206 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00385.png | 0/-0.289383 wrong | 1/0.39827 ok |
| 213 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00414.png | 1/0.680161 ok | 1/0.416748 ok |
| 221 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00449.png | 1/0.84758 ok | 1/0.392532 ok |
| 229 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00486.png | 1/1.25841 ok | 1/1.58804 ok |
| 237 | 1 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00521.png | 1/0.472885 ok | 1/0.390076 ok |

## Interpretation

- At least one secure-graph policy variant flips a baseline-wrong selected sample.
- Compare mean_abs_score and raw_score shifts here with public affine/temperature calibration reports; post-reveal calibration can improve BCE/confidence without changing the secret SPU graph.
