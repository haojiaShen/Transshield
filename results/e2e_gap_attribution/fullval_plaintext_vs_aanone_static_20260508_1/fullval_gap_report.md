# Binary Eval Gap Report

- base: `original_plaintext_fix3`
- candidate: `aanone_static_fullval`
- sample_count: `524`
- argmax_accuracy: `74.809160 -> 76.717557`
- threshold_accuracy: `75.381679 -> 91.984733`
- auc: `0.663525 -> 0.967876`
- bce_loss: `0.606093 -> 0.530501`
- score_correlation: `0.2642690056881764`
- same_sign_ratio: `0.5687022900763359`
- affine_boundary_shift_x_at_y0: `0.31084300003147913`
- base_best_threshold_accuracy: `75.381679`
- base_to_candidate_best_match_threshold: `0.26954724234072297`

## Correctness Buckets

- argmax: `{'both_correct': 284, 'base_only_correct': 108, 'candidate_only_correct': 118, 'both_wrong': 14}`
- threshold: `{'both_correct': 376, 'base_only_correct': 19, 'candidate_only_correct': 106, 'both_wrong': 23}`

## Top Abs Score Delta Samples

| index | target | base_score | candidate_score | delta | base_argmax | candidate_argmax | sample_path |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 43 | 0 | 0.573425 | -1.188418 | -1.761843 | 1 | 0 | `/data/wyb/pneumoniamnist_imagefolder_subset/val/0/00177.png` |
| 50 | 0 | 0.702900 | -0.975919 | -1.678818 | 1 | 0 | `/data/wyb/pneumoniamnist_imagefolder_subset/val/0/00199.png` |
| 83 | 0 | 0.549592 | -1.090048 | -1.639640 | 1 | 0 | `/data/wyb/pneumoniamnist_imagefolder_subset/val/0/00326.png` |
| 14 | 0 | 0.379341 | -1.255638 | -1.634979 | 1 | 0 | `/data/wyb/pneumoniamnist_imagefolder_subset/val/0/00046.png` |
| 99 | 0 | 0.627791 | -0.999813 | -1.627605 | 1 | 0 | `/data/wyb/pneumoniamnist_imagefolder_subset/val/0/00388.png` |
| 29 | 0 | 0.497440 | -1.121309 | -1.618749 | 1 | 0 | `/data/wyb/pneumoniamnist_imagefolder_subset/val/0/00130.png` |
| 103 | 0 | 0.509457 | -1.091798 | -1.601255 | 1 | 0 | `/data/wyb/pneumoniamnist_imagefolder_subset/val/0/00406.png` |
| 131 | 0 | 0.505878 | -1.088944 | -1.594822 | 1 | 0 | `/data/wyb/pneumoniamnist_imagefolder_subset/val/0/00505.png` |
| 121 | 0 | 0.478698 | -1.099685 | -1.578383 | 1 | 0 | `/data/wyb/pneumoniamnist_imagefolder_subset/val/0/00478.png` |
| 8 | 0 | 0.307813 | -1.267402 | -1.575215 | 1 | 0 | `/data/wyb/pneumoniamnist_imagefolder_subset/val/0/00033.png` |
| 89 | 0 | 0.411130 | -1.154429 | -1.565559 | 1 | 0 | `/data/wyb/pneumoniamnist_imagefolder_subset/val/0/00349.png` |
| 4 | 0 | 0.421192 | -1.131397 | -1.552589 | 1 | 0 | `/data/wyb/pneumoniamnist_imagefolder_subset/val/0/00022.png` |
