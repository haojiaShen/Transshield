# E2E Calibration Sample Report

- label: `e2e_heldout238_spuaware_sample_diagnosis_20260508_1`
- sample_count: `238`

| calibration | accuracy | wrong | low margin <0.25 | low margin <0.50 |
|---|---:|---:|---:|---:|
| static_bias | 90.7563 | 22 | 65 | 122 |
| spuaware_bias | 92.437 | 18 | 35 | 100 |
| e2e_smoke32_affine | 92.437 | 18 | 3 | 9 |
| e2e_smoke32_temperature | 92.437 | 18 | 1 | 8 |

## Category Counts

- `static_wrong_spuaware_correct`: `10`
- `static_correct_spuaware_wrong`: `6`
- `spuaware_wrong`: `18`

## static_wrong_spuaware_correct

| index | target | static | spuaware | affine | image |
|---:|---:|---:|---:|---:|---|
| 2 | 0 | 0.0143484 | -0.21817 | -1.1422 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00014.png |
| 92 | 0 | 0.044927 | -0.187592 | -0.973125 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00409.png |
| 0 | 0 | 0.048406 | -0.184113 | -0.953888 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00007.png |
| 83 | 0 | 0.0768789 | -0.15564 | -0.796454 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00365.png |
| 62 | 0 | 0.0784048 | -0.154114 | -0.788017 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00254.png |
| 40 | 0 | 0.117696 | -0.114822 | -0.570765 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00190.png |
| 50 | 0 | 0.163244 | -0.0692749 | -0.318922 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00215.png |
| 37 | 0 | 0.164388 | -0.0681305 | -0.312594 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00174.png |

## spuaware_wrong_low_margin

| index | target | static | spuaware | affine | image |
|---:|---:|---:|---:|---:|---|
| 217 | 1 | 0.181463 | -0.0510559 | -0.218184 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00434.png |
| 196 | 1 | 0.171834 | -0.0606842 | -0.271421 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00342.png |
| 216 | 1 | 0.166982 | -0.0655365 | -0.298251 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00430.png |
| 119 | 1 | 0.0984091 | -0.13411 | -0.677408 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00001.png |
| 23 | 0 | 0.376516 | 0.143997 | 0.860314 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00114.png |
| 129 | 1 | 0.0488027 | -0.183716 | -0.951694 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00050.png |
| 54 | 0 | 0.447393 | 0.214874 | 1.25221 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00223.png |
| 223 | 1 | 0.0138448 | -0.218674 | -1.14499 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00460.png |

## spuaware_wrong_high_margin

| index | target | static | spuaware | affine | image |
|---:|---:|---:|---:|---:|---|
| 121 | 1 | -0.459757 | -0.692276 | -3.76365 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00009.png |
| 220 | 1 | -0.408702 | -0.64122 | -3.48135 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00445.png |
| 167 | 1 | -0.320857 | -0.553375 | -2.99564 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00216.png |
| 227 | 1 | -0.248332 | -0.48085 | -2.59463 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00474.png |
| 21 | 0 | 0.558599 | 0.32608 | 1.8671 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00093.png |
| 206 | 1 | -0.0554606 | -0.287979 | -1.52819 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00385.png |
| 49 | 0 | 0.502279 | 0.26976 | 1.55569 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00210.png |
| 71 | 0 | 0.495519 | 0.263 | 1.51831 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00311.png |

## largest_affine_margin_gain

| index | target | static | spuaware | affine | image |
|---:|---:|---:|---:|---:|---|
| 192 | 1 | 1.88698 | 1.65446 | 9.21208 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00323.png |
| 140 | 1 | 1.80279 | 1.57027 | 8.74652 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00094.png |
| 231 | 1 | 1.77016 | 1.53764 | 8.56614 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00495.png |
| 194 | 1 | 1.76508 | 1.53256 | 8.53805 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00331.png |
| 163 | 1 | 1.76146 | 1.52895 | 8.51805 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00194.png |
| 177 | 1 | 1.69426 | 1.46175 | 8.14649 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00263.png |
| 182 | 1 | 1.68816 | 1.45564 | 8.11274 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00280.png |
| 155 | 1 | 1.68451 | 1.452 | 8.09257 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00153.png |

## Interpretation

- Rows in static_wrong_spuaware_correct explain why SPU-aware bias is now the accuracy-first default: they are boundary cases recovered by the SPU-aware public threshold.
- Use spuaware_wrong_high_margin for late-block numeric drift probes, because public threshold changes are unlikely to recover high-margin wrong decisions.
- Affine/temperature margin expansion is a post-reveal confidence/loss calibration layer; it does not change the secret-sharing SPU computation graph.
