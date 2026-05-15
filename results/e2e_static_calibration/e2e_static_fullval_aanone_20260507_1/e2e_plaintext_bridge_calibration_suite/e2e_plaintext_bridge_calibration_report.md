# E2E Plaintext Bridge Calibration

- label: `e2e_plaintext_bridge_calibration_suite`
- status: `bridge_candidate_not_better_than_current_spuaware_default`
- reason: Plaintext-static bridge calibration is numerically plausible, but on current held-out raw E2E logits it does not beat the existing spuaware_bias default in sample-weighted accuracy.
- best weighted-accuracy candidate: `spuaware_bias`
- best weighted-BCE candidate: `e2e_smoke32_affine`

## Weighted Summary

| candidate | weighted acc | weighted BCE |
|---|---:|---:|
| heldout64 | 90.697674 | 0.468384 |
| heldout128 | 91.860465 | 0.468636 |
| heldout238 | 91.627907 | 0.470773 |
| bridge_mean_bias | 91.860465 | 0.468842 |
| bridge_median_bias | 91.860465 | 0.468636 |
| static_bias | 90.000000 | 0.480843 |
| spuaware_bias | 92.093023 | 0.469540 |
| e2e_smoke32_affine | 92.093023 | 0.225934 |
| e2e_smoke32_temperature | 92.093023 | 0.227263 |

## Per Eval Dataset

| candidate | eval dataset | acc | BCE | wrong |
|---|---|---:|---:|---:|
| heldout64 | heldout64 | 90.625000 | 0.460057 | 6 |
| heldout64 | heldout128 | 89.843750 | 0.479451 | 13 |
| heldout64 | heldout238 | 91.176471 | 0.464671 | 21 |
| heldout128 | heldout64 | 92.187500 | 0.460310 | 5 |
| heldout128 | heldout128 | 90.625000 | 0.479720 | 12 |
| heldout128 | heldout238 | 92.436975 | 0.464914 | 18 |
| heldout238 | heldout64 | 92.187500 | 0.462447 | 5 |
| heldout238 | heldout128 | 90.625000 | 0.481900 | 12 |
| heldout238 | heldout238 | 92.016807 | 0.467028 | 19 |
| bridge_mean_bias | heldout64 | 92.187500 | 0.460516 | 5 |
| bridge_mean_bias | heldout128 | 90.625000 | 0.479932 | 12 |
| bridge_mean_bias | heldout238 | 92.436975 | 0.465117 | 18 |
| bridge_median_bias | heldout64 | 92.187500 | 0.460310 | 5 |
| bridge_median_bias | heldout128 | 90.625000 | 0.479720 | 12 |
| bridge_median_bias | heldout238 | 92.436975 | 0.464914 | 18 |
| static_bias | heldout64 | 92.187500 | 0.472509 | 5 |
| static_bias | heldout128 | 87.500000 | 0.492084 | 16 |
| static_bias | heldout238 | 90.756303 | 0.477039 | 22 |
| spuaware_bias | heldout64 | 92.187500 | 0.461214 | 5 |
| spuaware_bias | heldout128 | 91.406250 | 0.480646 | 11 |
| spuaware_bias | heldout238 | 92.436975 | 0.465806 | 18 |
| e2e_smoke32_affine | heldout64 | 92.187500 | 0.201445 | 5 |
| e2e_smoke32_affine | heldout128 | 91.406250 | 0.252806 | 11 |
| e2e_smoke32_affine | heldout238 | 92.436975 | 0.218068 | 18 |
| e2e_smoke32_temperature | heldout64 | 92.187500 | 0.201561 | 5 |
| e2e_smoke32_temperature | heldout128 | 91.406250 | 0.255607 | 11 |
| e2e_smoke32_temperature | heldout238 | 92.436975 | 0.218930 | 18 |
