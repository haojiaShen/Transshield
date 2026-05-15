# E2E Calibration Decision Report

- status: `promote_spuaware_bias_as_accuracy_first_default`
- reason: SPU-aware bias has cleared the heldout238 gate: sample-weighted held-out accuracy improves over static bias, heldout238 is non-regressive, and no held-out split regresses.
- accuracy-first choice: `spuaware_bias`
- loss-first choice: `e2e_smoke32_affine`
- default gate: Cleared: SPU-aware bias can replace static bias for accuracy-first E2E output calibration.

| split | calibration | accuracy | BCE loss | wrong | low margin <0.25 | mean abs margin |
|---|---|---:|---:|---:|---:|---:|
| heldout64 (n=64) | e2e_smoke32_affine | 92.1875 | 0.201445 | 5 | 0 | 3.458 |
| heldout64 (n=64) | e2e_smoke32_bias | 92.1875 | 0.460965 | 5 | 5 | 0.626126 |
| heldout64 (n=64) | e2e_smoke32_temperature | 92.1875 | 0.201561 | 5 | 0 | 4.0217 |
| heldout64 (n=64) | spuaware_bias | 92.1875 | 0.461214 | 5 | 6 | 0.625762 |
| heldout64 (n=64) | static_bias | 92.1875 | 0.472509 | 5 | 18 | 0.623033 |
| heldout128 (n=128) | e2e_smoke32_affine | 91.4062 | 0.252806 | 11 | 3 | 3.26524 |
| heldout128 (n=128) | e2e_smoke32_bias | 91.4062 | 0.480391 | 11 | 22 | 0.591628 |
| heldout128 (n=128) | e2e_smoke32_temperature | 91.4062 | 0.255607 | 11 | 2 | 3.79882 |
| heldout128 (n=128) | spuaware_bias | 91.4062 | 0.480646 | 11 | 22 | 0.591083 |
| heldout128 (n=128) | static_bias | 87.5 | 0.492084 | 16 | 39 | 0.595304 |
| heldout238 (n=238) | e2e_smoke32_affine | 92.437 | 0.218068 | 18 | 3 | 3.42919 |
| heldout238 (n=238) | e2e_smoke32_bias | 92.437 | 0.46556 | 18 | 31 | 0.621752 |
| heldout238 (n=238) | e2e_smoke32_temperature | 92.437 | 0.21893 | 18 | 1 | 3.9909 |
| heldout238 (n=238) | spuaware_bias | 92.437 | 0.465806 | 18 | 35 | 0.62097 |
| heldout238 (n=238) | static_bias | 90.7563 | 0.477039 | 22 | 65 | 0.620346 |

## Interpretation

- Use `spuaware_bias` when the priority is threshold / argmax accuracy.
- Use E2E-smoke32 affine or temperature when the priority is lower BCE loss and fewer low-margin outputs.
- Aggregate choices use sample-weighted held-out averages when sample counts are available.
- None of these public output calibrations changes the secret-sharing 2PC computation graph.
- None of these should be described as fixing late-block numeric drift; they are post-reveal public calibration layers.
