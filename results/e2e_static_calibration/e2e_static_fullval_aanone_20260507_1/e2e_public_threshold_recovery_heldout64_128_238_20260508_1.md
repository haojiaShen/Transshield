# E2E Public Threshold Recovery Report

- label: `e2e_public_threshold_recovery_heldout64_128_238_20260508_1`
- status: `within_subset_threshold_can_improve_but_transfer_not_proven`
- reason: Some subset can be improved by refitting the public threshold, but cross-subset transfer is not yet positive.

| dataset | samples | score scale | default acc | best acc | best threshold | delta |
|---|---:|---:|---:|---:|---:|---:|
| heldout64 | 64 | 1 | 92.1875 | 93.75 | -0.15992 | 1.5625 |
| heldout128 | 128 | 1 | 91.4062 | 91.4062 | 0 | 0 |
| heldout238 | 238 | 1 | 92.437 | 92.8571 | -0.0668335 | 0.420168 |

## Cross Dataset Eval

| source threshold | eval dataset | threshold | accuracy | wrong count |
|---|---|---:|---:|---:|
| heldout64 | heldout128 | -0.15992 | 89.8438 | 13 |
| heldout64 | heldout238 | -0.15992 | 91.1765 | 21 |
| heldout128 | heldout64 | 0 | 92.1875 | 5 |
| heldout128 | heldout238 | 0 | 92.437 | 18 |
| heldout238 | heldout64 | -0.0668335 | 92.1875 | 5 |
| heldout238 | heldout128 | -0.0668335 | 89.8438 | 13 |
