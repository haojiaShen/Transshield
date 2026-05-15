# E2E Public Threshold Recovery Report

- label: `e2e_aanone_public_threshold_recovery_smoke32_to_smoke64_20260507`
- status: `public_threshold_transfer_improves_eval_subset`
- reason: A threshold fitted on one E2E subset improved another E2E subset; SPU-aware public threshold calibration is a viable lightweight recovery axis.

| dataset | samples | score scale | default acc | best acc | best threshold | delta |
|---|---:|---:|---:|---:|---:|---:|
| smoke32_temp | 32 | 6.4983 | 87.5 | 96.875 | 0.232519 | 9.375 |
| smoke64_bias | 64 | 1 | 87.5 | 93.75 | 0.504247 | 6.25 |

## Cross Dataset Eval

| source threshold | eval dataset | threshold | accuracy | wrong count |
|---|---|---:|---:|---:|
| smoke32_temp | smoke64_bias | 0.232519 | 92.1875 | 5 |
| smoke64_bias | smoke32_temp | 0.504247 | 90.625 | 3 |
