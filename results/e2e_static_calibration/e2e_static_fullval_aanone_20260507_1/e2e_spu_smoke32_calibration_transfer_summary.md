# E2E SPU Smoke32 Calibration Transfer Summary

- status: `accuracy_neutral_loss_positive`
- fit subset: `e2e_aanone_exactln_clip0_smoke32_temp_nonisolated_20260507_1`
- heldout64 transfer report: `e2e_heldout64_spu_smoke32_calibration_transfer_report.json`
- heldout128 transfer report: `e2e_heldout128_spu_smoke32_calibration_transfer_report.json`

| split | calibration | accuracy | BCE loss | wrong | low margin abs(score)<0.25 |
|---|---|---:|---:|---:|---:|
| heldout64 | static bias | 92.1875 | 0.472509 | 5 | 18 |
| heldout64 | SPU-aware bias | 92.1875 | 0.461214 | 5 | 6 |
| heldout64 | E2E-smoke32 affine | 92.1875 | 0.201445 | 5 | 0 |
| heldout64 | E2E-smoke32 temperature | 92.1875 | 0.201561 | 5 | 0 |
| heldout128 | static bias | 87.5 | 0.492084 | 16 | 39 |
| heldout128 | SPU-aware bias | 91.40625 | 0.480646 | 11 | 22 |
| heldout128 | E2E-smoke32 affine | 91.40625 | 0.252806 | 11 | 3 |
| heldout128 | E2E-smoke32 temperature | 91.40625 | 0.255607 | 11 | 2 |

## Interpretation

- `SPU-aware bias` is the current accuracy-first candidate because it improves heldout128 accuracy over static bias.
- `E2E-smoke32 affine` and `E2E-smoke32 temperature` do not further improve heldout64/heldout128 accuracy, but they substantially reduce BCE loss and low-margin predictions.
- This gives a separate recovery path for confidence/loss calibration after reveal, without changing the secret-sharing 2PC computation graph.
- This is not a fix for late-block numeric drift; it is a public output calibration layer that makes the revealed score better calibrated.
