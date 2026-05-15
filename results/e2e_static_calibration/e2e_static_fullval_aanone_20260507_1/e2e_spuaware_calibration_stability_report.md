# E2E SPU-aware Calibration Stability Report

- status: `spuaware_public_bias_positive_or_neutral_on_current_smoke_checks`
- calibration: `e2e_static_output_calibration_public_logit_bias_spuaware_smoke32_threshold.json`
- bias: `0.3527068929`
- reason: The SPU-aware public bias fitted from smoke32 improves cross-subset replay and smoke96, is neutral on a smoke32-disjoint heldout64 run, and remains positive on heldout128 same-raw-logits comparison.

| check | samples | baseline / default | SPU-aware result | delta |
|---|---:|---:|---:|---:|
| smoke32 threshold fit | 32 | 87.5 | 96.875 | +9.375 |
| smoke32 threshold -> smoke64 replay | 64 | 87.5 | 92.1875 | +4.6875 |
| smoke96 actual E2E run | 96 | n/a | 95.8333 | n/a |
| smoke96 same raw logits: static bias | 96 | 92.7083 | 95.8333 | +3.125 |
| smoke96 same raw logits: temperature | 96 | 92.7083 | 95.8333 | +3.125 |
| smoke96 same raw logits: affine | 96 | 92.7083 | 95.8333 | +3.125 |
| smoke32-disjoint heldout64 actual E2E | 64 | n/a | 92.1875 | n/a |
| heldout64 same raw logits: static bias | 64 | 92.1875 | 92.1875 | 0 |
| heldout64 same raw logits: temperature | 64 | 92.1875 | 92.1875 | 0 |
| heldout64 same raw logits: affine | 64 | 92.1875 | 92.1875 | 0 |
| smoke32-disjoint heldout128 actual E2E | 128 | n/a | 91.40625 | n/a |
| heldout128 same raw logits: static bias | 128 | 87.5 | 91.40625 | +3.90625 |
| heldout128 same raw logits: temperature | 128 | 87.5 | 91.40625 | +3.90625 |
| heldout128 same raw logits: affine | 128 | 88.28125 | 91.40625 | +3.125 |

## Interpretation

- Current evidence is stable enough to keep `spuaware_bias` as an accuracy-first E2E candidate, but not enough to make it the default.
- The same smoke96 raw logits show `spuaware_bias` reduces wrong samples from `7` to `4`.
- The smoke32-disjoint heldout64 run shows no regression, but no improvement over other public calibrations.
- The smoke32-disjoint heldout128 run reduces wrong samples from `16` under static bias / temperature and `15` under affine to `11` under `spuaware_bias`.
- heldout128 also increases low-margin samples (`abs(score) < 0.25`) under the SPU-aware boundary, so this should be treated as boundary calibration evidence, not as a late-block numeric drift fix.
- This is still a public output calibration, not a fix for late-block numeric drift itself.
- Do not replace the default calibration until a larger held-out E2E subset confirms non-regression and a meaningful aggregate gain.
