# E2E Runtime Efficiency Report

- Label: `e2e_aanone_exactln_clip0_isolated_vs_nonisolated_20260507`
- Best speedup: `2.2774260356509277`

| Run | Samples | Finite | Threshold Acc | Elapsed (s) | Sec/sample | Speedup vs isolated | Privacy OK |
|---|---:|---|---:|---:|---:|---:|---|
| isolated_smoke8 | 8 | True | 75.0000 | 387.0394 | 48.3799 |  | True |
| nonisolated_smoke8 | 8 | True | 75.0000 | 187.0298 | 23.3787 | 2.0694 | True |
| isolated_smoke16 | 16 | True | 81.2500 | 768.6034 | 48.0377 |  | True |
| nonisolated_smoke16_bias | 16 | True | 81.2500 | 352.2982 | 22.0186 | 2.1817 | True |
| nonisolated_smoke16_affine | 16 | True | 87.5000 | 343.9620 | 21.4976 | 2.2346 | True |
| isolated_smoke32 | 32 | True | 90.6250 | 1522.9726 | 47.5929 |  | True |
| nonisolated_smoke32_legacy_bias | 32 | True | 90.6250 | 689.4148 | 21.5442 | 2.2091 | True |
| nonisolated_smoke32_affine_even | 32 | True | 87.5000 | 687.0904 | 21.4716 | 2.2166 | True |
| nonisolated_smoke32_temperature_even | 32 | True | 87.5000 | 668.7254 | 20.8977 | 2.2774 | True |
| nonisolated_smoke64_head | 64 | True | 64.0625 | 1345.3246 | 21.0207 |  | True |
| nonisolated_smoke64_even | 64 | True | 87.5000 | 1352.9082 | 21.1392 |  | True |

## Notes

- `speedup_vs_same_sample_isolated` is computed only when an isolated run with the same sample count is present.
- If present, `static_whole_forward_same_subset_*` is the primary comparator for the current secure-static whole-forward path.
- `original_plaintext_*` is retained only as context against the full bundle forward.
