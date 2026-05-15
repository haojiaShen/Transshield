# E2E Runtime Efficiency Report

- Label: `e2e_aanone_exactln_clip0_spuaware_heldout_20260508`
- Best speedup: `None`

| Run | Samples | Finite | Threshold Acc | Elapsed (s) | Sec/sample | Speedup vs isolated | Privacy OK |
|---|---:|---|---:|---:|---:|---:|---|
| smoke96_spuaware | 96 | True | 95.8333 | 2028.8420 | 21.1338 |  | True |
| heldout64_spuaware | 64 | True | 92.1875 | 1338.5611 | 20.9150 |  | True |
| heldout128_spuaware | 128 | True | 91.4062 | 2754.4067 | 21.5188 |  | True |
| heldout238_spuaware | 238 | True | 92.4370 | 4983.1371 | 20.9376 |  | True |

## Notes

- `speedup_vs_same_sample_isolated` is computed only when an isolated run with the same sample count is present.
- If present, `static_whole_forward_same_subset_*` is the primary comparator for the current secure-static whole-forward path.
- `original_plaintext_*` is retained only as context against the full bundle forward.
