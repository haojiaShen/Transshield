# Payload Precision Ablation

- default_dtype: `float16`
- stage_dtype_overrides: `{'1': 'float32'}`
- boundary_window: `4`
- stage_count: `3`
- total_float32_bytes: `899184`
- total_candidate_bytes: `612032`
- total_byte_ratio_vs_float32: `0.681x`
- all_exact_semantics_preserved: `true`

| Stage | Layer | Keep | Dtype | Boundary Window | Boundary FP32 Mean | Compact Shape | Bytes Ratio | Exact Semantics | Kth Max Abs Error | Exact Mask Match |
|---:|---:|---:|---|---:|---:|---|---:|---|---:|---:|
| 0 | 3 | 137 | `float16` | 4 | 9.00 | `[524, 196]` | 0.523x | true | 0.00000000 | 1.000000 |
| 1 | 6 | 96 | `float32` | 4 | 9.00 | `[524, 137]` | 1.000x | true | 0.00000000 | 1.000000 |
| 2 | 9 | 67 | `float16` | 4 | 9.00 | `[524, 96]` | 0.547x | true | 0.00000000 | 1.000000 |

- next_step_hint: If all_exact_semantics_preserved=true for the selected stage-dtype mix, the next engineering step is to add that mixed-precision payload path to the bridge and rerun secure profile / replay.

