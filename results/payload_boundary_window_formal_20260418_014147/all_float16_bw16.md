# Payload Precision Ablation

- default_dtype: `float16`
- stage_dtype_overrides: `{}`
- boundary_window: `16`
- stage_count: `3`
- total_float32_bytes: `899184`
- total_candidate_bytes: `553344`
- total_byte_ratio_vs_float32: `0.615x`
- all_exact_semantics_preserved: `false`

| Stage | Layer | Keep | Dtype | Boundary Window | Boundary FP32 Mean | Compact Shape | Bytes Ratio | Exact Semantics | Kth Max Abs Error | Exact Mask Match |
|---:|---:|---:|---|---:|---:|---|---:|---|---:|---:|
| 0 | 3 | 137 | `float16` | 16 | 33.00 | `[524, 196]` | 0.584x | true | 0.00000000 | 1.000000 |
| 1 | 6 | 96 | `float16` | 16 | 33.00 | `[524, 137]` | 0.620x | false | 0.00006300 | 0.996824 |
| 2 | 9 | 67 | `float16` | 16 | 33.00 | `[524, 96]` | 0.672x | true | 0.00000000 | 1.000000 |

- next_step_hint: If all_exact_semantics_preserved=true for the selected stage-dtype mix, the next engineering step is to add that mixed-precision payload path to the bridge and rerun secure profile / replay.

