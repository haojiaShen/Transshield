# Payload Precision Ablation

- default_dtype: `float32`
- stage_dtype_overrides: `{'1': 'float16'}`
- stage_count: `3`
- total_float32_bytes: `899184`
- total_candidate_bytes: `755608`
- total_byte_ratio_vs_float32: `0.840x`
- all_exact_semantics_preserved: `false`

| Stage | Layer | Keep | Dtype | Compact Shape | Bytes Ratio | Exact Semantics | Kth Max Abs Error | Exact Mask Match |
|---:|---:|---:|---|---|---:|---|---:|---:|
| 0 | 3 | 137 | `float32` | `[524, 196]` | 1.000x | true | 0.00000000 | 1.000000 |
| 1 | 6 | 96 | `float16` | `[524, 137]` | 0.500x | false | 0.00024408 | 0.925113 |
| 2 | 9 | 67 | `float32` | `[524, 96]` | 1.000x | true | 0.00000000 | 1.000000 |

- next_step_hint: If all_exact_semantics_preserved=true for the selected stage-dtype mix, the next engineering step is to add that mixed-precision payload path to the bridge and rerun secure profile / replay.

