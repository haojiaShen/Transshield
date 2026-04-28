# Payload Precision Ablation

- default_dtype: `float16`
- stage_dtype_overrides: `{}`
- boundary_window: `1`
- stage_count: `3`
- total_float32_bytes: `899184`
- total_candidate_bytes: `459024`
- total_byte_ratio_vs_float32: `0.510x`
- all_exact_semantics_preserved: `false`

| Stage | Layer | Keep | Dtype | Boundary Window | Boundary FP32 Mean | Compact Shape | Bytes Ratio | Exact Semantics | Kth Max Abs Error | Exact Mask Match |
|---:|---:|---:|---|---:|---:|---|---:|---|---:|---:|
| 0 | 3 | 137 | `float16` | 1 | 3.00 | `[524, 196]` | 0.508x | false | 0.00009009 | 0.999903 |
| 1 | 6 | 96 | `float16` | 1 | 3.00 | `[524, 137]` | 0.511x | false | 0.00054264 | 0.999972 |
| 2 | 9 | 67 | `float16` | 1 | 3.00 | `[524, 96]` | 0.516x | false | 0.00005680 | 0.999483 |

- next_step_hint: If all_exact_semantics_preserved=true for the selected stage-dtype mix, the next engineering step is to add that mixed-precision payload path to the bridge and rerun secure profile / replay.

