# Payload Precision Ablation

- dtype: `bfloat16`
- stage_count: `3`
- total_float32_bytes: `899184`
- total_candidate_bytes: `449592`
- total_byte_ratio_vs_float32: `0.500x`
- all_exact_semantics_preserved: `false`

| Stage | Layer | Keep | Compact Shape | Bytes Ratio | Exact Semantics | Kth Max Abs Error | Exact Mask Match |
|---:|---:|---:|---|---:|---|---:|---:|
| 0 | 3 | 137 | `[524, 196]` | 0.500x | false | 0.00192368 | 0.991159 |
| 1 | 6 | 96 | `[524, 137]` | 0.500x | false | 0.00194818 | 0.577227 |
| 2 | 9 | 67 | `[524, 96]` | 0.500x | false | 0.00097504 | 0.956345 |

- next_step_hint: If all_exact_semantics_preserved=true for float16, the next engineering step is to add a float16 payload path to the bridge and rerun secure profile / replay.

