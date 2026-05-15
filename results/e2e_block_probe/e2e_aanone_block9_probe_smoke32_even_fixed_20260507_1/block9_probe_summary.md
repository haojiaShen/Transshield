# E2E AA=none Block9 Probe Summary (Fixed Probe Semantics)

- status: `boundary_margin_and_cumulative_numeric_drift_are_primary_next_axes`
- reason: The wrong sample has abs bias margin 0.0336, while fixed block9 probe cosine remains ~0.999999. Accuracy work should prioritize margin robustness or cumulative drift reduction; attention probe semantics no longer support attention-direction-drift as the main cause.

| sample | target | correct | bias score | top drift stage | top cosine | top max abs err | top rel L2 err | attn rel L2 err |
|---|---:|---|---:|---|---:|---:|---:|---:|
| wrong_idx13 | 0 | False | 0.033559 | block_output_cls | 0.999999 | 0.742599 | 0.003540 | 0.001591 |
| correct_idx17 | 1 | True | 1.800390 | block_output_cls | 1.000000 | 1.032990 | 0.003542 | 0.001351 |

## Interpretation

- Earlier block9 probe outputs are superseded because SPU `attn_out_cls` was captured before projection while CPU captured the projected Attention output.
- With aligned semantics, both samples show high cosine similarity across block9 probe tensors.
- The wrong sample is a near-boundary case, so small cumulative numeric shifts can flip it.
- The next practical accuracy path is margin robustness / cumulative numeric drift reduction, not more output calibration and not an attention-policy change that the current trained uniform-attention bundle cannot support directly.
