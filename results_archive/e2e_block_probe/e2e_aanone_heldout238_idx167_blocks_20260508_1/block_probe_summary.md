# E2E AA=none Block Sweep Summary

- label: `e2e_aanone_heldout238_idx167_blocks_20260508_1`
- sample: `heldout238_idx167_spuaware_high_margin_probe`
- status: `late_block_cumulative_numeric_drift_with_high_cosine_alignment`
- reason: Block sweep on the selected sample shows high cosine alignment at every probed block, while block_output_cls max-abs drift grows toward late blocks. This supports cumulative numeric offset/amplitude drift plus insufficient boundary robustness, not a large attention-direction mismatch.

| block | largest stage | largest cosine | largest max abs | largest rel L2 | block_output max abs | block_output rel L2 | attn rel L2 | final logits max abs |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | mlp_out_cls | 1 | 0.328072 | 0.00349155 | 0.327412 | 0.0035207 | 0.00250879 | 0.00173478 |
| 6 | block_input_cls | 0.999999 | 0.400242 | 0.00480265 | 0.394169 | 0.00474401 | 0.00283986 | 0.000651408 |
| 9 | block_output_cls | 0.999999 | 0.814896 | 0.00399122 | 0.814896 | 0.00399122 | 0.00149206 | 0.00115495 |
| 12 | block_output_cls | 0.999999 | 2.42206 | 0.00320188 | 2.42206 | 0.00320188 | 0.0018522 | 0.00104814 |

## Interpretation

- The fixed probe semantics no longer support the earlier attention-direction-drift explanation.
- `attn_out_cls` remains high-cosine, and relative L2 is small across the probed blocks.
- The absolute CLS drift grows late, especially by block12, so the practical next axis is cumulative numeric drift reduction or boundary-margin robustness.
- Output calibration is still useful for CE/loss/probability repair, but it cannot recover samples whose raw E2E score is already on the wrong side of the boundary.
