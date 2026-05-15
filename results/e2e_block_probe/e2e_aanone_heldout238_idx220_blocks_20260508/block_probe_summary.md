# E2E AA=none Block Sweep Summary

- label: `e2e_aanone_heldout238_idx220_high_margin_wrong_block_probe_20260508`
- sample: `heldout238_idx220_spuaware_high_margin_wrong`
- status: `late_block_cumulative_numeric_drift_with_high_cosine_alignment`
- reason: Block sweep on the selected sample shows high cosine alignment at every probed block, while block_output_cls max-abs drift grows toward late blocks. This supports cumulative numeric offset/amplitude drift plus insufficient boundary robustness, not a large attention-direction mismatch.

| block | largest stage | largest cosine | largest max abs | largest rel L2 | block_output max abs | block_output rel L2 | attn rel L2 | final logits max abs |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | mlp_out_cls | 1 | 0.353287 | 0.00399431 | 0.352531 | 0.00402857 | 0.00254162 | 0.00168836 |
| 6 | block_output_cls | 0.999999 | 0.274254 | 0.00345239 | 0.274254 | 0.00345239 | 0.00237999 | 0.00246656 |
| 9 | block_output_cls | 0.999997 | 0.858231 | 0.00464487 | 0.858231 | 0.00464487 | 0.00139523 | 0.00286329 |
| 12 | block_output_cls | 0.999998 | 2.46857 | 0.00410572 | 2.46857 | 0.00410572 | 0.00182975 | 0.00263441 |

## Interpretation

- The fixed probe semantics no longer support the earlier attention-direction-drift explanation.
- `attn_out_cls` remains high-cosine, and relative L2 is small across the probed blocks.
- The absolute CLS drift grows late, especially by block12, so the practical next axis is cumulative numeric drift reduction or boundary-margin robustness.
- Output calibration is still useful for CE/loss/probability repair, but it cannot recover samples whose raw E2E score is already on the wrong side of the boundary.
