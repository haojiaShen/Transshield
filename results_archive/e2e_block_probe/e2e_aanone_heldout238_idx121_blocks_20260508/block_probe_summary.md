# E2E AA=none Block Sweep Summary

- label: `e2e_aanone_heldout238_idx121_high_margin_wrong_block_probe_20260508`
- sample: `heldout238_idx121_spuaware_high_margin_wrong`
- status: `late_block_cumulative_numeric_drift_with_high_cosine_alignment`
- reason: Block sweep on the selected sample shows high cosine alignment at every probed block, while block_output_cls max-abs drift grows toward late blocks. This supports cumulative numeric offset/amplitude drift plus insufficient boundary robustness, not a large attention-direction mismatch.

| block | largest stage | largest cosine | largest max abs | largest rel L2 | block_output max abs | block_output rel L2 | attn rel L2 | final logits max abs |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | mlp_out_cls | 1 | 0.312454 | 0.00367409 | 0.311752 | 0.00370623 | 0.0024361 | 0.0024192 |
| 6 | block_output_cls | 0.999999 | 0.336479 | 0.00471447 | 0.336479 | 0.00471447 | 0.00207226 | 0.00309059 |
| 9 | block_output_cls | 0.999997 | 0.803818 | 0.00500114 | 0.803818 | 0.00500114 | 0.00120914 | 0.00318214 |
| 12 | block_input_cls | 0.999998 | 1.88327 | 0.00427157 | 1.88031 | 0.00340614 | 0.00161707 | 0.00322792 |

## Interpretation

- The fixed probe semantics no longer support the earlier attention-direction-drift explanation.
- `attn_out_cls` remains high-cosine, and relative L2 is small across the probed blocks.
- The absolute CLS drift grows late, especially by block12, so the practical next axis is cumulative numeric drift reduction or boundary-margin robustness.
- Output calibration is still useful for CE/loss/probability repair, but it cannot recover samples whose raw E2E score is already on the wrong side of the boundary.
