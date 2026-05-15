# E2E AA=none Block Sweep Summary

- label: `e2e_aanone_block_sweep_wrong_idx13_20260507_1`
- sample: `wrong_idx13`
- status: `late_block_cumulative_numeric_drift_with_high_cosine_alignment`
- reason: Block sweep on the selected wrong low-margin sample shows high cosine alignment at every probed block, while block_output_cls max-abs drift grows toward late blocks. This supports cumulative numeric offset/amplitude drift plus low decision margin, not a large attention-direction mismatch.

| block | largest stage | largest cosine | largest max abs | largest rel L2 | block_output max abs | block_output rel L2 | attn rel L2 | final logits max abs |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | mlp_out_cls | 1 | 0.27327 | 0.00292884 | 0.272522 | 0.00295467 | 0.00252233 | 0.00287545 |
| 3 | block_output_cls | 1 | 0.224518 | 0.0026486 | 0.224518 | 0.0026486 | 0.00377086 | 0.00261605 |
| 6 | block_output_cls | 1 | 0.235825 | 0.00235441 | 0.235825 | 0.00235441 | 0.00293576 | 0.00258553 |
| 9 | block_output_cls | 0.999997 | 0.947922 | 0.00551215 | 0.947922 | 0.00551215 | 0.00161272 | 0.0034858 |
| 12 | block_output_cls | 0.999998 | 2.38269 | 0.00399718 | 2.38269 | 0.00399718 | 0.00189909 | 0.00269234 |

## Interpretation

- The fixed probe semantics no longer support the earlier attention-direction-drift explanation.
- `attn_out_cls` remains high-cosine, and relative L2 is small across the probed blocks.
- The absolute CLS drift grows late, especially by block12, so the practical next axis is cumulative numeric drift reduction or boundary-margin robustness.
- Output calibration is still useful for CE/loss/probability repair, but it cannot recover samples whose raw E2E score is already on the wrong side of the boundary.
