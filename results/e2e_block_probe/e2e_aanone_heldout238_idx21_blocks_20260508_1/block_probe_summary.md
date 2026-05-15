# E2E AA=none Block Sweep Summary

- label: `e2e_aanone_heldout238_idx21_blocks_20260508_1`
- sample: `heldout238_idx21_spuaware_high_margin_probe`
- status: `late_block_cumulative_numeric_drift_with_high_cosine_alignment`
- reason: Block sweep on the selected sample shows high cosine alignment at every probed block, while block_output_cls max-abs drift grows toward late blocks. This supports cumulative numeric offset/amplitude drift plus insufficient boundary robustness, not a large attention-direction mismatch.

| block | largest stage | largest cosine | largest max abs | largest rel L2 | block_output max abs | block_output rel L2 | attn rel L2 | final logits max abs |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | mlp_out_cls | 1 | 0.411781 | 0.00427013 | 0.411163 | 0.004304 | 0.00238926 | 0.0011681 |
| 6 | block_input_cls | 0.999999 | 0.399193 | 0.00497719 | 0.396118 | 0.0049323 | 0.00351013 | 0.000420421 |
| 9 | block_output_cls | 0.999999 | 1.23529 | 0.00506903 | 1.23529 | 0.00506903 | 0.00133351 | 0.00148854 |
| 12 | block_output_cls | 1 | 3.10205 | 0.00288436 | 3.10205 | 0.00288436 | 0.00138929 | 0.000607044 |

## Interpretation

- The fixed probe semantics no longer support the earlier attention-direction-drift explanation.
- `attn_out_cls` remains high-cosine, and relative L2 is small across the probed blocks.
- The absolute CLS drift grows late, especially by block12, so the practical next axis is cumulative numeric drift reduction or boundary-margin robustness.
- Output calibration is still useful for CE/loss/probability repair, but it cannot recover samples whose raw E2E score is already on the wrong side of the boundary.
