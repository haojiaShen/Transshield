# E2E Block Probe Batch Report

- label: `e2e_aanone_heldout238_high_margin_batch_20260508_1`
- status: `consistent_late_block_cumulative_drift_pattern_observed`
- reason: Every selected sample shows high attn cosine alignment plus late-block block_output drift growth; the residual error pattern is consistent with cumulative numeric drift, not a one-off attention-direction failure.

| probe | source index | target | spuaware score | growth first->last | min attn cosine | max final logits abs | image |
|---|---:|---:|---:|---:|---:|---:|---|
| idx121 | 121 | 1 | -0.692276 | 6.03142 | 0.999998 | 0.00322792 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00009.png |
| idx220 | 220 | 1 | -0.64122 | 7.0024 | 0.999997 | 0.00286329 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00445.png |
| idx167 | 167 | 1 | -0.553375 | 7.39759 | 0.999996 | 0.00173478 | /data/wyb/pneumoniamnist_imagefolder_subset/val/1/00216.png |
| idx21 | 21 | 0 | 0.32608 | 7.54457 | 0.999994 | 0.00148854 | /data/wyb/pneumoniamnist_imagefolder_subset/val/0/00093.png |
