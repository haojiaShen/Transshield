# E2E AA=none Block9 Probe Summary

- status: `attention_policy_is_primary_next_axis`
- reason: attn_out_cls has negative/near-zero cosine and largest max_abs/l2 error for both selected samples, while later residual/block outputs remain high-cosine. The next controlled experiment should compare uniform vs identity/standard/smoothed attention on the same small image list before changing training.

| sample | target | correct | bias score | top drift stage | attn cosine | attn max abs err | attn rel L2 err | block rel L2 err |
|---|---:|---|---:|---|---:|---:|---:|---:|
| wrong_idx13 | 0 | False | 0.033559 | attn_out_cls | -0.026590 | 14.082108 | 1.240614 | 0.004395 |
| correct_idx17 | 1 | True | 1.800390 | attn_out_cls | -0.017681 | 12.827559 | 1.237429 | 0.002019 |

## Interpretation

- `attn_out_cls` is the dominant drift point for both samples.
- The wrong sample is a near-boundary case (`abs margin = 0.0336`), so small downstream shifts flip it.
- The correct sample still has severe attention-output direction drift, but its final margin is large enough to survive.
- Next accuracy work should test attention-policy alternatives on the same image list; output calibration is now a loss/probability repair, not the main accuracy lever.
