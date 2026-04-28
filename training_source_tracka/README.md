# TrackA Source Training Snapshot

This directory vendors the original `DynamicViT_exp_square` training files used to produce the `tracka_lr3e5_timm_best_20260414` provenance chain.

Purpose:

- isolate training-stack drift from dataset / evaluator issues;
- run same-machine server checks from `Transshield_final`;
- keep the normal `training_compat/` stack separate from the original-source control path.

Small convenience / diagnosis flags added:

- `--stop_after_epoch`
- `--nonempty_keep_guard`

`--stop_after_epoch` stops after N completed epochs while preserving the original `--epochs` LR / weight-decay schedule.

`--nonempty_keep_guard` is disabled by default. When enabled, it prevents the mask-pruning training path from producing an all-zero per-sample keep mask, which otherwise can make `PredictorLG` divide by zero in the next pruning stage.

When `--debug_nan true` is enabled:

- `models/dyvit.py` prints `predictor_1/2_keep_diag`;
- `losses.py` prints `ratio_stage_i`.

These lines are diagnostics only and do not change training semantics.
