# Baseline comparison assets

This directory keeps the original/plaintext comparison assets needed by the single final `Transshield` repository.

## Dataset

These baseline assets are meant for:

- dataset: `pneumoniamnist_imagefolder_subset`
- training split provenance: `train`
- evaluation split provenance: `val`
- format: `ImageFolder`
- classes: `2`

## Source provenance

主目录当前保留：

- `baseline_plaintext_eval_checkpoint_light.pth`
- `original_plaintext_args_snapshot_fix3.json`
- `original_plaintext_threshold_best_fix3.json`
- `original_plaintext_threshold_search_fix3.csv`
- `manifest_fix3.json`

These files are used by:

- `artifacts/server_inference_friendly_pack/run_plaintext_eval.sh baseline`
- `artifacts/server_inference_friendly_pack/run_plaintext_model_compare.sh`

The lightweight checkpoint keeps only the model state and args snapshot required for evaluation, so the repo does not need a second standalone baseline code repository or a heavier training checkpoint.

Use guidance:

- use `baseline_plaintext_eval_checkpoint_light.pth` for the repo's default evaluation and comparison scripts
- use `original_plaintext_threshold_best_fix3.json` only with the paired `fix3` baseline checkpoints above

The threshold JSON in this directory has been regenerated to match the bundled `fix3` lightweight checkpoint, so the repo does not silently fall back to the earlier local baseline assets.
