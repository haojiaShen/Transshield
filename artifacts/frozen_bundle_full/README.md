# pneumonia_transshield_candidate_frozen_export_v1

Frozen export bundle for the audited modified Transshield plaintext candidate.

## Dataset

This bundle was produced for:

- dataset: `pneumoniamnist_imagefolder_subset`
- training split: `train`
- evaluation split: `val`
- format: `ImageFolder`
- classes: `2`

## Primary bundled assets

- source run provenance: `DynamicViT_exp_square/outputs/pneumonia_transshield_peproj_zero_fixstep0_lr1e5_actlr10_learnquadgelu_noamp_evalon_epoch8`
- portable state dict: `modified_plaintext_model_state_dict.pth`
- portable eval checkpoint: `modified_plaintext_eval_checkpoint_light.pth`
- args snapshot: `args_snapshot.json`
- threshold metadata: `threshold_best.json`
- threshold/eval stdout provenance: `train_stdout.log`, `eval_threshold_stdout.log`

## Reported metrics for the bundled modified checkpoint

- default argmax acc: `81.29771`
- thresholded acc: `85.87786`
- eval loss: `0.5782`
- AUC: `0.9231076835189945`

## Important note about omitted files

The original large training checkpoints are not kept in this final GitHub-oriented repo.

- omitted: `checkpoint-best.pth`
- omitted: `checkpoint-best-repro1.pth`

The replacement artifact for direct evaluation in this repo is:

- `modified_plaintext_eval_checkpoint_light.pth`

The archived full training checkpoint is kept separately at:

- `artifacts/archive/frozen_bundle_full/modified_plaintext_training_checkpoint_full.pth`

## Notes

- `modified_plaintext_model_state_dict.pth` keeps model parameters only and drops optimizer/scaler state.
- `modified_plaintext_eval_checkpoint_light.pth` is the repo-friendly eval checkpoint used by the bundled comparison scripts.
- thresholded binary eval uses `threshold_best.json`.
