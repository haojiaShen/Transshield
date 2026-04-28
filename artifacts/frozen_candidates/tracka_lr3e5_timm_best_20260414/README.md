# tracka_lr3e5_timm_best_20260414

Frozen export bundle for a Transshield inference-friendly training run.

## Primary candidate
- Historical source run: `/home/yclcg/DynamicViT_exp_square/outputs/pneumonia_transshield_tracka_lr3e5_timm`
- Current final-repo training entry: `training_compat/main.py`
- Full checkpoint file: `checkpoint-best.pth`
- Pure model weights: `modified_plaintext_model_state_dict.pth`
- Threshold metadata: `threshold_best.json`
- Default argmax acc: `93.70229244232178`
- Thresholded acc: `94.08397078514099`
- Eval loss: `0.47675642371177673`
- AUC: `0.972331702709198`
- Final epoch default eval acc: `85.87786539092319`
- Final epoch default eval loss: `0.5229347592050378`

## Notes
- The pure `state_dict` export keeps model parameters only and drops optimizer/scaler state.
- The full checkpoint is materialized in this bundle so the final repo no longer depends on the historical experiment repo symlink.
- Thresholded binary eval is generated separately and stored in `threshold_best.json`.
