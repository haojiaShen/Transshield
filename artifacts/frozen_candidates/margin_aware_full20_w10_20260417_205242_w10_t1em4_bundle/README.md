# margin_aware_full20_w10_20260417_205242_w10_t1em4_bundle

Frozen export bundle for a Transshield inference-friendly training run.

## Primary candidate
- Source run: `/data/wyb/Transshield_final/artifacts/train_runs/margin_aware_full20_w10_20260417_205242_w10_t1em4`
- Full checkpoint symlink: `checkpoint-best.pth`
- Pure model weights: `modified_plaintext_model_state_dict.pth`
- Threshold metadata: `threshold_best.json`
- Default argmax acc: `88.93129825592041`
- Thresholded acc: `90.2671754360199`
- Eval loss: `0.47089675068855286`
- AUC: `0.956507682800293`
- Final epoch default eval acc: `85.11450573870243`
- Final epoch default eval loss: `0.4769613200967962`

## Notes
- The pure `state_dict` export keeps model parameters only and drops optimizer/scaler state.
- The full checkpoint symlink preserves direct `main.py --resume ... --eval true` compatibility.
- Thresholded binary eval is generated separately and stored in `threshold_best.json`.
