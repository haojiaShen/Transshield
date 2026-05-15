# frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_aanone_20260507

Frozen export bundle for a Transshield inference-friendly training run.

## Primary candidate
- Source run: `/data/wyb/Transshield_final/artifacts/train_runs/secure_static_accprof_epoch8_20260507_aanone_1`
- Full checkpoint symlink: `checkpoint-best.pth`
- Pure model weights: `modified_plaintext_model_state_dict.pth`
- Threshold metadata: `threshold_best.json`
- Default argmax acc: `76.7175555229187`
- Thresholded acc: `91.9847309589386`
- Eval loss: `0.530501127243042`
- AUC: `0.9678758382797241`
- Final epoch default eval acc: `72.70992535307207`
- Final epoch default eval loss: `0.561071742664684`

## Notes
- The pure `state_dict` export keeps model parameters only and drops optimizer/scaler state.
- The full checkpoint symlink preserves direct `main.py --resume ... --eval true` compatibility.
- Thresholded binary eval is generated separately and stored in `threshold_best.json`.
