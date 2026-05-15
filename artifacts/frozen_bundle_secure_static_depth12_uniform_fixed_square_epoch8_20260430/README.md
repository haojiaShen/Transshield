# frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_20260430

Frozen export bundle for a Transshield inference-friendly training run.

## Primary candidate
- Source run: `/data/wyb/Transshield_final/artifacts/train_runs/secure_static_depth12_uniform_fixed_square_lr1e5_b8_gpu2_workers0_epoch8_20260430`
- Full checkpoint symlink: `checkpoint-best.pth`
- Pure model weights: `modified_plaintext_model_state_dict.pth`
- Threshold metadata: `threshold_best.json`
- Default argmax acc: `75.95419883728027`
- Thresholded acc: `89.69465494155884`
- Eval loss: `0.5419613122940063`
- AUC: `0.9539369940757751`
- Final epoch default eval acc: `87.21374302055999`
- Final epoch default eval loss: `0.5201374529437586`

## Notes
- The pure `state_dict` export keeps model parameters only and drops optimizer/scaler state.
- The full checkpoint symlink preserves direct `main.py --resume ... --eval true` compatibility.
- Thresholded binary eval is generated separately and stored in `threshold_best.json`.
