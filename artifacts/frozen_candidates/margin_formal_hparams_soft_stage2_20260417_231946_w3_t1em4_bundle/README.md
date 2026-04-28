# margin_formal_hparams_soft_stage2_20260417_231946_w3_t1em4_bundle

Frozen export bundle for a Transshield inference-friendly training run.

## Primary candidate
- Source run: `/data/wyb/Transshield_final/artifacts/train_runs/margin_formal_hparams_soft_stage2_20260417_231946_w3_t1em4`
- Full checkpoint symlink: `checkpoint-best.pth`
- Pure model weights: `modified_plaintext_model_state_dict.pth`
- Threshold metadata: `threshold_best.json`
- Default argmax acc: `85.11450290679932`
- Thresholded acc: `91.60305261611938`
- Eval loss: `0.4679286479949951`
- AUC: `0.967475950717926`
- Final epoch default eval acc: `80.72519124737222`
- Final epoch default eval loss: `0.4833294695073908`

## Notes
- The pure `state_dict` export keeps model parameters only and drops optimizer/scaler state.
- The full checkpoint symlink preserves direct `main.py --resume ... --eval true` compatibility.
- Thresholded binary eval is generated separately and stored in `threshold_best.json`.
