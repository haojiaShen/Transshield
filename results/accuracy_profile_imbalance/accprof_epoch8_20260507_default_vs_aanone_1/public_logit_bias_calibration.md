# Public Logit Bias Calibration: secure_static_accprof_epoch8_20260507_aanone_1

## 1. Conclusion

- status: `public_bias_recovers_threshold_argmax`
- reason: 公开 class-1 logit bias 将最优 threshold 等价搬到 argmax 边界；该操作只需一次公开加法，不改变 token pruning / SPU 主体算子。
- threshold: `0.3577311038970947`
- class1_logit_bias: `0.5852264595359804`

## 2. Metrics

- original_argmax_accuracy: `76.7175572519084`
- threshold_accuracy: `91.98473282442748`
- calibrated_argmax_accuracy: `91.98473282442748`
- original_ce_loss: `0.5305012234754626`
- calibrated_ce_loss: `0.4286648295761729`
- calibrated_minus_original_ce_loss: `-0.10183639389928967`
- calibrated_auc: `0.9678758449966676`

## 3. Deployment Note

- Add the public scalar to the class-1 logit before final argmax.
- This is a public post-processing add and does not change the secure ViT operator family.
- AUC ranking is unchanged because the bias is a monotonic shift of the binary score.
