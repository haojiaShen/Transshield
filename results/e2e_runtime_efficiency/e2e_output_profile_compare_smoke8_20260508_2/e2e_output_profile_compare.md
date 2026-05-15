# E2E Output Profile Compare

- run_count: `3`
- same_sample_signature: `true`
- same_sample_count: `true`
- baseline_label: `accuracy_first`
- best_threshold_accuracy: `loss_first_affine` (100.000000)
- best_argmax_accuracy: `loss_first_affine` (100.000000)
- best_lowest_calibrated_bce: `loss_first_temperature` (0.023480)
- best_fastest: `loss_first_affine` (188.937261)
- best_lowest_total_bytes: `loss_first_temperature` (1764378721)
- best_lowest_raw_logits_mae_vs_static: `loss_first_affine` (0.002226)

| label | profile | th_acc | argmax_acc | calibrated_bce | raw_th_acc | raw_argmax_acc | raw_logits_mae | elapsed_sec | total_bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| accuracy_first | accuracy_first | 100.000000 | 100.000000 | 0.355974 | 100.000000 | 87.500000 | 0.002460 | 194.064289 | 1764397552 |
| loss_first_affine | loss_first_affine | 100.000000 | 100.000000 | 0.031657 | 100.000000 | 87.500000 | 0.002226 | 188.937261 | 1764380509 |
| loss_first_temperature | loss_first_temperature | 100.000000 | 100.000000 | 0.023480 | 100.000000 | 87.500000 | 0.002238 | 190.488374 | 1764378721 |

## Delta Vs Baseline

Baseline: `accuracy_first`

| label | d_th_acc | d_argmax_acc | d_calibrated_bce | d_elapsed_sec | d_total_bytes | d_raw_logits_mae |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accuracy_first | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0.000000 |
| loss_first_affine | 0.000000 | 0.000000 | -0.324317 | -5.127028 | -17043 | -0.000235 |
| loss_first_temperature | 0.000000 | 0.000000 | -0.332493 | -3.575914 | -18831 | -0.000222 |
