# MPCViT PneumoniaMNIST Multi-Seed Summary

## Per-Seed Best-By-Argmax

| Seed | Epoch | Argmax Acc (%) | Threshold Acc (%) | AUC | Elapsed Sec |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 20 | 96.183206 | 96.183206 | 0.992564 | 69.2465 |

## Aggregate

| Selection | Metric | Mean | Std | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: |
| best_by_threshold | argmax_accuracy | 91.030534 | 0.000000 | 91.030534 | 91.030534 |
| best_by_threshold | threshold_accuracy | 96.564885 | 0.000000 | 96.564885 | 96.564885 |
| best_by_threshold | auc | 0.990269 | 0.000000 | 0.990269 | 0.990269 |
| best_by_argmax | argmax_accuracy | 96.183206 | 0.000000 | 96.183206 | 96.183206 |
| best_by_argmax | threshold_accuracy | 96.183206 | 0.000000 | 96.183206 | 96.183206 |
| best_by_argmax | auc | 0.992564 | 0.000000 | 0.992564 | 0.992564 |
| best_by_auc | argmax_accuracy | 95.038168 | 0.000000 | 95.038168 | 95.038168 |
| best_by_auc | threshold_accuracy | 95.992366 | 0.000000 | 95.992366 | 95.992366 |
| best_by_auc | auc | 0.993297 | 0.000000 | 0.993297 | 0.993297 |
