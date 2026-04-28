# MPCViT PneumoniaMNIST Multi-Seed Summary

## Per-Seed Best-By-Argmax

| Seed | Epoch | Argmax Acc (%) | Threshold Acc (%) | AUC | Elapsed Sec |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 11 | 95.992366 | 96.374046 | 0.991060 | 48.2644 |
| 2 | 17 | 97.328244 | 97.519084 | 0.995839 | 49.4656 |

## Aggregate

| Selection | Metric | Mean | Std | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: |
| best_by_threshold | argmax_accuracy | 96.374046 | 0.763359 | 95.610687 | 97.137405 |
| best_by_threshold | threshold_accuracy | 97.232824 | 0.286260 | 96.946565 | 97.519084 |
| best_by_threshold | auc | 0.992131 | 0.002575 | 0.989555 | 0.994706 |
| best_by_argmax | argmax_accuracy | 96.660305 | 0.667939 | 95.992366 | 97.328244 |
| best_by_argmax | threshold_accuracy | 96.946565 | 0.572519 | 96.374046 | 97.519084 |
| best_by_argmax | auc | 0.993449 | 0.002390 | 0.991060 | 0.995839 |
| best_by_auc | argmax_accuracy | 96.564885 | 0.763359 | 95.801527 | 97.328244 |
| best_by_auc | threshold_accuracy | 96.946565 | 0.572519 | 96.374046 | 97.519084 |
| best_by_auc | auc | 0.993873 | 0.001966 | 0.991907 | 0.995839 |
