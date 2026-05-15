# e2e_aanone_exactln_clip0_smoke8_accuracyfirst_20260508_1_plaintext_static_gap

## Stage Metrics
- `plaintext_full_model`: argmax_acc=`50.000000`, threshold_acc=`50.000000`, auc=`1.000000000`, ce_loss=`0.532467280`
- `static_whole_forward`: argmax_acc=`87.500000`, threshold_acc=`100.000000`, auc=`1.000000000`, ce_loss=`0.360592408`

## Score Alignment
- `argmax_match_ratio = 0.375000`
- `threshold_match_ratio = 0.500000`
- `score_correlation = 0.991371`
- `same_sign_ratio = 0.375000`
- `affine static_score ~= a * plaintext_score + b`: `a = 1.927835`, `b = -1.388062`, `x_at_y0 = 0.720010`

## Threshold Sweep
- `plaintext zero-threshold accuracy = 50.000000`
- `plaintext best-threshold accuracy = 100.000000` at threshold `0.465708`
- `best threshold to match static argmax = 0.926744` with match `1.000000`

## Judgement
- status: `ranking_preserved_but_zero_boundary_misaligned`
- reason: `Plaintext full-path class scores stay strongly rank-correlated with static scores, but the zero boundary is badly shifted; sweeping a public threshold on plaintext scores nearly recovers static-level accuracy.`
