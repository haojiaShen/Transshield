# e2e_plaintext_static_gap_heldout128

## Stage Metrics
- `plaintext_full_model`: argmax_acc=`51.562500`, threshold_acc=`50.000000`, auc=`0.952636719`, ce_loss=`0.623240713`
- `static_whole_forward`: argmax_acc=`86.718750`, threshold_acc=`88.281250`, auc=`0.957275391`, ce_loss=`0.486554001`

## Score Alignment
- `argmax_match_ratio = 0.382812`
- `threshold_match_ratio = 0.554688`
- `score_correlation = 0.960693`
- `same_sign_ratio = 0.382812`
- `affine static_score ~= a * plaintext_score + b`: `a = 1.871330`, `b = -1.464497`, `x_at_y0 = 0.782597`

## Threshold Sweep
- `plaintext zero-threshold accuracy = 51.562500`
- `plaintext best-threshold accuracy = 89.843750` at threshold `0.623077`
- `best threshold to match static argmax = 0.798052` with match `0.929688`

## Judgement
- status: `ranking_preserved_but_zero_boundary_misaligned`
- reason: `Plaintext full-path class scores stay strongly rank-correlated with static scores, but the zero boundary is badly shifted; sweeping a public threshold on plaintext scores nearly recovers static-level accuracy.`
