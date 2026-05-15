# e2e_plaintext_static_gap_heldout64

## Stage Metrics
- `plaintext_full_model`: argmax_acc=`51.562500`, threshold_acc=`50.000000`, auc=`0.964843750`, ce_loss=`0.612092948`
- `static_whole_forward`: argmax_acc=`89.062500`, threshold_acc=`92.187500`, auc=`0.971679688`, ce_loss=`0.467208805`

## Score Alignment
- `argmax_match_ratio = 0.406250`
- `threshold_match_ratio = 0.515625`
- `score_correlation = 0.966689`
- `same_sign_ratio = 0.406250`
- `affine static_score ~= a * plaintext_score + b`: `a = 1.865846`, `b = -1.475392`, `x_at_y0 = 0.790736`

## Threshold Sweep
- `plaintext zero-threshold accuracy = 51.562500`
- `plaintext best-threshold accuracy = 93.750000` at threshold `0.657084`
- `best threshold to match static argmax = 0.760129` with match `0.953125`

## Judgement
- status: `ranking_preserved_but_zero_boundary_misaligned`
- reason: `Plaintext full-path class scores stay strongly rank-correlated with static scores, but the zero boundary is badly shifted; sweeping a public threshold on plaintext scores nearly recovers static-level accuracy.`
