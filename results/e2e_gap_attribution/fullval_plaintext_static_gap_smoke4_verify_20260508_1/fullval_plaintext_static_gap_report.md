# fullval_plaintext_static_gap_smoke4_verify_20260508_1

## Stage Metrics
- `plaintext_full_model`: argmax_acc=`0.000000`, threshold_acc=`0.000000`, auc=`nan`, ce_loss=`0.880794134`
- `static_whole_forward`: argmax_acc=`100.000000`, threshold_acc=`50.000000`, auc=`nan`, ce_loss=`0.372004539`

## Score Alignment
- `argmax_match_ratio = 0.000000`
- `threshold_match_ratio = 0.500000`
- `score_correlation = 0.981731`
- `same_sign_ratio = 0.000000`
- `affine static_score ~= a * plaintext_score + b`: `a = 1.496509`, `b = -1.328587`, `x_at_y0 = 0.887791`

## Threshold Sweep
- `plaintext zero-threshold accuracy = 0.000000`
- `plaintext best-threshold accuracy = 100.000000` at threshold `0.535719`
- `best threshold to match static argmax = 0.535719` with match `1.000000`

## Judgement
- status: `ranking_preserved_but_zero_boundary_misaligned`
- reason: `Plaintext full-path class scores stay strongly rank-correlated with static scores, but the zero boundary is badly shifted; sweeping a public threshold on plaintext scores nearly recovers static-level accuracy.`
