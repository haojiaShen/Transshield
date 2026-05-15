# e2e_plaintext_static_gap_heldout238

## Stage Metrics
- `plaintext_full_model`: argmax_acc=`50.840336`, threshold_acc=`50.000000`, auc=`0.958830591`, ce_loss=`0.614260453`
- `static_whole_forward`: argmax_acc=`86.134454`, threshold_acc=`90.756303`, auc=`0.965115458`, ce_loss=`0.471812465`

## Score Alignment
- `argmax_match_ratio = 0.369748`
- `threshold_match_ratio = 0.533613`
- `score_correlation = 0.965601`
- `same_sign_ratio = 0.369748`
- `affine static_score ~= a * plaintext_score + b`: `a = 1.894506`, `b = -1.474863`, `x_at_y0 = 0.778495`

## Threshold Sweep
- `plaintext zero-threshold accuracy = 50.840336`
- `plaintext best-threshold accuracy = 91.176471` at threshold `0.568822`
- `best threshold to match static argmax = 0.815891` with match `0.945378`

## Judgement
- status: `ranking_preserved_but_zero_boundary_misaligned`
- reason: `Plaintext full-path class scores stay strongly rank-correlated with static scores, but the zero boundary is badly shifted; sweeping a public threshold on plaintext scores nearly recovers static-level accuracy.`
