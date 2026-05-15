# runtime_pruning_vs_static_fullval_20260508

## Stage Metrics
- `plaintext_full_model`: argmax_acc=`74.618321`, threshold_acc=`74.236641`, auc=`0.963724650`, ce_loss=`0.479971828`
- `static_whole_forward`: argmax_acc=`76.717557`, threshold_acc=`91.984733`, auc=`0.967875845`, ce_loss=`0.530501223`

## Score Alignment
- `argmax_match_ratio = 0.513359`
- `threshold_match_ratio = 0.730916`
- `score_correlation = 0.962371`
- `same_sign_ratio = 0.513359`
- `affine static_score ~= a * plaintext_score + b`: `a = 1.938300`, `b = -1.509774`, `x_at_y0 = 0.778916`

## Threshold Sweep
- `plaintext zero-threshold accuracy = 74.618321`
- `plaintext best-threshold accuracy = 92.748092` at threshold `0.500032`
- `best threshold to match static argmax = 0.815891` with match `0.942748`

## Judgement
- status: `ranking_related_but_boundary_and_scale_both_shifted`
- reason: `Plaintext and static scores are still correlated, but the boundary/scale drift is large enough that a simple zero-threshold decision is not reliable.`
