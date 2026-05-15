# e2e_aanone_heldout238_gap_calibrated_20260508_1

## Stage Metrics
- `reference_static_plaintext` [logits]: argmax_acc=`86.134454`, threshold_acc=`90.756303`, auc=`0.965115458`, ce_loss=`0.471812465`
- `cpu_static_candidate` [logits]: argmax_acc=`86.134454`, threshold_acc=`90.756303`, auc=`0.965115458`, ce_loss=`0.471812465`
- `spu_static_candidate` [logits]: argmax_acc=`92.436975`, threshold_acc=`92.436975`, auc=`0.965044841`, ce_loss=`0.465806137`

## Pairwise Compare
- `reference_vs_cpu`: argmax_match=`1.000000`, threshold_match=`1.000000`, logit_max_abs=`0.000000000`, logit_mean_abs=`0.000000000`
- `cpu_vs_spu`: argmax_match=`0.894958`, threshold_match=`0.932773`, logit_max_abs=`0.363180399`, logit_mean_abs=`0.178705517`
- `reference_vs_spu`: argmax_match=`0.894958`, threshold_match=`0.932773`, logit_max_abs=`0.363180399`, logit_mean_abs=`0.178705517`

## Judgement
- status: `spu_specific_gap_dominant`
- reason: `CPU static and plaintext static whole-forward are effectively identical; the remaining gap is introduced on the SPU-side output path.`
- reference_to_cpu_argmax_accuracy_delta: `0.0`
- cpu_to_spu_argmax_accuracy_delta: `6.30252100840336`
- reference_to_spu_argmax_accuracy_delta: `6.30252100840336`
- reference_to_cpu_ce_loss_delta: `0.0`
- cpu_to_spu_ce_loss_delta: `-0.006006328284406837`
- reference_to_spu_ce_loss_delta: `-0.006006328284406837`
