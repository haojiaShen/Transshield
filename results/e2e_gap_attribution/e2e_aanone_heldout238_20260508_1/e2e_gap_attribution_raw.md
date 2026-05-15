# e2e_aanone_heldout238_gap_raw_20260508_1

## Stage Metrics
- `reference_static_plaintext` [logits]: argmax_acc=`86.134454`, threshold_acc=`90.756303`, auc=`0.965115458`, ce_loss=`0.471812465`
- `cpu_static_candidate` [logits]: argmax_acc=`86.134454`, threshold_acc=`90.756303`, auc=`0.965115458`, ce_loss=`0.471812465`
- `spu_static_candidate` [raw_logits_before_output_calibration]: argmax_acc=`86.134454`, threshold_acc=`90.756303`, auc=`0.965044841`, ce_loss=`0.471704888`

## Pairwise Compare
- `reference_vs_cpu`: argmax_match=`1.000000`, threshold_match=`1.000000`, logit_max_abs=`0.000000000`, logit_mean_abs=`0.000000000`
- `cpu_vs_spu`: argmax_match=`1.000000`, threshold_match=`1.000000`, logit_max_abs=`0.004115105`, logit_mean_abs=`0.002251724`
- `reference_vs_spu`: argmax_match=`1.000000`, threshold_match=`1.000000`, logit_max_abs=`0.004115105`, logit_mean_abs=`0.002251724`

## Judgement
- status: `spu_numeric_drift_present_but_decision_negligible`
- reason: `SPU introduces measurable logit drift, but on this subset it does not change argmax or threshold decisions.`
- reference_to_cpu_argmax_accuracy_delta: `0.0`
- cpu_to_spu_argmax_accuracy_delta: `0.0`
- reference_to_spu_argmax_accuracy_delta: `0.0`
- reference_to_cpu_ce_loss_delta: `0.0`
- cpu_to_spu_ce_loss_delta: `-0.00010757671134742353`
- reference_to_spu_ce_loss_delta: `-0.00010757671134742353`
