# distill_compensation Pair Compare

## 1. 结论

- status: `no_clear_distill_benefit_yet`
- reason: 候选 run 虽已观测到有效 distill term，但当前 full-val compare 还没有形成明确收益。

## 2. 训练指标

- baseline `distill_comp_pair_epoch3_20260506_cls_only1_nodistill` test_acc1: `74.23664122137404`
- candidate `distill_comp_pair_epoch3_20260506_cls_only1_cls_only` test_acc1: `75.763358938785`
- delta candidate-baseline test_acc1: `1.5267177174109605`
- delta candidate-baseline train_loss: `0.10691285680751406`

## 3. 明文评估对照

- threshold_accuracy delta: `-0.38167834281921387`
- auc delta: `-0.0015805008092925066`
- argmax_accuracy delta: `0.9541988372802734`

## 4. Distill

- baseline distill status: `distill_disabled_reference`
- candidate distill status: `distill_terms_observed`
- candidate mean_effective_cls_term: `0.06735`
- candidate mean_effective_token_term: `0.0`
- candidate nonzero_effective_distill_line_count: `4`

## 5. 配置差异

- `cls_distill_weight`: baseline=`0.0` candidate=`1.0`
