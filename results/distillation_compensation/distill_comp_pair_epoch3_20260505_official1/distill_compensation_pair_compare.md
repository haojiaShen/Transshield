# distill_compensation Pair Compare

## 1. 结论

- status: `no_clear_distill_benefit_yet`
- reason: 候选 run 虽已观测到有效 distill term，但当前 full-val compare 还没有形成明确收益。

## 2. 训练指标

- baseline `distill_comp_pair_epoch3_20260505_official1_nodistill` test_acc1: `74.23664122137404`
- candidate `distill_comp_pair_epoch3_20260505_official1_official` test_acc1: `76.33587797907472`
- delta candidate-baseline test_acc1: `2.0992367577006803`
- delta candidate-baseline train_loss: `0.11587431743031451`

## 3. 明文评估对照

- threshold_accuracy delta: `-0.5725204944610596`
- auc delta: `-0.001923260020946338`
- argmax_accuracy delta: `1.9083976745605469`

## 4. Distill

- baseline distill status: `distill_disabled_reference`
- candidate distill status: `distill_terms_observed`
- candidate mean_effective_cls_term: `0.06755`
- candidate mean_effective_token_term: `0.009122`
- candidate nonzero_effective_distill_line_count: `4`

## 5. 配置差异

- `cls_distill_weight`: baseline=`0.0` candidate=`1.0`
- `token_distill_weight`: baseline=`0.0` candidate=`0.02`
