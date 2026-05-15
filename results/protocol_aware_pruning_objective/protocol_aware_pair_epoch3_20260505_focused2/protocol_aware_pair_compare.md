# protocol_aware_pruning Pair Compare

## 1. 结论

- status: `candidate_margin_not_observed`
- reason: 候选 protocol-aware run 还没有产出可解析的 pruning margin stats。

## 2. 训练指标

- baseline `protocol_aware_pair_epoch3_20260505_focused2_baseline` test_acc1: `76.33587797907472`
- candidate `protocol_aware_pair_epoch3_20260505_focused2_focused` test_acc1: `76.33587797907472`
- delta candidate-baseline test_acc1: `0.0`
- delta candidate-baseline train_loss: `0.0`

## 3. 明文评估对照

- threshold_accuracy delta: `0.0`
- auc delta: `0.0`
- argmax_accuracy delta: `0.0`

## 4. Margin / Boundary

- candidate margin status: `loss_info_present_but_no_stage_margin_stats`
- focus_stage_index: `1`
- candidate focus_stage_violation_ratio: `None`
- candidate focus_stage_margin_mean: `None`
- baseline focus_stage_violation_ratio: `None`

## 5. 配置差异

- `pruning_margin_stage_weights`: baseline=`None` candidate=`1.40,1.90,0.90`
