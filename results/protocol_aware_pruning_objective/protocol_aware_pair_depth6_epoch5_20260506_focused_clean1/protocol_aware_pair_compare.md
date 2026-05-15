# protocol_aware_pruning Pair Compare

## 1. 结论

- status: `no_boundary_relief_yet`
- reason: 候选 run 的 focus stage violation ratio 仍接近 1.0，当前更像是 objective 已接线但尚未缓解边界歧义。

## 2. 训练指标

- baseline `protocol_aware_pair_depth6_epoch5_20260506_focused_clean1_baseline` test_acc1: `85.4961867587257`
- candidate `protocol_aware_pair_depth6_epoch5_20260506_focused_clean1_focused` test_acc1: `86.06870514381933`
- delta candidate-baseline test_acc1: `0.5725183850936304`
- delta candidate-baseline train_loss: `0.00011713245288036411`

## 3. 明文评估对照

- threshold_accuracy delta: `-0.1908421516418457`
- auc delta: `-0.0031800437970104234`
- argmax_accuracy delta: `0.3816843032836914`

## 4. Margin / Boundary

- candidate margin status: `protocol_margin_stats_available`
- focus_stage_index: `0`
- candidate focus_stage_violation_ratio: `1.0`
- candidate focus_stage_margin_mean: `5.588000000000001e-09`
- baseline focus_stage_violation_ratio: `None`

## 5. 配置差异

- `pruning_margin_weight`: baseline=`0.0` candidate=`3.0`
- `pruning_margin_target`: baseline=`1e-4` candidate=`3.9e-05`
- `pruning_margin_stage_weights`: baseline=`None` candidate=`1.40,1.90,0.90`
