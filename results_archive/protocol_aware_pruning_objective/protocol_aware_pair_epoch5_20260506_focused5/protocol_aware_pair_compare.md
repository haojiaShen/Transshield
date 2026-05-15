# protocol_aware_pruning Pair Compare

## 1. 结论

- status: `no_boundary_relief_yet`
- reason: 候选 run 的 focus stage violation ratio 仍接近 1.0，当前更像是 objective 已接线但尚未缓解边界歧义。

## 2. 训练指标

- baseline `protocol_aware_pair_epoch5_20260506_focused5_baseline` test_acc1: `75.76335898246474`
- candidate `protocol_aware_pair_epoch5_20260506_focused5_focused` test_acc1: `76.14503818249885`
- delta candidate-baseline test_acc1: `0.3816792000341138`
- delta candidate-baseline train_loss: `0.00011712920909023872`

## 3. 明文评估对照

- threshold_accuracy delta: `-0.38167834281921387`
- auc delta: `0.0009330667428353312`
- argmax_accuracy delta: `0.19083619117736816`

## 4. Margin / Boundary

- candidate margin status: `protocol_margin_stats_available`
- focus_stage_index: `1`
- candidate focus_stage_violation_ratio: `1.0`
- candidate focus_stage_margin_mean: `0.0`
- baseline focus_stage_violation_ratio: `None`

## 5. 配置差异

- `pruning_margin_weight`: baseline=`0.0` candidate=`3.0`
- `pruning_margin_target`: baseline=`1e-4` candidate=`3.9e-05`
- `pruning_margin_stage_weights`: baseline=`None` candidate=`1.40,1.90,0.90`
