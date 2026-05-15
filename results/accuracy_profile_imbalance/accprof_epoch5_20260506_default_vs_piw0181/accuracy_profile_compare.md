# accuracy_profile_imbalance_epoch5 Pair Compare

## 1. 结论

- status: `candidate_eval_not_improved`
- reason: 候选 run 在当前配对比较中没有显示出明确收益，应先收敛训练预算或重新审视目标配置。

## 2. 训练指标

- baseline `secure_static_accprof_epoch5_20260506_default1` test_acc1: `79.58015511782115`
- candidate `secure_static_accprof_epoch5_20260506_piw0181` test_acc1: `76.14504026456643`
- delta candidate-baseline test_acc1: `-3.435114853254717`
- delta candidate-baseline train_loss: `0.014477448804037896`

## 3. 明文评估对照

- threshold_accuracy delta: `-0.19083619117736816`
- auc delta: `-0.0021708083404741174`
- argmax_accuracy delta: `-1.7175555229187012`

## 4. 配置差异

- `class_weight_mode`: baseline=`none` candidate=`power_inverse_freq`
- `class_weight_power`: baseline=`1.0` candidate=`0.18`
