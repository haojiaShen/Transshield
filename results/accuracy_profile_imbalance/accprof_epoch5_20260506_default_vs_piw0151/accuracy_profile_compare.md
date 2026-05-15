# accuracy_profile_imbalance_epoch5 Pair Compare

## 1. 结论

- status: `candidate_eval_not_improved`
- reason: 候选 run 在当前配对比较中没有显示出明确收益，应先收敛训练预算或重新审视目标配置。

## 2. 训练指标

- baseline `secure_static_accprof_epoch5_20260506_default1` test_acc1: `79.58015511782115`
- candidate `secure_static_accprof_epoch5_20260506_piw0151` test_acc1: `76.90839962558891`
- delta candidate-baseline test_acc1: `-2.6717554922322364`
- delta candidate-baseline train_loss: `0.012168845351861468`

## 3. 明文评估对照

- threshold_accuracy delta: `-0.19083619117736816`
- auc delta: `-0.0020755974483480655`
- argmax_accuracy delta: `-1.7175555229187012`

## 4. 配置差异

- `class_weight_mode`: baseline=`none` candidate=`power_inverse_freq`
- `class_weight_power`: baseline=`1.0` candidate=`0.15`
