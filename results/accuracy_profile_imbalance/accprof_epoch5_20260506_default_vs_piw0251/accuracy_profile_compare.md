# accuracy_profile_imbalance_epoch5 Pair Compare

## 1. 结论

- status: `candidate_eval_not_improved`
- reason: 候选 run 在当前配对比较中没有显示出明确收益，应先收敛训练预算或重新审视目标配置。

## 2. 训练指标

- baseline `secure_static_accprof_epoch5_20260506_default1` test_acc1: `79.58015511782115`
- candidate `secure_static_accprof_epoch5_20260506_piw0251` test_acc1: `74.8091630426072`
- delta candidate-baseline test_acc1: `-4.770992075213954`
- delta candidate-baseline train_loss: `0.01967356237424467`

## 3. 明文评估对照

- threshold_accuracy delta: `-0.19083619117736816`
- auc delta: `-0.002227934875749793`
- argmax_accuracy delta: `-2.8625965118408203`

## 4. 配置差异

- `class_weight_mode`: baseline=`none` candidate=`power_inverse_freq`
- `class_weight_power`: baseline=`1.0` candidate=`0.25`
