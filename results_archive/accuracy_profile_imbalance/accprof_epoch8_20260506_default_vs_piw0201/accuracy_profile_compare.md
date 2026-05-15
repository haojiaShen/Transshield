# accuracy_profile_imbalance_epoch8 Pair Compare

## 1. 结论

- status: `candidate_eval_not_improved`
- reason: 候选 run 在当前配对比较中没有显示出明确收益，应先收敛训练预算或重新审视目标配置。

## 2. 训练指标

- baseline `secure_static_accprof_epoch8_20260506_default1` test_acc1: `78.81679540736074`
- candidate `secure_static_accprof_epoch8_20260506_piw0201` test_acc1: `75.19084184952365`
- delta candidate-baseline test_acc1: `-3.625953557837093`
- delta candidate-baseline train_loss: `0.01589037325917453`

## 3. 明文评估对照

- threshold_accuracy delta: `-0.38167834281921387`
- auc delta: `-0.0017709225935447215`
- argmax_accuracy delta: `-1.526719331741333`

## 4. 配置差异

- `class_weight_mode`: baseline=`none` candidate=`power_inverse_freq`
- `class_weight_power`: baseline=`1.0` candidate=`0.20`
