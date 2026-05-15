# accuracy_profile_imbalance_epoch3 Pair Compare

## 1. 结论

- status: `candidate_eval_not_improved`
- reason: 候选 run 在当前配对比较中没有显示出明确收益，应先收敛训练预算或重新审视目标配置。

## 2. 训练指标

- baseline `secure_static_accprof_epoch3_20260506_default1` test_acc1: `79.38931588908189`
- candidate `secure_static_accprof_epoch3_20260506_fixstep1` test_acc1: `79.96183497305135`
- delta candidate-baseline test_acc1: `0.5725190839694676`
- delta candidate-baseline train_loss: `0.0014569333621433644`

## 3. 明文评估对照

- threshold_accuracy delta: `-0.38167834281921387`
- auc delta: `-0.00045701228220518253`
- argmax_accuracy delta: `6.488549709320068`

## 4. 配置差异

- `pretrained_fix_step`: baseline=`0` candidate=`1`
