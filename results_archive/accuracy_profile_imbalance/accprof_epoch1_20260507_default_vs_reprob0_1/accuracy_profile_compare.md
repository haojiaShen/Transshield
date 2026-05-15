# accuracy_profile_imbalance_epoch1 Pair Compare

## 1. 结论

- status: `candidate_eval_not_improved`
- reason: 候选 run 在当前配对比较中没有显示出明确收益，应先收敛训练预算或重新审视目标配置。

## 2. 训练指标

- baseline `secure_static_accprof_epoch1_20260506_default1` test_acc1: `81.4885515984688`
- candidate `secure_static_accprof_epoch1_20260507_reprob0_1` test_acc1: `82.82443039289868`
- delta candidate-baseline test_acc1: `1.335878794429874`
- delta candidate-baseline train_loss: `-0.005304792705847272`

## 3. 明文评估对照

- threshold_accuracy delta: `-0.38167834281921387`
- auc delta: `-0.0038846043987432743`
- argmax_accuracy delta: `1.3358771800994873`

## 4. 配置差异

- `reprob`: baseline=`0.25` candidate=`0.0`
