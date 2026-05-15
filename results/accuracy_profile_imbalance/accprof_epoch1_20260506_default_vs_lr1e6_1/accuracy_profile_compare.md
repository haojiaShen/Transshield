# accuracy_profile_imbalance_epoch1 Pair Compare

## 1. 结论

- status: `candidate_eval_not_improved`
- reason: 候选 run 在当前配对比较中没有显示出明确收益，应先收敛训练预算或重新审视目标配置。

## 2. 训练指标

- baseline `secure_static_accprof_epoch1_20260506_default1` test_acc1: `81.4885515984688`
- candidate `secure_static_accprof_epoch1_20260506_lr1e6_1` test_acc1: `83.01527032051378`
- delta candidate-baseline test_acc1: `1.5267187220449756`
- delta candidate-baseline train_loss: `0.005372145548969698`

## 3. 明文评估对照

- threshold_accuracy delta: `0.0`
- auc delta: `-0.0006855184233076628`
- argmax_accuracy delta: `1.526719331741333`

## 4. 配置差异

- `lr`: baseline=`3e-6` candidate=`1e-6`
