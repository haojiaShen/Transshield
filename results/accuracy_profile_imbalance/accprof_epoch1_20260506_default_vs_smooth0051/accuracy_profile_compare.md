# accuracy_profile_imbalance_epoch1 Pair Compare

## 1. 结论

- status: `candidate_eval_not_improved`
- reason: 候选 run 在当前配对比较中没有显示出明确收益，应先收敛训练预算或重新审视目标配置。

## 2. 训练指标

- baseline `secure_static_accprof_epoch1_20260506_default1` test_acc1: `81.4885515984688`
- candidate `secure_static_accprof_epoch1_20260506_smooth0051` test_acc1: `82.25191061005337`
- delta candidate-baseline test_acc1: `0.7633590115845692`
- delta candidate-baseline train_loss: `-0.010876060951323763`

## 3. 明文评估对照

- threshold_accuracy delta: `0.0`
- auc delta: `-0.00011425307055124012`
- argmax_accuracy delta: `0.7633566856384277`

## 4. 配置差异

- `smoothing`: baseline=`0.1` candidate=`0.05`
