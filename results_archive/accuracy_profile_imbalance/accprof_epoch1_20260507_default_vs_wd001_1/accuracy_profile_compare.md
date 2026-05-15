# accuracy_profile_imbalance_epoch1 Pair Compare

## 1. 结论

- status: `candidate_eval_not_worse`
- reason: 候选 run 的 threshold/AUC 没有低于 baseline，可以继续做更长训练或更完整验证。

## 2. 训练指标

- baseline `secure_static_accprof_epoch1_20260506_default1` test_acc1: `81.4885515984688`
- candidate `secure_static_accprof_epoch1_20260507_wd001_1` test_acc1: `81.4885515984688`
- delta candidate-baseline test_acc1: `0.0`
- delta candidate-baseline train_loss: `-2.007095181166818e-07`

## 3. 明文评估对照

- threshold_accuracy delta: `0.0`
- auc delta: `0.0`
- argmax_accuracy delta: `0.0`

## 4. 配置差异

- `weight_decay`: baseline=`0.05` candidate=`0.01`
