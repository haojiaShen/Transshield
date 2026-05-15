# accuracy_profile_imbalance_epoch1 Pair Compare

## 1. 结论

- status: `candidate_eval_not_worse`
- reason: 候选 run 的 threshold/AUC 没有低于 baseline，可以继续做更长训练或更完整验证。

## 2. 训练指标

- baseline `secure_static_accprof_epoch1_20260507_aanone_1` test_acc1: `72.13740731741636`
- candidate `secure_static_accprof_epoch1_20260507_aanone_ema1` test_acc1: `72.13740731741636`
- delta candidate-baseline test_acc1: `0.0`
- delta candidate-baseline train_loss: `0.0`

## 3. 明文评估对照

- threshold_accuracy delta: `0.0`
- auc delta: `0.0`
- argmax_accuracy delta: `0.0`

## 4. 配置差异

- `model_ema`: baseline=`false` candidate=`true`
