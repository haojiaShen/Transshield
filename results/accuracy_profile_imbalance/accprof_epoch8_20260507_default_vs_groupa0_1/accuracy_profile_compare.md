# accuracy_profile_imbalance_epoch8 Pair Compare

## 1. 结论

- status: `candidate_eval_not_worse`
- reason: 候选 run 的 threshold/AUC 没有低于 baseline，可以继续做更长训练或更完整验证。

## 2. 训练指标

- baseline `secure_static_accprof_epoch8_20260506_default1` test_acc1: `78.81679540736074`
- candidate `secure_static_accprof_epoch8_20260507_groupa0_1` test_acc1: `78.81679540736074`
- delta candidate-baseline test_acc1: `0.0`
- delta candidate-baseline train_loss: `0.0001139786778663332`

## 3. 明文评估对照

- threshold_accuracy delta: `0.0`
- auc delta: `0.0003427592116538314`
- argmax_accuracy delta: `0.0`

## 4. 配置差异

- `groupa_lr_scale`: baseline=`0.1` candidate=`0.0`
