# accuracy_profile_imbalance_epoch5 Pair Compare

## 1. 结论

- status: `candidate_eval_not_worse`
- reason: 候选 run 的 threshold/AUC 没有低于 baseline，可以继续做更长训练或更完整验证。

## 2. 训练指标

- baseline `secure_static_accprof_epoch5_20260506_default1` test_acc1: `79.58015511782115`
- candidate `secure_static_accprof_epoch5_20260506_piw0201` test_acc1: `75.95420033695133`
- delta candidate-baseline test_acc1: `-3.6259547808698187`
- delta candidate-baseline train_loss: `0.015990277131398445`

## 3. 明文评估对照

- threshold_accuracy delta: `0.0`
- auc delta: `0.0003808435685043188`
- argmax_accuracy delta: `-2.4809181690216064`

## 4. 配置差异

- `class_weight_mode`: baseline=`none` candidate=`power_inverse_freq`
- `class_weight_power`: baseline=`1.0` candidate=`0.20`
