# accuracy_profile_imbalance_epoch3 Pair Compare

## 1. 结论

- status: `candidate_eval_not_worse`
- reason: 候选 run 的 threshold/AUC 没有低于 baseline，可以继续做更长训练或更完整验证。

## 2. 训练指标

- baseline `secure_static_accprof_epoch3_20260506_default1` test_acc1: `79.38931588908189`
- candidate `secure_static_accprof_epoch3_20260506_piw0201` test_acc1: `76.14504026456643`
- delta candidate-baseline test_acc1: `-3.2442756245154527`
- delta candidate-baseline train_loss: `0.015648308659897414`

## 3. 明文评估对照

- threshold_accuracy delta: `0.0`
- auc delta: `0.0003237170332285322`
- argmax_accuracy delta: `-2.0992398262023926`

## 4. 配置差异

- `class_weight_mode`: baseline=`none` candidate=`power_inverse_freq`
- `class_weight_power`: baseline=`1.0` candidate=`0.20`
