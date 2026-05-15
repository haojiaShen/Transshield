# accuracy_profile_imbalance_epoch5 Pair Compare

## 1. 结论

- status: `candidate_eval_not_worse`
- reason: 候选 run 的 threshold/AUC 没有低于 baseline，可以继续做更长训练或更完整验证。

## 2. 训练指标

- baseline `secure_static_accprof_epoch5_20260506_default1` test_acc1: `79.58015511782115`
- candidate `secure_static_accprof_epoch5_20260506_piw0221` test_acc1: `75.1908428978374`
- delta candidate-baseline test_acc1: `-4.389312219983751`
- delta candidate-baseline train_loss: `0.017480974700175178`

## 3. 明文评估对照

- threshold_accuracy delta: `0.0`
- auc delta: `0.0003427592116538314`
- argmax_accuracy delta: `-2.8625965118408203`

## 4. 配置差异

- `class_weight_mode`: baseline=`none` candidate=`power_inverse_freq`
- `class_weight_power`: baseline=`1.0` candidate=`0.22`
