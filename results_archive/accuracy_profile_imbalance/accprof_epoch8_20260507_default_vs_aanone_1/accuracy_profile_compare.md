# accuracy_profile_imbalance_epoch8 Pair Compare

## 1. 结论

- status: `candidate_eval_not_worse`
- reason: 候选 run 的 threshold/AUC 没有低于 baseline，可以继续做更长训练或更完整验证。

## 2. 训练指标

- baseline `secure_static_accprof_epoch8_20260506_default1` test_acc1: `78.81679540736074`
- candidate `secure_static_accprof_epoch8_20260507_aanone_1` test_acc1: `72.70992535307207`
- delta candidate-baseline test_acc1: `-6.106870054288677`
- delta candidate-baseline train_loss: `-0.011154394571473047`

## 3. 明文评估对照

- threshold_accuracy delta: `2.2900760173797607`
- auc delta: `0.02096543844615817`
- argmax_accuracy delta: `-5.725193023681641`

## 4. Public Logit-Bias Calibration

- baseline calibration status: `public_bias_recovers_threshold_argmax`
- candidate calibration status: `public_bias_recovers_threshold_argmax`
- candidate class1_logit_bias: `0.5852264595359804`
- calibrated_argmax_accuracy delta: `2.2900763358778704`
- calibrated_ce_loss delta: `-0.015017706667024455`
- calibrated_auc delta: `0.02096543844615817`

## 5. 配置差异

- `aa`: baseline=`rand-m9-mstd0.5-inc1` candidate=``
