# accuracy_profile_imbalance_epoch5 Pair Compare

## 1. 结论

- status: `candidate_eval_not_improved`
- reason: 候选 run 在当前配对比较中没有显示出明确收益，应先收敛训练预算或重新审视目标配置。

## 2. 训练指标

- baseline `secure_static_accprof_epoch5_20260506_default1` test_acc1: `79.58015511782115`
- candidate `secure_static_accprof_epoch5_20260506_ema1` test_acc1: `80.91603321337517`
- delta candidate-baseline test_acc1: `1.3358780955540226`
- delta candidate-baseline train_loss: `-0.0052561013876986085`

## 3. 明文评估对照

- threshold_accuracy delta: `-0.19083619117736816`
- auc delta: `-0.00022850614110248024`
- argmax_accuracy delta: `5.534350872039795`

## 4. 配置差异

- `model_ema`: baseline=`false` candidate=`true`
