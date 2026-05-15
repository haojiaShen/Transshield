# accuracy_profile_imbalance_epoch1 Pair Compare

## 1. 结论

- status: `candidate_eval_not_improved`
- reason: 候选 run 在当前配对比较中没有显示出明确收益，应先收敛训练预算或重新审视目标配置。

## 2. 训练指标

- baseline `secure_static_accprof_epoch1_20260507_aanone_1` test_acc1: `72.13740731741636`
- candidate `secure_static_accprof_epoch1_20260507_aanone_ema1` test_acc1: `72.13740731741636`
- delta candidate-baseline test_acc1: `0.0`
- delta candidate-baseline train_loss: `0.0`

## 3. 明文评估对照

- threshold_accuracy delta: `-0.7633566856384277`
- auc delta: `-0.011615728839379225`
- argmax_accuracy delta: `15.839695930480957`

## 4. 配置差异

- `model_ema`: baseline=`false` candidate=`true`
