# accuracy_profile_imbalance_epoch3 Pair Compare

## 1. 结论

- status: `candidate_eval_not_improved`
- reason: 候选 run 在当前配对比较中没有显示出明确收益，应先收敛训练预算或重新审视目标配置。

## 2. 训练指标

- baseline `secure_static_accprof_epoch3_20260506_default1` test_acc1: `79.38931588908189`
- candidate `secure_static_accprof_epoch3_20260506_freezepatchweight1` test_acc1: `79.38931588908189`
- delta candidate-baseline test_acc1: `0.0`
- delta candidate-baseline train_loss: `2.5349409401798084e-05`

## 3. 明文评估对照

- threshold_accuracy delta: `0.0`
- auc delta: `-0.000304674854803455`
- argmax_accuracy delta: `-0.3816843032836914`

## 4. 配置差异

- `freeze_patch_embed_weight`: baseline=`false` candidate=`true`
