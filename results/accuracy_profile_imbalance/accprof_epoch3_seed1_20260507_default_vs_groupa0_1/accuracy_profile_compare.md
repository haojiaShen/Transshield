# accuracy_profile_imbalance_epoch3 Pair Compare

## 1. 结论

- status: `candidate_eval_not_improved`
- reason: 候选 run 在当前配对比较中没有显示出明确收益，应先收敛训练预算或重新审视目标配置。

## 2. 训练指标

- baseline `secure_static_accprof_epoch3_seed1_20260507_default1` test_acc1: `79.77099504543625`
- candidate `secure_static_accprof_epoch3_seed1_20260507_groupa0_1` test_acc1: `79.96183427417552`
- delta candidate-baseline test_acc1: `0.1908392287392644`
- delta candidate-baseline train_loss: `3.3033948366290034e-05`

## 3. 明文评估对照

- threshold_accuracy delta: `0.0`
- auc delta: `-3.8084356850376366e-05`
- argmax_accuracy delta: `0.0`

## 4. 配置差异

- `groupa_lr_scale`: baseline=`0.1` candidate=`0.0`
