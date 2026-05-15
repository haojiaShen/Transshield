# accuracy_profile_imbalance_epoch3 Pair Compare

## 1. 结论

- status: `candidate_eval_not_worse`
- reason: 候选 run 的 threshold/AUC 没有低于 baseline，可以继续做更长训练或更完整验证。

## 2. 训练指标

- baseline `secure_static_accprof_epoch3_seed1_20260507_default1` test_acc1: `79.77099504543625`
- candidate `secure_static_accprof_epoch3_seed1_20260507_aanone_1` test_acc1: `72.13740696797844`
- delta candidate-baseline test_acc1: `-7.633588077457816`
- delta candidate-baseline train_loss: `-0.006942505333699334`

## 3. 明文评估对照

- threshold_accuracy delta: `0.7633566856384277`
- auc delta: `0.013405693611349134`
- argmax_accuracy delta: `-7.63358473777771`

## 4. 配置差异

- `aa`: baseline=`rand-m9-mstd0.5-inc1` candidate=``
