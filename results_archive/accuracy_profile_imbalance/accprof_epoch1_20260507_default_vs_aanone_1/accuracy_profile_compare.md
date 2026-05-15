# accuracy_profile_imbalance_epoch1 Pair Compare

## 1. 结论

- status: `candidate_eval_not_worse`
- reason: 候选 run 的 threshold/AUC 没有低于 baseline，可以继续做更长训练或更完整验证。

## 2. 训练指标

- baseline `secure_static_accprof_epoch1_20260506_default1` test_acc1: `81.4885515984688`
- candidate `secure_static_accprof_epoch1_20260507_aanone_1` test_acc1: `72.13740731741636`
- delta candidate-baseline test_acc1: `-9.351144281052441`
- delta candidate-baseline train_loss: `0.003667243483926219`

## 3. 明文评估对照

- threshold_accuracy delta: `0.9541988372802734`
- auc delta: `0.012605922117490231`
- argmax_accuracy delta: `-9.351146221160889`

## 4. 配置差异

- `aa`: baseline=`rand-m9-mstd0.5-inc1` candidate=``
