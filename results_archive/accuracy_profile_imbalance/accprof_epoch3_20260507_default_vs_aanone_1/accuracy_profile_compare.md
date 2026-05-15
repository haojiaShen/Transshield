# accuracy_profile_imbalance_epoch3 Pair Compare

## 1. 结论

- status: `candidate_eval_not_worse`
- reason: 候选 run 的 threshold/AUC 没有低于 baseline，可以继续做更长训练或更完整验证。

## 2. 训练指标

- baseline `secure_static_accprof_epoch3_20260506_default1` test_acc1: `79.38931588908189`
- candidate `secure_static_accprof_epoch3_20260507_aanone_1` test_acc1: `71.37404725751804`
- delta candidate-baseline test_acc1: `-8.015268631563842`
- delta candidate-baseline train_loss: `-0.0014901226069651274`

## 3. 明文评估对照

- threshold_accuracy delta: `1.526719331741333`
- auc delta: `0.015271827097019908`
- argmax_accuracy delta: `-8.206111192703247`

## 4. 配置差异

- `aa`: baseline=`rand-m9-mstd0.5-inc1` candidate=``
