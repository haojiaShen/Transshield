# accuracy_profile_imbalance_epoch1 Pair Compare

## 1. 结论

- status: `candidate_eval_not_improved`
- reason: 候选 run 在当前配对比较中没有显示出明确收益，应先收敛训练预算或重新审视目标配置。

## 2. 训练指标

- baseline `secure_static_accprof_epoch1_20260506_default1` test_acc1: `81.4885515984688`
- candidate `secure_static_accprof_epoch1_20260507_aanone_cj0_1` test_acc1: `75.57252118059696`
- delta candidate-baseline test_acc1: `-5.916030417871838`
- delta candidate-baseline train_loss: `-0.027390180801858732`

## 3. 明文评估对照

- threshold_accuracy delta: `0.0`
- auc delta: `-0.002437398838427085`
- argmax_accuracy delta: `-5.916029214859009`

## 4. 配置差异

- `color_jitter`: baseline=`0.4` candidate=`0.0`
- `aa`: baseline=`rand-m9-mstd0.5-inc1` candidate=``
