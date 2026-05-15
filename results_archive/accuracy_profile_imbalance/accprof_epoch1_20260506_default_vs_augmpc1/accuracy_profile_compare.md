# accuracy_profile_imbalance_epoch1 Pair Compare

## 1. 结论

- status: `candidate_eval_not_improved`
- reason: 候选 run 在当前配对比较中没有显示出明确收益，应先收敛训练预算或重新审视目标配置。

## 2. 训练指标

- baseline `secure_static_accprof_epoch1_20260506_default1` test_acc1: `81.4885515984688`
- candidate `secure_static_accprof_epoch1_20260506_augmpc1` test_acc1: `87.97710302221866`
- delta candidate-baseline test_acc1: `6.488551423749854`
- delta candidate-baseline train_loss: `-0.021697526075402096`

## 3. 明文评估对照

- threshold_accuracy delta: `-0.5725204944610596`
- auc delta: `-0.0035037608302389556`
- argmax_accuracy delta: `6.488549709320068`

## 4. 配置差异

- `augmentation_profile`: baseline=`timm` candidate=`mpcvit_like`
