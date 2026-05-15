# accuracy_profile_imbalance_epoch1 Pair Compare

## 1. 结论

- status: `candidate_eval_not_improved`
- reason: 候选 run 在当前配对比较中没有显示出明确收益，应先收敛训练预算或重新审视目标配置。

## 2. 训练指标

- baseline `secure_static_accprof_epoch1_20260506_default1` test_acc1: `81.4885515984688`
- candidate `secure_static_accprof_epoch1_20260506_warmup0_1` test_acc1: `81.6793915260839`
- delta candidate-baseline test_acc1: `0.1908399276151016`
- delta candidate-baseline train_loss: `-0.0010491169634319064`

## 3. 明文评估对照

- threshold_accuracy delta: `0.0`
- auc delta: `-0.0005141388174807471`
- argmax_accuracy delta: `0.1908421516418457`

## 4. 配置差异

- `warmup_steps`: baseline=`20` candidate=`0`
