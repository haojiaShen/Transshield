# accuracy_profile_imbalance_epoch1 Pair Compare

## 1. 结论

- status: `candidate_eval_not_worse`
- reason: 候选 run 的 threshold/AUC 没有低于 baseline，可以继续做更长训练或更完整验证。

## 2. 训练指标

- baseline `secure_static_accprof_epoch1_20260506_default1` test_acc1: `81.4885515984688`
- candidate `secure_static_accprof_epoch1_20260506_fixstep1` test_acc1: `88.16794225095792`
- delta candidate-baseline test_acc1: `6.6793906524891185`
- delta candidate-baseline train_loss: `0.01315956334678492`

## 3. 明文评估对照

- threshold_accuracy delta: `0.0`
- auc delta: `0.0008568980291344674`
- argmax_accuracy delta: `6.679391860961914`

## 4. 配置差异

- `pretrained_fix_step`: baseline=`0` candidate=`1`
