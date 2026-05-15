# accuracy_profile_imbalance_epoch1 Pair Compare

## 1. 结论

- status: `candidate_eval_not_worse`
- reason: 候选 run 的 threshold/AUC 没有低于 baseline，可以继续做更长训练或更完整验证。

## 2. 训练指标

- baseline `secure_static_accprof_epoch1_20260506_default1` test_acc1: `81.4885515984688`
- candidate `secure_static_accprof_epoch1_20260506_freezepatch1` test_acc1: `81.4885515984688`
- delta candidate-baseline test_acc1: `0.0`
- delta candidate-baseline train_loss: `4.913470372047701e-05`

## 3. 明文评估对照

- threshold_accuracy delta: `0.1908421516418457`
- auc delta: `9.521089212605194e-05`
- argmax_accuracy delta: `0.0`

## 4. 配置差异

- `freeze_patch_embed_proj`: baseline=`false` candidate=`true`
