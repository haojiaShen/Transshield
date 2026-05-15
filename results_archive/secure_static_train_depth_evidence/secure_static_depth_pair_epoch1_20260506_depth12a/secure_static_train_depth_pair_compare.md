# secure_static_train_depth Pair Compare

## 1. 结论

- status: `no_clear_depth_benefit_yet`
- reason: 当前 depth pair-study 已实现单因子控制，但更深的 secure_static_train_depth 还没有形成明确收益。

## 2. 训练指标

- baseline `secure_static_depth_pair_epoch1_20260506_depth12a_depth0` test_acc1: `78.05343540570207`
- candidate `secure_static_depth_pair_epoch1_20260506_depth12a_depth12` test_acc1: `81.4885515984688`
- delta candidate-baseline test_acc1: `3.4351161927667277`
- delta candidate-baseline train_loss: `-3.380840327471546e-06`

## 3. 明文评估对照

- threshold_accuracy delta: `-1.526719331741333`
- auc delta: `-0.0116728553746549`
- argmax_accuracy delta: `4.007631540298462`

## 4. 配置差异

- `secure_static_train_depth`: baseline=`0` candidate=`12`
