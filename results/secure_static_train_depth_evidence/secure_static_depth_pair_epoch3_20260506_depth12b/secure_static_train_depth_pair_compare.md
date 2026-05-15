# secure_static_train_depth Pair Compare

## 1. 结论

- status: `no_clear_depth_benefit_yet`
- reason: 当前 depth pair-study 已实现单因子控制，但更深的 secure_static_train_depth 还没有形成明确收益。

## 2. 训练指标

- baseline `secure_static_depth_pair_epoch3_20260506_depth12b_depth0` test_acc1: `76.33587797907472`
- candidate `secure_static_depth_pair_epoch3_20260506_depth12b_depth12` test_acc1: `79.38931588908189`
- delta candidate-baseline test_acc1: `3.053437910007162`
- delta candidate-baseline train_loss: `5.829901922327352e-06`

## 3. 明文评估对照

- threshold_accuracy delta: `-0.9541988372802734`
- auc delta: `-0.009749595353708451`
- argmax_accuracy delta: `5.534350872039795`

## 4. 配置差异

- `secure_static_train_depth`: baseline=`0` candidate=`12`
