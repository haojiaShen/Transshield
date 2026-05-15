# P1 第二项：`secure_static_train_depth` 证据化

最后更新：`2026-05-10`

## 1. 当前收口方式

这一项当前也不应被包装成“已经证明 `secure_static_train_depth` 带来稳定正收益”。

当前正式收口方式是：

- 先把 **当前可验证的 deployment-aligned 证据** 固化成正式报告；
- 再把 **缺失的单因子 paired control** 收束成正式 wrapper 与正式 compare；
- 避免把 deployment-oriented 证据误写成 clean causal proof，也避免把第一条 paired result 误写成“已证明收益”。

对应工具与结果：

- 工具：`tools/transshield_secure_static_depth_evidence.py`
- server wrapper：`artifacts/server_inference_friendly_pack/run_secure_static_depth_evidence.sh`
- paired-study wrapper：`artifacts/server_inference_friendly_pack/run_secure_static_depth_pair_study.sh`
- 当前结果：
  - `results/secure_static_train_depth_evidence/secure_static_train_depth_20260505_clean/secure_static_train_depth_evidence.json`
  - `results/secure_static_train_depth_evidence/secure_static_train_depth_20260505_clean/secure_static_train_depth_evidence.md`
  - `results/secure_static_train_depth_evidence/secure_static_train_depth_20260510_full/secure_static_train_depth_evidence.json`
  - `results/secure_static_train_depth_evidence/secure_static_train_depth_20260510_full/secure_static_train_depth_evidence.md`
  - `results/secure_static_train_depth_evidence/secure_static_depth_pair_epoch1_20260506_depth12a/secure_static_train_depth_pair_compare.json`
  - `results/secure_static_train_depth_evidence/secure_static_depth_pair_epoch1_20260506_depth12a/secure_static_train_depth_pair_compare.md`
  - `results/secure_static_train_depth_evidence/secure_static_depth_pair_epoch3_20260506_depth12b/secure_static_train_depth_pair_compare.json`
  - `results/secure_static_train_depth_evidence/secure_static_depth_pair_epoch3_20260506_depth12b/secure_static_train_depth_pair_compare.md`

## 2. 当前已经能支持的说法

当前已经能比较扎实支持的，不是“`secure_static_train_depth` 单独造成了全部收益”，而是：

1. 当前官方 bundle 确实是在一条 **static whole-forward 对齐训练语义** 上训练出来的：
   - `secure_static_train_depth=12`
   - `secure_static_skip_pruning=true`
   - `approx_attn_mode=uniform`
   - `square_activation_mode=fixed_square`

2. 相比当前仓里保留的 baseline depth0 line，当前 official modified line 在 full-val 上有明显提升：
   - `threshold_accuracy +14.6947 pt`
   - `AUC +0.294906`

3. 这条 official modified line 能被当前 deployable secret path 承接：
   - `secret_blockwise_stage + public_calibrated + uniform + fixed_square + clip0 + depth6 + bsz1`
   - guarded secret runtime `complete=true`
   - same-policy replay consistency 仍然较高

也就是说，当前证据足以支持：

> `secure_static_train_depth` 是当前官方 secure-friendly line 的一部分，并且它处在一条已经能被实际 deployable secret path 承接的训练-部署链路里。

## 3. 当前不能支持的说法

当前仍不能诚实支持的说法是：

> “只因为打开了 `secure_static_train_depth`，所以指标就提升了这么多。”

原因很简单：

- 当前 official uplift 不是来自一条历史上就保留下来的 **仅修改 `secure_static_train_depth`、其它保持一致** 的 paired control bundle；
- 当前 baseline 和 official modified line 同时还存在：
  - `uniform attention`
  - `fixed_square`
  - `use_square_gelu`
  - `secure_static_skip_pruning`
  - 训练配置差异

虽然当前仓里已经补上了正式的 depth pair-study wrapper，并且已经拿到第一条 `depth0 vs depth12` 单因子 compare，但这条新 paired result 本身并没有显示出清晰正收益。

所以现在能给出的，是：

- official line 的 **deployment-aligned evidence**
- 以及 depth 单因子控制已经补齐、但 `epoch1/epoch3` 都仍为 `no_clear_depth_benefit_yet` 的 paired evidence

而不是 “`secure_static_train_depth` 已经被 clean causal proof 证明更优”。

## 4. 为什么这个证据仍然有价值

这并不意味着这项 `P1` 没价值。

相反，它把一件重要事情讲清楚了：

- 当前主线不是“先训练一个普通模型，再勉强塞进 secure runtime”；
- 当前 official line 从训练语义开始，就已经朝 static secret whole-forward 的方向靠拢；
- 而且它最终确实形成了：
  - 可评测的 plaintext full-val
  - 可运行的 guarded secret path
  - 可验证的 same-policy consistency

这就使得 `secure_static_train_depth` 不再只是代码参数，而是当前正式交付线的一部分。

## 5. 当前 paired control 已经补到哪里

当前仓里已经把这条对照入口补成正式 wrapper：

- `artifacts/server_inference_friendly_pack/run_secure_static_depth_pair_study.sh`

它的默认口径是：

- baseline：
  - `secure_static_train_depth=0`
- candidate：
  - `secure_static_train_depth=12`
- 其余关键配置保持一致：
  - `uniform`
  - `fixed_square`
  - `secure_static_skip_pruning=true`
  - 同一 teacher / 同一数据 / 同一 distill 配置
  - `cls_distill_weight=1.0`
  - `token_distill_weight=0.02`

它会自动完成：

- baseline / candidate 训练
- threshold search / eval
- plaintext checkpoint eval
- paired compare report

当前已经拿到第一条最小单因子结果：

- pair：`secure_static_depth_pair_epoch1_20260506_depth12a`
- baseline：`secure_static_depth_pair_epoch1_20260506_depth12a_depth0`
- candidate：`secure_static_depth_pair_epoch1_20260506_depth12a_depth12`
- changed keys：只有 `secure_static_train_depth`
- compare judgement：`no_clear_depth_benefit_yet`
- `threshold_accuracy_delta = -1.5267 pt`
- `auc_delta = -0.0116729`
- `argmax_accuracy_delta = +4.0076 pt`

当前也已经拿到第一条更长一点的 follow-up：

- pair：`secure_static_depth_pair_epoch3_20260506_depth12b`
- baseline：`secure_static_depth_pair_epoch3_20260506_depth12b_depth0`
- candidate：`secure_static_depth_pair_epoch3_20260506_depth12b_depth12`
- changed keys：只有 `secure_static_train_depth`
- compare judgement：`no_clear_depth_benefit_yet`
- `threshold_accuracy_delta = -0.9542 pt`
- `auc_delta = -0.0097496`
- `argmax_accuracy_delta = +5.5344 pt`

这条结果能支持的说法是：

- `secure_static_train_depth` 的单因子控制已经补齐；
- compare 工具已经能验证“是否真的只改了 depth”；
- `epoch1` 与 `epoch3` 两条 paired result 的方向保持一致：
  - argmax 上升
  - 但 threshold / AUC 仍为负
- 因此当前不能把更深 depth 写成正式正向结论。

因此，后续这条线不应再靠手工切环境变量拼命令。

截至当前，更合理的正式收口是：

- `P1-2` 已经不再缺 paired control；
- 更深 `secure_static_train_depth` 在当前 `epoch1` 与 `epoch3` 下都没有形成明确收益；
- 这条线当前可以暂时收口，不再把 train-depth 当作“还没验证完的正向增强项”。

除非后续 accuracy-improvement 分支有强理由重开，否则当前不建议继续盲目把同一对照拉到更长 epoch。

## 6. 复现实验命令

直接生成当前证据报告：

```bash
bash artifacts/server_inference_friendly_pack/run_secure_static_depth_evidence.sh
```

跑当前最小 paired control：

```bash
export PAIR_EPOCHS=1
export BASELINE_SECURE_STATIC_DEPTH=0
export CANDIDATE_SECURE_STATIC_DEPTH=12
bash artifacts/server_inference_friendly_pack/run_secure_static_depth_pair_study.sh suite
```

跑当前更有判断力的同口径 follow-up：

```bash
export PAIR_EPOCHS=3
export BASELINE_SECURE_STATIC_DEPTH=0
export CANDIDATE_SECURE_STATIC_DEPTH=12
bash artifacts/server_inference_friendly_pack/run_secure_static_depth_pair_study.sh suite
```

若要显式指定当前 clean 结果：

```bash
cd /data/wyb/Transshield_final
export REPO_ROOT=/data/wyb/Transshield_final
export RUN_NAME=delivery_line_suite_20260505_clean
export BUNDLE_DIR=$REPO_ROOT/artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_20260430
export SECURE_RUN_DIR=$REPO_ROOT/artifacts/server_pipeline_run/$RUN_NAME
export SECRET_RUN_DIR=$REPO_ROOT/artifacts/server_pipeline_run/secret_depth6_clip0_guarded_eval_20260505_clean
export ACCEPTANCE_JSON=$REPO_ROOT/results/delivery_acceptance/delivery_acceptance_20260505_clean/delivery_acceptance_report.json
export OUTPUT_DIR=$REPO_ROOT/results/secure_static_train_depth_evidence/secure_static_train_depth_20260505_clean
bash artifacts/server_inference_friendly_pack/run_secure_static_depth_evidence.sh
```
