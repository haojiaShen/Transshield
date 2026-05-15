# P1 第四项：蒸馏补偿

最后更新：`2026-05-06`

## 1. 当前定位

蒸馏补偿不是新的主创新。

它在当前总路线里的角色是：

- 作为 `secure-friendly` 近似模型的精度回补机制；
- 服务当前 official line：
  - `uniform` attention
  - `fixed_square` activation
  - `secure_static_train_depth`
  - `secret_blockwise_stage + public_calibrated` secret path
- 判断“去掉一部分原始 ViT 语义之后，蒸馏是否能把 full-val 指标拉回可交付区间”。

因此，这一项当前要证明的不是“蒸馏本身很新”，而是：

- 在当前 secure-friendly line 上，
- 蒸馏是否确实带来可复现的长期收益，
- 并且这种收益是否值得保留为正式训练默认值。

## 2. 当前仓里的正式入口

- 训练 wrapper：
  - `artifacts/server_inference_friendly_pack/run_secure_static_distill_train.sh`
- 新增日志解析工具：
  - `tools/transshield_distill_log_report.py`
- 新增成对对照 wrapper：
  - `artifacts/server_inference_friendly_pack/run_distill_compensation_pair_study.sh`
- 对照汇总复用：
  - `tools/transshield_training_pair_compare.py`

## 3. 当前 pair-study 的正式口径

当前蒸馏补偿不再靠手工改两次环境变量，而是固定成一条 paired study：

- baseline：
  - `CLS_DISTILL_WEIGHT=0.0`
  - `TOKEN_DISTILL_WEIGHT=0.0`
- candidate：
  - 默认沿用当前 official line：
    - `CLS_DISTILL_WEIGHT=1.0`
    - `TOKEN_DISTILL_WEIGHT=0.02`

两侧保持相同：

- base bundle
- teacher checkpoint
- train/val dataset
- `secure_static_train_depth`
- `uniform + fixed_square`
- threshold search / eval 口径

因此，这条 compare 的意义是：

- 直接回答“当前官方蒸馏配置，相对 no-distill 参考，到底有没有补偿收益”。

## 4. 日志证据怎么读

`tools/transshield_distill_log_report.py` 会解析训练日志里的：

- `cls_kl=...`
- `token_kl=...`

并结合 `command.sh` 里的：

- `cls_distill_weight`
- `token_distill_weight`

给出：

- `mean_cls_kl`
- `mean_token_kl`
- `mean_effective_cls_term`
- `mean_effective_token_term`
- `nonzero_effective_distill_line_count`

当前希望回答的不是“teacher-student gap 是否存在”，而是：

- 当前 run 是否真的把 distill term 以非零权重放进了总 loss。

## 5. 第一条正式 paired result

当前已经拿到第一条正式对照：

- pair：
  - `results/distillation_compensation/distill_comp_pair_epoch3_20260505_official1/`
- baseline report：
  - `results/distillation_compensation/distill_comp_pair_epoch3_20260505_official1_nodistill/distill_log_report.json`
- candidate report：
  - `results/distillation_compensation/distill_comp_pair_epoch3_20260505_official1_official/distill_log_report.json`
- compare：
  - `results/distillation_compensation/distill_comp_pair_epoch3_20260505_official1/distill_compensation_pair_compare.json`

当前已确认：

- baseline = `distill_disabled_reference`
- candidate = `distill_terms_observed`
- candidate `nonzero_effective_distill_line_count = 4`
- `threshold_accuracy delta = -0.5725 pt`
- `auc delta = -0.00192326`
- `argmax_accuracy delta = +1.9084 pt`
- compare `status = no_clear_distill_benefit_yet`

这说明：

- 当前 official distill 的确已经真实进入 loss；
- 但当前 3-epoch pair-study 还没有形成可以支持升级为正式收益证据的 full-val compare。

进一步的轻量单变量也已经完成：

- pair：
  - `results/distillation_compensation/distill_comp_pair_epoch3_20260506_cls_only1/`
- candidate：
  - `CLS_DISTILL_WEIGHT=1.0`
  - `TOKEN_DISTILL_WEIGHT=0.0`
- 当前读数：
  - `threshold_accuracy delta = -0.3817 pt`
  - `auc delta = -0.00158050`
  - `argmax_accuracy delta = +0.9542 pt`
  - compare `status = no_clear_distill_benefit_yet`

这说明：

- 去掉 token distill 后，负效应相对 `official1` 略有收敛；
- 但 distill 线整体仍未形成明确收益；
- 因而 token distill 不是当前问题的唯一来源。

## 6. 当前仍然不能宣称的东西

即使第一条正式 paired result 已经产出，当前仍然不能宣称：

- 蒸馏已经明确提升了当前 official line 的 threshold/AUC
- 当前 `cls=1.0, token=0.02` 已经是最优蒸馏配置
- token distill 一定比 cls distill 更关键
- 蒸馏收益一定大于 protocol-aware pruning objective 的收益

## 7. 推荐命令

直接跑当前标准对照：

```bash
export PAIR_NAME=distill_comp_pair_epoch3_20260505_official1
export DISTILL_PROFILE=official
export PAIR_EPOCHS=3
export BASELINE_CLS_DISTILL_WEIGHT=0.0
export BASELINE_TOKEN_DISTILL_WEIGHT=0.0
export CANDIDATE_CLS_DISTILL_WEIGHT=1.0
export CANDIDATE_TOKEN_DISTILL_WEIGHT=0.02
bash artifacts/server_inference_friendly_pack/run_distill_compensation_pair_study.sh suite
```

如果之后要测试更强 token distill，可在不改 wrapper 的前提下直接覆盖：

```bash
export DISTILL_PROFILE=token04
export CANDIDATE_TOKEN_DISTILL_WEIGHT=0.04
bash artifacts/server_inference_friendly_pack/run_distill_compensation_pair_study.sh suite
```

## 8. 当前结论

截至 `2026-05-06`，这一项的正式收口是：

- 蒸馏补偿已被收束成当前仓内的正式 paired-study 流程；
- 不再靠手工拼命令；
- 第一条正式 paired result 已经拿到，并且已经证明 official distill 确实接线；
- `cls-only1` 也已说明去掉 token distill 仍不能把这条线转成明确正收益；
- 因此暂不把 distill 直接升级为“已证明更优”的正式默认值；
- 当前默认应暂停继续扩 distill 剂量轴；若还要继续做精度修正，应另起新的更轻量单变量假设，而不是回到既有 `P1-3` 或既有 distill/profile 轴继续加预算。
