# `ACCURACY_PROFILE` 不平衡修正短验证

最后更新：`2026-05-06`

## 1. 这条分支的定位

这不是新的主方法分支。

它的定位是：

- 在 **不改主模型语义** 的前提下；
- 用当前 wrapper 已支持的 `ACCURACY_PROFILE`；
- 先验证 train / val 类别不平衡修正，是否能给当前 official secure-friendly line 带来更好的 full-val 指标。

当前固定不变的主线语义仍然是：

- `ViT / DynamicViT`
- `uniform + fixed_square`
- `secure_static_train_depth=12`
- `secure_static_skip_pruning=true`
- `cls_distill_weight=1.0`
- `token_distill_weight=0.02`

也就是说，这条线只改：

- `ACCURACY_PROFILE`

而不改：

- 主模型结构
- pruning 语义
- distill 剂量
- protocol-aware margin objective
- train-depth

## 2. 背景

当前服务器数据分布不平衡：

- train：`class0=1214`，`class1=3494`
- val：`class0=135`，`class1=389`

因此当前最自然的下一条精度修正，不是继续改结构，而是先看：

- sampler 修正
- class-weight 修正

仓内 wrapper 已支持：

- `ACCURACY_PROFILE=weighted_sqrt_sampler`
- `ACCURACY_PROFILE=sqrt_class_weight`
- `run_accuracy_profile_pair_study.sh`
  - 可在不改主模型语义的前提下，对 `class_weight_mode / class_weight_power / train_sampler_mode / model_ema / smoothing` 做正式 paired compare；
  - 因此像 `class_weight_mode=power_inverse_freq` 这类更轻量单变量假设，不需要再手工拼 baseline / candidate / eval / compare。

## 3. 本轮验证内容

### 3.1 `weighted_sqrt_sampler`：`epoch1`

- baseline：
  - `secure_static_accprof_epoch1_20260506_default1`
- candidate：
  - `secure_static_accprof_epoch1_20260506_weightedsqrt1`
- 结果：
  - compare：`results/accuracy_profile_imbalance/accprof_epoch1_20260506_default_vs_weightedsqrt1/accuracy_profile_compare.json`
  - `threshold_accuracy_delta = -0.3817 pt`
  - `auc_delta = -0.004056`
  - `argmax_accuracy_delta = -6.8702 pt`
  - judgement = `candidate_eval_not_improved`

当前结论：

- `weighted_sqrt_sampler` 在 `epoch1` 下已经给出明确负信号；
- 它不适合作为当前默认继续推进的精度修正分支。

### 3.2 `sqrt_class_weight`：`epoch1`

- baseline：
  - `secure_static_accprof_epoch1_20260506_default1`
- candidate：
  - `secure_static_accprof_epoch1_20260506_sqrtcw1`
- 结果：
  - compare：`results/accuracy_profile_imbalance/accprof_epoch1_20260506_default_vs_sqrtcw1/accuracy_profile_compare.json`
  - `threshold_accuracy_delta = 0.0`
  - `auc_delta = +0.000076`
  - `argmax_accuracy_delta = -7.2519 pt`
  - judgement = `candidate_eval_not_worse`

当前结论：

- `sqrt_class_weight` 通过了最小 `epoch1` gate；
- threshold/AUC 没有低于 default；
- 但 argmax 明显回落，因此还不能直接升级成正向结论。

### 3.3 `sqrt_class_weight`：`epoch3`

- baseline：
  - `secure_static_accprof_epoch3_20260506_default1`
- candidate：
  - `secure_static_accprof_epoch3_20260506_sqrtcw1`
- 结果：
  - compare：`results/accuracy_profile_imbalance/accprof_epoch3_20260506_default_vs_sqrtcw1/accuracy_profile_compare.json`
  - `threshold_accuracy_delta = -0.1908 pt`
  - `auc_delta = +0.000571`
  - `argmax_accuracy_delta = -7.2519 pt`
  - judgement = `candidate_eval_not_improved`

当前结论：

- `sqrt_class_weight` 在 `epoch3` 下没有把 `epoch1` 的“勉强不差”转成明确收益；
- 它表现为：
  - AUC 略升
  - 但 threshold_accuracy 回落
  - argmax 仍明显回落

### 3.4 `power_inverse_freq=0.20`：`epoch1`

- baseline：
  - `secure_static_accprof_epoch1_20260506_default1`
- candidate：
  - `secure_static_accprof_epoch1_20260506_piw0201`
- 结果：
  - compare：`results/accuracy_profile_imbalance/accprof_epoch1_20260506_default_vs_piw0201/accuracy_profile_compare.json`
  - `threshold_accuracy_delta = 0.0`
  - `auc_delta = +0.00009521`
  - `argmax_accuracy_delta = -1.7176 pt`
  - judgement = `candidate_eval_not_worse`

当前结论：

- `power_inverse_freq=0.20` 在 `epoch1` 下通过最小 gate；
- 虽然 AUC 增幅略低于 `0.25`，但 argmax 回落明显更小；
- 这说明它可能比 `0.25` 更接近“稳态非劣”配置。

### 3.5 `power_inverse_freq=0.20`：`epoch3`

- baseline：
  - `secure_static_accprof_epoch3_20260506_default1`
- candidate：
  - `secure_static_accprof_epoch3_20260506_piw0201`
- 结果：
  - compare：`results/accuracy_profile_imbalance/accprof_epoch3_20260506_default_vs_piw0201/accuracy_profile_compare.json`
  - `threshold_accuracy_delta = 0.0`
  - `auc_delta = +0.00032372`
  - `argmax_accuracy_delta = -2.0992 pt`
  - judgement = `candidate_eval_not_worse`

当前结论：

- `power_inverse_freq=0.20` 在 `epoch3` 下继续保持 `threshold/AUC` 非劣；
- 相比 `0.25`，它的 argmax 回落仍更小；
- 这使它升级为当前这条轴上的最强轻量候选。

### 3.6 `power_inverse_freq=0.20`：`epoch5`

- baseline：
  - `secure_static_accprof_epoch5_20260506_default1`
- candidate：
  - `secure_static_accprof_epoch5_20260506_piw0201`
- 结果：
  - compare：`results/accuracy_profile_imbalance/accprof_epoch5_20260506_default_vs_piw0201/accuracy_profile_compare.json`
  - `threshold_accuracy_delta = 0.0`
  - `auc_delta = +0.00038084`
  - `argmax_accuracy_delta = -2.4809 pt`
  - judgement = `candidate_eval_not_worse`

当前结论：

- `power_inverse_freq=0.20` 没有像 `0.25` 一样在更长预算下翻负；
- 到 `epoch5` 仍保持 `threshold_accuracy` 持平、`AUC` 小幅更优；
- 这是当前第一条在 `epoch1 / epoch3 / epoch5` 三个预算下都保持 `candidate_eval_not_worse` 的 class-weight 假设。

### 3.7 `power_inverse_freq=0.25`：`epoch1 / epoch3 / epoch5`

- `epoch1`：
  - compare：`results/accuracy_profile_imbalance/accprof_epoch1_20260506_default_vs_piw0251/accuracy_profile_compare.json`
  - `threshold_accuracy_delta = 0.0`
  - `auc_delta = +0.00013330`
  - `argmax_accuracy_delta = -2.6718 pt`
  - judgement = `candidate_eval_not_worse`
- `epoch3`：
  - compare：`results/accuracy_profile_imbalance/accprof_epoch3_20260506_default_vs_piw0251/accuracy_profile_compare.json`
  - `threshold_accuracy_delta = 0.0`
  - `auc_delta = +0.00043797`
  - `argmax_accuracy_delta = -2.8626 pt`
  - judgement = `candidate_eval_not_worse`
- `epoch5`：
  - compare：`results/accuracy_profile_imbalance/accprof_epoch5_20260506_default_vs_piw0251/accuracy_profile_compare.json`
  - `threshold_accuracy_delta = -0.1908 pt`
  - `auc_delta = -0.00222793`
  - `argmax_accuracy_delta = -2.8626 pt`
  - judgement = `candidate_eval_not_improved`

当前结论：

- `power_inverse_freq=0.25` 在 `epoch1 / 3` 的 AUC 略好于 `0.20`；
- 但它在 `epoch5` 已经重新转负；
- 因此它当前更适合作为“更强但不稳定”的对照，不再是主推荐候选。

### 3.8 `power_inverse_freq=0.15`：`epoch5`

- baseline：
  - `secure_static_accprof_epoch5_20260506_default1`
- candidate：
  - `secure_static_accprof_epoch5_20260506_piw0151`
- 结果：
  - compare：`results/accuracy_profile_imbalance/accprof_epoch5_20260506_default_vs_piw0151/accuracy_profile_compare.json`
  - `threshold_accuracy_delta = -0.1908 pt`
  - `auc_delta = -0.00207560`
  - `argmax_accuracy_delta = -1.7176 pt`
  - judgement = `candidate_eval_not_improved`

当前结论：

- `power_inverse_freq=0.15` 在 `epoch5` 下已经给出负信号；
- 这说明继续把 power 往 `0.20` 以下压，并没有换来更稳的主指标；
- 结合 `0.25` 的长程转负，当前已可把 `0.20` 视为这段局部区间里的最佳点。

### 3.9 `power_inverse_freq=0.20`：`epoch8`

- baseline：
  - `secure_static_accprof_epoch8_20260506_default1`
- candidate：
  - `secure_static_accprof_epoch8_20260506_piw0201`
- 结果：
  - compare：`results/accuracy_profile_imbalance/accprof_epoch8_20260506_default_vs_piw0201/accuracy_profile_compare.json`
  - `threshold_accuracy_delta = -0.3817 pt`
  - `auc_delta = -0.00177092`
  - `argmax_accuracy_delta = -1.5267 pt`
  - judgement = `candidate_eval_not_improved`

当前结论：

- `power_inverse_freq=0.20` 虽然在 `epoch1 / 3 / 5` 都保持非劣，但到 `epoch8` 也已经转负；
- 这说明它当前更适合作为**短中预算候选**，而不是可无限继续拉长训练预算的稳定默认配置；
- 因此当前不要再把“继续给 `0.20` 加 epoch”当作默认下一步。

### 3.10 `power_inverse_freq=0.18`：`epoch5`

- baseline：
  - `secure_static_accprof_epoch5_20260506_default1`
- candidate：
  - `secure_static_accprof_epoch5_20260506_piw0181`
- 结果：
  - compare：`results/accuracy_profile_imbalance/accprof_epoch5_20260506_default_vs_piw0181/accuracy_profile_compare.json`
  - `threshold_accuracy_delta = -0.1908 pt`
  - `auc_delta = -0.00217081`
  - `argmax_accuracy_delta = -1.7176 pt`
  - judgement = `candidate_eval_not_improved`

当前结论：

- `0.18` 在 `epoch5` 已经直接转负；
- 这说明把 power 从 `0.20` 继续往下压，并没有换来更稳的 `threshold/AUC`；
- 因此 `0.18` 不能取代 `0.20` 成为新的局部候选。

### 3.11 `power_inverse_freq=0.22`：`epoch5`

- baseline：
  - `secure_static_accprof_epoch5_20260506_default1`
- candidate：
  - `secure_static_accprof_epoch5_20260506_piw0221`
- 结果：
  - compare：`results/accuracy_profile_imbalance/accprof_epoch5_20260506_default_vs_piw0221/accuracy_profile_compare.json`
  - `threshold_accuracy_delta = 0.0`
  - `auc_delta = +0.00034276`
  - `argmax_accuracy_delta = -2.8626 pt`
  - judgement = `candidate_eval_not_worse`

当前结论：

- `0.22` 在 `epoch5` 仍保持非劣；
- 但它并没有超过 `0.20`：
  - `threshold_accuracy` 同样持平；
  - `auc` 增幅略低于 `0.20`；
  - `argmax` 回落反而更大；
- 因此 `0.22` 只能算“邻近可行点”，不能替代 `0.20` 成为 `epoch5` 局部最优。

### 3.12 `power_inverse_freq=0.22`：`epoch8`

- baseline：
  - `secure_static_accprof_epoch8_20260506_default1`
- candidate：
  - `secure_static_accprof_epoch8_20260506_piw0221`
- 结果：
  - compare：`results/accuracy_profile_imbalance/accprof_epoch8_20260506_default_vs_piw0221/accuracy_profile_compare.json`
  - `threshold_accuracy_delta = -0.3817 pt`
  - `auc_delta = -0.00177092`
  - `argmax_accuracy_delta = -1.7176 pt`
  - judgement = `candidate_eval_not_improved`

当前结论：

- `0.22` 到 `epoch8` 同样转负；
- 这意味着 `0.22` 并没有补上 `0.20` 的长预算短板；
- 结合 `0.20 @ epoch8` 的负结果，当前这条 class-weight 邻域里还没有出现稳定的长预算候选。

### 3.13 `MODEL_EMA=true`：`epoch5`

- baseline：
  - `secure_static_accprof_epoch5_20260506_default1`
- candidate：
  - `secure_static_accprof_epoch5_20260506_ema1`
- 评估口径：
  - candidate 使用 `checkpoint-best-ema.pth`
  - checkpoint state dict key 使用 `model_ema`
- 结果：
  - compare：`results/accuracy_profile_imbalance/accprof_epoch5_20260506_default_vs_ema1/accuracy_profile_compare.json`
  - `threshold_accuracy_delta = -0.1908 pt`
  - `auc_delta = -0.00022851`
  - `argmax_accuracy_delta = +5.5344 pt`
  - judgement = `candidate_eval_not_improved`

当前结论：

- EMA 对 `argmax_accuracy` 有明显正向作用；
- 但当前正式主指标仍以 `threshold_accuracy / AUC` 为主，而 EMA 在这两项上没有形成收益；
- 因此 `MODEL_EMA=true` 当前不能直接升级为默认精度修正；
- 这条结果更像是“校准/阈值形态发生变化”，而不是稳定提升 full-val 主指标。

### 3.14 `SMOOTHING=0.05`：`epoch1`

- baseline：
  - `secure_static_accprof_epoch1_20260506_default1`
  - `smoothing = 0.1`
- candidate：
  - `secure_static_accprof_epoch1_20260506_smooth0051`
  - `smoothing = 0.05`
- 结果：
  - compare：`results/accuracy_profile_imbalance/accprof_epoch1_20260506_default_vs_smooth0051/accuracy_profile_compare.json`
  - `threshold_accuracy_delta = 0.0 pt`
  - `auc_delta = -0.00011425`
  - `argmax_accuracy_delta = +0.7634 pt`
  - `eval_loss_delta = -0.00757`
  - judgement = `candidate_eval_not_improved`

当前结论：

- 降低 smoothing 对 `argmax_accuracy` 和 `eval_loss` 有轻微改善；
- 但正式主指标 `threshold_accuracy` 持平、`AUC` 略降；
- 因此 `SMOOTHING=0.05` 只能记为校准形态变化，当前不能升级为默认 accuracy fix。

### 3.15 `GROUPA_LR_SCALE=1.0`：`epoch1`

- baseline：
  - `secure_static_accprof_epoch1_20260506_default1`
  - `groupa_lr_scale = 0.1`
- candidate：
  - `secure_static_accprof_epoch1_20260506_groupa1x1`
  - `groupa_lr_scale = 1.0`
- 结果：
  - compare：`results/accuracy_profile_imbalance/accprof_epoch1_20260506_default_vs_groupa1x1/accuracy_profile_compare.json`
  - `threshold_accuracy_delta = 0.0 pt`
  - `auc_delta = +0.00047605`
  - `argmax_accuracy_delta = +0.1908 pt`
  - `eval_loss_delta = +0.00143`
  - judgement = `candidate_eval_not_worse`

当前结论：

- 这是当前新的非 class-weight 单变量里第一条正式主指标非劣且 AUC 改善的信号；
- 但 `threshold_accuracy` 仍未提升，且 `eval_loss` 略升；
- 因此它只能进入 `epoch3` 验证，不能直接升级为默认配置。

### 3.16 `GROUPA_LR_SCALE=1.0`：`epoch3`

- baseline：
  - `secure_static_accprof_epoch3_20260506_default1`
  - `groupa_lr_scale = 0.1`
- candidate：
  - `secure_static_accprof_epoch3_20260506_groupa1x1`
  - `groupa_lr_scale = 1.0`
- 结果：
  - compare：`results/accuracy_profile_imbalance/accprof_epoch3_20260506_default_vs_groupa1x1/accuracy_profile_compare.json`
  - `threshold_accuracy_delta = 0.0 pt`
  - `auc_delta = -0.00238027`
  - `argmax_accuracy_delta = 0.0 pt`
  - `eval_loss_delta = -0.00083`
  - judgement = `candidate_eval_not_improved`

当前结论：

- `GROUPA_LR_SCALE=1.0` 的 `epoch1` AUC 改善没有延续到 `epoch3`；
- 到 `epoch3` 时 threshold 仍持平，AUC 明显转负；
- 因此这条轴不应继续追加更长 epoch，也不能升级为默认配置。

### 3.17 `CLS_TOKEN_FULL_LR=true`：`epoch1`

- baseline：
  - `secure_static_accprof_epoch1_20260506_default1`
  - `cls_token_full_lr = false`
- candidate：
  - `secure_static_accprof_epoch1_20260506_clsfull1`
  - `cls_token_full_lr = true`
- 结果：
  - compare：`results/accuracy_profile_imbalance/accprof_epoch1_20260506_default_vs_clsfull1/accuracy_profile_compare.json`
  - `threshold_accuracy_delta = 0.0 pt`
  - `auc_delta = 0.0`
  - `argmax_accuracy_delta = 0.0 pt`
  - `eval_loss_delta = -0.00000143`
  - judgement = `candidate_eval_not_worse`

当前结论：

- 只放开 `cls_token` 学习率几乎不改变 full-val 指标；
- 这说明 `GROUPA_LR_SCALE=1.0` 的 `epoch1` 小幅 AUC 正信号不是单独来自 `cls_token`；
- 因此这条轴不应继续追加更长 epoch。

### 3.18 `TRAIN_POS_EMBED=true`：`epoch1`

- baseline：
  - `secure_static_accprof_epoch1_20260506_default1`
  - `train_pos_embed = false`
- candidate：
  - `secure_static_accprof_epoch1_20260506_trainpos1`
  - `train_pos_embed = true`
- 结果：
  - compare：`results/accuracy_profile_imbalance/accprof_epoch1_20260506_default_vs_trainpos1/accuracy_profile_compare.json`
  - `threshold_accuracy_delta = 0.0 pt`
  - `auc_delta = 0.0`
  - `argmax_accuracy_delta = 0.0 pt`
  - `eval_loss_delta = -0.00001776`
  - judgement = `candidate_eval_not_worse`

当前结论：

- 只让位置编码参与微调基本不改变 full-val 指标；
- 这条结果可以回应“位置编码是否有可直接收益”：当前训练口径下没有观察到主指标收益；
- 因此这条轴不应继续追加更长 epoch。

### 3.19 `PRETRAINED_FIX_STEP=1`：`epoch1 / epoch3`

`epoch1`：

- baseline：
  - `secure_static_accprof_epoch1_20260506_default1`
  - `pretrained_fix_step = 0`
- candidate：
  - `secure_static_accprof_epoch1_20260506_fixstep1`
  - `pretrained_fix_step = 1`
- 结果：
  - compare：`results/accuracy_profile_imbalance/accprof_epoch1_20260506_default_vs_fixstep1/accuracy_profile_compare.json`
  - `threshold_accuracy_delta = 0.0 pt`
  - `auc_delta = +0.00085690`
  - `argmax_accuracy_delta = +6.6794 pt`
  - `eval_loss_delta = +0.01693`
  - judgement = `candidate_eval_not_worse`

`epoch3`：

- baseline：
  - `secure_static_accprof_epoch3_20260506_default1`
  - `pretrained_fix_step = 0`
- candidate：
  - `secure_static_accprof_epoch3_20260506_fixstep1`
  - `pretrained_fix_step = 1`
- 结果：
  - compare：`results/accuracy_profile_imbalance/accprof_epoch3_20260506_default_vs_fixstep1/accuracy_profile_compare.json`
  - `threshold_accuracy_delta = -0.3817 pt`
  - `auc_delta = -0.00045701`
  - `argmax_accuracy_delta = +6.4885 pt`
  - `eval_loss_delta = +0.02004`
  - judgement = `candidate_eval_not_improved`

当前结论：

- `PRETRAINED_FIX_STEP=1` 在 `epoch1` 下确实给出过很强的 argmax 正信号，且 AUC 小幅更优；
- 但它同时让 `eval_loss` 变差，说明校准/概率形态没有稳定改善；
- 到 `epoch3` 时正式主指标 `threshold_accuracy / AUC` 均转负；
- 因此这条轴不应继续追加更长 epoch，也不能升级为默认 accuracy fix。

### 3.20 `LR=1e-6`：`epoch1`

- baseline：
  - `secure_static_accprof_epoch1_20260506_default1`
  - `lr = 3e-6`
- candidate：
  - `secure_static_accprof_epoch1_20260506_lr1e6_1`
  - `lr = 1e-6`
- 结果：
  - compare：`results/accuracy_profile_imbalance/accprof_epoch1_20260506_default_vs_lr1e6_1/accuracy_profile_compare.json`
  - `threshold_accuracy_delta = 0.0 pt`
  - `auc_delta = -0.00068552`
  - `argmax_accuracy_delta = +1.5267 pt`
  - `eval_loss_delta = +0.01210`
  - judgement = `candidate_eval_not_improved`

当前结论：

- 降低整体 LR 可以改善 argmax，但没有改善正式主指标；
- AUC 与 eval loss 同时变差，说明这不是更稳的校准方向；
- 因此 `LR=1e-6` 不应继续追加 `epoch3`，也不能升级为默认 accuracy fix。

### 3.21 `WARMUP_STEPS=0`：`epoch1`

- baseline：
  - `secure_static_accprof_epoch1_20260506_default1`
  - `warmup_steps = 20`
- candidate：
  - `secure_static_accprof_epoch1_20260506_warmup0_1`
  - `warmup_steps = 0`
- 结果：
  - compare：`results/accuracy_profile_imbalance/accprof_epoch1_20260506_default_vs_warmup0_1/accuracy_profile_compare.json`
  - `threshold_accuracy_delta = 0.0 pt`
  - `auc_delta = -0.00051414`
  - `argmax_accuracy_delta = +0.1908 pt`
  - `eval_loss_delta = -0.000794`
  - judgement = `candidate_eval_not_improved`

当前结论：

- 去掉 warmup 能让 eval loss 小幅改善；
- 但正式主指标仍是 `threshold_accuracy` 持平、`AUC` 转负；
- 因此 `WARMUP_STEPS=0` 不应继续追加 `epoch3`，也不能升级为默认 accuracy fix。

### 3.22 `AUGMENTATION_PROFILE=mpcvit_like`：`epoch1`

- baseline：
  - `secure_static_accprof_epoch1_20260506_default1`
  - `augmentation_profile = timm`
- candidate：
  - `secure_static_accprof_epoch1_20260506_augmpc1`
  - `augmentation_profile = mpcvit_like`
- 结果：
  - compare：`results/accuracy_profile_imbalance/accprof_epoch1_20260506_default_vs_augmpc1/accuracy_profile_compare.json`
  - `threshold_accuracy_delta = -0.5725 pt`
  - `auc_delta = -0.00350376`
  - `argmax_accuracy_delta = +6.4885 pt`
  - `eval_loss_delta = -0.05379`
  - judgement = `candidate_eval_not_improved`

当前结论：

- 去掉 `timm` 重增强后，argmax 和 eval loss 明显改善；
- 但正式主指标 `threshold_accuracy / AUC` 同时转负，说明概率排序/阈值校准变差；
- 因此 `AUGMENTATION_PROFILE=mpcvit_like` 不应继续追加 `epoch3`，也不能升级为默认 accuracy fix。

### 3.23 `FREEZE_PATCH_EMBED_PROJ=true`：`epoch1 / epoch3`

这条轴用于回应“词向量嵌入 / patch embedding 是否有优化空间”的问题。当前只做 plaintext 训练侧的单变量验证，不改变 secure runtime 语义，也不把它上升为新的 OpenBumbleBee/SPU 协议优化。

`epoch1`：

- baseline：
  - `secure_static_accprof_epoch1_20260506_default1`
  - `freeze_patch_embed_proj = false`
- candidate：
  - `secure_static_accprof_epoch1_20260506_freezepatch1`
  - `freeze_patch_embed_proj = true`
- 结果：
  - compare：`results/accuracy_profile_imbalance/accprof_epoch1_20260506_default_vs_freezepatch1/accuracy_profile_compare.json`
  - `threshold_accuracy_delta = +0.1908 pt`
  - `auc_delta = +0.00009521`
  - `argmax_accuracy_delta = 0.0 pt`
  - `eval_loss_delta = +0.000794`
  - judgement = `candidate_eval_not_worse`

`epoch3`：

- baseline：
  - `secure_static_accprof_epoch3_20260506_default1`
  - `freeze_patch_embed_proj = false`
- candidate：
  - `secure_static_accprof_epoch3_20260506_freezepatch1`
  - `freeze_patch_embed_proj = true`
- 结果：
  - compare：`results/accuracy_profile_imbalance/accprof_epoch3_20260506_default_vs_freezepatch1/accuracy_profile_compare.json`
  - `threshold_accuracy_delta = 0.0 pt`
  - `auc_delta = -0.00030467`
  - `argmax_accuracy_delta = -0.3817 pt`
  - `eval_loss_delta = +0.001249`
  - judgement = `candidate_eval_not_improved`

当前结论：

- 冻结 patch embedding projection 在 `epoch1` 下给出过 `threshold/AUC` 短信号；
- 但 `epoch3` 下 AUC、argmax 和 eval loss 都转差，短信号没有延续；
- 因此这条轴只能记录为“embedding 侧已做过最小训练验证，但未形成稳定收益”，不能升级成默认 accuracy fix。

### 3.24 `FREEZE_PATCH_EMBED_WEIGHT=true`：`epoch1 / epoch3`

这条轴用于拆分 `FREEZE_PATCH_EMBED_PROJ=true` 的短正信号来源，只冻结 `patch_embed.proj.weight`，不冻结 bias。

`epoch1`：

- candidate：
  - `secure_static_accprof_epoch1_20260506_freezepatchweight1`
- 结果：
  - compare：`results/accuracy_profile_imbalance/accprof_epoch1_20260506_default_vs_freezepatchweight1/accuracy_profile_compare.json`
  - `threshold_accuracy_delta = +0.1908 pt`
  - `auc_delta = +0.00009521`
  - `argmax_accuracy_delta = 0.0 pt`
  - `eval_loss_delta = +0.000793`
  - judgement = `candidate_eval_not_worse`

`epoch3`：

- candidate：
  - `secure_static_accprof_epoch3_20260506_freezepatchweight1`
- 结果：
  - compare：`results/accuracy_profile_imbalance/accprof_epoch3_20260506_default_vs_freezepatchweight1/accuracy_profile_compare.json`
  - `threshold_accuracy_delta = 0.0 pt`
  - `auc_delta = -0.00030467`
  - `argmax_accuracy_delta = -0.3817 pt`
  - `eval_loss_delta = +0.001250`
  - judgement = `candidate_eval_not_improved`

当前结论：

- `FREEZE_PATCH_EMBED_WEIGHT=true` 基本复现了 full projection freeze 的走势；
- 因此 `FREEZE_PATCH_EMBED_PROJ=true` 的 epoch1 短正信号主要来自 weight，而不是 bias；
- 但它同样没有延续到 `epoch3`，不能升级为默认 accuracy fix。

### 3.25 `FREEZE_PATCH_EMBED_BIAS=true`：`epoch1`

这条轴只冻结 `patch_embed.proj.bias`。

- candidate：
  - `secure_static_accprof_epoch1_20260506_freezepatchbias1`
- 结果：
  - compare：`results/accuracy_profile_imbalance/accprof_epoch1_20260506_default_vs_freezepatchbias1/accuracy_profile_compare.json`
  - `threshold_accuracy_delta = 0.0 pt`
  - `auc_delta = 0.0`
  - `argmax_accuracy_delta = 0.0 pt`
  - `eval_loss_delta = +0.00000077`
  - judgement = `candidate_eval_not_worse`

当前结论：

- 单独冻结 patch embedding bias 基本等同 baseline；
- 它不是 `FREEZE_PATCH_EMBED_PROJ=true` 短正信号的来源；
- 因此不需要继续补 `epoch3`。

### 3.26 `PATCH_EMBED_BIAS_INIT_MODE=zero`：`epoch1`

这条轴用于判断是否需要保留预训练 patch embedding bias。它不冻结参数，只把 student 的 `patch_embed.proj.bias` 从预训练值改为 zero init。

- candidate：
  - `secure_static_accprof_epoch1_20260506_patchbiaszero1`
- 结果：
  - compare：`results/accuracy_profile_imbalance/accprof_epoch1_20260506_default_vs_patchbiaszero1/accuracy_profile_compare.json`
  - `threshold_accuracy_delta = 0.0 pt`
  - `auc_delta = -0.00001904`
  - `argmax_accuracy_delta = 0.0 pt`
  - `eval_loss_delta = -0.00000542`
  - judgement = `candidate_eval_not_improved`

当前结论：

- zero init patch embedding bias 对 loss 有极小改善；
- 但正式主指标没有改善，AUC 还略降；
- 因此当前仍应保留预训练 patch embedding bias，不继续补 `epoch3`。

### 3.27 `BATCH_SIZE=16`：`epoch1`

这条轴用于验证更小 batch 是否能改善小数据集上的优化噪声和校准形态。它不改模型结构、secure runtime 或 distill/pruning 语义。

- baseline：
  - `secure_static_accprof_epoch1_20260506_default1`
  - `batch_size = 32`
- candidate：
  - `secure_static_accprof_epoch1_20260507_bsz16_1`
  - `batch_size = 16`
- 结果：
  - compare：`results/accuracy_profile_imbalance/accprof_epoch1_20260507_default_vs_bsz16_1/accuracy_profile_compare.json`
  - `threshold_accuracy_delta = 0.0 pt`
  - `auc_delta = -0.00306579`
  - `argmax_accuracy_delta = +0.5725 pt`
  - `eval_loss_delta = -0.010924`
  - judgement = `candidate_eval_not_improved`

当前结论：

- 更小 batch 明显改善了 loss，并小幅改善 argmax；
- 但 AUC 明显下降，说明概率排序/阈值校准变差；
- 因此 `BATCH_SIZE=16` 当前不能作为正式 accuracy fix，不继续补 `epoch3`。

### 3.28 `WEIGHT_DECAY=0.01`：`epoch1`

这条轴用于验证降低 AdamW weight decay 是否能改善 fine-tune 校准。它不改模型结构、secure runtime 或 distill/pruning 语义。

- baseline：
  - `secure_static_accprof_epoch1_20260506_default1`
  - `weight_decay = 0.05`
- candidate：
  - `secure_static_accprof_epoch1_20260507_wd001_1`
  - `weight_decay = 0.01`
- 结果：
  - compare：`results/accuracy_profile_imbalance/accprof_epoch1_20260507_default_vs_wd001_1/accuracy_profile_compare.json`
  - `threshold_accuracy_delta = 0.0 pt`
  - `auc_delta = 0.0`
  - `argmax_accuracy_delta = 0.0 pt`
  - `eval_loss_delta = -0.00000221`
  - judgement = `candidate_eval_not_worse`

当前结论：

- `WEIGHT_DECAY=0.01` 几乎完全等同 baseline；
- loss 的改善只有 `2e-6` 量级，不能视为有效信号；
- 因此这条轴不继续补 `epoch3`。

### 3.29 `CLIP_GRAD=2.0`：`epoch1`

这条轴用于验证放宽 gradient clipping 是否能改善 fine-tune 更新。它不改模型结构、secure runtime 或 distill/pruning 语义。

- baseline：
  - `secure_static_accprof_epoch1_20260506_default1`
  - `clip_grad = 1.0`
- candidate：
  - `secure_static_accprof_epoch1_20260507_clip2_1`
  - `clip_grad = 2.0`
- 结果：
  - compare：`results/accuracy_profile_imbalance/accprof_epoch1_20260507_default_vs_clip2_1/accuracy_profile_compare.json`
  - `threshold_accuracy_delta = 0.0 pt`
  - `auc_delta = -0.00083786`
  - `argmax_accuracy_delta = +0.1908 pt`
  - `eval_loss_delta = +0.00004053`
  - judgement = `candidate_eval_not_improved`

当前结论：

- 放宽 gradient clipping 带来很小的 argmax 改善；
- 但 AUC 和 loss 都转差，正式主指标没有改善；
- 因此这条轴不继续补 `epoch3`。

### 3.30 `GROUPA_LR_SCALE=0.0`：`epoch1 / epoch3 / epoch5 / epoch8`

这条轴用于验证更保守地冻结 GroupA 更新，是否能减少小数据集 fine-tune 漂移。它不改模型结构、secure runtime 或 distill/pruning 语义。

- baseline：
  - `secure_static_accprof_epoch{1,3,5,8}_20260506_default1`
  - `groupa_lr_scale = 0.1`
- candidate：
  - `secure_static_accprof_epoch{1,3,5,8}_20260507_groupa0_1`
  - `groupa_lr_scale = 0.0`
- 结果：
  - `epoch1`：`threshold_accuracy_delta = +0.1908 pt`，`auc_delta = +0.00009521`，`argmax_accuracy_delta = +0.1908 pt`，`eval_loss_delta = -0.00008118`
  - `epoch3`：`threshold_accuracy_delta = 0.0 pt`，`auc_delta = +0.00028563`，`argmax_accuracy_delta = 0.0 pt`，`eval_loss_delta = +0.00000417`
  - `epoch5`：`threshold_accuracy_delta = 0.0 pt`，`auc_delta = +0.00032372`，`argmax_accuracy_delta = -0.3817 pt`，`eval_loss_delta = +0.00006258`
  - `epoch8`：`threshold_accuracy_delta = 0.0 pt`，`auc_delta = +0.00034276`，`argmax_accuracy_delta = 0.0 pt`，`eval_loss_delta = +0.00008100`
  - judgement 全部为 `candidate_eval_not_worse`

当前结论：

- `GROUPA_LR_SCALE=0.0` 是目前少数在 `epoch1 / epoch3 / epoch5 / epoch8` 都保持 threshold 非劣且 AUC 小幅正向的单变量；
- 但它没有稳定提升 threshold，且 `epoch5/epoch8` 的 loss 小幅转差，`epoch5` 的 argmax 也转负；
- 因此当前只能把它记为“AUC/calibration 候选”，不能直接升级成全面更优的正式默认值；
- 如果继续做精度提升，下一步比继续扩同一 seed 更合理的是做独立 seed paired check 或更接近正式 bundle 的复现实验。

### 3.31 `GROUPA_LR_SCALE=0.0`：`seed1` `epoch3`

这条轴是对上面候选的独立 seed 复现，用来检查正向信号是否只是一条 seed 噪声。

- baseline：
  - `secure_static_accprof_epoch3_seed1_20260507_default1`
  - `groupa_lr_scale = 0.1`
  - `seed = 1`
- candidate：
  - `secure_static_accprof_epoch3_seed1_20260507_groupa0_1`
  - `groupa_lr_scale = 0.0`
  - `seed = 1`
- 结果：
  - compare：`results/accuracy_profile_imbalance/accprof_epoch3_seed1_20260507_default_vs_groupa0_1/accuracy_profile_compare.json`
  - `threshold_accuracy_delta = 0.0 pt`
  - `auc_delta = -3.8084e-05`
  - `argmax_accuracy_delta = 0.0 pt`
  - `eval_loss_delta = +0.00009984`
  - judgement = `candidate_eval_not_improved`

当前结论：

- 这个独立 seed 复现没有延续原先的 AUC 小正信号；
- 因此 `GROUPA_LR_SCALE=0.0` 仍不能直接升级为稳定默认值；
- 当前更准确的表述是：它是一个 seed 敏感的 AUC/calibration 候选，而不是已经证明稳健更优的正式默认值。

## 4. 当前正式判断

截至 `2026-05-07`，这条 `ACCURACY_PROFILE` 不平衡修正轴的正式结论是：

1. `weighted_sqrt_sampler`
   - `epoch1` 已经可以判负；
   - 当前不要继续给它更多训练预算。

2. `sqrt_class_weight`
   - `epoch1` 只是通过最小 gate；
   - `epoch3` 仍未形成明确收益；
   - 当前不应把它写成“已证明更优”的 accuracy fix。

3. `power_inverse_freq=0.15`
   - `epoch5` 已经转负；
   - 当前不值得再补 `epoch1 / epoch3`。

4. `power_inverse_freq=0.18`
   - `epoch5` 已经转负；
   - 当前不能作为 `0.20` 的更稳替代。

5. `power_inverse_freq=0.20`
   - `epoch1 / epoch3 / epoch5` 全部保持 `candidate_eval_not_worse`；
   - `epoch5` 仍是当前已测区间里的局部最优点；
   - 但 `epoch8` 已重新转负；
   - 当前只能把它写成“截至 `epoch5` 的最佳候选”，还不能写成稳定长预算默认配置。

6. `power_inverse_freq=0.22`
   - `epoch5` 仍保持 `candidate_eval_not_worse`；
   - 但没有超过 `0.20`，且 `argmax` 回落更大；
   - `epoch8` 同样重新转负；
   - 当前只能写成“邻近可行但不优于 `0.20` 的点”，不能写成新的长预算候选。

7. `power_inverse_freq=0.25`
   - `epoch1` 与 `epoch3` 都通过了最小 gate；
   - 但 `epoch5` 重新转负；
   - 当前不能作为稳定候选继续推进。

8. `MODEL_EMA=true`
   - `epoch5` 下 `argmax_accuracy` 明显提升；
   - 但 `threshold_accuracy / AUC` 没有提升；
   - 当前不能作为正式默认精度修正。

9. `SMOOTHING=0.05`
   - `epoch1` 下 `argmax_accuracy` 与 `eval_loss` 改善；
   - 但 `threshold_accuracy` 持平、`AUC` 略降；
   - 当前不能作为正式默认精度修正。

10. `GROUPA_LR_SCALE=1.0`
   - `epoch1` 下 `threshold_accuracy` 持平、`AUC` 小幅改善、`argmax_accuracy` 小幅改善；
   - `epoch3` 下 `threshold_accuracy` 仍持平，但 `AUC` 明显转负；
   - 当前不能作为稳定候选继续推进。

11. `CLS_TOKEN_FULL_LR=true`
   - `epoch1` 下正式主指标和 argmax 完全持平；
   - 当前不能作为稳定候选继续推进。

12. `TRAIN_POS_EMBED=true`
   - `epoch1` 下正式主指标和 argmax 完全持平；
   - 当前不能作为稳定候选继续推进。

13. `PRETRAINED_FIX_STEP=1`
   - `epoch1` 下 `threshold_accuracy` 持平、`AUC` 小幅改善、`argmax_accuracy` 大幅改善；
   - 但 `eval_loss` 变差，且 `epoch3` 下 `threshold_accuracy / AUC` 已转负；
   - 当前不能作为稳定候选继续推进。

14. `LR=1e-6`
   - `epoch1` 下 `threshold_accuracy` 持平、`argmax_accuracy` 上升；
   - 但 `AUC` 与 `eval_loss` 同时变差；
   - 当前不能作为稳定候选继续推进。

15. `WARMUP_STEPS=0`
   - `epoch1` 下 `eval_loss` 小幅改善；
   - 但 `threshold_accuracy` 持平、`AUC` 下降；
   - 当前不能作为稳定候选继续推进。

16. `AUGMENTATION_PROFILE=mpcvit_like`
   - `epoch1` 下 `argmax_accuracy` 与 `eval_loss` 明显改善；
   - 但 `threshold_accuracy / AUC` 明显下降；
   - 当前不能作为稳定候选继续推进。

17. `FREEZE_PATCH_EMBED_PROJ=true`
   - `epoch1` 下 `threshold_accuracy` 与 `AUC` 有短正信号；
   - 但 `epoch3` 下 `AUC / argmax / eval_loss` 转差；
   - 当前只能作为“patch embedding 训练侧已验证过”的证据，不能升级为默认精度修正。

18. `FREEZE_PATCH_EMBED_WEIGHT=true`
   - 复现了 full projection freeze 的短正信号；
   - 但 `epoch3` 同样转为 `AUC / argmax / eval_loss` 变差；
   - 当前不能作为稳定候选继续推进。

19. `FREEZE_PATCH_EMBED_BIAS=true`
   - `epoch1` 下几乎完全等同 baseline；
   - 当前不需要继续补 `epoch3`。

20. `PATCH_EMBED_BIAS_INIT_MODE=zero`
   - `epoch1` 下 loss 极小改善；
   - 但 `threshold_accuracy` 持平、`AUC` 略降；
   - 当前不作为稳定候选继续推进。

21. `BATCH_SIZE=16`
   - `epoch1` 下 loss 和 argmax 改善；
   - 但 AUC 明显下降；
   - 当前不作为稳定候选继续推进。

22. `WEIGHT_DECAY=0.01`
   - `epoch1` 下正式主指标完全持平；
   - loss 只出现微小变化；
   - 当前不作为稳定候选继续推进。

23. `CLIP_GRAD=2.0`
   - `epoch1` 下 argmax 小幅改善；
   - 但 AUC 和 loss 转差；
   - 当前不作为稳定候选继续推进。

24. `GROUPA_LR_SCALE=0.0`
   - `epoch1 / epoch3 / epoch5 / epoch8` 都保持 `candidate_eval_not_worse`；
   - AUC 持续小幅正向，threshold 不低于 baseline；
   - 但 loss/argmax 不一致，当前只能作为 AUC/calibration 候选。

25. `GROUPA_LR_SCALE=0.0` 的 `seed1 epoch3`
   - 没有复现原先的小正 AUC 信号；
   - 因此它仍是 seed 敏感候选，不能升级为稳定默认值。

因此，当前更合理的收口是：

- `ACCURACY_PROFILE` 这条不平衡修正轴已经不再只有“最小验证”层面的结论；
- 在当前已经测过的 `epoch5` 邻域 `{0.18, 0.20, 0.22}` 里，`0.20` 已是局部最优点：
  - `0.18` 已负；
  - `0.22` 虽非劣，但没有超过 `0.20`；
- 在更宽的已测区间 `{0.15, 0.18, 0.20, 0.22, 0.25}` 里，也还没有出现能替代 `0.20` 的更优配置；
- 但 `0.20` 与 `0.22` 到 `epoch8` 都已经转负；
- 因此当前这条 class-weight 轴仍没有找到稳定长预算候选，也不应继续默认沿同一邻域盲目扩展。
- EMA 与 `SMOOTHING=0.05` 是当前新增的非 class-weight 单变量验证；二者都改善 argmax 或 loss，但没有改善正式主指标，因此同样不能直接升级为默认配置。
- `GROUPA_LR_SCALE=1.0` 已从 `epoch1` 正向信号转为 `epoch3` 负信号，当前也不应继续加预算。
- `CLS_TOKEN_FULL_LR=true` 基本等同 baseline，当前也不应继续加预算。
- `TRAIN_POS_EMBED=true` 基本等同 baseline，当前也不应继续加预算。
- `PRETRAINED_FIX_STEP=1` 已从 `epoch1` 非劣信号转为 `epoch3` 负信号，当前也不应继续加预算。
- `LR=1e-6` 在 `epoch1` 已经 AUC/loss 转负，当前也不应继续加预算。
- `WARMUP_STEPS=0` 在 `epoch1` 已经 AUC 转负，当前也不应继续加预算。
- `AUGMENTATION_PROFILE=mpcvit_like` 在 `epoch1` 已经 threshold/AUC 转负，当前也不应继续加预算。
- `FREEZE_PATCH_EMBED_PROJ=true` 在 `epoch1` 有短正信号，但到 `epoch3` 没有延续，当前也不应继续加预算。
- `FREEZE_PATCH_EMBED_WEIGHT=true` 说明 projection freeze 的短正信号主要来自 weight，但同样没有延续到 `epoch3`，当前也不应继续加预算。
- `FREEZE_PATCH_EMBED_BIAS=true` 基本等同 baseline，当前也不应继续加预算。
- `PATCH_EMBED_BIAS_INIT_MODE=zero` 没有改善正式主指标，当前也不应继续加预算。
- `BATCH_SIZE=16` 改善 loss/argmax，但 AUC 明显转负，当前也不应继续加预算。
- `WEIGHT_DECAY=0.01` 基本等同 baseline，当前也不应继续加预算。
- `CLIP_GRAD=2.0` 只改善 argmax，AUC/loss 转差，当前也不应继续加预算。
- `GROUPA_LR_SCALE=0.0` 是当前新出现的稳定非劣候选：`epoch1/3/5/8` 均未压低 threshold/AUC，且 AUC 持续小幅正向；但它没有稳定改善 threshold/loss/argmax，因此不能直接升级成正式默认值。
- `GROUPA_LR_SCALE=0.0` 的 `seed1 epoch3` 未复现原先的小正 AUC，因此它仍是 seed 敏感候选，不能直接升级成正式默认值。

## 5. 当前可引用的结果文件

- `results/accuracy_profile_imbalance/accprof_epoch1_20260506_default_vs_weightedsqrt1/accuracy_profile_compare.json`
- `results/accuracy_profile_imbalance/accprof_epoch1_20260506_default_vs_sqrtcw1/accuracy_profile_compare.json`
- `results/accuracy_profile_imbalance/accprof_epoch3_20260506_default_vs_sqrtcw1/accuracy_profile_compare.json`
- `results/accuracy_profile_imbalance/accprof_epoch1_20260506_default_vs_piw0201/accuracy_profile_compare.json`
- `results/accuracy_profile_imbalance/accprof_epoch3_20260506_default_vs_piw0201/accuracy_profile_compare.json`
- `results/accuracy_profile_imbalance/accprof_epoch5_20260506_default_vs_piw0201/accuracy_profile_compare.json`
- `results/accuracy_profile_imbalance/accprof_epoch8_20260506_default_vs_piw0201/accuracy_profile_compare.json`
- `results/accuracy_profile_imbalance/accprof_epoch5_20260506_default_vs_piw0151/accuracy_profile_compare.json`
- `results/accuracy_profile_imbalance/accprof_epoch5_20260506_default_vs_piw0181/accuracy_profile_compare.json`
- `results/accuracy_profile_imbalance/accprof_epoch5_20260506_default_vs_piw0221/accuracy_profile_compare.json`
- `results/accuracy_profile_imbalance/accprof_epoch8_20260506_default_vs_piw0221/accuracy_profile_compare.json`
- `results/accuracy_profile_imbalance/accprof_epoch5_20260506_default_vs_ema1/accuracy_profile_compare.json`
- `results/accuracy_profile_imbalance/accprof_epoch1_20260506_default_vs_smooth0051/accuracy_profile_compare.json`
- `results/accuracy_profile_imbalance/accprof_epoch1_20260506_default_vs_groupa1x1/accuracy_profile_compare.json`
- `results/accuracy_profile_imbalance/accprof_epoch3_20260506_default_vs_groupa1x1/accuracy_profile_compare.json`
- `results/accuracy_profile_imbalance/accprof_epoch1_20260506_default_vs_clsfull1/accuracy_profile_compare.json`
- `results/accuracy_profile_imbalance/accprof_epoch1_20260506_default_vs_trainpos1/accuracy_profile_compare.json`
- `results/accuracy_profile_imbalance/accprof_epoch1_20260506_default_vs_fixstep1/accuracy_profile_compare.json`
- `results/accuracy_profile_imbalance/accprof_epoch3_20260506_default_vs_fixstep1/accuracy_profile_compare.json`
- `results/accuracy_profile_imbalance/accprof_epoch1_20260506_default_vs_lr1e6_1/accuracy_profile_compare.json`
- `results/accuracy_profile_imbalance/accprof_epoch1_20260506_default_vs_warmup0_1/accuracy_profile_compare.json`
- `results/accuracy_profile_imbalance/accprof_epoch1_20260506_default_vs_augmpc1/accuracy_profile_compare.json`
- `results/accuracy_profile_imbalance/accprof_epoch1_20260506_default_vs_freezepatch1/accuracy_profile_compare.json`
- `results/accuracy_profile_imbalance/accprof_epoch3_20260506_default_vs_freezepatch1/accuracy_profile_compare.json`
- `results/accuracy_profile_imbalance/accprof_epoch1_20260506_default_vs_freezepatchweight1/accuracy_profile_compare.json`
- `results/accuracy_profile_imbalance/accprof_epoch3_20260506_default_vs_freezepatchweight1/accuracy_profile_compare.json`
- `results/accuracy_profile_imbalance/accprof_epoch1_20260506_default_vs_freezepatchbias1/accuracy_profile_compare.json`
- `results/accuracy_profile_imbalance/accprof_epoch1_20260506_default_vs_patchbiaszero1/accuracy_profile_compare.json`
- `results/accuracy_profile_imbalance/accprof_epoch1_20260507_default_vs_bsz16_1/accuracy_profile_compare.json`
- `results/accuracy_profile_imbalance/accprof_epoch1_20260507_default_vs_wd001_1/accuracy_profile_compare.json`
- `results/accuracy_profile_imbalance/accprof_epoch1_20260507_default_vs_clip2_1/accuracy_profile_compare.json`
- `results/accuracy_profile_imbalance/accprof_epoch1_20260507_default_vs_groupa0_1/accuracy_profile_compare.json`
- `results/accuracy_profile_imbalance/accprof_epoch3_20260507_default_vs_groupa0_1/accuracy_profile_compare.json`
- `results/accuracy_profile_imbalance/accprof_epoch5_20260507_default_vs_groupa0_1/accuracy_profile_compare.json`
- `results/accuracy_profile_imbalance/accprof_epoch8_20260507_default_vs_groupa0_1/accuracy_profile_compare.json`
- `results/accuracy_profile_imbalance/accprof_epoch1_20260506_default_vs_piw0251/accuracy_profile_compare.json`
- `results/accuracy_profile_imbalance/accprof_epoch3_20260506_default_vs_piw0251/accuracy_profile_compare.json`
- `results/accuracy_profile_imbalance/accprof_epoch5_20260506_default_vs_piw0251/accuracy_profile_compare.json`

以及各自的：

- `plaintext_eval.json`
- `command.sh`
- `log.txt`
- `threshold_best.json`
- `threshold_eval.json`

均已回传到本地仓。

## 6. 本轮命令

`weighted_sqrt_sampler` `epoch1`：

```bash
export RUN_NAME=secure_static_accprof_epoch1_20260506_weightedsqrt1
export ACCURACY_PROFILE=weighted_sqrt_sampler
export EPOCHS=1
bash artifacts/server_inference_friendly_pack/run_secure_static_distill_train.sh epoch1
```

`sqrt_class_weight` `epoch1`：

```bash
export RUN_NAME=secure_static_accprof_epoch1_20260506_sqrtcw1
export ACCURACY_PROFILE=sqrt_class_weight
export EPOCHS=1
bash artifacts/server_inference_friendly_pack/run_secure_static_distill_train.sh epoch1
```

`sqrt_class_weight` `epoch3`：

```bash
export RUN_NAME=secure_static_accprof_epoch3_20260506_sqrtcw1
export ACCURACY_PROFILE=sqrt_class_weight
export EPOCHS=3
bash artifacts/server_inference_friendly_pack/run_secure_static_distill_train.sh epoch1
```

`power_inverse_freq=0.20` paired study：

```bash
export PAIR_EPOCHS=1
export PAIR_NAME=accprof_epoch1_20260506_default_vs_piw0201
export BASELINE_RUN_NAME=secure_static_accprof_epoch1_20260506_default1
export CANDIDATE_RUN_NAME=secure_static_accprof_epoch1_20260506_piw0201
export CANDIDATE_CLASS_WEIGHT_MODE=power_inverse_freq
export CANDIDATE_CLASS_WEIGHT_POWER=0.20
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh train-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh post-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh compare
```

后续同口径只改：

- `PAIR_EPOCHS=3` / `PAIR_NAME=accprof_epoch3_20260506_default_vs_piw0201`
- `PAIR_EPOCHS=5` / `PAIR_NAME=accprof_epoch5_20260506_default_vs_piw0201`

`power_inverse_freq=0.18` `epoch5`：

```bash
export PAIR_EPOCHS=5
export PAIR_NAME=accprof_epoch5_20260506_default_vs_piw0181
export BASELINE_RUN_NAME=secure_static_accprof_epoch5_20260506_default1
export CANDIDATE_RUN_NAME=secure_static_accprof_epoch5_20260506_piw0181
export CANDIDATE_CLASS_WEIGHT_MODE=power_inverse_freq
export CANDIDATE_CLASS_WEIGHT_POWER=0.18
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh train-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh post-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh compare
```

`power_inverse_freq=0.22` `epoch5`：

```bash
export PAIR_EPOCHS=5
export PAIR_NAME=accprof_epoch5_20260506_default_vs_piw0221
export BASELINE_RUN_NAME=secure_static_accprof_epoch5_20260506_default1
export CANDIDATE_RUN_NAME=secure_static_accprof_epoch5_20260506_piw0221
export CANDIDATE_CLASS_WEIGHT_MODE=power_inverse_freq
export CANDIDATE_CLASS_WEIGHT_POWER=0.22
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh train-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh post-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh compare
```

`power_inverse_freq=0.22` `epoch8`：

```bash
export PAIR_EPOCHS=8
export PAIR_NAME=accprof_epoch8_20260506_default_vs_piw0221
export BASELINE_RUN_NAME=secure_static_accprof_epoch8_20260506_default1
export CANDIDATE_RUN_NAME=secure_static_accprof_epoch8_20260506_piw0221
export CANDIDATE_CLASS_WEIGHT_MODE=power_inverse_freq
export CANDIDATE_CLASS_WEIGHT_POWER=0.22
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh train-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh post-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh compare
```

`MODEL_EMA=true` `epoch5`：

```bash
export PAIR_EPOCHS=5
export PAIR_NAME=accprof_epoch5_20260506_default_vs_ema1
export BASELINE_RUN_NAME=secure_static_accprof_epoch5_20260506_default1
export CANDIDATE_RUN_NAME=secure_static_accprof_epoch5_20260506_ema1
export CANDIDATE_MODEL_EMA=true
export CANDIDATE_CHECKPOINT_NAME=checkpoint-best-ema.pth
export CANDIDATE_CHECKPOINT_MODEL_KEY=model_ema
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh train-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh post-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh compare
```

`SMOOTHING=0.05` `epoch1`：

```bash
export PAIR_EPOCHS=1
export PAIR_NAME=accprof_epoch1_20260506_default_vs_smooth0051
export BASELINE_RUN_NAME=secure_static_accprof_epoch1_20260506_default1
export CANDIDATE_RUN_NAME=secure_static_accprof_epoch1_20260506_smooth0051
export BASELINE_SMOOTHING=0.1
export CANDIDATE_SMOOTHING=0.05
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh train-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh post-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh compare
```

`GROUPA_LR_SCALE=1.0` `epoch1`：

```bash
export PAIR_EPOCHS=1
export PAIR_NAME=accprof_epoch1_20260506_default_vs_groupa1x1
export BASELINE_RUN_NAME=secure_static_accprof_epoch1_20260506_default1
export CANDIDATE_RUN_NAME=secure_static_accprof_epoch1_20260506_groupa1x1
export BASELINE_GROUPA_LR_SCALE=0.1
export CANDIDATE_GROUPA_LR_SCALE=1.0
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh train-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh post-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh compare
```

后续同口径只改：

- `PAIR_EPOCHS=3`
- `PAIR_NAME=accprof_epoch3_20260506_default_vs_groupa1x1`
- `BASELINE_RUN_NAME=secure_static_accprof_epoch3_20260506_default1`
- `CANDIDATE_RUN_NAME=secure_static_accprof_epoch3_20260506_groupa1x1`

`CLS_TOKEN_FULL_LR=true` `epoch1`：

```bash
export PAIR_EPOCHS=1
export PAIR_NAME=accprof_epoch1_20260506_default_vs_clsfull1
export BASELINE_RUN_NAME=secure_static_accprof_epoch1_20260506_default1
export CANDIDATE_RUN_NAME=secure_static_accprof_epoch1_20260506_clsfull1
export BASELINE_CLS_TOKEN_FULL_LR=false
export CANDIDATE_CLS_TOKEN_FULL_LR=true
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh train-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh post-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh compare
```

`TRAIN_POS_EMBED=true` `epoch1`：

```bash
export PAIR_EPOCHS=1
export PAIR_NAME=accprof_epoch1_20260506_default_vs_trainpos1
export BASELINE_RUN_NAME=secure_static_accprof_epoch1_20260506_default1
export CANDIDATE_RUN_NAME=secure_static_accprof_epoch1_20260506_trainpos1
export BASELINE_TRAIN_POS_EMBED=false
export CANDIDATE_TRAIN_POS_EMBED=true
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh train-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh post-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh compare
```

`PRETRAINED_FIX_STEP=1` `epoch1`：

```bash
export PAIR_EPOCHS=1
export PAIR_NAME=accprof_epoch1_20260506_default_vs_fixstep1
export BASELINE_RUN_NAME=secure_static_accprof_epoch1_20260506_default1
export CANDIDATE_RUN_NAME=secure_static_accprof_epoch1_20260506_fixstep1
export BASELINE_PRETRAINED_FIX_STEP=0
export CANDIDATE_PRETRAINED_FIX_STEP=1
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh train-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh post-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh compare
```

`PRETRAINED_FIX_STEP=1` `epoch3`：

```bash
export PAIR_EPOCHS=3
export PAIR_NAME=accprof_epoch3_20260506_default_vs_fixstep1
export BASELINE_RUN_NAME=secure_static_accprof_epoch3_20260506_default1
export CANDIDATE_RUN_NAME=secure_static_accprof_epoch3_20260506_fixstep1
export BASELINE_PRETRAINED_FIX_STEP=0
export CANDIDATE_PRETRAINED_FIX_STEP=1
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh train-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh post-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh compare
```

`LR=1e-6` `epoch1`：

```bash
export PAIR_EPOCHS=1
export PAIR_NAME=accprof_epoch1_20260506_default_vs_lr1e6_1
export BASELINE_RUN_NAME=secure_static_accprof_epoch1_20260506_default1
export CANDIDATE_RUN_NAME=secure_static_accprof_epoch1_20260506_lr1e6_1
export BASELINE_LR=3e-6
export CANDIDATE_LR=1e-6
export BASELINE_MIN_LR=1e-7
export CANDIDATE_MIN_LR=1e-7
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh train-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh post-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh compare
```

`WARMUP_STEPS=0` `epoch1`：

```bash
export PAIR_EPOCHS=1
export PAIR_NAME=accprof_epoch1_20260506_default_vs_warmup0_1
export BASELINE_RUN_NAME=secure_static_accprof_epoch1_20260506_default1
export CANDIDATE_RUN_NAME=secure_static_accprof_epoch1_20260506_warmup0_1
export BASELINE_WARMUP_STEPS=20
export CANDIDATE_WARMUP_STEPS=0
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh train-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh post-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh compare
```

`AUGMENTATION_PROFILE=mpcvit_like` `epoch1`：

```bash
export PAIR_EPOCHS=1
export PAIR_NAME=accprof_epoch1_20260506_default_vs_augmpc1
export BASELINE_RUN_NAME=secure_static_accprof_epoch1_20260506_default1
export CANDIDATE_RUN_NAME=secure_static_accprof_epoch1_20260506_augmpc1
export BASELINE_AUGMENTATION_PROFILE=timm
export CANDIDATE_AUGMENTATION_PROFILE=mpcvit_like
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh train-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh post-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh compare
```

`FREEZE_PATCH_EMBED_PROJ=true` paired study：

```bash
export PAIR_EPOCHS=1
export PAIR_NAME=accprof_epoch1_20260506_default_vs_freezepatch1
export BASELINE_RUN_NAME=secure_static_accprof_epoch1_20260506_default1
export CANDIDATE_RUN_NAME=secure_static_accprof_epoch1_20260506_freezepatch1
export BASELINE_FREEZE_PATCH_EMBED_PROJ=false
export CANDIDATE_FREEZE_PATCH_EMBED_PROJ=true
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh train-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh post-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh compare
```

后续同口径只改：

- `PAIR_EPOCHS=3` / `PAIR_NAME=accprof_epoch3_20260506_default_vs_freezepatch1`
- `CANDIDATE_RUN_NAME=secure_static_accprof_epoch3_20260506_freezepatch1`

`FREEZE_PATCH_EMBED_WEIGHT=true` paired study：

```bash
export PAIR_EPOCHS=1
export PAIR_NAME=accprof_epoch1_20260506_default_vs_freezepatchweight1
export BASELINE_RUN_NAME=secure_static_accprof_epoch1_20260506_default1
export CANDIDATE_RUN_NAME=secure_static_accprof_epoch1_20260506_freezepatchweight1
export BASELINE_FREEZE_PATCH_EMBED_WEIGHT=false
export CANDIDATE_FREEZE_PATCH_EMBED_WEIGHT=true
export BASELINE_FREEZE_PATCH_EMBED_BIAS=false
export CANDIDATE_FREEZE_PATCH_EMBED_BIAS=false
export BASELINE_FREEZE_PATCH_EMBED_PROJ=false
export CANDIDATE_FREEZE_PATCH_EMBED_PROJ=false
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh train-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh post-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh compare
```

后续同口径只改：

- `PAIR_EPOCHS=3` / `PAIR_NAME=accprof_epoch3_20260506_default_vs_freezepatchweight1`
- `CANDIDATE_RUN_NAME=secure_static_accprof_epoch3_20260506_freezepatchweight1`

`FREEZE_PATCH_EMBED_BIAS=true` paired study：

```bash
export PAIR_EPOCHS=1
export PAIR_NAME=accprof_epoch1_20260506_default_vs_freezepatchbias1
export BASELINE_RUN_NAME=secure_static_accprof_epoch1_20260506_default1
export CANDIDATE_RUN_NAME=secure_static_accprof_epoch1_20260506_freezepatchbias1
export BASELINE_FREEZE_PATCH_EMBED_WEIGHT=false
export CANDIDATE_FREEZE_PATCH_EMBED_WEIGHT=false
export BASELINE_FREEZE_PATCH_EMBED_BIAS=false
export CANDIDATE_FREEZE_PATCH_EMBED_BIAS=true
export BASELINE_FREEZE_PATCH_EMBED_PROJ=false
export CANDIDATE_FREEZE_PATCH_EMBED_PROJ=false
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh train-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh post-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh compare
```

`PATCH_EMBED_BIAS_INIT_MODE=zero` paired study：

```bash
export PAIR_EPOCHS=1
export PAIR_NAME=accprof_epoch1_20260506_default_vs_patchbiaszero1
export BASELINE_RUN_NAME=secure_static_accprof_epoch1_20260506_default1
export CANDIDATE_RUN_NAME=secure_static_accprof_epoch1_20260506_patchbiaszero1
export BASELINE_PATCH_EMBED_BIAS_INIT_MODE=pretrained
export CANDIDATE_PATCH_EMBED_BIAS_INIT_MODE=zero
export BASELINE_SKIP_PATCH_EMBED_BIAS_PRETRAINED=false
export CANDIDATE_SKIP_PATCH_EMBED_BIAS_PRETRAINED=false
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh train-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh post-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh compare
```

`BATCH_SIZE=16` paired study：

```bash
export PAIR_EPOCHS=1
export PAIR_NAME=accprof_epoch1_20260507_default_vs_bsz16_1
export BASELINE_RUN_NAME=secure_static_accprof_epoch1_20260506_default1
export CANDIDATE_RUN_NAME=secure_static_accprof_epoch1_20260507_bsz16_1
export BASELINE_BATCH_SIZE=32
export CANDIDATE_BATCH_SIZE=16
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh train-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh post-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh compare
```

`WEIGHT_DECAY=0.01` paired study：

```bash
export PAIR_EPOCHS=1
export PAIR_NAME=accprof_epoch1_20260507_default_vs_wd001_1
export BASELINE_RUN_NAME=secure_static_accprof_epoch1_20260506_default1
export CANDIDATE_RUN_NAME=secure_static_accprof_epoch1_20260507_wd001_1
export BASELINE_WEIGHT_DECAY=0.05
export CANDIDATE_WEIGHT_DECAY=0.01
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh train-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh post-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh compare
```

`CLIP_GRAD=2.0` paired study：

```bash
export PAIR_EPOCHS=1
export PAIR_NAME=accprof_epoch1_20260507_default_vs_clip2_1
export BASELINE_RUN_NAME=secure_static_accprof_epoch1_20260506_default1
export CANDIDATE_RUN_NAME=secure_static_accprof_epoch1_20260507_clip2_1
export BASELINE_CLIP_GRAD=1.0
export CANDIDATE_CLIP_GRAD=2.0
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh train-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh post-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh compare
```

`GROUPA_LR_SCALE=0.0` paired study：

```bash
export PAIR_EPOCHS=1
export PAIR_NAME=accprof_epoch1_20260507_default_vs_groupa0_1
export BASELINE_RUN_NAME=secure_static_accprof_epoch1_20260506_default1
export CANDIDATE_RUN_NAME=secure_static_accprof_epoch1_20260507_groupa0_1
export BASELINE_GROUPA_LR_SCALE=0.1
export CANDIDATE_GROUPA_LR_SCALE=0.0
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh train-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh post-candidate
bash artifacts/server_inference_friendly_pack/run_accuracy_profile_pair_study.sh compare
```

同一设置已按 `PAIR_EPOCHS=3/5/8` 分别完成：

- `accprof_epoch3_20260507_default_vs_groupa0_1`
- `accprof_epoch5_20260507_default_vs_groupa0_1`
- `accprof_epoch8_20260507_default_vs_groupa0_1`

当前如果继续沿这条轴推进，优先顺序应改成：

1. 先把 `power_inverse_freq=0.20` 固定为当前 `epoch<=5` 的局部 best candidate；
2. 承认 `0.18 / 0.22` 窄区间验证已经做完，且没有出现更优点；
3. 承认 `0.20 / 0.22` 到 `epoch8` 都已转负，因此这条 class-weight 轴当前没有稳定长预算候选；
4. 承认 `MODEL_EMA=true` 已经验证过，当前只改善 argmax，不改善正式主指标；
5. 如果还要继续精度修正，默认不再沿当前 `power_inverse_freq` 邻域或 EMA 轴追加预算，而应切到新的单变量假设；
6. `SMOOTHING=0.05` 已验证为只改善 argmax/loss、不改善正式主指标，后续也不要默认沿这条追加预算。
7. `GROUPA_LR_SCALE=1.0` 已验证为 `epoch1` 非劣但 `epoch3` 转负，后续也不要默认沿这条追加预算。
8. `CLS_TOKEN_FULL_LR=true` 已验证为基本等同 baseline，后续也不要默认沿这条追加预算。
9. `TRAIN_POS_EMBED=true` 已验证为基本等同 baseline，后续也不要默认沿这条追加预算。
10. `PRETRAINED_FIX_STEP=1` 已验证为 `epoch1` 有 argmax/AUC 短信号但 `epoch3` 主指标转负，后续也不要默认沿这条追加预算。
11. `LR=1e-6` 已验证为 `epoch1` 只改善 argmax、不改善 AUC/loss，后续也不要默认沿这条追加预算。
12. `WARMUP_STEPS=0` 已验证为 `epoch1` 只改善 loss、不改善 AUC，后续也不要默认沿这条追加预算。
13. `AUGMENTATION_PROFILE=mpcvit_like` 已验证为 `epoch1` 只改善 argmax/loss、不改善 threshold/AUC，后续也不要默认沿这条追加预算。
14. `FREEZE_PATCH_EMBED_PROJ=true` 已验证为 `epoch1` 有 patch embedding 短正信号，但 `epoch3` 未延续，后续也不要默认沿这条追加预算。
15. `FREEZE_PATCH_EMBED_WEIGHT=true` 已验证为短正信号来源，但 `epoch3` 未延续，后续也不要默认沿这条追加预算。
16. `FREEZE_PATCH_EMBED_BIAS=true` 已验证为基本等同 baseline，后续也不要默认沿这条追加预算。
17. `PATCH_EMBED_BIAS_INIT_MODE=zero` 已验证为不改善正式主指标，后续也不要默认沿这条追加预算。
18. `BATCH_SIZE=16` 已验证为改善 loss/argmax 但 AUC 明显转负，后续也不要默认沿这条追加预算。
19. `WEIGHT_DECAY=0.01` 已验证为基本等同 baseline，后续也不要默认沿这条追加预算。
20. `CLIP_GRAD=2.0` 已验证为只改善 argmax，AUC/loss 转差，后续也不要默认沿这条追加预算。
21. `GROUPA_LR_SCALE=0.0` 已验证为 `epoch1/3/5/8` 稳定非劣，AUC 持续小幅正向，但 threshold/loss/argmax 不是全面更优；当前可作为 AUC/calibration 候选，不直接升级默认值。
22. 如果继续精度修正，应另起新的单变量假设；不要把上面任何一条已经关闭或 seed 敏感的轴写成“已证明更优”的正式默认值。

## 2026-05-07 追加：AutoAugment 关闭轴

为避免继续重复已关闭的 class-weight、EMA、LR、freeze 与 group-A LR 轴，本轮新增了增强参数透传：

- `run_secure_static_distill_train.sh` 现在显式透传 `COLOR_JITTER`、`AA`、`REPROB`。
- `run_accuracy_profile_pair_study.sh` 现在支持 `BASELINE_/CANDIDATE_COLOR_JITTER`、`BASELINE_/CANDIDATE_AA`、`BASELINE_/CANDIDATE_REPROB`。
- `AA=none` 是 wrapper 哨兵值，进入训练命令时转换为空字符串，从而关闭 timm RandAugment，并回退到 ColorJitter。
- `tools/transshield_training_pair_compare.py` 已把 `color_jitter / aa / reprob` 纳入 config compare。

已完成的单因子结果：

- `REPROB=0.0` vs default `0.25`，epoch1：`threshold_accuracy_delta = -0.3817 pt`，`auc_delta = -0.00388460`，`argmax_accuracy_delta = +1.3359 pt`；只改善 argmax/loss，正式主指标转负，关闭。
- `AA=none` vs default RandAugment，epoch1：`threshold_accuracy_delta = +0.9542 pt`，`auc_delta = +0.01260592`，`argmax_accuracy_delta = -9.3511 pt`；出现强 threshold/AUC 校准收益，但牺牲 argmax/loss。
- `AA=none` vs default RandAugment，epoch3 seed0：`threshold_accuracy_delta = +1.5267 pt`，`auc_delta = +0.01527183`，`argmax_accuracy_delta = -8.2061 pt`。
- `AA=none` vs default RandAugment，epoch3 seed1：`threshold_accuracy_delta = +0.7634 pt`，`auc_delta = +0.01340569`，`argmax_accuracy_delta = -7.6336 pt`；threshold/AUC 改善通过 seed1 复验。
- `AA=none + COLOR_JITTER=0.0`，epoch1：`threshold_accuracy_delta = 0.0 pt`，`auc_delta = -0.00243740`，`argmax_accuracy_delta = -5.9160 pt`；关闭 ColorJitter 后主指标收益消失，关闭。
- `AA=none` vs default RandAugment，epoch5 seed0：candidate `threshold_accuracy = 91.2214%`，candidate `auc = 0.96271541`，`threshold_accuracy_delta = +1.5267 pt`，`auc_delta = +0.01576692`，`argmax_accuracy_delta = -8.7786 pt`。
- `AA=none` vs default RandAugment，epoch8 seed0：candidate `threshold_accuracy = 91.9847%`，candidate `auc = 0.96787584`，`threshold_accuracy_delta = +2.2901 pt`，`auc_delta = +0.02096544`，`argmax_accuracy_delta = -5.7252 pt`。

当前判定：

- `AA=none` 是目前最强的 accuracy-improvement 候选，正式主指标应按 threshold accuracy / AUC 记录：epoch8 达到 `91.9847% / 0.96787584`。
- 它不能直接写成“全面精度更优”，因为默认 argmax 与 eval loss 仍显著差于 default；论文/答辩里只能表述为“阈值校准口径和排序能力提升”。
- `AA=none + MODEL_EMA=true` 已做 epoch1 EMA 权重评估：相对 `AA=none` baseline，`argmax_accuracy_delta = +15.8397 pt`、`eval_loss_delta = -0.05074`，但 `threshold_accuracy_delta = -0.7634 pt`、`auc_delta = -0.01161573`；它修复 argmax/loss 但破坏正式主指标，不能作为当前主线。
- 新增 `tools/transshield_public_logit_bias_calibration.py` 用于把 best threshold 转成公开 class-1 logit bias。
- 对 `AA=none epoch8` 的 public bias 报告：
  - report：`results/accuracy_profile_imbalance/accprof_epoch8_20260507_default_vs_aanone_1/public_logit_bias_calibration.json`
  - `effective_class1_logit_bias = 0.5852264595359804`
  - `original_argmax_accuracy = 76.7176%`
  - `calibrated_argmax_accuracy = 91.9847%`
  - `original_ce_loss = 0.53050122`
  - `calibrated_ce_loss = 0.42866483`
  - `calibrated_auc = 0.96787584`
- 这说明 `AA=none` 的 argmax/loss 损失主要来自公开决策边界偏移；通过在最终 class-1 logit 上加公开标量即可恢复 calibrated argmax 和 loss，不需要重训，也不改变 secure ViT 主体算子。
- 已把 public logit-bias calibration 接入 `tools/transshield_training_pair_compare.py` 和 `run_accuracy_profile_pair_study.sh compare`：
  - pair compare 现在可直接读取 baseline/candidate 的 `plaintext_eval.csv` 和 `threshold_eval.json`；
  - report 内新增 `public_logit_bias_calibration_compare`；
  - `accprof_epoch8_20260507_default_vs_aanone_1` 重新 compare 后，calibrated candidate-baseline delta 为 `argmax_accuracy +2.2901 pt`、`auc +0.02096544`、`CE loss -0.01501771`。
- `tools/transshield_public_logit_bias_calibration.py` 现在还可生成 E2E/OpenBumbleBee 已支持的 `--output-calibration-json`：
  - output：`results/accuracy_profile_imbalance/accprof_epoch8_20260507_default_vs_aanone_1/e2e_output_calibration_public_logit_bias.json`
  - schema：`weights=[-1.0, 1.0]`、`bias=0.5852264595359804`、`threshold=0.5`
  - 语义：`class1_score = logits @ [-1, 1] + public_class1_logit_bias`
- E2E/OpenBumbleBee smoke 已验证该 output calibration 能在部署路径实际生效：
  - `e2e_approx_eval_public_bias_smoke2_20260507_1`：`sample_count=2`、`finite_logits=true`、`threshold_match_ratio=1.0`
  - `e2e_approx_eval_public_bias_smoke4_20260507_1`：`sample_count=4`、`output_calibration` 写入 metrics、`e2e_threshold_accuracy=100%`、same-subset plaintext threshold accuracy `100%`、`threshold_accuracy_gap=0.0pp`
  - `e2e_approx_eval_public_bias_smoke8_20260507_1`：`sample_count=8`、`finite_logits=true`、`output_calibration` 写入 metrics、`e2e_threshold_accuracy=100%`、same-subset plaintext threshold accuracy `100%`、`threshold_accuracy_gap=0.0pp`
  - `e2e_approx_eval_public_bias_smoke16_20260507_1`：`sample_count=16`、`finite_logits=true`，但 `e2e_threshold_accuracy=68.75%`、same-subset plaintext threshold accuracy `87.5%`、`threshold_accuracy_gap=-18.75pp`、`threshold_match_ratio=0.6875`
  - `e2e_approx_eval_public_bias_smoke16_chunk3_20260507_1`：设置 `E2E_SPU_BLOCK_CHUNK_SIZE=3` 后没有改善
  - mismatch report：`artifacts/server_pipeline_run/e2e_approx_eval_public_bias_smoke16_20260507_1/e2e_secure_poc/e2e_public_bias_smoke16_mismatch_report.json`
  - smoke 结论只证明 calibrated argmax / threshold-calibrated decision 在 E2E 路径中完成了接入；不能写成 raw plaintext argmax 一致，也不能写成 16 样本已经稳定。
- 若要继续提高，不应回到已关闭的 `REPROB=0.0`、`AA=none + COLOR_JITTER=0.0`、`MODEL_EMA=true`、class-weight 邻域或 group-A LR 轴；下一步只保留“公开校准层更细化”的轻量路线，例如 class bias + temperature 或 class-prior logit adjustment。

## 2026-05-07 E2E calibrated path 追加诊断

- public logit-bias calibration 在 full-val formal eval 上仍然成立：它恢复的是公开决策边界，不改变 AUC，也不改变 secure ViT 主体算子。
- smoke16 的新增 CPU/SPU drift 报告表明，public bias 本身不是失配来源；CPU static/public-bias 已达到 `threshold_match_ratio = 0.75`，而原 SPU publiccalib LN + `fixed_square clip3.0` 只有 `0.6875`。
- 已修复 block probe 入口，使 probe 支持 `public_calibrated` LN、calibration JSON 和 activation clip value。
- sample `index=5` 的 block 1 probe 将 SPU-specific drift 定位到 MLP：`exact LN + clip3.0` 的 `mlp_out_cls max_abs_error = 93.5489`，`exact LN + clip0` 降到 `0.3636`。
- 16 样本完整验证显示 `exact LN + clip0` 的 SPU 输出与 CPU static/public-bias 的 threshold 决策完全同型，错配均为 `1,3,7,15`。
- 当前精度解释需要拆成两层：`activation_clip=3.0` 是 SPU-specific 额外漂移；剩余 `1,3,7,15` 是 CPU static approximation 与原 plaintext threshold 的差距。
- 后续若继续提高 argmax/loss 或 E2E calibrated accuracy，应优先处理 static approximation 对齐或重新生成 clip-specific public LN calibration，而不是回到已关闭的训练超参轴。

## 2026-05-07 E2E static full-val 校准

- 为避免只看 smoke 子集，新增 `tools/transshield_e2e_static_calibration_report.py` 并在当前 frozen bundle 上跑 full-val CPU static 校准。
- 报告：`results/e2e_static_calibration/e2e_static_fullval_20260507_1/e2e_static_calibration_report.json`。
- 当前 frozen bundle 的 full-val CPU static 指标为 `argmax_accuracy = 88.1679%`、`best_threshold_accuracy = 89.3130%`、`auc = 0.94670094`。
- 该结果与 `AA=none epoch8` 的 formal eval `threshold_accuracy = 91.9847%`、`auc = 0.96787584` 不一致，说明 E2E bundle 本身还没有升级到最新精度候选。
- 所以后续精度路线应先做 checkpoint-to-E2E-bundle 导出，而不是继续在旧 bundle 上微调 public bias。

## 2026-05-07 AA=none E2E bundle 导出

- 已导出 `AA=none epoch8` candidate 为 `artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_aanone_20260507`。
- 新 bundle manifest 记录 `threshold_accuracy = 91.9847%`、`auc = 0.96787584`，与 formal eval 对齐。
- 新 bundle full-val CPU static 校准也复现 `best_threshold_accuracy = 91.9847%`、`auc = 0.96787584`。
- 这一步解决了“E2E frozen bundle 精度滞后”的问题；下一步只需验证 SPU smoke 是否在 `exact LN + clip0` accuracy-first 配置下跟上 CPU static。

## 2026-05-07 AA=none E2E smoke

- `smoke8` 已跑通：`sample_count = 8`，`finite_logits = true`，`e2e_threshold_accuracy = 75.0%`。
- `smoke16` 已跑通：`sample_count = 16`，`finite_logits = true`，`e2e_threshold_accuracy = 81.25%`。
- 两次 smoke 均使用 `exact LN + clip0 + static-path public output calibration`。
- smoke16 的 privacy 字段确认：party-local share load，host 不加载 plaintext pixel/private share tensor，private paths redacted。
- 当前可写成：`AA=none` 高精度候选已经从 formal eval 进入 E2E bundle，并完成 16 样本 SPU smoke；但由于 E2E 当前是 static approximate path，不能把 prediction match vs original plaintext 动态路径写成 100% 一致。
