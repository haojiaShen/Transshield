# Margin-aware pruning 研究记录

最后更新：`2026-04-18`

本页只记录 `margin-aware pruning` 的**研究性 ablation 结论**，不代表当前 Web demo 默认模型已经切换。

## 当前保留的两条有用证据

### 1. `w10`：最强的协议友好性证据

- 候选 bundle：`artifacts/frozen_candidates/margin_aware_full20_w10_20260417_205242_w10_t1em4_bundle`
- 服务器报告目录：`/data/wyb/Transshield_final/results/margin_aware_pruning_ablation/margin_aware_full20_w10_20260417_205242`
- 效果：
  - Argmax Acc：`88.9313%`
  - Threshold Acc：`90.2672%`
  - AUC：`0.956508`
- Stage-wise 关键读数：
  - Stage 1 margin：`1.368x`
  - Stage 2 margin：`243.532x`
  - Stage 3 margin：`2.986x`
  - Stage 2 `<=1e-4`：`98.66% -> 5.92%`
- secure 检查：
  - `argmax_match_ratio = 1.0`
  - `threshold_match_ratio = 1.0`
  - `logits_max_abs_error = 0.0`
  - `probabilities_max_abs_error = 0.0`

一句话定位：**这条是“margin-aware 思路有效”的最强证据**。它清楚说明了 Stage 2 边界可以被显著拉开，而且 secure 一致性没有被破坏。

### 2. `w3 + formal hparams + tok0.02`：当前最好的折中证据

- 候选 bundle：`artifacts/frozen_candidates/margin_formal_hparams_soft_stage2_20260417_231946_w3_t1em4_bundle`
- 服务器报告目录：`/data/wyb/Transshield_final/results/margin_aware_pruning_ablation/margin_formal_hparams_soft_stage2_20260417_231946`
- 效果：
  - Argmax Acc：`85.1145%`
  - Threshold Acc：`91.6031%`
  - AUC：`0.967476`
- Stage-wise 关键读数：
  - Stage 1 margin：`0.820x`
  - Stage 2 margin：`20.032x`
  - Stage 3 margin：`3.001x`
  - Stage 2 `<=1e-4`：`98.66% -> 42.56%`
  - Stage 3 `<=1e-4`：`44.66% -> 34.16%`

一句话定位：**这条不是最强的协议证据，但它是目前“精度 / 协议友好性”最均衡的一条**。

## 已验证但不继续推进的结果

### `Stage2-only + delayed start`

- 候选 bundle：`artifacts/frozen_candidates/margin_stage2_only_delayed_retry_20260417_230013_w10_t1em4_bundle`
- 效果：
  - Argmax Acc：`74.2366%`
  - Threshold Acc：`74.2366%`
  - AUC：`0.647815`

结论：**失败配置**。说明只压 Stage 2、并且后半程再开，并没有自动把精度拉回来。

### `w2 + formal hparams`

- 候选 bundle：`artifacts/frozen_candidates/margin_formal_hparams_soft_stage2_w2_20260417_234051_w2_t1em4_bundle`
- 效果：
  - Argmax Acc：`87.4046%`
  - Threshold Acc：`90.0763%`
  - AUC：`0.952718`

结论：Stage 2 margin 更强，但 Stage 3 明显变坏，不如 `w3 + tok0.02`。

### `w3 + formal hparams + tok0.04`

- 候选 bundle：`artifacts/frozen_candidates/margin_formal_hparams_soft_stage2_w3_tok004_20260417_235928_w3_t1em4_bundle`
- 效果：
  - Argmax Acc：`90.2672%`
  - Threshold Acc：`90.6489%`
  - AUC：`0.952966`
- Stage-wise 关键读数：
  - Stage 2 margin：`53.758x`
  - Stage 3 margin：`0.063x`
  - Stage 3 `<=1e-4`：`44.66% -> 100.00%`

结论：提高 `token_distill_weight` 到 `0.04` 会把 `argmax` 顶上去，但会破坏后段 pruning 分布与 AUC / threshold 表现，不是当前想要的方向。

## 当前研究结论

### 1. `w10` 仍然有用，而且非常有用

它不是“可直接替换正式模型”的候选，但它仍然是：

- 最强的 Stage 2 boundary 拉开证据；
- 最完整的 secure 一致性闭环证据；
- 后续 `network-kth` / payload 优化叙事的重要前置论据。

### 2. 当前最值得保留的不是某个“新默认模型”，而是两条证据链

- `w10`：说明**协议友好分布是可以学出来的**
- `w3 + tok0.02`：说明在更温和配置下，**可以部分保住精度并保留一部分协议收益**

### 3. Phase 2 搜索可以先停

当前已经足够说明：

- 思路是对的；
- 但继续靠扫 margin 训练超参，短期内不太像能把 `Threshold / AUC` 拉回到正式默认模型水平；
- 更值得把时间投入到 `Phase 3 network-kth` 与 `Phase 4 payload`。

## 推荐的后续衔接方式

后续如果要在答辩或文档里引用 `margin-aware pruning`，建议这样讲：

1. `w10` 作为**最强研究性正结果**
   - Stage 2 margin `243.532x`
   - Stage 2 near-boundary `98.66% -> 5.92%`
   - secure 一致性 `100%`
2. `w3 + tok0.02` 作为**折中版本**
   - Threshold `91.6031%`
   - AUC `0.967476`
   - Stage 2 margin `20.032x`
3. 说明当前不把它替换成正式默认模型
   - 因为默认展示仍以 `94.083971% threshold / 0.972313 AUC` 为准
4. 接下来把重点转到：
   - `network-kth` 的层次化 / 分块化
   - sidecar payload 压缩

## 明确禁止的写法

- 不要把这些 ablation 结果写成“当前正式模型成绩”
- 不要把它们写进 Web demo 主卡片
- 不要把负结果混进答辩主表
