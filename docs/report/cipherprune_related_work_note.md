# CipherPrune 相关工作实验说明

本文档记录一次围绕 `CipherPrune` 风格安全剪枝思路的**并行实验**结果。该实验只用于相关工作比较与后续方案判断，**不构成当前正式主线能力**。

## 实验范围与边界

- 正式主线未替换，当前项目正式方案仍为：
  - `DynamicViT` 安全化改写
  - `uniform attention`
  - `fixed_square` 激活
  - 两方 SPU whole-forward 近似执行链
- 本次实验分为两类：
  - **接口态对比**：仅把运行路径改成 external keep-mask 回放，不改变原有 keep masks
  - **论文态最小近似**：用 `CLS -> token attention` 平均注意力作为 token importance，生成一组 `CipherPrune` 风格近似 keep masks
- 该实验**不是完整 CipherPrune 复现**，也没有重新训练模型。

## 实验设置

- bundle：`artifacts/frozen_bundle_medical_dynamic_mainline`
- `static_depth_limit=10`
- CPU 对比：`2` 个样本
- SPU 对比：`1` 个样本
- SPU 输入模式：`party_local_debug_share_load`
- SPU 路径：external keep-mask whole-forward 回放

## 结果摘要

### 1. 接口态对比

在 keep masks 完全相同的前提下：

- CPU `dynamicvit_runtime_pruning_reference`：`0.7383s`
- CPU `cipherprune_experimental + external keep-mask replay`：`0.7357s`

输出一致性：

- `logits max_abs_error = 0.0`
- `probabilities max_abs_error = 0.0`
- `argmax_match_ratio = 1.0`
- `threshold_match_ratio = 1.0`

结论：

- 仅把内部剪枝决策改成 external keep-mask 回放，**几乎没有额外收益**。

### 2. 论文态最小近似

采用 `CLS -> token attention` 平均值作为 token importance，生成近似 `CipherPrune` 风格 keep masks。

#### `base_rate=0.70`

- keep counts：`137 / 96 / 67`
- CPU `dynamicvit_runtime_pruning`：`0.7226s`
- CPU `cipherprune_style_attention`：`0.7504s`
- SPU `cipherprune_style_attention`：`118.42s`

说明：

- 保留数量与当前主线相同，但保留 token 身份不同。
- 在该组下，stage keep masks 与当前主线存在明显差异，但未带来加速收益。

#### `base_rate=0.65`

- keep counts：`127 / 82 / 53`
- CPU `dynamicvit_runtime_pruning`：`0.7127s`
- CPU `cipherprune_style_attention`：`0.7402s`
- SPU `cipherprune_style_attention`：`114.56s`

说明：

- 相比 `0.70`，SPU 单样本时延有小幅下降，约 `3.3%`。
- 该组是目前实验中**最值得继续观察**的一组。

#### `base_rate=0.60`

- keep counts：`117 / 70 / 42`
- CPU `dynamicvit_runtime_pruning`：`0.7280s`
- CPU `cipherprune_style_attention`：`0.7297s`
- SPU `cipherprune_style_attention`：`115.12s`

说明：

- 相比 `0.65`，SPU 时延未继续明显下降。
- 输出概率继续漂移，但在当前小样本实验中未出现 `argmax` 或 `threshold` 翻转。

## 综合判断

- `CipherPrune` 风格思路对当前项目**不是完全没有启发**，但在现有主线约束下，新增收益有限。
- 当前实现瓶颈并不只由 token 数决定，还受到：
  - `uniform attention`
  - 固定图 whole-forward
  - SPU kernel / 通信协议开销
  的共同影响。
- 在不重训模型的情况下，仅通过更激进的 keep-mask 推理侧替换，暂未观察到足以支撑替换正式主线的明显收益。

## 当前可用于相关工作的结论

可采用如下口径：

> 我们对 `CipherPrune` 风格密态渐进剪枝思路进行了并行实验。结果显示，在当前已采用近似注意力、固定图激活和两方 SPU whole-forward 改写的实现框架下，仅通过 external keep-mask 回放或最小 attention-based 剪枝近似，收益较为有限；其中 `base_rate=0.65` 在单样本 SPU 场景下相对当前主线有小幅时延下降，但总体尚不足以支持替换现有正式主线方案。

## 解释约束

- 不应将本文档中的 `cipherprune_style_attention` 表述为“完整 CipherPrune 实现”
- 不应将本文档中的结果表述为“正式主线收益”
- 如需进一步论证 `CipherPrune` 的真实价值，仍需：
  - 补完整 keep-mask / token importance 生成逻辑
  - 进行更大样本量 CPU/SPU 对比
  - 在必要时重新训练或重新蒸馏模型
