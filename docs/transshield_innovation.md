# Transshield 创新点文档

最后更新：`2026-05-20`

本文档只总结 **当前正式主线中已采纳并实际落地的创新点**，用于支撑最终作品报告、答辩展示和代码附录。默认前提如下：

- 医疗正式主线：`dynamic secure pruning + full privacy`
- 金融：`boundary stress validation only`
- 本文不把过程性对照、fallback 或失败实验写成正式创新点

## 1. 创新点总览

| 创新点 | 具体做法 | 解决的问题 | 实际收益 |
|---|---|---|---|
| 1 | 把删 token、阈值比较改写成可密态执行的 mask 选择 | 动态剪枝原本很难直接进入 MPC 图 | 医疗正式主线保留 dynamic secure pruning |
| 2 | 用带索引跟踪的安全排序完成 top-k 与 tie 处理 | 普通 top-k 在密态下不稳定，也容易引入复杂分支 | 动态裁剪决策链稳定，正式 gate 与参考路径保持一致 |
| 3 | 把 Predictor、阈值判断和 tie 处理全部放进安全环境内部执行 | 否则动态剪枝只能先在外部明文跑一遍，正式系统会退化成半隐私 | 输入、参数和动态决策同时受保护 |
| 4 | 统一采用更适合 MPC 的注意力、归一化和激活算子 | 标准算子在安全计算里代价高、稳定性差 | 医疗正式链路稳定运行，失败变体被明确排除 |
| 5 | 对医疗动态路径单独做公开阈值校准 | 动态路径不能直接沿用静态路径旧阈值 | 524 张验证集恢复到 `92.7481%` |
| 6 | 在前端与后端补上轻量控制面 | 纯 secure pipeline 默认把输入视为天然安全 | 医疗现场演示新增 DQA、审计哈希链、服务端权威快检闭环 |

## 2. 创新点一：pruning boundary 的协议友好重写

### 用到的技术

- `masking -> F_mux`
- `threshold compare -> F_less`
- dynamic pruning 决策边界的 secure-friendly 表达

### 产生的收益

- DynamicViT 的核心价值来自按样本变化的 token 保留决策，而不是一个固定结构分类 head。
- 这项重写把原本不适合 MPC 的删除式表达，变成可以进入 SPU 图执行的安全表达。
- 没有这一步，项目最多只能落到静态安全推理；有了这一步，医疗正式主线才能保留 dynamic secure pruning。

## 3. 创新点二：Encoded-Key Bitonic Sort 安全 Top-K

### 用到的技术

- encoded-key tie-breaking
- index-tracking bitonic sort
- 直接由排序索引构造 keep-mask

### 产生的收益

- 避免普通 top-k 对数据相关分支和不稳定 tie-breaking 的依赖。
- 让 secure pruning 决策链在固定图结构中保持稳定。
- 支撑 `stage_decision_match_ratio` 和最终 `argmax / threshold match` 的稳定对齐。

## 4. 创新点三：PredictorLG in-SPU + 双向隐私 runtime

### 用到的技术

- `secure_internal_pruning`
- `party_local_debug_share_load`
- `secret` 参数模式
- `reveal_policy = final_logits_only`

### 产生的收益

- 医院这一侧不需要交出原始输入，AI 公司这一侧不需要交出模型参数，双向隐私真正同时成立。
- 动态裁剪决策链不再依赖外部 keep-mask 回放，正式系统本身就具备完整动态能力。
- 正式落地可以明确写成“两方服务器在 2PC 环境里联合完成推理”。

## 5. 创新点四：MPC-Friendly 算子族

### 用到的技术

- `exact LN`
- `uniform attention`
- `fixed_square`

### 产生的收益

- 支撑 dynamic full-privacy 路径在真实运行时保持稳定。
- 让医疗主线围绕同一套 operator family 保持工程一致性。
- 已验证无收益或不稳定的变体被明确排除出正式主线：
  - `public-calibrated LN + clip0`
  - `clip3`

## 6. 创新点五：dynamic-path public threshold calibration

### 用到的技术

- dynamic path 概率输出的公开阈值搜索
- full-val reference threshold sweep
- 阈值回代到部署 gate

### 产生的收益

- 如果沿用静态路径旧阈值，医疗动态路径会在 32 张正式 gate 上退化到 `50%`。
- 单独对动态路径做阈值校准后：
  - 524 张验证集上的正式阈值精度达到 `92.7481%`
  - 把同一阈值回代到 32 张正式 gate 时，threshold accuracy 达到 `93.75%`
- 这项创新把“dynamic secure pruning 能跑”提升成“dynamic secure pruning 能被正确部署”。

## 7. 创新点六：前端 + 服务端轻量控制面

### 用到的技术

- 前端 worker 中的本地 DQA 与审计哈希链
- 服务端 share / tensor 权威快检
- 反重放与限频保护
- 审计落盘

### 产生的收益

- 医疗现场演示不再是“默认输入绝对安全”的脆弱管道。
- 前端先做质量预检、审计摘要和分片；服务端进入 SPU 前再做合法性检查与最终质量裁决。
- 页面能够显式展示 `quality_assurance`、`audit`、`control_plane_metrics` 三组闭环证据。

## 8. 不计入正式创新点的方向

以下方向可以保留为详细验证中的过程说明，但不再计入正式创新点：

- `true static no-pruning` 作为第二条正式主线
- distillation
- token recycle
- token_ratio speedup
- clip3
- public-calibrated LN + clip0
- decomposed LRD
- 历史 keep-mask finance 链
- 把金融包装成第二条正式落地主线

## 9. 答辩与报告中的推荐写法

- 前半部分先写医疗正式落地模型，再写创新点。
- 创新点只写“用到了什么技术、落在什么位置、带来了什么收益”。
- 金融只作为边界压力验证区出现，不写成“正式双域落地”。
- 失败项、fallback、历史链和 rejected 方向，统一放到后面的“详细验证与采用原因”章节再解释。
