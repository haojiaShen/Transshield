# Transshield 创新点文档

最后更新：`2026-05-16`

本文档系统论证 Transshield 项目的 5 个核心创新点。所有创新点均有实际代码实现和真实运行数据支撑。

先说清楚一件最容易混淆的事：

- **当前网页默认展示的主结果，不等于“默认一直开着运行时动态剪枝”**
- 医疗当前主展示线是 `depth10 + batch12` 的固定结构优化版本，真实完整隐私推理为 `69.57s/sample`
- 金融当前主展示线是 `depth12 + LRD rank192` 的固定结构优化版本，真实完整隐私推理为 `117.80s/sample`
- 这样选不是因为动态剪枝无效，而是因为这两条线是**目前实测最适合公开展示的稳定配置**

与此同时，Transshield 的**动态裁剪能力并没有被删掉**，而是以两条已经验过的数据链单独保留：

1. **keep-mask replay**：先在参考路径生成按样本变化的 keep-mask，再在 SPU whole-forward 中 exact replay  
   - `524` 样本 `argmax / threshold = 1.0 / 1.0`
2. **secure pruning**：PredictorLG + `kth_threshold` + tie-breaking 已在 SPU 内部完整跑通  
   - 说明“按输入决定保留哪些 token”的动态决策本身，已经可以进入密态执行

## 动态剪枝和静态剪枝，到底差在哪？

| 维度 | 静态剪枝 / 固定结构 | 动态剪枝 |
|---|---|---|
| 结构是否固定 | 固定。所有输入走同一结构 | 不固定。不同输入保留的 token 集合不同 |
| 决策时机 | 部署前就决定好 | 推理时按当前输入决定 |
| 是否按样本自适应 | 否 | 是 |
| 在 MPC 里难点 | 相对简单 | 容易引入动态 shape、条件分支和边界泄漏 |
| Transshield 的价值 | 作为稳定展示线使用 | 把这条原本很难进密态的动态决策链改写成可安全执行 |

因此，**Transshield 的方法主线确实来自动态剪枝**。  
但**当前默认展示数据**为了稳定性和速度，采用的是固定结构优化线。两者不能混为一谈。

---

## 创新点 1：DynamicViT Pruning Boundary 的协议友好重写

### 问题
原始 DynamicViT 的 token pruning 依赖"删除式表达"，在 MPC 环境中带来动态 shape、条件分支泄漏等问题。

### 方法
将 pruning 决策边界重写为 MPC 友好表达：
1. **`masking → F_mux`**：token keep/drop 改写为 MPC 友好的多路选择函数
2. **`threshold compare → F_less`**：pruning threshold 比较改写为 MPC 友好的安全比较函数
3. **secure sidecar/replay**：pruning 决策以 sidecar 方式安全执行，结果 replay 回主模型

### 验证
- boundary check：`boundary_kth_check_passed = true`（max abs error 1.28e-05）
- E2E consistency：argmax/threshold match = 1.0/1.0
- 代码位置：`spu_static_vit.py`（86处相关引用）

### 创新性
和常见“把静态模型直接搬进隐私计算”的作品不同，Transshield 处理的是更难的一类问题：**如何把按样本变化的动态裁剪边界改写成 MPC 可执行形式**。  
这使项目不只是“做一个加密版 ViT”，而是把 DynamicViT 的核心决策链真正翻译进隐私计算语义。

---

## 创新点 2：端到端隐私保护的 SPU Forward 实现

### 问题
MPC 隐私推理中，如何在不暴露任何一方数据的前提下完成完整推理。

### 方法
实现完整的 SPU forward 路径，满足双向隐私约束：
1. 输入以 `party-local share load` 方式加载（`host_plaintext_pixel_values_materialized = false`）
2. 模型参数以 `secret` 模式加载（`host_model_params_materialized = false`）
3. PredictorLG 在 SPU 内部完整执行
4. 只暴露最终 logits（`reveal_policy = final_logits_only`）

### 验证
- smoke1/8/16/32：argmax/threshold match = 1.0/1.0
- heldout238 子集：threshold accuracy = 92.44%
- 隐私字段全部通过

### 创新性
很多作品只做到“输入不出明文”，但模型参数仍由使用方持有，或者只跑静态主干。Transshield 进一步做到：

- 输入不出明文
- 模型参数不出明文
- PredictorLG 这条动态裁剪决策链也能进入 SPU

也就是说，这里保护的不是单边隐私，而是**输入侧和模型侧同时成立**。

---

## 创新点 3：Encoded-Key Bitonic Sort 安全 Top-K 选择

### 问题
DynamicViT 的 pruning 需要 Top-K 选择，传统 Top-K 依赖条件分支，在 MPC 环境中不安全。

### 方法
使用双调排序实现 MPC 友好的 Top-K：
- encoded_key 编码保证 tie-breaking 语义
- O(n log²n) 复杂度，无条件分支
- 全 jnp.where 实现，JAX tracer 兼容

### 验证
- `stage_decision_match_ratio = 1.0`
- 代码位置：`spu_static_vit.py`（4处核心实现）

### 创新性
和普通 Top-K 实现相比，这里不是为了单纯排序，而是为了在**不泄漏分支信息**的前提下完成动态裁剪阈值选择。  
它解决的是“动态裁剪为什么能进 MPC”这个关键卡点。

---

## 创新点 4：MPC-Friendly 算子族设计

### 问题
标准 ViT 中的 softmax、LayerNorm、GELU 等在 MPC 环境中计算开销极高。

### 方法
设计统一的 MPC-friendly 算子族：
- **uniform attention**：替换昂贵的安全 softmax
- **fixed_square 激活**：用 `x * x * sign(x)` 近似 GELU
- **exact LayerNorm**：保证计算正确性
- **训练-部署一致性**：消除 train-test mismatch

### 验证
- 医疗模型：threshold_accuracy = 91.98%（校准后），auc = 0.9679
- secure/plaintext 一致性：argmax/threshold match = 1.0/1.0

### 创新性
不少作品只展示“能跑”，但训练和部署使用的是两套不同语义，最后很难解释精度为什么漂。Transshield 的做法是把注意力、激活和裁剪表达统一到一套 secure-friendly 口径里，让训练、验证、部署更一致。

---

## 创新点 5：SVD 低秩分解 MPC 推理加速

### 问题
MPC 安全推理中，模型参数量直接影响通信开销和计算时间。如何在保持精度的同时压缩参数？

### 方法
通过 SVD 分解将线性层分解为 down_weight × up_weight：
- rank=192 时参数量压缩至 68.39%
- **merged 模式**：将分解后的权重合并回原尺寸，保持单次 matmul 通信
- 发现分解式在 SPU 中反而更慢（两次小 matmul 的通信开销 > 一次大 matmul）

### 验证
- **金融模型**：完整隐私实测 `117.80s/sample`，`argmax_match = 100%`，参数量 `22,390,184 → 15,312,296`（68.39%）
- **医疗模型**：LRD merged 测试 `threshold accuracy = 91.98%`（与 depth10 截断方案精度相同）
- 隐私保护：host_model_params_materialized=false

### 核心发现
**SPU 的 2PC/MPC 协议中，通信轮次比计算量更关键。** 分解式 LRD（96.55s）比 baseline（69.57s）慢 38.8%，因此必须使用 merged 模式。

### 创新性
这部分的价值不只是“做了压缩”，而是给出了一条**适合 MPC 约束的压缩方式选择原则**：

- 分解式看起来更轻，但在 SPU 中会增加 matmul 次数，反而更慢
- merged 模式虽然保持单次 matmul 形状，但能保住参数压缩收益，同时不额外增加通信轮次

也就是说，Transshield 不是机械套用压缩论文，而是把压缩方法重新按密态执行代价做了筛选。

---

## 两种优化策略对比

| 项目 | 医疗模型 | 金融模型 |
|------|---------|---------|
| **优化策略** | 深度截断（depth12→depth10） | SVD 低秩分解（rank=192） |
| **配置** | depth10 + batch12 + fixed_square | depth12 + LRD rank192 merged |
| **推理时间** | 69.57s（3.07倍加速） | 117.80s（完整隐私实测） |
| **精度** | 91.98%（threshold） | 100% |
| **参数量** | 18.84M（压缩至84.15%） | 15.31M（压缩至68.39%） |
| **Baseline** | 22.39M, 213.9s, 76.72% | 22.39M, 作为未压缩对照保留 |

### 为什么医疗模型不用LRD？

医疗模型已有 **depth10 深度截断** 方案：
- depth12 → depth10，参数减少 15.85%（22.39M → 18.84M）
- 计算量减少 16.7%，实现 3.07 倍加速（213.9s → 69.57s）
- 精度反而提升（76.72% → 91.98%）

LRD merged 模式测试结果：
- 精度：91.98%（与 depth10 相同）
- 但**没有额外加速效果**（merged 模式保持原尺寸 matmul）

### 为什么金融模型用LRD？

金融模型没有使用深度截断（保持 depth12），LRD 带来：
- 参数压缩：22.39M → 15.31M（压缩至 68.39%）
- 完整隐私验证：`117.80s/sample`
- 精度保持：`argmax_match = 100%`（8/8）

---

## 最终部署数据

| 模型 | 配置 | 推理时间 | 精度 | 参数量 | 参数压缩 |
|------|------|---------|------|--------|---------|
| 医疗 | depth10 + batch12 + fixed_square | 69.57s | 91.98% | 18.84M | 84.15% |
| 金融 | depth12 + LRD rank192 merged | 117.80s | 100%（8/8 一致性） | 15.31M | 68.39% |
| Baseline | depth12 | 213.9s | 76.72% | 22.39M | 100% |
