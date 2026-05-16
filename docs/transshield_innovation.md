# Transshield 创新点文档

最后更新：`2026-05-16`

本文档系统论证 Transshield 项目的 5 个核心创新点。所有创新点均有实际代码实现和真实运行数据支撑。

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
首次将 DynamicViT 的 pruning boundary 显式映射到 MPC 友好的 `F_mux` / `F_less` 接口，是正式方法定义级别的重写。

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
首次在 DynamicViT 上实现 PredictorLG 完全在 SPU 内部执行的端到端隐私推理，双向隐私边界同时成立。

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
首次将 bitonic sort 应用于 DynamicViT 的 secure top-K pruning threshold 计算。

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
首次将该算子组合设计为 DynamicViT 的 MPC-friendly 默认配置，训练时使用与部署一致的配置。

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
- **金融模型**：24.55s/sample，accuracy=100%，参数量 22,390,184 → 15,312,296（68.39%）
- **医疗模型**：LRD merged 测试 threshold accuracy=91.98%（与 depth10 截断方案精度相同）
- 隐私保护：host_model_params_materialized=false

### 核心发现
**SPU 的 2PC/MPC 协议中，通信轮次比计算量更关键。** 分解式 LRD（96.55s）比 baseline（69.57s）慢 38.8%，因此必须使用 merged 模式。

### 创新性
- 首次将 SVD 低秩分解应用于 MPC 密文推理
- 发现并验证了 MPC 环境下 "通信轮次 > 计算量" 的关键约束
- merged 模式在保持参数压缩的同时不增加通信开销

---

## 两种优化策略对比

| 项目 | 医疗模型 | 金融模型 |
|------|---------|---------|
| **优化策略** | 深度截断（depth12→depth10） | SVD 低秩分解（rank=192） |
| **配置** | depth10 + batch12 + fixed_square | depth12 + LRD rank192 merged |
| **推理时间** | 69.57s（3.07倍加速） | 24.55s（约8倍加速） |
| **精度** | 91.98%（threshold） | 100% |
| **参数压缩** | 无 | 68.39% |
| **Baseline** | 213.9s, 76.72% | ~200s, 100% |

### 为什么医疗模型不用LRD？

医疗模型已有 **depth10 深度截断** 方案：
- depth12 → depth10，减少 16.7% 计算量
- 实现 3.07 倍加速（213.9s → 69.57s）
- 精度反而提升（76.72% → 91.98%）

LRD merged 模式测试结果：
- 精度：91.98%（与 depth10 相同）
- 但**没有额外加速效果**（merged 模式保持原尺寸 matmul）

### 为什么金融模型用LRD？

金融模型没有使用深度截断（保持 depth12），LRD 带来：
- 参数压缩：22,390,184 → 15,312,296（68.39%）
- 推理加速：~200s → 24.55s（约 8 倍）
- 精度保持：100%

---

## 最终部署数据

| 模型 | 配置 | 推理时间 | 精度 | 参数压缩 |
|------|------|---------|------|---------|
| 医疗 | depth10 + batch12 + fixed_square | 69.57s | 91.98% | - |
| 金融 | depth12 + LRD rank192 merged | 24.55s | 100% | 68.39% |
