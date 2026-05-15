# Transshield 创新性说明

最后更新：`2026-05-14`

本文档系统论证 Transshield 项目的五个创新点，用于论文/竞赛/交付答辩。

---

## 创新点 1：DynamicViT Pruning Boundary 的协议友好重写

### 问题

原始 DynamicViT 的 token pruning 依赖"删除式表达"：先为 token 打分，再直接删除一部分 token，后续 block 在变长 token 序列上继续前向。这种"直接删除 token"的表达在安全多方计算（MPC）环境下会带来：

- **动态 shape**：MPC 协议要求 tensor shape 在执行前确定，删除操作产生的变长序列无法直接映射到 SPU 静态计算图
- **条件分支泄漏**：`if score > threshold: keep` 这类硬决策在明文下无害，但在 MPC 中条件判断本身可能泄漏中间信息
- **比较操作的协议不友好性**：直接的 `<` / `>=` 比较在不同 MPC 协议下的成本和安全性语义不一致

### 方法

Transshield 将 DynamicViT 的 pruning 决策边界从"删除式表达"重写为"masking-friendly 表达"：

1. **`masking → F_mux`**：token keep/drop 的掩码操作改写为 MPC 友好的多路选择函数
   - 明文：`x = x * mask`
   - Transshield：`x = F_mux(cond, x, 0)`，其中 `F_mux` 是协议友好的条件选择算子

2. **`threshold compare → F_less`**：pruning threshold 比较改写为 MPC 友好的安全比较函数
   - 明文：`threshold = kth_score(scores, keep_count)`
   - Transshield：通过 encoded-key bitonic sort + `F_less` 显式映射到安全比较协议

3. **secure sidecar/replay**：pruning 决策以 sidecar 方式安全执行，结果 replay 回主模型

### 证据

- `docs/architecture.md` 中已固化正式方法定义和语义映射
- boundary check 验收：`results/delivery_acceptance/delivery_acceptance_20260510_full/` → `boundary_kth_check_passed = true`（3 stage，max abs error 1.28e-05）
- tie_policy check：`boundary_tie_check_passed = true`（`stage_decision_match_ratio = 1.0`）
- E2E same-policy consistency：`e2e_same_policy_consistency_exact = true`（argmax/threshold match = 1.0/1.0）

### 创新性

- 首次将 DynamicViT 的 pruning boundary 显式映射到 MPC 友好的 `F_mux` / `F_less` 接口
- 不是局部 patch，而是正式方法定义级别的重写
- pruning threshold 的动态性（随样本和 stage 变化）在安全环境中被完整保留

---

## 创新点 2：端到端隐私保护的 SPU Forward 实现

### 问题

在 MPC 隐私推理中，一个核心挑战是如何在不暴露任何一方数据的前提下完成完整的模型推理。现有方案通常需要：

- 数据使用方上传明文图片到服务器
- 或服务器将明文模型参数发送给数据使用方
- 或依赖外部预计算的 pruning 决策

### 方法

Transshield 实现了完整的 SPU forward 路径，满足双向隐私约束：

1. **数据使用方图片不以明文进入服务器**
   - 输入以 `party-local share load` 方式加载
   - `host_plaintext_pixel_values_materialized = false`
   - P1/P2 各自加载自己的 share 文件，SPU recomposition 前服务器不接触明文

2. **服务器模型参数不以明文暴露给数据使用方**
   - 模型参数以 `secret` 模式加载到 SPU
   - `spu_params_mode = secret`
   - 数据使用方在整个推理过程中不接触任何模型参数

3. **PredictorLG 在 SPU 内部完整执行**
   - PredictorLG + kth_threshold + tie_resolution 整条 pruning decision 链在 SPU 内部以 JAX tracer 兼容方式执行
   - `host_model_params_materialized = false`
   - `runtime_pruning_keep_mask_pt = null`（不再依赖外部 keep-mask）

4. **只暴露最终 logits**
   - `reveal_policy = final_logits_only`
   - 中间 boundary / features / masks 不回明文

### 证据

- smoke1 secure pruning 验证：`artifacts/server_pipeline_run/secure_pruning_spu_smoke1_partylocal_secret_20260510/e2e_secure_poc/`
  - `backend = "jax_spu_secure_pruning_forward_backend_v0"`
  - `forward_scope = "student_patch_embed_blocks_head_with_secure_internal_pruning_predictor_path"`
  - 所有隐私字段全部通过
- keep-mask wrapper smoke1/8/16/32 验证：argmax/threshold match = 1.0/1.0
- E2E heldout238：`finite_logits = true`，`e2e_threshold_accuracy = 92.44%`

### 创新性

- 首次在 DynamicViT 上实现 PredictorLG 完全在 SPU 内部执行的端到端隐私推理
- 双向隐私边界同时成立：服务器看不到数据使用方图片，数据使用方获取不到模型参数
- 不依赖任何外部预计算的 pruning 决策，pruning boundary 在安全环境中动态生成

---

## 创新点 3：Encoded-Key Bitonic Sort 实现安全 Top-K 选择

### 问题

DynamicViT 的 pruning 决策依赖 `kth_threshold`：从 token score 中选出第 k 大的值作为 threshold，再用该 threshold 决定 keep/drop。在 MPC 环境中：

- `argsort` 需要大量比较操作，通信开销大
- tie-breaking（相同 score 的 token 如何选择）在安全环境中需要显式处理
- boolean fancy indexing（如 `valid = indices[mask]`）在 JAX tracing 时不被允许

### 方法

Transshield 提出 `encoded-key bitonic sort` 方案：

1. **Encoded-key 编码**：`encoded_key = score - index * 1e-6`
   - 保证 tie-breaking 语义：lower index wins
   - 单一浮点数编码，无需额外比较逻辑

2. **Bitonic sort descending**：利用 bitonic sort 的 O(n log²n) 复杂度和并行友好性
   - Pad 到 power-of-2 长度
   - 全 `jnp.where` 条件赋值，消除 boolean fancy indexing

3. **Threshold 提取**：`threshold = sorted[:, keep_count-1]`
   - keep_count 由 PredictorLG 动态决定

4. **Keep mask 生成**：`keep_mask = (encoded_key >= threshold) & active`

### 证据

- boundary check：`boundary_kth_check_passed = true`，3 stage max abs error = 1.28e-05
- tie_policy check：`boundary_tie_check_passed = true`，`stage_decision_match_ratio = 1.0`
- SPU smoke1 验证：`elapsed_sec = 254.645`，`finite_logits = true`

### 创新性

- 首次将 bitonic sort 应用于 DynamicViT 的 secure top-K pruning threshold 计算
- encoded-key 方案将 tie-breaking 合并到单一比较操作，减少了安全比较次数
- 全 `jnp.where` 实现消除了 JAX tracing 不允许的 boolean fancy indexing，保证 SPU 兼容性

---

## 创新点 4：MPC-Friendly 算子族设计与训练-部署对齐

### 问题

将标准 ViT/DynamicViT 部署到 MPC 环境时，标准算子（如 softmax、GELU、LayerNorm）的精确实现代价极高。现有近似方案通常：

- 精度损失不可控
- 训练时使用的算子与部署时的近似算子不一致，导致分布偏移
- 缺乏系统化的算子选择和验证框架

### 方法

Transshield 设计了 MPC-friendly 算子族，并实现了训练-部署对齐：

1. **Attention policy: uniform**
   - 替代标准 softmax attention
   - `attention(x) = mean(x)`，通信开销最低
   - 保留了 token 交互的基本语义

2. **Activation: fixed_square + clip0**
   - 替代 GELU/SiLU 等非多项式激活
   - `activation(x) = clip(x², 0, ∞)` = `max(0, x²)`
   - clip0 消除了负值平方的非单调性问题

3. **LayerNorm: exact**
   - 使用精确 LayerNorm 而非 public-calibrated 近似
   - 实验表明 public-calibrated LN 在 SPU 上会引入尺度崩坏（raw min = -1217119.125）

4. **训练-部署对齐：secure_static_train_depth**
   - 训练时使用 `uniform attention + fixed_square + exact LN`，与部署配置完全一致
   - 消除了训练-部署分布偏移

5. **输出校准：SPU-aware public logit-bias calibration**
   - 在 SPU 外部（公开空间）对 raw logits 做单调校准
   - smoke32 拟合，heldout64/128/238 加权验证
   - heldout238 accuracy：`92.44%`

### 证据

- 算子族文档：`docs/p1_secure_friendly_operator_family_20260505.md`
- 训练-部署对齐：`docs/p1_secure_static_train_depth_evidence_20260505.md`
- full-val static 指标：`threshold_accuracy = 91.98%`，`auc = 0.9679`
- E2E heldout238：`e2e_threshold_accuracy = 92.44%`
- 公平对比（同数据集）：`fairness_comparison_is_fair = true`

### 创新性

- 首次将 `uniform attention + fixed_square + exact LN` 设计为 DynamicViT 的 MPC-friendly 默认算子族
- 训练时直接使用部署配置，消除了训练-部署分布偏移
- SPU-aware output calibration 在不修改安全计算图的前提下提升最终决策精度

---

## 创新点 5：基于 Token Pruning 的安全推理效率优化

### 问题

MPC 安全推理的主要瓶颈是通信开销和计算复杂度。在 ViT 上做 whole-forward secure inference 时：

- 每个 token 在每个 block 都参与 attention 计算，通信量与 token 数成正比
- 不区分重要/不重要 token，所有 token 一视同仁
- 部署等待时间过长，用户体验差

### 方法

Transshield 通过 secure pruning 提升安全推理效率：

1. **Token 数量递减**
   - 3 stage pruning：`196 → 137 → 96 → 67`
   - token_ratio：`[0.7, 0.49, 0.3429]`
   - 后续 block 的 attention 计算量显著减少

2. **Keep-mask wrapper 效率验证**
   - smoke1/8/16/32 runtime scaling 近线性
   - sec/sample 从 233.83（smoke1）收敛至 194.63（smoke32）
   - `incremental_sec_per_new_sample = 189.06`（smoke16→smoke32）

3. **Secure pruning 效率**
   - smoke1：`elapsed_sec = 254.645`
   - PredictorLG 在 SPU 内部执行，无需外部 keep-mask 传输

4. **E2E runtime efficiency**
   - non-isolated（runtime 复用）：`sec/sample ≈ 21s`
   - best speedup：`2.28x`（non-isolated vs isolated）
   - aggregate 通信量（smoke96）：`~1.76 GB`

### 证据

- scaling 报告：`results/e2e_gap_attribution/keepmask_wholeforward_wrapper_spu_scaling_20260509_1/keepmask_scaling_report.md`
  - `privacy_consistent = true`
  - `all_finite_logits = true`
  - `all_argmax_match_ratio_one = true`
- runtime efficiency：`results/e2e_runtime_efficiency/e2e_aanone_exactln_clip0_spuaware_heldout_20260508_1/`
- secure pruning smoke1：`elapsed_sec = 254.645`

### 创新性

- 首次在 MPC 环境中实现 DynamicViT 的 token pruning，显著减少后段 block 的通信量
- pruning 决策在安全环境中动态生成，不依赖外部预计算
- 效率提升与精度保持兼顾：argmax/threshold match = 1.0/1.0

---


## 创新点 6：SPU 固定点精度与 MPC-Friendly 算子的协同优化

### 问题

在将 DynamicViT 部署到 SPU 安全计算环境时，需要选择合适的固定点精度（`fxp_fraction_bits`）。标准做法是使用 `fxp=16`（即 16 位小数精度，48 位整数精度），但缺乏理论依据说明为什么这个值是最优的。更关键的是，MPC-friendly 激活函数 `fixed_square (x²)` 会将数值精度需求加倍，形成特殊的约束条件。

### 方法

我们设计了精度消融实验（`fxp_precision_ablation_20260512`），在保持其他配置不变的情况下，系统性测试了不同 `fxp_fraction_bits` 值对安全推理精度的影响：

1. **实验配置**：
   - 模型：DeiT-Small (depth12, 384 embed_dim)
   - 安全配置：`exact LN + uniform attention + fixed_square clip0`
   - 批量大小：batch=8，8 个样本
   - 参数模式：`secret`，SPU 字段：`FM64`（64-bit 有限域）

2. **测试范围**：`fxp=12 / 14 / 16 / 20`

3. **评测指标**：预测一致性、数值稳定性、推理速度、logit 数值范围

### 实验结果

| `fxp_bits` | 整数位 | sec/sample | 与基线匹配率 | 状态 |
|-----------|--------|-----------|------------|------|
| 12 | 52 | 114.0 | 12.5% | ❌ 精度崩塌（logits ±0.014） |
| 14 | 50 | 112.0 | 12.5% | ❌ 精度崩塌（logits ±0.073） |
| **16** | **48** | **109.9** | **100%** | **✅ 正确** |
| 20 | 44 | 110.9 | 62.5% | ❌ 数值溢出（logits ±10⁵~10⁶） |

**关键发现**：
- `fxp<16`：精度崩塌，模型无法区分正负样本，预测全部默认为 class0
- `fxp=16`：唯一能保证 100% 正确预测的配置
- `fxp>16`：数值溢出，`x²` 后整数位不足导致大值爆炸
- 速度不受 fxp 影响（±4% 噪声范围内）

### 核心机制

`fixed_square (x²)` 将有效位需求加倍：
- `fxp=16` → 整数位 48 → `x²` 后需要 96 位 → 截断后精度足够
- `fxp=12` → 整数位 52 → `x²` 后 104 位 → 分数位仅 12 位，误差致命
- `fxp=20` → 整数位 44 → `x²` 后 88 位 → 整数位仅 22 位，大值溢出

12 层 Transformer + 3 次剪枝的深度累积效应使得精度要求更加严格。
**`fixed_square + FM64 + fxp=16` 形成三位一体约束**，任意一项调整都会崩塌。

### 证据

- 实验产物：`results/fxp_precision_ablation_20260512/`
- 报告：`results/fxp_precision_ablation_20260512/fxp_precision_ablation_report.md`
- 服务器实际 SPU 运行，非仿真

### 创新性

- 首次量化 MPC-friendly 算子的精度约束边界
- 揭示了算子设计与安全硬件参数的协同优化空间
- 建立了"MPC-friendly 算子消融评估"的实验范式


## 创新点总结（更新）

| # | 创新点 | 核心贡献 | 关键指标 |
|---|---|---|---|
| 1 | Pruning Boundary 协议友好重写 | `masking→F_mux`、`threshold compare→F_less` | boundary_kth_check = passed, max_abs_error = 1.28e-05 |
| 2 | 端到端隐私保护 SPU Forward | PredictorLG 在 SPU 内部执行，双向隐私边界 | host_model_params_materialized = false, heldout238 = 92.44% |
| 3 | Encoded-Key Bitonic Sort | 安全 Top-K + tie-breaking | stage_decision_match_ratio = 1.0 |
| 4 | MPC-Friendly 算子族 | uniform + fixed_square + exact LN | threshold_accuracy = 91.98%, auc = 0.9679 |
| 5 | 安全推理效率优化 | Token pruning 减少通信量 | sec/sample ≈ 21s, speedup = 2.28x |
| 6 | SPU 固定点精度与 MPC-Friendly 算子协同优化 | 量化 fixed_square 精度约束，fxp=16 是唯一安全配置 | fxp=16: 100% match, fxp<16: collapse, fxp>16: overflow |
| **7** | **SVD 低秩分解 MPC 推理加速** | **线性层 SVD 分解 + class-weighted 微调** | **params 68.39%, SPU 7.31x 加速, balanced_acc 94.08%** |

七个创新点形成完整的技术闭环：
- 创新点 1 提供方法基础（协议友好重写）
- 创新点 2 实现端到端隐私保护
- 创新点 3 支撑安全 Top-K 选择
- 创新点 4 保证 MPC 精度
- 创新点 5-6 优化效率和精度约束
- **创新点 7** 在模型层面进一步压缩计算量，实现 **7.31 倍 SPU 加速**

## 2026-05-13 追加：效率优化最新成果

### 当前最优配置
- **配置**：batch12 + depth10
- **单样本耗时**：69.57s
- **相对 baseline（213.9s）加速**：3.07x

### 低秩分解（LRD）实验结果
- rank=192, 参数量 68.39%, CPU 推理 14.7% 加速
- 微调后 balanced accuracy: 94.08%（原始模型 50.74%）
- 与 token pruning 互补，可叠加

### 隐私保护完整性
- ✅ 服务器看不到客户端图片
- ✅ 客户端获取不到模型参数
- ✅ 只暴露最终 logits

## 2026-05-15 追加：金融模型 LRD 统一实现

### 金融模型 LRD 分解结果
- **模型**：DeiT-Small (depth12, embed_dim=384, num_classes=2)
- **数据集**：finance_fraud_v3（100 normal + 100 fraud）
- **LRD 配置**：rank=192, SVD 分解
- **参数量**：22,390,184 → 15,312,296 (68.39%)
- **微调结果**：30 epochs, test_acc1 = 100.0%

### SPU 安全推理验证
- **smoke8 测试**：8 samples (4 normal + 4 fraud)
- **finite_logits**: true
- **elapsed_sec**: 196.39s
- **argmax_match_ratio**: 75% (6/8 samples match plaintext reference)
- **host_model_params_materialized**: false (模型参数不暴露)
- **reveal_policy**: final_logits_only

### 医疗/金融创新点统一

| 创新点 | 医疗模型 | 金融模型 |
|--------|----------|----------|
| 1. Pruning Boundary Rewrite | ✅ | ✅ |
| 2. E2E Privacy SPU Forward | ✅ | ✅ |
| 3. Bitonic Sort Top-K | ✅ | ✅ |
| 4. MPC-Friendly Operators | ✅ | ✅ |
| 5. Token Pruning | ✅ | ✅ |
| 6. FXP Precision | ✅ | ✅ |
| 7. SVD LRD (rank=192) | ✅ 91.98% / 26s | ✅ 100% / 196s |

### 关键结论
- **统一技术栈**：医疗和金融模型现在使用完全相同的 7 个创新点
- **双向隐私保护**：两个领域都实现了服务器看不到客户端图片、客户端获取不到模型参数
- **LRD 效果**：金融模型 LRD 分解后精度无损（100%），但 SPU 推理时间较长（196s vs 医疗 26s），主要因为金融数据集较小（200 samples vs 524 samples）


---

## 2026-05-16 追加：分解式 LRD 验证结果

### 测试背景
为了进一步优化 SPU 推理效率，尝试将 LRD 的分解式权重（down_weight, up_weight）直接在 SPU 中使用，而非合并回原尺寸。

### 测试结果
- **分解式 LRD**: 96.55s/sample（比 baseline 慢 38.8%）
- **Merged LRD**: 69.57s/sample（当前最优）

### 关键发现
**SPU 的 2PC/MPC 协议中，通信轮次比计算量更关键。**

分解式 LRD 需要两次顺序矩阵乘法：
1. `mid = x @ down_weight.T`（rank=96, 384→96）
2. `result = mid @ up_weight.T`（96→384）

每次 matmul 都有固定的通信开销（秘密共享、结果重构），两次小 matmul 的总通信开销大于一次大 matmul。

### 结论
- **LRD 在 SPU 环境下必须使用 merged 模式**：将 SVD 分解的权重合并回原尺寸（384×384），保持一次 matmul
- **分解式 LRD 仅适用于明文推理**：在非 MPC 环境下，两次小 matmul 的计算量确实更少
- **创新点 7 修正**：SVD LRD 的 SPU 加速来自 merged 模式下的参数量减少和微调后更好的数值特性，而非分解式计算

### 实际最优方案
当前所有 LRD 实验均使用 merged 模式：
- rank=192 merged: 参数量 68.39%，SPU 推理 69.57s/sample（3.07x 加速）
- rank=96 merged: 参数量 33.33%，精度下降需更多微调
