# Transshield 创新点文档

最后更新：`2026-05-16`

本文档系统论证 Transshield 项目的 6 个核心创新点。

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
2. **`threshold compare → F_less`**：pruning threshold 比较改写为 MPC 友好的安全比较函数
3. **secure sidecar/replay**：pruning 决策以 sidecar 方式安全执行，结果 replay 回主模型

### 验证
- boundary check：`boundary_kth_check_passed = true`（max abs error 1.28e-05）
- tie_policy check：`boundary_tie_check_passed = true`（`stage_decision_match_ratio = 1.0`）
- E2E consistency：`e2e_same_policy_consistency_exact = true`（argmax/threshold match = 1.0/1.0）

### 创新性
- 首次将 DynamicViT 的 pruning boundary 显式映射到 MPC 友好的 `F_mux` / `F_less` 接口
- 不是局部 patch，而是正式方法定义级别的重写
- pruning threshold 的动态性（随样本和 stage 变化）在安全环境中被完整保留

---

## 创新点 2：端到端隐私保护的 SPU Forward 实现

### 问题
在 MPC 隐私推理中，核心挑战是如何在不暴露任何一方数据的前提下完成完整的模型推理。

### 方法
Transshield 实现了完整的 SPU forward 路径，满足双向隐私约束：
1. **数据使用方图片不以明文进入服务器**
   - 输入以 `party-local share load` 方式加载
   - `host_plaintext_pixel_values_materialized = false`
2. **服务器模型参数不以明文暴露给数据使用方**
   - 模型参数以 `secret` 模式加载到 SPU
   - `spu_params_mode = secret`
3. **PredictorLG 在 SPU 内部完整执行**
   - PredictorLG + kth_threshold + tie_resolution 整条 pruning decision 链在 SPU 内部执行
   - `host_model_params_materialized = false`
4. **只暴露最终 logits**
   - `reveal_policy = final_logits_only`

### 验证
- smoke1 secure pruning：所有隐私字段全部通过
- keep-mask wrapper smoke1/8/16/32：argmax/threshold match = 1.0/1.0
- E2E heldout238：`finite_logits = true`，`e2e_threshold_accuracy = 92.44%`

### 创新性
- 首次在 DynamicViT 上实现 PredictorLG 完全在 SPU 内部执行的端到端隐私推理
- 双向隐私边界同时成立：服务器看不到数据使用方图片，数据使用方获取不到模型参数

---

## 创新点 3：Encoded-Key Bitonic Sort 安全 Top-K 选择

### 问题
DynamicViT 的 pruning 需要对 token score 进行 Top-K 选择，传统 Top-K 依赖条件分支（if/else），在 MPC 环境中不安全。

### 方法
使用双调排序（Bitonic Sort）实现 MPC 友好的 Top-K 选择：
- **encoded_key 编码**：将 (score, index) 编码为单一值，保证 tie-breaking 语义
- **比较器网络**：O(n log²n) 复杂度的安全排序，无条件分支
- **全 jnp.where 实现**：JAX tracer 兼容，可在 SPU 内部执行

### 验证
- `stage_decision_match_ratio = 1.0`（与明文 Top-K 完全一致）

### 创新性
- 首次将 bitonic sort 应用于 DynamicViT 的 secure top-K pruning threshold 计算
- encoded_key 编码保证了 tie-breaking 语义在安全环境中的正确性

---

## 创新点 4：MPC-Friendly 算子族设计

### 问题
标准 ViT 中的 attention softmax、LayerNorm、GELU 激活函数等在 MPC 环境中计算开销极高或不安全。

### 方法
设计统一的 MPC-friendly 算子族：
- **uniform attention**：将 softmax 注意力替换为均匀注意力，避免昂贵的安全 softmax 计算
- **fixed_square 激活**：用 `x * x * sign(x)` 近似 GELU，仅需乘法和符号判断
- **exact LayerNorm**：使用精确的安全层归一化，保证计算正确性
- **训练-部署一致性**：训练时使用与部署一致的算子配置，消除 train-test mismatch

### 验证
- threshold_accuracy = 91.98%（校准后），auc = 0.9679
- secure/plaintext 一致性：argmax/threshold match = 1.0/1.0

### 创新性
- 首次将该算子组合设计为 DynamicViT 的 MPC-friendly 默认配置
- 训练-部署一致性设计消除了传统方案中的 approximation gap

---

## 创新点 5：LUT GELU 高精度激活函数

### 问题
fixed_square 激活函数（`x * x * sign(x)`）虽然 MPC 友好，但精度损失较大（76.72%），限制了实际应用。

### 方法
设计 16 段分段线性 GELU 近似：
- **均匀采样**：在 [-8, 8] 区间均匀采样 16 个点
- **分段线性插值**：仅需比较和线性运算，无需特殊函数（erf, tanh 等）
- **二分搜索优化**：O(log N) 安全比较，16 段仅需 4 次安全比较
- **MPC 友好**：全部操作可映射到 SPU 的安全比较和安全乘法

### 验证
- 明文精度：**97.33%**（相比 fixed_square 的 76.72% 提升 20.61pp）
- AUC：0.9937
- SPU 端到端：threshold_match_ratio = 1.0

### 创新性
- 首次提出适用于 MPC 环境的分段线性 GELU 近似方案
- 二分搜索优化将安全比较次数从 O(N) 降至 O(log N)
- 精度提升 20.61 个百分点，使 MPC 推理精度接近明文模型

---

## 精度-效率权衡

| 配置 | 明文精度 | SPU速度 | 适用场景 |
|------|---------|---------|---------|
| LUT GELU 16段 | 97.33% | 430.8s/sample | 高精度需求 |
| fixed_square + depth10 | 91.98%（校准后） | 69.57s/sample | 生产部署 |

## 部署方案

| 方案 | 目标场景 | 配置 | 推理时间 | 精度 |
|------|---------|------|---------|------|
| A | 城市三甲医院 | depth10 + batch12 | ~69s | 92% |
| B | 县级医院 | depth10 + batch8 | ~100s | ~90% |
| C | 乡村医院 | depth10 + batch4 | ~150s | ~88% |

---

## 创新点 6：FXP 定点精度系统消融

### 问题
SPU 使用定点算术（fixed-point arithmetic）进行安全计算，`fxp_fraction_bits` 参数决定精度。但目前没有针对 MPC 安全推理场景的系统性 fxp 精度消融研究，参数选择缺乏依据。

### 方法
在 SPU 实际运行环境中系统测试 fxp=12 / 14 / 16 / 20：
- fxp=12：精度崩塌，安全比较产生大量错误
- fxp=14：精度崩塌，与 fxp=12 类似
- **fxp=16**：100% match，唯一正确配置
- fxp=20：溢出，FM64 无法容纳

### 验证
- fxp=16：argmax/threshold match = 1.0/1.0
- fxp<16：precision collapse（安全比较误差累积）
- fxp>16：overflow（FM64 位宽不够）

### 核心发现
**fixed_square + FM64 + fxp=16 形成三位一体约束**，任意一项调整都会导致精度崩塌。该发现为 MPC 定点推理的参数选择提供了明确指导。

### 创新性
- 首次在 MPC 密文推理环境中系统性地进行 fxp 精度消融
- 揭示了 fixed_square 激活函数、FM64 域大小、fxp 位宽三者之间的耦合约束
- 所有数据来自服务器实际 SPU 运行，非仿真
