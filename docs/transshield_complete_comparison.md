# Transshield 完整对比数据报告

最后更新：`2026-05-14`

---

## 一、核心模型精度对比（CPU full-val, 524 samples）

| 模型配置 | threshold_acc | argmax_acc | AUC | 参数量 | 状态 |
|---------|--------------|------------|-----|--------|------|
| **Baseline** (depth12, 384dim, fixed_square) | 91.98% | 76.72% | 0.9679 | 100% | ✅ 主力 |
| **LRD** (rank192, 192dim) | 91.98% | 90.27% | 0.9666 | 68.39% | ✅ 主力 |
| **LUT GELU** (16-segment piecewise) | 97.33% | 97.33% | 0.9937 | 100% | ⚠️ SPU 超时 |
| **Mixed Attention** (uniform 0-5 + softmax 6-11) | 91.60% | 88.36% | — | 100% | ✅ 验证通过 |
| **Knowledge Distillation** (DeiT-Tiny student) | 89.69% | 75.95% | 0.9539 | 26.34% | ⚠️ SPU 未验证 |

**关键发现：**
- LRD 在参数量减少 31.61% 的情况下，argmax_acc 从 76.72% 提升到 90.27%（+13.55pp）
- LUT GELU 精度最高（97.33%），但 SPU 执行超时，暂无法部署
- Mixed Attention 解决了全 uniform 模式的类别不平衡问题（balanced_acc 从 50% 提升到 82%）

---

## 二、SPU 安全推理效率对比（服务器实测）

### 2.1 Baseline (fixed_square, depth12, 384dim)

| 测试 | 样本数 | 批量 | 总耗时 | sec/sample | finite | argmax |
|------|--------|------|--------|-----------|--------|--------|
| smoke1 | 1 | 1 | 254.6s | 254.6s | ✅ | [1] |
| smoke8 batch4 | 8 | 4 | 1284.9s | 160.6s | ✅ | [1,1,1,1,0,0,1,1] |
| smoke8 batch8 | 8 | 8 | 906.2s | 113.3s | ✅ | [1,1,1,1,1,0,1,1] |
| depth10 smoke8 | 8 | 8 | 804.3s | 100.5s | ✅ | [1,1,1,1,1,1,1,1] |
| **depth10 smoke12 batch12** | **12** | **12** | **834.9s** | **69.6s** | ✅ | **[1,1,1,1,1,0,1,1,1,1,1,1]** |

### 2.2 LRD (rank192, 192dim)

| 测试 | 样本数 | 批量 | 总耗时 | sec/sample | finite | argmax |
|------|--------|------|--------|-----------|--------|--------|
| smoke1 | 1 | 1 | 61.4s | 61.4s | ✅ | [0] |
| smoke8 | 8 | 8 | 207.8s | 26.0s | ✅ | [0,1,0,0,0,0,0,1] |

### 2.3 Mixed Attention (uniform 0-5 + softmax 6-11)

| 测试 | 样本数 | 批量 | 总耗时 | sec/sample | finite | argmax |
|------|--------|------|--------|-----------|--------|--------|
| smoke1 | 1 | 1 | 246.5s | 246.5s | ✅ | [1] |
| smoke8 | 8 | 8 | 1719.7s | 215.0s | ✅ | [1,1,1,1,1,0,1,1] |

### 2.4 效率汇总

| 配置 | sec/sample | 相对 Baseline 加速 | finite | 状态 |
|------|-----------|-------------------|--------|------|
| Baseline (depth12, batch8) | 113.3s | 1.00x | ✅ | 基线 |
| Baseline depth10 (batch8) | 100.5s | 1.13x | ✅ | ✅ |
| **Baseline depth10 (batch12)** | **69.6s** | **1.63x** | ✅ | **✅ 最优** |
| LRD (batch8) | 26.0s | **4.36x** | ✅ | ✅ 最快 |
| Mixed Attention (batch8) | 215.0s | 0.53x (更慢) | ✅ | ⚠️ 效率低 |

---

## 三、创新点验证状态

| # | 创新点 | CPU 精度 | SPU 验证 | 效率提升 | 状态 |
|---|--------|---------|---------|---------|------|
| 1 | Pruning Boundary 协议友好重写 | — | ✅ smoke32 | — | ✅ 完成 |
| 2 | 端到端隐私保护 SPU Forward | — | ✅ smoke32 | — | ✅ 完成 |
| 3 | Encoded-Key Bitonic Sort | match=1.0 | ✅ smoke32 | — | ✅ 完成 |
| 4 | MPC-Friendly 算子族 | 91.98% | ✅ smoke32 | — | ✅ 完成 |
| 5 | Token Pruning 效率优化 | — | ✅ smoke32 | 2.28x | ✅ 完成 |
| 6 | fxp=16 精度约束 | 100% match | ✅ smoke8 | — | ✅ 完成 |
| 7 | SVD 低秩分解 | 91.98% (90.27% argmax) | ✅ smoke8 | **4.36x** | ✅ 完成 |
| 8 | 分层混合注意力 | 88.36% argmax, 82% bacc | ✅ smoke8 | 0.53x | ✅ 完成 |
| 9 | LUT GELU 激活函数 | **97.33%** | ❌ 超时 | — | ⚠️ CPU only |

---

## 四、隐私保护验证

| 隐私约束 | 验证结果 | 证据 |
|---------|---------|------|
| 服务器看不到客户端图片 | ✅ `host_plaintext_pixel_values_materialized = false` | smoke32 |
| 客户端获取不到模型参数 | ✅ `host_model_params_materialized = false` | smoke32 |
| 只暴露最终 logits | ✅ `reveal_policy = final_logits_only` | smoke32 |
| PredictorLG 在 SPU 内部执行 | ✅ `runtime_pruning_keep_mask_pt = null` | smoke1 |

---

## 五、最优展示配置

### 推荐配置 A：最高精度（用于展示精度优势）
- **模型**：LRD (rank192, 192dim)
- **配置**：depth12, batch8
- **精度**：threshold_acc=91.98%, argmax_acc=90.27%
- **效率**：26.0s/sample（4.36x 加速）
- **隐私**：双向隐私保护完整

### 推荐配置 B：最高效率（用于展示效率优势）
- **模型**：Baseline depth10
- **配置**：depth10, batch12
- **精度**：argmax match=91.67%, threshold match=100%
- **效率**：69.6s/sample（1.63x 加速）
- **隐私**：双向隐私保护完整

### 推荐配置 C：最高绝对精度（CPU only）
- **模型**：LUT GELU (16-segment)
- **配置**：depth12, uniform attention
- **精度**：threshold_acc=97.33%, auc=0.994
- **效率**：CPU 5.8s, SPU 超时
- **隐私**：CPU 验证通过，SPU 待优化

---

## 六、fxp 精度消融实验

| fxp_bits | 整数位 | sec/sample | 与基线匹配率 | logits 范围 | 状态 |
|----------|--------|-----------|------------|------------|------|
| 12 | 52 | 114.0s | 12.5% | ±0.014 | ❌ 精度崩塌 |
| 14 | 50 | 112.0s | 12.5% | ±0.073 | ❌ 精度崩塌 |
| **16** | **48** | **109.9s** | **100%** | **±0.33** | **✅ 正确** |
| 20 | 44 | 110.9s | 62.5% | ±10⁵~10⁶ | ❌ 数值溢出 |

**结论**：`fixed_square + FM64 + fxp=16` 形成三位一体约束，fxp=16 是唯一安全操作点。

---

## 七、E2E 精度增强路径

| 路径 | heldout238 accuracy | 状态 |
|------|---------------------|------|
| Static baseline (bias-only) | 90.76% | ✅ |
| SPU-aware public bias | **92.44%** | ✅ 最优 |
| Affine calibration | 91.86% | ✅ |
| Temperature calibration | 92.10% | ✅ |
| Bridge calibration | 91.86% | ✅ |

---

## 八、待解决问题

| 问题 | 优先级 | 状态 |
|------|--------|------|
| LUT GELU SPU 超时 | 高 | 待优化（jnp.interp 在 MPC 中开销过大） |
| Knowledge Distillation SPU 验证 | 中 | 待执行 |
| Mixed Attention 效率优化 | 低 | 已记录，非阻塞 |

