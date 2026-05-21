# Transshield 修改改进全记录

最后更新：`2026-05-19`

本文档完整记录 Transshield 项目自立项至今的所有技术修改与改进，按模块分类整理。
可直接作为竞赛作品报告"系统实现"、"工程创新"和"实验结果"章节的素材来源。

---

## 目录

1. [协议层改造：BumbleBee / SPU 底层适配](#1-协议层改造)
2. [模型层改造：DynamicViT 安全推理适配](#2-模型层改造)
3. [算子替换：MPC-Friendly 算子族设计](#3-算子替换)
4. [训练策略改进](#4-训练策略改进)
5. [推理效率优化](#5-推理效率优化)
6. [精度优化：输出校准](#6-精度优化)
7. [模型压缩：SVD 低秩分解](#7-模型压缩)
8. [金融领域扩展](#8-金融领域扩展)
9. [实验数据汇总表](#9-实验数据汇总表)

---

## 1. 协议层改造

> 改造目标：使 SPU + BumbleBee Cheetah 2PC 协议引擎能正确、稳定地执行 Transshield 的安全推理流程。

### 1.1 Mixed Compare Mode — 混合协议调度

**改造动机**：DynamicViT 的 pruning 依赖 token score 比较，探索是否可通过混合协议路径优化 `Less` 比较。

**修改文件（5 个 C++ 文件 + 1 个 Proto 文件）**：

| 文件 | 修改内容 |
|---|---|
| `spu_vendored/libspu/spu.proto` | `CheetahConfig` 新增 `mixed_compare_mode` 字段（field 5） |
| `spu_vendored/libspu/mpc/cheetah/arithmetic.h` | 新增 `LessAA_viaBoolean` 内核声明 |
| `spu_vendored/libspu/mpc/cheetah/arithmetic.cc` | 实现 `LessAA_viaBoolean::proc`（ring_sub → PRSS A2B → extract MSB → boolean share） |
| `spu_vendored/libspu/mpc/cheetah/protocol.cc` | 注册 `LessAA_viaBoolean` kernel 到 `regCheetahProtocol` |
| `spu_vendored/libspu/kernel/hal/ring.cc` | 添加 `x.isSecret() && y.isSecret()` 类型守卫，防止公开值误走混合路径 |

**关键 Bug 修复**：原始实现对所有 `Less` 操作（含 Pub2k 公开值比较）都 dispatch 到 `LessAA_viaBoolean`，导致 `AShrTy` 类型断言失败。修复后仅对双方秘密共享的输入使用混合路径。

**实验结果**：

| 配置 | sec/sample | 精度 | 加速比 |
|---|---|---|---|
| 原生 Cheetah（mode=0） | 254.6s | argmax=1.0 | 1.00x |
| mixed_compare_mode=1 | 246.5s | argmax=1.0 | 1.033x |

**结论**：经 C++ 源码对比确认，原生 `MsbA2B` 已采用 MSB-only 提取（非 full A2B），两者通信量实质相同。实测加速仅 ~3.3%，属系统噪声范围。代码保留作为未来 SPU 底层 per-op protocol switching 的基础设施。

### 1.2 enable_mul_lsb_error 稳定性修复

**问题**：旧版配置中 `cheetah_2pc_config.enable_mul_lsb_error = true`，在全深度 ViT 运行中偶尔产生百万量级的异常 logits。

**修复**：在 `configs/openbumblebee/2pc.json` 中设置 `"enable_mul_lsb_error": false`。SPU runtime setup helper 也默认强制使用此稳定配置。

### 1.3 Bazel 编译与部署

**难点**：服务器无外网访问，`bcr.bazel.build` 不可达。

**解决**：
- 在 `/data/wyb/spu_src/` 克隆 SPU 源码（tag: `0.9.3.dev20241118`）
- 使用 `bazel build //spu:libspu.so -c opt --jobs=40 --fetch=false` 离线编译（383s, 4524 targets）
- 修复 Python 版本兼容：`MODULE.bazel` 的 `DEFAULT_PYTHON_VERSION` 从 `"3.11"` 改为 `"3.9"`（服务器环境为 Python 3.9.25）
- 替换运行时库到 transshield conda env

---

## 2. 模型层改造

> 改造目标：将 DynamicViT 的动态裁剪逻辑改写为 MPC 可执行形式，同时修复 JAX tracer 兼容性问题。

### 2.1 Pruning Boundary 协议友好重写

**核心问题**：原始 DynamicViT 的 token pruning 依赖"删除式表达"（直接删除 token），在 MPC 环境中带来动态 shape 和条件分支泄漏。

**改写方法**：

| 原始表达 | 协议友好表达 | 说明 |
|---|---|---|
| 直接删除 token | `masking → F_mux` | token 保留在张量里，通过掩码表达 keep/drop |
| 硬阈值裁剪 | `threshold compare → F_less` | pruning threshold 比较改写为 MPC 友好的安全比较函数 |
| 外部裁剪决策 | `secure sidecar/replay` | pruning 决策以 sidecar 方式安全执行，结果 replay 回主模型 |

**验证**：
- boundary check：`boundary_kth_check_passed = true`（max abs error 1.28e-05）
- E2E consistency：argmax/threshold match = 1.0/1.0
- 代码位置：`spu_static_vit.py`

### 2.2 Encoded-Key Bitonic Sort — 安全 Top-K 选择

**问题**：DynamicViT 的 pruning 需要 Top-K 选择，传统 Top-K 依赖条件分支，在 MPC 中不安全。

**实现**：
- encoded_key 编码：`encoded_key = score - index * 1e-6`（tie-breaking: lower index wins）
- O(n log²n) 复杂度，无条件分支
- 全 `jnp.where` 实现，JAX tracer 兼容

**验证**：`stage_decision_match_ratio = 1.0`（与 CPU reference 完全一致）

### 2.3 JAX Tracer 兼容性修复

SPU 通过 JAX tracer 将 Python 代码编译为安全计算图，但 tracer 对某些 Python 操作有严格限制。

| 问题 | 原因 | 修复方式 | 代码位置 |
|---|---|---|---|
| frozenset concrete 传参 | SPU JAX tracer 不支持动态 frozenset | 将 `pruning_metadata` 改为闭包 concrete Python 值，不经过 SPU 参数通道 | `spu_static_vit.py` |
| boolean fancy indexing | JAX tracer 不支持 `x[bool_mask]` | 全部改写为 `jnp.where` 模式 | `_bitonic_sort_desc()` |
| `jax.nn.logsumexp` 不兼容 | `stablehlo.is_finite` 在 SPU 中不可用 | 手动实现 `max + log(sum(exp(...)))` | PredictorLG 的 `log_softmax` |
| `pruning_metadata` 参数 | Python dict 不可 trace | 通过闭包捕获，不作为函数参数传入 | `forward_with_secure_pruning_fn()` |

**手动 logsumexp 实现**：
```python
# 原始（SPU 不支持）
log_probs = jnn.log_softmax(logits, axis=-1)

# 修复后
max_logits = jnp.max(logits, axis=-1, keepdims=True)
log_sum = max_logits + jnp.log(jnp.sum(jnp.exp(logits - max_logits), axis=-1, keepdims=True))
log_probs = logits - log_sum
```

**全 jnp.where bitonic sort 修复**：
```python
# 原始（JAX tracer 不支持）
x[left_mask] = high[left_mask]

# 修复后
new_val = jnp.where(is_left, left_val, right_val)
x = jnp.where(has_partner, new_val, x_at_p)
```

### 2.4 SPU Intrinsic 自定义算子

BumbleBee SPU 支持通过 JAX custom call 注册 C++ intrinsic 算子。仓库中已注册的 intrinsic：

| Intrinsic | 功能 | 当前状态 |
|---|---|---|
| `spu_vit_gelu` | ViT 专用 GELU 激活 | 注册完成，当前用 `fixed_square` 替代 |
| `spu_gelu` | 通用 GELU 激活 | 注册完成 |
| `spu_silu` | SiLU 激活 | 注册完成 |
| `spu_nexp` | 负指数函数 | 注册完成 |

当前选择 `fixed_square`（`x² × sign(x)`）而非 `spu_vit_gelu` 的原因：`fixed_square` 只需 `SquareA` 一次算子调用，通信开销最低。

---

## 3. 算子替换

> 设计原则：统一的 MPC-friendly 算子族，保证训练、验证、部署一致性。

| 原始算子 | MPC-Friendly 替代 | 理由 |
|---|---|---|
| Softmax Attention | **Uniform Attention**（均匀注意力） | 安全 softmax 需要 `exp + div`，通信开销极高 |
| GELU 激活 | **Fixed Square**（`x² × sign(x)`） | 仅需 `SquareA` 一次算子调用 |
| LayerNorm | **Exact LayerNorm** | 保持数值正确性 |
| — | **Clip0** | 激活值截断到非负，减少数值范围 |

**创新性说明**：不是简单"换掉不好算的算子"，而是保证训练和部署使用同一套 secure-friendly 口径，消除 train-test mismatch。很多作品只展示"能跑"，但训练和部署使用两套不同语义，最终精度很难解释。

---

## 4. 训练策略改进

### 4.1 Secure-Static 训练

**目标**：让模型在训练阶段就适应部署时的近似算子配置（uniform attention + fixed_square + clip0）。

**最终训练配置**：
- exact LayerNorm + uniform attention + fixed_square clip0
- 8 epochs，class-weighted loss
- 输出校准：static-path public output calibration

**结果**：
- full-val threshold accuracy = **91.98%**
- AUC = **0.9679**

### 4.2 Secure-Static Train Depth 对齐

**实验**：在训练时使用 `depth=10`（与部署 depth 截断对齐），与 depth=12 训练对比。

**结果**：epoch1/epoch3 均 no_clear_depth_benefit_yet（收益未证明），默认收口。

### 4.3 Protocol-Aware Pruning Loss

**实验**：针对 margin/tie/active-set 稳定性设计 protocol-aware pruning objective。

**结果**：focused5 epoch5 pair-study 已拿到，no_boundary_relief_yet，默认暂停。

### 4.4 蒸馏补偿

**实验**：尝试用教师模型蒸馏补偿近似算子带来的精度损失。

**结果**：official + cls-only paired result 均 no_clear_distill_benefit_yet，默认暂停。

---

## 5. 推理效率优化

### 5.1 Batch Size Scaling

共享 SPU 通信协议初始化开销（每次 spu_run 含约 28s 协议初始化），batch 越大越划算。

| 配置 | batch_size | depth | spu_run 次数 | sec/sample | 相对 baseline |
|---|---|---|---|---|---|
| baseline | 1 | 12 | 8 | 213.9s | 1.00x |
| batch4 | 4 | 12 | 2 | 160.6s | **1.33x** |
| batch8 | 8 | 12 | 1 | 113.3s | **1.89x** |
| **batch12 + depth10** | **12** | **10** | **1** | **69.57s** | **3.07x** |

**关键发现**：batch16 depth10 尝试失败——SPU 节点 OOM（62GB RAM 不足以承载 16 样本同时 in-SPU 计算）。

### 5.2 Depth Truncation

去掉最后 2 个 block（depth12 → depth10）：
- 参数减少 15.85%（22.39M → 18.84M）
- 计算量减少 16.7%
- argmax 反而提升 +3.24pp，threshold 仅下降 0.57pp
- **深度截断是医疗模型的主优化策略**

### 5.3 Merged LRD（vs 分解式）

| 模式 | sec/sample | 与 baseline 对比 | 说明 |
|---|---|---|---|
| 原始（无 LRD） | 69.57s | 1.00x | batch12 + depth10 |
| 分解式 LRD rank=96 | 96.55s | **0.72x（更慢 38.8%）** | 两次小 matmul 通信 > 一次大 matmul |
| **Merged LRD rank=192** | **69.57s** | **1.00x** | 权重合并回原尺寸，通信轮次不变 |

**核心发现**：**MPC 协议中通信轮次比计算量更关键。** 分解式虽然 FLOPs 更少，但增加了 matmul 次数，每多一次 matmul 就多一轮 2PC 通信。

### 5.4 Mixed Protocol 实验

经 C++ 源码对比确认，原生 MsbA2B 已是 MSB-only 路径，混合协议无实质收益（仅 ~3.3% 加速，属系统噪声）。

### 5.5 其他尝试（已放弃）

| 尝试 | 结果 | 原因 |
|---|---|---|
| token_ratio 加速 | 不提速 | secure pruning 仍为 full-shape masking |
| recycle=0.1 | 无收益 | 70.85s vs 69.57s |
| fxp_exp_iters=3 | 不推荐 | threshold 精度下降（66.67% vs 100%） |

---

## 6. 精度优化

### 6.1 SPU 定点精度约束验证（fxp 消融）

SPU 使用定点数表示秘密浮点值。`fixed_square` 激活函数（`x²`）将有效位需求加倍。

| fxp | sec/sample | 与 baseline 匹配率 | 状态 |
|---|---|---|---|
| 12 | 114.0 | 12.5% | ❌ 精度崩塌 |
| 14 | 112.0 | 12.5% | ❌ 精度崩塌 |
| **16** | **109.9** | **100%** | **✅ 唯一正确** |
| 20 | 110.9 | 62.5% | ❌ 数值溢出 |

**核心发现**：`fixed_square + FM64 + fxp=16` 三位一体约束——选择 `fixed_square` 作为激活函数的同时，fxp 精度被锁定为 16。这是一条从模型设计到协议参数的硬约束链。

### 6.2 输出校准（不重训恢复路径）

| 方法 | calibrated argmax acc | CE loss | 说明 |
|---|---|---|---|
| bias-only (0.5852) | 91.79% | 0.4287 | 最简单 |
| affine | 91.79% | 0.2025 | 更低 loss |
| temperature | 91.79% | 0.1984 | 最低 loss |
| **SPU-aware bias (heldout-weighted)** | **92.09%** | — | 当前默认 |

- SPU-aware bias calibration 来源：smoke32 拟合，heldout64/128/238 加权验证
- 决策：`promote_spuaware_bias_as_accuracy_first_default`

### 6.3 E2E 精度一致性验证（Secure vs Plaintext）

| Run | 样本数 | argmax match | threshold match | logit max_abs_error |
|---|---|---|---|---|
| keep-mask smoke1 | 1 | 1.0 | 1.0 | 0.00259 |
| keep-mask smoke8 | 8 | 1.0 | 1.0 | 0.00279 |
| keep-mask smoke16 | 16 | 1.0 | 1.0 | 0.00263 |
| keep-mask smoke32 | 32 | 1.0 | 1.0 | 0.00355 |

- **全部 1.0/1.0**，`privacy_consistent = true`
- E2E heldout238（最大 heldout）：**92.44%** threshold accuracy

---

## 7. 模型压缩

### 7.1 SVD 低秩分解（LRD）

**分解目标层**：blocks.*.attn.qkv, attn.proj, mlp.fc1, mlp.fc2（共 48 层）

| rank | 参数量 | 压缩率 |
|---|---|---|
| 96 | 8,234,408 | 36.78% |
| 128 | 10,593,704 | 47.31% |
| **192** | **15,312,296** | **68.39%** |
| 256 | 20,030,888 | 89.46% |
| 原始 | 22,390,184 | 100% |

**微调策略**：class-weighted CrossEntropy + WeightedRandomSampler, lr=5e-5, 10 epochs

**微调训练曲线（rank=192）**：

| Epoch | Train Acc | Val Acc | Balanced Acc |
|---|---|---|---|
| 0 | - | 74.24% | 50.00% |
| 1 | 88.21% | 85.11% | 89.49% |
| 3 | 92.91% | 94.08% | 91.90% |
| **4** | **94.44%** | **94.08%** | **94.08%** |

**关键发现**：
1. SVD 分解后必须微调，否则模型退化为 majority-class 预测
2. Class-weighted loss 对恢复精度至关重要
3. CPU 推理速度提升 14.7%，SPU 上 merged 模式保持原速

### 7.2 Merged vs 分解式 LRD

| 方案 | 原理 | SPU 效果 |
|---|---|---|
| 分解式 | 两次小 matmul（down → up） | ❌ 更慢（通信轮次翻倍） |
| **Merged** | 合并回原尺寸单次 matmul | ✅ 保持原速 + 参数压缩 |

**LRD merged 模式是 SPU 环境下的最优选择。**

---

## 8. 金融领域扩展

### 8.1 训练

- **Bundle**: `artifacts/frozen_bundle_finance_lrd_rank192_20260515/`
- **配置**: LRD rank192 merged, depth12, 30 epochs
- **参数量**: 22,390,184 → 15,312,296 (68.39%)
- **test_acc1**: 100.0%

### 8.2 完整隐私验证

- **smoke8 配置**: LRD rank192 + party-local secret + batch8 + uniform attention
- **隐私**: `host_plaintext_pixel_values_materialized = false` ✅, `spu_params_mode = secret` ✅
- **效率**: `sec_per_sample = 117.80s`
- **精度**: `argmax_match = 100%`（8/8）, `logits max_abs_error = 0.001373`

**意义**：金融模型首次实现完整隐私保护推理，与医疗模型保持一致的隐私边界。

---

## 9. 实验数据汇总表

### 9.1 最终部署配置

| 模型 | 配置 | 推理时间 | 精度 | 参数量 | 参数压缩 |
|---|---|---|---|---|---|
| 医疗 | depth10 + batch12 + fixed_square | **69.57s** | 91.98% | 18.84M | 84.15% |
| 金融 | depth12 + LRD rank192 merged | **117.80s** | 100%（8/8） | 15.31M | 68.39% |
| Baseline | depth12 | 213.9s | 76.72% | 22.39M | 100% |

### 9.2 BumbleBee 算子使用频率

| 算子类型 | 用途 | 每样本调用频率 |
|---|---|---|
| `MatMulAP` | 主力：QKV / MLP / Head 全部线性层 | ~100+ 次 |
| `MatMulAA` | PredictorLG 内部矩阵乘法 | ~15 次 |
| `MulAA` | attention × value、prev_decision × token | ~60+ 次 |
| `SquareA` | fixed_square 激活 | ~72 次 |
| `TruncA` | 每次乘法后定点截断 | ~500+ 次 |
| `MsbA2B` / `A2B` | 比较操作（bitonic sort） | ~6000+ 次 |

### 9.3 技术栈层级

```
应用层    Transshield 模型代码（JAX / Python）
            ↓ JAX tracer 编译
编译层    SPU（Secure Processing Unit）—— Google 的 MPC 编译框架
            ↓ 算子 dispatch
协议层    OpenBumbleBee Cheetah —— 蚂蚁集团的 2PC 协议引擎
            ↓ 底层密码原语
原语层    Ferret OT（YACL 库）—— 不经意传输基础协议
```

---

## 附录：修改文件索引

| 类别 | 关键文件 |
|---|---|
| SPU 协议层源码 | `spu_vendored/libspu/mpc/cheetah/` |
| Proto 定义 | `spu_vendored/libspu/spu.proto` |
| Intrinsic 实现 | `spu_vendored/spu_python/intrinsic/` |
| 运行时配置 | `configs/openbumblebee/2pc.json` |
| SPU forward 实现 | `integrations/openbumblebee/e2e_secure_vit/spu_static_vit.py` |
| PredictorLG 模型 | `models/dyvit.py` |
| 低秩分解工具 | `tools/transshield_low_rank_decompose.py` |
| LRD 微调脚本 | `tools/lrd_finetune_balanced.py` |
| 混合协议测试 | `transshield_spu_extension/test_mixed_compare_sim.py` |
| fxp 消融结果 | `results/fxp_precision_ablation_20260512/` |
| E2E 漂移归因 | `results/e2e_gap_attribution/` |
| 效率优化报告 | `results/e2e_runtime_efficiency/` |
| 金融模型结果 | `artifacts/server_pipeline_run/finance_lrd_rank192_partylocal_secret_spu_smoke8_20260516_v2/` |
| 详细 BumbleBee 改造记录 | `docs/transshield_bumblebee_spu_modifications.md` |

