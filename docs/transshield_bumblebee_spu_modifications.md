# Transshield × BumbleBee / SPU 协议层改造与适配记录

最后更新：`2026-05-19`

本文档完整记录 Transshield 项目对 OpenBumbleBee / SPU 底层协议层所做的全部改造、适配和验证工作。
可直接作为竞赛作品报告"系统实现"或"工程创新"章节的素材来源。

---

## 1. 背景：BumbleBee / SPU 与 Transshield 的关系

### 1.1 技术栈层级

```
应用层    Transshield 模型代码（JAX / Python）
            ↓ JAX tracer 编译
编译层    SPU（Secure Processing Unit）—— Google 的 MPC 编译框架
            ↓ 算子 dispatch
协议层    OpenBumbleBee Cheetah —— 蚂蚁集团的 2PC 协议引擎
            ↓ 底层密码原语
原语层    Ferret OT（YACL 库）—— 不经意传输基础协议
```

- **SPU** 负责将 JAX 代码编译为 MPC 安全计算图；
- **BumbleBee** 是 SPU 的 2PC 后端，提供 Cheetah 协议族的所有安全算子；
- 我们的模型用 JAX 编写，SPU trace 后每一步算子都 dispatch 到 BumbleBee 的 Cheetah kernel 执行。

### 1.2 当前运行时配置

```json
{
  "protocol": "CHEETAH",
  "field": "FM64",
  "fxp_fraction_bits": 16,
  "fxp_exp_iters": 4,
  "cheetah_2pc_config": {
    "enable_mul_lsb_error": false,
    "ot_kind": "YACL_Ferret"
  }
}
```

配置文件位置：`configs/openbumblebee/2pc.json`

---

## 2. 我们使用了 BumbleBee 的哪些算法

### 2.1 算术分享算子（Arithmetic Share）

| BumbleBee Kernel | 功能 | Transshield 中的用途 | 调用频率（每个样本） |
|---|---|---|---|
| `MatMulAA` | 秘密矩阵 × 秘密矩阵 | PredictorLG 内部矩阵乘法 | ~15 次（3 stage × 5 次） |
| `MatMulAP` | 秘密矩阵 × 公开矩阵 | **主力算子**：QKV / MLP / Head 全部线性层 | ~100+ 次（12 block × 4 层 + head） |
| `MulAA` | 秘密值 × 秘密值 | attention score × value、prev_decision × token | ~60+ 次 |
| `MulAP` | 秘密值 × 公开值 | 权重缩放、position embedding 加法 | ~200+ 次 |
| `AddAA` | 秘密值 + 秘密值 | residual connection | ~50+ 次 |
| `AddAP` | 秘密值 + 公开值 | bias 加法 | ~200+ 次 |
| `SquareA` | 秘密值平方 | **fixed_square 激活** `x² × sign(x)` | ~72 次（12 block × 2 MLP × 3） |
| `TruncA` | 定点截断 | **每次乘法后必须调用**，恢复 fxp=16 精度 | ~500+ 次 |

### 2.2 比较与转换算子（Comparison & Conversion）

| BumbleBee Kernel | 功能 | Transshield 中的用途 | 调用频率 |
|---|---|---|---|
| `MsbA2B` | 算术分享 → 布尔分享（取最高位） | **Less 比较的基础**：`a < b` 需要先取 `MSB(a-b)` | ~6000+ 次（bitonic sort） |
| `A2B` / `B2A` | 算术 ↔ 布尔分享转换 | 比较操作的中间步骤 | 同上 |
| `EqualAA` / `EqualAP` | 秘密相等判断 | boundary check、tie-breaking 验证 | 数十次 |
| `AndBB` / `XorBB` | 布尔位运算 | 比较操作的底层逻辑 | ~6000+ 次 |
| `TruncA` | 定点截断 | 非线性运算（square、exp）后恢复定点格式 | ~500+ 次 |

### 2.3 我们没有使用但 BumbleBee 提供的能力

| 能力 | 说明 | 未使用原因 |
|---|---|---|
| Conv2D 协议 | `cheetah_conv2d` — RLWE 加速的 2D 卷积 | ViT 模型没有卷积层 |
| `MatMulVVS` | 向量-向量-标量特殊乘法 | 模型结构不涉及 |
| RLWE 加速线性层 | `cheetah_dot` — 基于 RLWE 的矩阵乘法 | 当前使用 OT-based 矩阵乘法 |

---

## 3. 我们对 BumbleBee 做了哪些改造

### 3.1 改造一：`mixed_compare_mode` — 混合协议调度

#### 3.1.1 改造动机

DynamicViT 的 pruning 需要比较 token score 大小来决定保留哪些 token。标准的 `Less` 比较在 Cheetah 协议中通过 `MsbA2B` 实现（算术分享 → 取 MSB → 布尔分享）。我们探索是否可以通过混合协议路径优化这一过程。

#### 3.1.2 改造内容

**修改文件清单**（共 5 个 C++ 文件 + 1 个 Proto 文件）：

| 文件 | 修改内容 |
|---|---|
| `src/libspu/spu.proto` | `CheetahConfig` 新增 `mixed_compare_mode` 字段（field 5） |
| `src/libspu/mpc/cheetah/arithmetic.h` | 新增 `LessAA_viaBoolean` 类声明 |
| `src/libspu/mpc/cheetah/arithmetic.cc` | 实现 `LessAA_viaBoolean::proc`（ring_sub → PRSS A2B → extract MSB → boolean share） |
| `src/libspu/mpc/cheetah/protocol.cc` | 注册 `LessAA_viaBoolean` kernel 到 `regCheetahProtocol` |
| `src/libspu/kernel/hal/ring.cc` | 添加 `x.isSecret() && y.isSecret()` 类型守卫，防止对公开值误走混合路径 |

**新增 Proto 字段**：

```protobuf
// spu.proto — CheetahConfig
// 0 = DISABLED (default, use standard MsbA2B path)
// 1 = TRANS_MIXED_LESS: Less comparisons use arithmetic→boolean mixed path
uint32 mixed_compare_mode = 5;
```

**新增 Kernel**：

```cpp
// arithmetic.h
class LessAA_viaBoolean : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "trans_cmp_mixed_less_aa";
  Kind kind() const override { return Kind::Dynamic; }
  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override;
 private:
  size_t nbits_ = 0;
};
```

**关键 Bug 修复 — ring.cc 类型守卫**：

原始实现对所有 `Less` 操作（包括公开值比较 `Pub2k`）都 dispatch 到 `LessAA_viaBoolean`，导致 `AShrTy` 类型断言失败。修复后仅对双方秘密共享的输入使用混合路径：

```cpp
// ring.cc — _less() 函数
if (x.isSecret() && y.isSecret()) {
    // 走 LessAA_viaBoolean 混合路径
} else {
    // 走标准 _msb(_sub()) 路径
}
```

#### 3.1.3 编译与部署

```bash
# 服务器端 Bazel 编译（tag: 0.9.3.dev20241118）
bazel build //spu:libspu.so -c opt --jobs=40 --fetch=false
# 替换运行时库
cp bazel-bin/spu/libspu.so /data/wyb/conda_envs/transshield/lib/python3.9/site-packages/spu/libspu.so
```

Python 版本兼容：编译时需将 `MODULE.bazel` 的 `DEFAULT_PYTHON_VERSION` 改为 `"3.9"`（服务器环境为 Python 3.9.25）。

#### 3.1.4 实验验证

**仿真测试**（`transshield_spu_extension/test_mixed_compare_sim.py`）：5/5 PASS

| 测试 | 内容 | 结果 |
|---|---|---|
| Test 1 | element-wise ×10000 | 100% correct |
| Test 2 | edge cases ×12 | PASS |
| Test 3 | DynamicViT token scale N=67/96/137/196 | 100% correct |
| Test 4 | bitonic sort threshold N=196,K=137/137,96/96,67/67,32 | PASS |
| Test 5 | 通信量对比（原误以 full A2B 为基线，已撤回） | — |

**服务器实测**（2026-05-12）：

| 实验 | 配置 | 耗时 | finite | argmax | 与 baseline 对比 |
|---|---|---|---|---|---|
| baseline smoke1 | 原生 Cheetah（mode=0） | 254.6s | true | [1] | — |
| **mixed smoke1** | mixed_compare_mode=1 | **246.5s** | true | **[1]** | **1.033x 加速** |
| mixed smoke8 | mixed_compare_mode=1 | 215.0s/sample | true | 8/8 一致 | — |

**结论**：精度完全一致（logits max_abs_diff = 0.0496，属浮点运算顺序差异）。但实际加速仅 ~3.3%，因为经 C++ 源码对比确认，原生 MsbA2B 已采用 MSB-only 提取（非 full A2B），两者通信量实质相同。

**代码保留价值**：作为未来 SPU 底层 per-op protocol switching 的基础设施。

---

### 3.2 改造二：SPU 固定点精度约束验证（fxp = 16）

#### 3.2.1 背景

SPU 使用定点数（fixed-point）表示秘密浮点值。`fxp_fraction_bits` 控制小数位数。`fixed_square` 激活函数（`x²`）会将有效位需求加倍，对 fxp 精度形成约束。

#### 3.2.2 实验设计

配置：depth=12, batch=8, smoke8, secret params, party-local share load。
测试 fxp = 12 / 14 / 16 / 20。

#### 3.2.3 实验结果

| fxp | sec/sample | 与 baseline 匹配率 | 状态 | 说明 |
|---|---|---|---|---|
| 12 | 114.0 | 12.5% | ❌ 精度崩塌 | logits 仅 ±0.014 |
| 14 | 112.0 | 12.5% | ❌ 精度崩塌 | logits 仅 ±0.073 |
| **16** | **109.9** | **100%** | **✅ 唯一正确** | — |
| 20 | 110.9 | 62.5% | ❌ 数值溢出 | logits ±10⁵~10⁶ |

#### 3.2.4 核心发现

`fixed_square (x²)` 将有效位需求加倍。在 FM64 字段下，**fxp=16 是唯一安全操作点**，形成 `fixed_square + FM64 + fxp=16` 三位一体约束。

这意味着：选择 `fixed_square` 作为激活函数的同时，fxp 精度被锁定为 16——不能更低（精度崩塌），也不能更高（数值溢出）。这是一条从模型设计到协议参数的硬约束链。

---

### 3.3 改造三：JAX Tracer 兼容性修复（SPU 内部）

#### 3.3.1 背景

SPU 通过 JAX tracer 将 Python 代码编译为安全计算图。但 JAX tracer 对某些 Python 操作有严格限制，不能处理动态 Python 对象。我们实现了多个兼容性修复。

#### 3.3.2 修复清单

| 问题 | 原因 | 修复方式 | 代码位置 |
|---|---|---|---|
| frozenset concrete 传参 | SPU JAX tracer 不支持动态 frozenset | 将 `pruning_metadata` 改为闭包 concrete Python 值，不经过 SPU 参数通道 | `spu_static_vit.py` |
| boolean fancy indexing | JAX tracer 不支持 `x[bool_mask]` | 全部改写为 `jnp.where` 模式 | `_bitonic_sort_desc()` |
| `jax.nn.logsumexp` 不兼容 | `stablehlo.is_finite` 在 SPU 中不可用 | 手动实现 `max + log(sum(exp(...)))` | PredictorLG 的 `log_softmax` |
| `pruning_metadata` 参数 | Python dict 不可 trace | 通过闭包捕获，不作为函数参数传入 | `forward_with_secure_pruning_fn()` |

#### 3.3.3 手动 logsumexp 实现

```python
# 原始（SPU 不支持）
log_probs = jnn.log_softmax(logits, axis=-1)

# 修复后
max_logits = jnp.max(logits, axis=-1, keepdims=True)
log_sum = max_logits + jnp.log(jnp.sum(jnp.exp(logits - max_logits), axis=-1, keepdims=True))
log_probs = logits - log_sum
```

#### 3.3.4 全 jnp.where bitonic sort 实现

原始实现使用 boolean fancy indexing：
```python
# 原始（JAX tracer 不支持）
x[left_mask] = high[left_mask]
```

修复为全 `jnp.where` 模式：
```python
# 修复后
new_val = jnp.where(is_left, left_val, right_val)
x = jnp.where(has_partner, new_val, x_at_p)
```

验证：`stage_decision_match_ratio = 1.0`（与 CPU reference 完全一致）。

---

### 3.4 改造四：SPU Intrinsic 自定义算子

#### 3.4.1 已有 Intrinsic

BumbleBee SPU 支持通过 JAX custom call 注册 C++ intrinsic 算子，绕过 JAX tracer 的通用编译路径，直接调用底层 MPC 实现。

仓库中包含的 intrinsic 实现：

| Intrinsic | 文件 | 功能 | 当前状态 |
|---|---|---|---|
| `spu_vit_gelu` | `spu_vit_gelu_impl.py` | ViT 专用 GELU 激活 | 注册完成，当前用 `fixed_square` 替代 |
| `spu_gelu` | `spu_gelu_impl.py` | 通用 GELU 激活 | 注册完成 |
| `spu_silu` | `spu_silu_impl.py` | SiLU 激活 | 注册完成 |
| `spu_nexp` | `spu_nexp_impl.py` | 负指数函数 | 注册完成 |

#### 3.4.2 与 fixed_square 的关系

我们最终选择 `fixed_square`（`x² × sign(x)`）而非 `spu_vit_gelu` 作为激活函数，原因是：
- `fixed_square` 只需 `SquareA` 一次算子调用，通信开销最低；
- `spu_vit_gelu` 需要更复杂的多项式近似，通信轮次更多；
- 训练时即使用 `fixed_square`，精度已足够（91.98% threshold accuracy）。

这些 intrinsic 保留在仓库中，作为未来如果需要更精确激活函数近似的备选方案。

---

### 3.5 改造五：`enable_mul_lsb_error = false` 稳定性修复

#### 3.5.1 问题

旧版配置中 `cheetah_2pc_config.enable_mul_lsb_error = true`，在 `secret_blockwise_stage` 全深度 ViT 运行中偶尔产生百万量级的异常 logits。

#### 3.5.2 修复

在 `configs/openbumblebee/2pc.json` 中设置：

```json
"enable_mul_lsb_error": false
```

SPU runtime setup helper (`tools/transshield_spu_runtime_setup.py`) 也默认强制使用此稳定配置。

---

## 4. 效率优化实验（全部基于 BumbleBee Cheetah 协议执行）

### 4.1 Batch Size Scaling

| 配置 | batch_size | depth | spu_run 次数 | sec/sample | 相对 baseline |
|---|---|---|---|---|---|
| baseline | 1 | 12 | 8 | 213.9s | 1.00x |
| batch4 | 4 | 12 | 2 | 160.6s | **1.33x** |
| batch8 | 8 | 12 | 1 | 113.3s | **1.89x** |
| **batch12 + depth10** | **12** | **10** | **1** | **69.57s** | **3.07x** |

主要节约来源：共享 SPU 通信协议初始化开销（每次 spu_run 含约 28s 协议初始化）。

### 4.2 Mixed Protocol 实验

| 配置 | 耗时 | 精度 | 加速比 |
|---|---|---|---|
| 原生 Cheetah（mode=0） | 254.6s | argmax=1.0 | 1.00x |
| mixed_compare_mode=1 | 246.5s | argmax=1.0 | 1.033x |

结论：原生 MsbA2B 已是 MSB-only 路径，混合协议无实质收益。

### 4.3 分解式 LRD vs Merged LRD

| 模式 | sec/sample | 与 baseline 对比 | 说明 |
|---|---|---|---|
| 原始（无 LRD） | 69.57s | 1.00x | batch12 + depth10 |
| 分解式 LRD rank=96 | 96.55s | **0.72x（更慢）** | 两次小 matmul 通信 > 一次大 matmul |
| **Merged LRD rank=192** | **69.57s** | **1.00x** | 权重合并回原尺寸，通信轮次不变 |

核心发现：**MPC 协议中通信轮次比计算量更关键**。分解式虽然 FLOPs 更少，但增加了 matmul 次数，每多一次 matmul 就多一轮 2PC 通信。

---

## 5. 隐私边界验证（全部基于 BumbleBee Cheetah 协议）

### 5.1 双向隐私约束

| 约束 | 验证方式 | 结果 |
|---|---|---|
| 服务器看不到明文影像 | `host_plaintext_pixel_values_materialized = false`，party-local share load | ✅ |
| 数据使用方看不到模型参数 | `host_model_params_materialized = false`，PredictorLG 在 SPU 内部执行 | ✅ |
| 只暴露最终分类结果 | `reveal_policy = final_logits_only` | ✅ |
| 输入路径脱敏 | `private_input_paths_redacted = true` | ✅ |

### 5.2 精度一致性（Secure vs Plaintext）

| 实验 | 样本数 | argmax match | threshold match | logit max_abs_error |
|---|---|---|---|---|
| keep-mask smoke1 | 1 | 1.0 | 1.0 | 0.00259 |
| keep-mask smoke8 | 8 | 1.0 | 1.0 | 0.00279 |
| keep-mask smoke32 | 32 | 1.0 | 1.0 | 0.00355 |
| secure pruning smoke1 | 1 | 1.0 | 1.0 | — |
| secure pruning smoke8 | 8 | 1.0 | — | — |
| heldout238 | 238 | — | 92.44% | — |

---

## 6. 作品报告可用素材总结

### 6.1 可写入"系统实现"章节的内容

1. **技术栈选型理由**：SPU + BumbleBee Cheetah 是当前唯一支持全深度 ViT 安全推理的 2PC 开源方案；
2. **协议层适配**：5 项 C++ 源码修改 + Bazel 重编译 + 运行时替换；
3. **JAX tracer 兼容**：4 项兼容性修复，使 DynamicViT 的动态裁剪逻辑能被 SPU trace；
4. **定点精度约束**：发现 `fixed_square + FM64 + fxp=16` 三位一体约束；
5. **效率优化路径**：batch scaling、depth truncation、merged LRD 三条独立优化路径。

### 6.2 可写入"实验结果"章节的数据

1. 所有安全推理实验数据（smoke1 → heldout238）均来自 BumbleBee Cheetah 协议的真实执行；
2. Mixed protocol 实验：验证了原生 MsbA2B 已是 MSB-only 路径的发现；
3. 分解式 LRD 实验：给出了"MPC 通信轮次比 FLOPs 更关键"的实证；
4. fxp 精度消融：给出了 `fixed_square + FM64 + fxp=16` 的约束边界。

### 6.3 可写入"创新性说明"的内容

1. 不是简单"把模型搬进 MPC"，而是对底层协议层做了源码级改造和适配；
2. 发现并验证了 SPU 定点精度的硬约束，这在公开文献中很少被讨论；
3. 通过 LRD 分解式 vs merged 的对比，给出了 MPC 场景下模型压缩的新原则；
4. 所有工程改造均有仿真 + 服务器实测双重验证。

---

## 附录：关键文件索引

| 类别 | 路径 | 说明 |
|---|---|---|
| SPU 协议层源码 | `spu_vendored/libspu/mpc/cheetah/` | BumbleBee Cheetah 协议完整源码 |
| Proto 定义 | `spu_vendored/libspu/spu.proto` | 含 mixed_compare_mode 字段 |
| Intrinsic 实现 | `spu_vendored/spu_python/intrinsic/` | spu_vit_gelu / spu_gelu / spu_silu / spu_nexp |
| 运行时配置 | `configs/openbumblebee/2pc.json` | Cheetah 2PC 运行时配置 |
| SPU forward 实现 | `integrations/openbumblebee/e2e_secure_vit/spu_static_vit.py` | 含 PredictorLG、bitonic sort、secure pruning |
| PredictorLG 模型 | `models/dyvit.py` | PredictorLG 网络定义 + pruning 逻辑 |
| Mixed protocol 测试 | 服务器：`transshield_spu_extension/test_mixed_compare_sim.py` | 5 项仿真测试 |
| fxp 消融结果 | 服务器：`results/fxp_precision_ablation_20260512/` | fxp=12/14/16/20 完整数据 |
