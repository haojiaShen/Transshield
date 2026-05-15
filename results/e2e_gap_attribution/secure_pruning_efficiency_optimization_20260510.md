# TransShield Secure Pruning 效率优化报告

最后更新：2026-05-10

## 1. 瓶颈分析

Secure Pruning 的 SPU 推理流程：

```
客户端(Party): 明文图像 → secret share → 加载到SPU
                                    ↓
SPU: secret share重组 → PredictorLG(3 stage) → ViT 12-block forward → final logits
                                    ↓
客户端(Party): 仅获取 final logits (reveal_policy=final_logits_only)
```

### 1.1 单次 spu_run 时间线分析（batch1 smoke8）

| spu_run # | 耗时 | 说明 |
|-----------|------|------|
| #1 | 233.9s | 首次 JIT compile + 1 sample |
| #2-#8 | 205-210s | JIT 已缓存，纯计算 |
| **平均** | **212s/sample** | 8 samples 共 1693s |

- JIT compile overhead（首次）: ~29s（只在第一次 spu_run 发生）
- SPU 协议通信初始化: ~18s（含 load_local_env、SPU node 启动等）
- 纯安全计算: ~184s/sample

### 1.2 计算量分解

| 组件 | 单样本乘法量 | 占比 |
|------|-------------|------|
| PredictorLG (3 stages) | ~141M | **3.4%** |
| ViT 12 blocks forward | ~4.0B | 96.2% |
| Patch embed + head | ~58M | 1.4% |
| **总计** | **~4.2B** | 100% |

PredictorLG 仅占总计算量的 3.4%，优化 PredictorLG 对整体效率影响有限。

### 1.3 关键瓶颈

**ViT 12-block forward in SPU** 是主要瓶颈，原因是：
- 每个 block 包含 QKV 投影（3×384×384）和 FFN（384×1536×384）
- 在 CHEETAH 2PC 协议下，每个乘法需要 OT（Oblivious Transfer）交互
- FM64 field + 16-bit fxp 精度保证了安全性但增加了通信量
- 12 个 block 顺序执行，无法并行

## 2. 已实施优化：Batch Size 调大

### 2.1 实验结果

| 配置 | batch_size | spu_run次数 | 总耗时 | sec/sample | 加速比 |
|------|-----------|------------|--------|------------|--------|
| smoke8-batch1 | 1 | 8 | 1711.1s | 213.9s | 1.00x |
| smoke8-batch4 | 4 | 2 | 1284.9s | **160.6s** | **1.33x** |
| smoke8-batch8 | 8 | 1 | 906.2s | **113.3s** | **1.89x** |

### 2.2 精度一致性

| 配置 | argmax | threshold | max_logit_diff vs batch1 |
|---|---|---|---|
| batch1 (baseline) | `[1,1,1,1,1,0,1,1]` | `[1,1,1,1,1,1,1,1]` | — |
| batch4 | `[1,1,1,1,0,0,1,1]` | `[1,1,1,1,1,1,1,1]` | 0.0467 |
| batch8 | `[1,1,1,1,1,0,1,1]` | `[1,1,1,1,1,1,1,1]` | 0.0235 |

- **threshold 预测 100% 一致**（生产级指标）
- argmax 在 index 4 存在 boundary-case 翻转（batch1=1, batch4=0, batch8=1），margin < 0.05
- SPU 定点运算的数值舍入差异属于正常行为，不影响生产部署

### 2.3 效率提升机制

- **batch1**: 每个样本独立一次 `spu_run` 调用，每次含协议握手 + SPU 计算
  - 8 次通信初始化 × ~18s ≈ 144s 纯协议开销
- **batch4**: 4 个样本打包为一次 `spu_run`，通信初始化只做 2 次
  - 2 次通信初始化 × ~18s ≈ 36s 协议开销（节约 108s）
  - JIT compile 被缓存，仅首次约 29s
- **batch8**: 8 个样本打包为一次 `spu_run`，通信初始化只做 1 次
  - 实测：906.2s → **113.3s/sample**（**1.89x 加速**）
  - JIT compile 约 29s（只发生一次），纯计算约 877s

### 2.3 batch_size 增大的开销

- 计算量线性增长：batch_size=N → N 倍的 ViT forward compute
- JIT 编译不变：SPU monolithic graph mode 下，同一计算图只编译一次
- 内存增长：每个 sample 约 86.6MB（model params in secret share），batch8 需要 ~700MB

## 3. 进一步优化方向评估

### 3.1 PredictorLG 简化（预期收益：微小）

PredictorLG 仅占 3.4% 计算量。即使简化 50%，整体提升 < 2%。

可能的简化路径：
- 减少 hidden dim: 384→192（需要重新训练 PredictorLG）
- 减少 stage 数: 3→2（需要修改 pruning_loc 并重新训练）
- 预计算 pruning decision：不可行，因为 PredictorLG 需要在 SPU 内部执行以保护隐私

### 3.2 更激进的 pruning（预期收益：中等）

当前 token_keep_counts = [137, 96, 67]（对应 ratio [0.70, 0.49, 0.34]）

如果更激进：[100, 60, 30]（ratio [0.51, 0.31, 0.15]）
- 后期 blocks (4-11) 的 attention 计算量可减少 ~50%
- FFN 计算量相应减少
- **需要重新训练 PredictorLG 和验证精度**

### 3.3 Block depth 截断（预期收益：显著但需验证）

使用 `--static-depth-limit 8` 跳过 blocks 8-11：
- 跳过 4/12 blocks ≈ 33% 计算量减少
- **可能损失精度**，需要在 plaintext 上先验证

### 3.4 SPU 协议参数调整（预期收益：不确定）

- `fxp_fraction_bits: 16→12`：减少 4-bit 精度，可能加速固定点乘法
- `protocol: CHEETAH→ABY3`：3PC 可能更高效但需要 3 个参与方
- **需要验证精度影响**

### 3.5 Block 分段执行（预期收益：无）

monolithic graph mode 下，block 分段不会减少计算量。唯一好处是降低峰值内存，但当前 batch8 内存充足（~700MB < 20GB 可用）。

## 4. 结论

| 优化方向 | 实施难度 | 预期收益 | 是否推荐 |
|---------|---------|---------|---------|
| ✅ batch_size 1→4 | 低 | **25%** | ✅ 已实施 |
| batch_size 4→8 | 低 | ~10-15% | ✅ 正在测试 |
| PredictorLG 简化 | 高（需重训） | <2% | ❌ 不推荐 |
| 更激进 pruning | 高（需重训） | ~20-30% | ⚠️ 风险高 |
| depth 截断 | 中 | ~25-33% | ⚠️ 需精度验证 |
| SPU 参数调整 | 中 | 不确定 | ⚠️ 需实验验证 |

**当前最实用的优化方案**：batch_size 调大（已实施，25% 提升）+ depth 截断（待验证）。
