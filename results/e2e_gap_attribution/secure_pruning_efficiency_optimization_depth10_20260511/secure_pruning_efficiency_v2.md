# TransShield Secure Pruning 效率优化报告 v2

最后更新：`2026-05-11`

## 1. 实验演进

### 1.1 baseline：batch1 depth12（smoke8）

| 指标 | 值 |
|---|---|
| elapsed_sec | 1711.1s |
| sec_per_sample | **213.9s** |
| finite_logits | true |
| argmax_match | 1.0 (8/8) |

### 1.2 优化一：batch8 depth12

| 指标 | 值 |
|---|---|
| elapsed_sec | 906.2s |
| sec_per_sample | **113.3s** |
| 加速比 vs batch1 | **1.89x** |
| finite_logits | true |
| argmax_match | 1.0 (8/8) |

### 1.3 优化二：batch8 depth10 ✅ NEW

| 指标 | 值 |
|---|---|
| elapsed_sec | **804.3s** |
| sec_per_sample | **100.5s** |
| 加速比 vs batch1 | **2.13x** |
| 加速比 vs batch8-d12 | **1.13x** |
| finite_logits | true |
| depth_limit | 10 |
| threshold_match vs d12-batch8 | 1.0 (8/8) |
| argmax_match vs d12-batch8 | 0.875 (7/8) |
| logits_max_abs_error vs d12-batch8 | 0.147 |

### 1.4 优化三：batch16 depth10 🔄 RUNNING

| 指标 | 预期 |
|---|---|
| sec_per_sample | ~90-95s（进一步摊薄协议开销） |
| 加速比 vs batch1 | ~2.3x |

## 2. 综合对比表

| 配置 | batch_size | depth | spu_runs | 总耗时 | sec/sample | 加速比 | threshold_match | argmax_match |
|------|:---------:|:-----:|:--------:|:------:|:----------:|:------:|:---------------:|:------------:|
| baseline | 1 | 12 | 8 | 1711.1s | **213.9s** | 1.00x | — | — |
| batch4 | 4 | 12 | 2 | 1284.9s | **160.6s** | 1.33x | 1.0 | 1.0 |
| batch8 | 8 | 12 | 1 | 906.2s | **113.3s** | 1.89x | 1.0 | 1.0 |
| **batch8+d10** | **8** | **10** | **1** | **804.3s** | **100.5s** | **2.13x** | **1.0** | **0.875** |
| batch16+d10 | 16 | 10 | 1 | TBD | ~90-95s | ~2.3x | TBD | TBD |

## 3. 精度分析

### 3.1 Plaintext CPU 精度（full-val 524 样本）

| depth | argmax_acc | threshold_acc | argmax Δ vs d12 | threshold Δ vs d12 | 计算量 |
|:-----:|:----------:|:-------------:|:----------------:|:-------------------:|:------:|
| 6 | 25.76% | 30.73% | -50.96pp | -61.25pp | 67.1% |
| 8 | 26.34% | 51.53% | -50.38pp | -40.45pp | 80.0% |
| 9 | 66.60% | 85.50% | -10.12pp | -6.48pp | 86.5% |
| **10** | **79.96%** | **91.41%** | **+3.24pp** | **-0.57pp** | **91.0%** |
| 11 | 79.01% | 91.22% | +2.29pp | -0.76pp | 95.5% |
| 12 | 76.72% | 91.98% | 0.00 | 0.00 | 100.0% |

### 3.2 SPU 实测精度（smoke8，vs depth12 SPU baseline）

- threshold_match = **1.0** (8/8)：深度截断不改变 threshold 决策
- argmax_match = **0.875** (7/8)：1 个 boundary-case 翻转（margin < 0.05）
- logits_max_abs_error = **0.147**：SPU 定点 + depth 截断的累积差异
- 与 batch size 优化的观察一致：SPU 数值边界翻转是正常行为

### 3.3 depth=10 为什么 argmax 反而提升

- depth12 argmax=76.72% 低于 threshold=91.98%，是因为 argmax 使用固定阈值 0.5
- 最后 2 个 block 的累积数值漂移会将部分 boundary 样本推过 0.5 边界
- 去掉最后 2 个 block 减少了 late-block drift，反而让 argmax 预测更准确
- 这与 E2E drift 诊断中的 "late-block cumulative drift" 观察一致

## 4. 隐私边界（depth10 实验）

| 字段 | 值 |
|---|---|
| host_plaintext_pixel_values_materialized | **false** |
| host_model_params_materialized | **false** |
| spu_params_mode | secret |
| runtime_pruning_keep_mask_pt | null |
| reveal_policy | final_logits_only |
| has_predictor_params | true |
| backend | jax_spu_secure_pruning_forward_backend_v0 |
| forward_scope | student_patch_embed_blocks_head_with_secure_internal_pruning_predictor_path |

**双向隐私边界完整保持**：PredictorLG 仍在 SPU 内部执行，客户端无模型参数副本。

## 5. 效率提升分解

| 优化手段 | 绝对节省 | 来源 |
|---|---|---|
| batch1→batch8 | 100.6s/sample | 协议开销摊薄（8×18s → 1×18s） |
| depth12→depth10 | 12.8s/sample | 少 2 个 block（67 tokens/block） |
| **总节省** | **113.4s/sample** | **2.13x 加速** |

## 6. 下一步

1. 等待 batch16 depth10 结果（预期 ~90-95s/sample，~2.3x 加速）
2. 考虑 batch16 depth12 对比（纯 batch 优化到极致）
3. 如果精度可接受，depth=10 + batch≥8 可作为新的默认配置
