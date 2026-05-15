# Depth Truncation 精度分析报告

最后更新：`2026-05-11`

## 1. 实验目的

评估在 ViT forward 中截断最后若干 block 的精度影响，为 SPU 效率优化提供 plaintext 参考。

## 2. 方法

- 使用 `AA=none epoch8` bundle（`artifacts/frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_aanone_20260507`）
- 全量 524 样本（`fullval_pixel_values.pt`）
- Static whole-forward reference（无 runtime pruning，token 序列保持 196）
- 评测阈值：`0.3577311038970947`
- 截断点：depth=6/8/9/10/11/12（baseline）

## 3. 结果

| depth | argmax_acc | threshold_acc | argmax Δ vs d12 | threshold Δ vs d12 | 计算量（线性） |
|:-----:|:----------:|:-------------:|:----------------:|:-------------------:|:-------------:|
| 6 | 25.76% | 30.73% | -50.96pp | -61.25pp | 67.1% |
| 8 | 26.34% | 51.53% | -50.38pp | -40.45pp | 80.0% |
| 9 | 66.60% | 85.50% | -10.12pp | -6.48pp | 86.5% |
| **10** | **79.96%** | **91.41%** | **+3.24pp** | **-0.57pp** | **91.0%** |
| 11 | 79.01% | 91.22% | +2.29pp | -0.76pp | 95.5% |
| 12 | 76.72% | 91.98% | 0.00 | 0.00 | 100.0% |

## 4. 关键发现

### 4.1 depth=10 是最有趣的候选
- **argmax 反升 +3.24pp**：去掉最后 2 个 block（处理 67 tokens）反而提高了 argmax 准确率
- **threshold 仅微降 -0.57pp**：91.41% vs 91.98%，差距在统计噪声范围内
- **计算节省**：约 9%（线性 token 计算量代理）

### 4.2 反直觉的 argmax 提升解释
- depth=12 的 argmax accuracy（76.72%）低于 threshold accuracy（91.98%）是因为 argmax 使用固定阈值 0.5
- 最后 2 个 block 的微小数值扰动可能将部分 boundary 样本从 class0 推到 class1 或反之
- 去掉这些 block 减少了累积数值漂移，反而让 argmax 预测更准确
- 这与之前 E2E drift 诊断中的 "late-block cumulative drift" 观察一致

### 4.3 不可行的 depth 选择
- depth=6/8：精度崩塌，不可用
- depth=9：threshold 下降 6.48pp，不可接受
- depth=11：精度接近 depth=10 但节省更少（4.5% vs 9%）

## 5. 当前 SPU 实验

- 服务器正在运行 `secure_pruning_spu_smoke8_batch8_depth10`
- 目标：验证 SPU 定点运算下 depth=10 的精度和效率
- 预期：~8-9% 计算节省 + SPU 协议开销摊薄后，sec/sample 可能从 ~113s 降至 ~103-105s

## 6. 注意事项

- 计算量节省主要来自最后 2 个 block 少处理 67 tokens（而非 196）
- 更大的节省需要 depth=8 或更浅，但精度损失不可接受
- 当前 batch8 + depth=10 组合预期总加速为 ~1.98x vs baseline batch1 depth12
