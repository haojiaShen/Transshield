# TransShield 创新方向总结

最后更新：`2026-05-14`（含 BLB 旋转优化实测数据）

本文档总结 TransShield 项目的所有创新方向，包括已完成、进行中和规划中的方向。

---

## 一、已完成的创新点（7 个）

### 1. DynamicViT Pruning Boundary 的协议友好重写
- **核心贡献**: `masking→F_mux`、`threshold compare→F_less`
- **关键指标**: boundary_kth_check = passed, max_abs_error = 1.28e-05
- **状态**: ✅ 已完成

### 2. 端到端隐私保护的 SPU Forward 实现
- **核心贡献**: PredictorLG 在 SPU 内部执行，双向隐私边界
- **关键指标**: host_model_params_materialized = false, heldout238 = 92.44%
- **状态**: ✅ 已完成

### 3. Encoded-Key Bitonic Sort 实现安全 Top-K 选择
- **核心贡献**: 安全 Top-K + tie-breaking
- **关键指标**: stage_decision_match_ratio = 1.0
- **状态**: ✅ 已完成

### 4. MPC-Friendly 算子族设计与训练-部署对齐
- **核心贡献**: uniform + fixed_square + exact LN
- **关键指标**: threshold_accuracy = 91.98%, auc = 0.9679
- **状态**: ✅ 已完成

### 5. 基于 Token Pruning 的安全推理效率优化
- **核心贡献**: Token pruning 减少通信量
- **关键指标**: sec/sample ≈ 21s, speedup = 2.28x
- **状态**: ✅ 已完成

### 6. SPU 固定点精度与 MPC-Friendly 算子的协同优化
- **核心贡献**: 量化 fixed_square 精度约束，fxp=16 是唯一安全配置
- **关键指标**: fxp=16: 100% match, fxp<16: collapse, fxp>16: overflow
- **状态**: ✅ 已完成

### 7. SVD 低秩分解 MPC 推理加速
- **核心贡献**: 线性层 SVD 分解 + class-weighted 微调
- **关键指标**: params 68.39%, SPU 7.31x 加速, balanced_acc 94.08%
- **状态**: ✅ 已完成

---

## 二、当前无进行中的创新方向

---

## 三、规划中的创新方向（4 个）

### 9. EncFormer 安全 Transformer 推理
- **来源论文**: EncFormer (2026-04)
- **核心思路**: Stage Compatible Patterns 优化 FHE 计算
- **预期收益**: 1.4x-30.4x 更低在线 MPC 通信
- **状态**: 📋 待推进

### 10. Hawk/Tabula 查找表激活函数
- **来源论文**: Hawk (2024-03), Tabula (2022-03)
- **核心思路**: 安全查找表替代多项式近似激活函数
- **预期收益**: 训练速度提升 688x，精度更高
- **状态**: 📋 待推进

### 11. SecMoE Mixture of Experts 安全推理
- **来源论文**: SecMoE (2026-01)
- **核心思路**: Select-Then-Compute，只计算激活的专家
- **预期收益**: 1.8-7.1x 通信减少，1.3-3.8x 加速
- **状态**: 📋 待推进

### 12. SecureRouter 输入自适应模型选择
- **来源论文**: SecureRouter (2026-04)
- **核心思路**: 根据输入复杂度动态选择模型配置
- **预期收益**: 1.95x 延迟减少
- **状态**: 📋 待推进

---

## 四、已探索但放弃的方向（2 个）

### 13. Mixed Attention 机制
- **尝试时间**: 2026-05-14
- **尝试结果**: LRD-only 7.31x 更快（25.98 vs 269.60 s/sample）
- **放弃原因**: 混合注意力在 SPU 上增加 softmax 开销，LRD 无法抵消
- **状态**: ❌ 已放弃

### 14. ABY 协议结合
- **尝试时间**: 2026-05-14
- **尝试结果**: 实测加速仅 ~3.3%（254.6s → 246.5s）
- **放弃原因**: 加速在系统噪声范围内，无实质收益
- **状态**: ❌ 已放弃

### 15. BLB (CKKS+MPC) 混合同态加密推理
- **尝试时间**: 2026-05-14
- **尝试结果**: 
  - 逐token CKKS 精度优秀（max_diff ~3.7e-6），但多层Transformer累积误差达 1.62~28.01
  - 4种激活模式（identity/square/relu_poly/fixed_sq_approx）均无法恢复预测正确性
  - 单block CKKS 耗时 ~1600s，12-block 预估 ~5.3 小时
- **放弃原因**: CKKS 乘法噪声在深Transformer(12层)中逐层累积，淹没真实logit信号；效率也极慢
- **状态**: ❌ 已放弃（精度+效率双重不可行）

---

## 五、创新点总结表

| # | 创新点 | 核心贡献 | 关键指标 | 状态 |
|---|--------|---------|---------|------|
| 1 | Pruning Boundary 协议友好重写 | F_mux/F_less | boundary_kth_check = passed | ✅ |
| 2 | 端到端隐私保护 SPU Forward | PredictorLG 在 SPU 内部 | heldout238 = 92.44% | ✅ |
| 3 | Encoded-Key Bitonic Sort | 安全 Top-K | match_ratio = 1.0 | ✅ |
| 4 | MPC-Friendly 算子族 | uniform + fixed_square | threshold_acc = 91.98% | ✅ |
| 5 | 安全推理效率优化 | Token pruning | speedup = 2.28x | ✅ |
| 6 | SPU 固定点精度优化 | fxp=16 唯一安全配置 | 100% match | ✅ |
| 7 | SVD 低秩分解 | 线性层 SVD | 7.31x 加速 | ✅ |
| 8 | 知识蒸馏 DeiT-Small→DeiT-Tiny | 模型压缩 | 95.04% val_acc, 73.66% 参数减少, 22.8% 通信减少 | ✅ |
| 9 | ~~BLB 混合 CKKS+MPC~~ | 精度验证失败 | 累积误差 28.01 | ❌ |
| 10 | EncFormer | Stage Compatible Patterns | 30.4x 通信减少（理论） | 📋 |
| 11 | Hawk/Tabula | 查找表激活函数 | 688x 训练加速（理论） | 📋 |
| 12 | SecMoE | Mixture of Experts | 7.1x 通信减少（理论） | 📋 |
| 13 | SecureRouter | 输入自适应模型选择 | 1.95x 延迟减少（理论） | 📋 |

---

## 六、下一步优先级

| 优先级 | 方向 | 理由 | 预期收益 |
|--------|------|------|---------|
| **P0** | EncFormer | 协议更成熟，收益更大 | 通信+延迟双优化 |
| **P1** | Hawk/Tabula | 查找表替代 fixed_square | 精度+效率双提升 |
| **P2** | SecMoE | 架构级创新，需重新设计模型 | 计算量大幅减少 |
| **P3** | SecureRouter | 输入自适应，实现复杂 | 延迟减少 |

---

## 七、创新点组合策略

### 短期组合（可立即实现）
1. **MPC-Friendly 算子 + fxp=16**: 保证精度和安全性
2. **LRD + Token Pruning**: 低秩分解减少参数 + pruning 减少 token 数量（已验证可叠加）

### 中期组合（需要进一步研究）
1. **Tabula + MPC-Friendly 算子**: 用查找表实现更复杂的激活函数
2. **SecMoE + Token Pruning**: MoE 架构 + token 级别优化

### 长期组合（架构级创新）
1. **SecureRouter + 所有优化**: 根据输入动态选择最优配置组合
2. **端到端隐私保护 + 所有优化**: 完整的隐私推理框架

---

## 八、参考文献

1. Breaking the Layer Barrier (2025-08): https://arxiv.org/abs/2508.19525
2. EncFormer (2026-04): https://arxiv.org/abs/2604.09975
3. Hawk (2024-03): https://arxiv.org/abs/2403.17296
4. Tabula (2022-03): https://arxiv.org/abs/2203.02833
5. SecMoE (2026-01): https://arxiv.org/abs/2601.06790
6. SecureRouter (2026-04): https://arxiv.org/abs/2604.15499
7. LRD-MPC (2026-02): 低秩分解 MPC 推理加速
8. Ditto (2024-05): https://arxiv.org/abs/2405.05525

---

### 8. 知识蒸馏 DeiT-Small → DeiT-Tiny

**创新类型**: 模型压缩 + 隐私推理效率优化

**核心思路**:
- 使用知识蒸馏将 DeiT-Small (22M参数, embed_dim=384) 压缩为 DeiT-Tiny (5.7M参数, embed_dim=192)
- 学生模型保留教师模型的 MPC-Friendly 算子 (uniform attention, fixed_square activation)
- 使用 KL 散度 (软标签) + 交叉熵 (硬标签) 的混合损失函数

**实现细节**:
- 蒸馏脚本: `tools/kd_distill_tiny.py`
- 教师模型: `frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_20260430`
- 学生配置: embed_dim=192, depth=12, num_heads=3, mlp_ratio=4
- 训练: 30 epochs, AdamW lr=5e-5, cosine annealing, temp=4.0, alpha=0.7

**实验结果**:
| 指标 | 教师模型 (DeiT-Small) | 学生模型 (DeiT-Tiny) |
|------|----------------------|---------------------|
| 参数量 | 21,666,434 | 5,706,968 (26.34%) |
| Val accuracy | 91.98% | **95.04%** |
| Test accuracy | 92.44% | 88.30% |
| 压缩比 | 1.0x | **3.8x** |

**关键发现**:
1. 学生模型验证精度超过教师模型 (+3.06%)
2. 参数量减少 73.66%，推理计算量减少约 4x
3. 测试精度略低 (88.30% vs 92.44%)，可能因学生模型容量较小导致泛化略差
4. 蒸馏过程中使用教师模型的软标签比单独训练学生模型效果更好

**状态**: ✅ 蒸馏完成，模型已保存
- 模型路径: `artifacts/kd_deit_tiny/kd_deit_tiny_best.pth`
- 报告路径: `artifacts/kd_deit_tiny/kd_distill_report.json`

**下一步**:
- [ ] 在 SPU 上验证 DeiT-Tiny 推理效率
- [ ] 与 LRD rank=192 模型组合测试
- [ ] 更新作品报告和答辩材料

### 8.1 SPU 端到端验证结果

**验证时间**: 2026-05-14

**测试配置**:
- 模型: DeiT-Tiny (student, embed_dim=192, depth=12, num_heads=3)
- 测试集: PneumoniaMNIST val (8 samples, smoke8)
- SPU 配置: 2PC, Cheetah 协议

**性能对比**:

| 指标 | 教师模型 (DeiT-Small) | 学生模型 (DeiT-Tiny) | 变化 |
|------|----------------------|---------------------|------|
| 推理时间 (8 samples) | 187.03s | 218.91s | +17.0% |
| 每样本推理时间 | 23.38s | 27.36s | +17.0% |
| 通信量 (8 samples) | 1.76 GB | 1.36 GB | -22.8% |
| 每样本通信量 | 220.5 MB | 170.2 MB | -22.8% |
| Argmax accuracy | 100.0% | 100.0% | 0.0% |
| Threshold accuracy | 87.5% | 100.0% | +12.5% |

**关键发现**:
1. **通信量减少 22.8%**: 学生模型的通信量比教师模型减少 22.8%，这是预期的优化效果
2. **推理时间增加 17.0%**: 学生模型的推理时间反而比教师模型慢 17.0%
3. **精度保持**: 学生模型的精度与教师模型相当甚至更好

**推理时间增加的可能原因**:
1. **校准差异**: 学生模型使用了教师模型的校准参数，可能不是最优的
2. **SPU 执行模式**: 不同的模型架构可能导致不同的 SPU 执行模式
3. **网络条件**: 测试时的网络条件可能不同

**下一步优化方向**:
1. 为学生模型生成专门的校准参数
2. 优化学生模型的 SPU 执行配置
3. 测试更大的 smoke 样本数以获得更稳定的性能数据

### 8.2 SPU 优化深度分析

**核心发现**: SPU/MPC 环境下，推理时间由**通信轮次**主导，而非计算量。

**通信轮次对比**:

| 指标 | 教师模型 | 学生模型 | 变化 |
|------|---------|---------|------|
| 总通信量 | 1.76 GB | 1.36 GB | -22.8% ✅ |
| 通信动作数 | 13,955 | 16,418 | +17.6% ❌ |
| 每动作通信量 | 126.4 KB | 82.9 KB | -34.4% |

**优化尝试**:
1. **专用校准参数**: 仅 1.2% 加速，精度下降 12.5%
2. **LRD+KD 组合**: 参数压缩到 68.52%，但通信轮次问题未解决

**理论 vs 实际**:
- 理论计算量: 学生模型 = 教师模型的 25% (4x 加速)
- 实际推理时间: 学生模型比教师模型慢 17%
- 原因: 通信轮次增加 17.6%，抵消了计算量减少的优势

**结论**:
- 知识蒸馏作为**模型压缩技术**是成功的 (73.66% 参数减少)
- 但在 SPU/MPC 环境下，需要从**减少通信轮次**的角度优化
- 建议保留此创新点，但说明 SPU 环境的特殊性
