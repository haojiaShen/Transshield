# Transshield 金融领域扩展计划

最后更新：`2026-05-11`

## 1. 目标

将 Transshield 从"医疗影像隐私推理系统"扩展为"通用隐私保护推理框架"，新增金融领域验证，证明方法的跨领域通用性。

## 2. 核心思路

Transshield 的核心方法（masking → F_mux、threshold compare → F_less、bitonic sort 安全 Top-K）不绑定任何特定数据类型。金融数据的适配点在于：

- 金融表格数据的每一行/每个特征编码为一个 token
- 复用 DynamicViT + PredictorLG 的 token 剪枝结构
- 复用 SPU 安全推理链路
- 只需新增数据编码器和数据集

## 3. 数据集选择

### 首选：Credit Card Fraud Detection（信用卡欺诈检测）

- 来源：Kaggle / UCI，公开可用
- 特征：28 个 PCA 特征 + Time + Amount = 30 个数值特征
- 标签：正常/欺诈（严重不平衡，欺诈占 0.17%）
- 样本数：284,807
- 适合原因：标准金融二分类任务，与胸片二分类任务结构一致

### 备选：合成金融数据

- 如果无法获取真实数据集，可用 sklearn 生成合成分类数据
- 保持二分类结构，模拟金融场景

## 4. 技术方案

### 4.1 数据编码器

将 30 个数值特征编码为 token 序列：

```
金融数据 (N, 30) 
  → 分组为 N 个样本，每样本 30 个特征
  → 每个特征经 Linear 投影到 embed_dim=384
  → 位置编码：10 个 token（每 3 个特征一组，类似 image patch）
  → 送入 ViT 处理
```

### 4.2 模型架构

复用 Transshield 的 DynamicViT 架构，但调整输入层：

- patch_size: 3 → 1（每特征一个 token，或分组为 patch）
- image_size: 224 → 10（token 数量从 196 降至 10）
- depth: 12 → 4（token 少，不需要太多层）
- 其余：uniform attention、fixed_square activation、exact LN

### 4.3 训练

- 复用 main.py 训练框架
- 数据集路径：datasets.py 新增 FinanceDataset
- 输出：冻结 bundle

### 4.4 SPU 推理

- 复用 server_inference_friendly_pack 入口
- 新增金融数据的输入 share 生成脚本
- 验证双向隐私边界

## 5. 实施步骤

| 步骤 | 内容 | 预计时间 |
|---|---|---|
| P-F1 | 数据准备：下载/生成金融数据集，编写 FinanceDataset | 30min |
| P-F2 | 数据编码器：金融特征 → token 序列 | 30min |
| P-F3 | 模型适配：调整 ViT 输入层，适配金融数据形状 | 30min |
| P-F4 | 训练：在金融数据上训练金融 ViT 模型 | 1-2h |
| P-F5 | 本地验证：plaintext 推理精度验证 | 15min |
| P-F6 | SPU 推理：在服务器上验证安全推理 | 30min |
| P-F7 | 文档更新：更新作品报告、创新性说明 | 30min |

## 6. 预期结果

| 指标 | 医疗（胸片） | 金融（欺诈检测） |
|---|---|---|
| 任务 | 二分类 | 二分类 |
| 数据类型 | 图像 | 表格 |
| 输入格式 | 196 tokens (14×14 patches) | 10 tokens (特征分组) |
| threshold accuracy | 91.98% | 预期 >95%（欺诈检测通常精度更高） |
| 隐私保护 | ✅ 双向完整 | ✅ 双向完整 |
| SPU 推理 | ✅ 已验证 | ✅ 复用链路 |

## 7. 答辩价值

- 证明 Transshield 不是"医疗专用"，而是"通用隐私推理框架"
- 跨领域验证增强方法可信度
- 金融场景的隐私需求（银行间联合风控、反洗钱）与医疗同样强烈
- 竞赛评审对"通用性"通常给予加分

## 8. 风险

| 风险 | 影响 | 应对 |
|---|---|---|
| 金融数据 token 数过少（10 vs 196） | pruning 的效果可能不明显 | 调整 token 分组策略，或使用更多特征 |
| 训练数据不平衡 | 欺诈样本极少 | 使用 weighted loss 或采样策略 |
| 时间不够 | 无法完成 SPU 验证 | 至少完成 plaintext 训练 + 精度验证 |

## 9. 完成状态（2026-05-11）

| 步骤 | 内容 | 状态 |
|---|---|---|
| P-F1 | 数据准备：`tools/prepare_finance_v3.py` → `data/finance_fraud_v3/` | ✅ 完成 |
| P-F2 | 数据编码器：v1 fail → v2 fail → **v3 image-like encoding** 成功 | ✅ 完成 |
| P-F3 | 模型适配：复用 DeiT-S 12 层，零架构修改 | ✅ 完成 |
| P-F4 | 训练：`finance_v3_20260511_125609`，val accuracy **99.5%** | ✅ 完成 |
| P-F5 | 本地验证：plaintext 99.5% on 200 val images | ✅ 完成 |
| P-F6 | SPU 推理：`finance_keepmask_smoke8_20260511_131750`，argmax_match=1.0 | ✅ 完成 |
| P-F7 | 文档更新：current_work_status / handoff-next / report / bundle README | ✅ 完成 |

### 关键经验

1. **编码方式决定一切**：金融数值数据必须编码为"类图像"的平滑区域才能复用 ViT 的 patch embedding
2. **蒸馏必须禁用**：医疗域 teacher 完全不适合金融域，开启蒸馏会把 accuracy 压死
3. **核心方法 domain-agnostic**：F_mux / F_less / bitonic sort 在金融域上零修改直接复用
4. **双向隐私完整**：服务器看不到金融数据明文，数据使用方获取不到模型参数

### 最终成果

Transshield 已从"医疗影像隐私推理系统"升级为**通用隐私保护推理框架**。
