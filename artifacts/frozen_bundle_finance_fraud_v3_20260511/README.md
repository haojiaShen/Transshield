# Finance Fraud Detection Bundle (v3)

冻结 bundle：基于 `finance_v3_20260511_125609` 训练 run。

## 概况

| 属性 | 值 |
|---|---|
| 数据集 | `data/finance_fraud_v3/`（信用卡欺诈检测，v3 image-like 编码） |
| 编码方式 | 30 个 PCA 特征 → 224x224 灰度图（normal=smooth gradient，fraud=high-contrast patch） |
| 训练集 | 500 normal + 500 fraud（1000 张） |
| 验证集 | 100 normal + 100 fraud（200 张） |
| 模型 | DeiT-S（depth=12, embed_dim=384, num_heads=6） |
| epochs | 15 |
| lr | 1e-4 |
| batch_size | 16 |
| smoothing | 0.0 |
| warmup_epochs | 2 |
| distillation | 无（ratio_weight=0, cls_distill=0, token_distill=0） |
| 最佳 val accuracy | **99.5%** |
| 训练时间 | ~2 分钟（服务器 GPU） |

## 文件列表

| 文件 | 说明 |
|---|---|
| `checkpoint-best.pth` | 最佳 checkpoint（model state dict） |
| `modified_plaintext_model_state_dict.pth` | plaintext 推理用 state dict |
| `args_snapshot.json` | 训练参数快照（SPU 推理需要 `imagenet_default_mean_and_std` 等字段） |
| `threshold_best.json` | 最佳阈值 + 精度统计 |
| `train_stdout.log` | 训练日志 |

## SPU 安全推理验证

Run: `finance_keepmask_smoke8_20260511_131750`

| 指标 | 值 |
|---|---|
| argmax_match_ratio | 1.0 (8/8) |
| max_abs_logits_error | 0.000935 |
| host_plaintext_pixel_values_materialized | false |
| elapsed_sec | 82.96s |

## 训练经验

- **关键发现**：金融域训练必须**禁用蒸馏**。医疗域 teacher 信号（pneumoniamnist）完全不适合金融域（fraud detection），开启蒸馏会把 accuracy 压死在 57-66%。
- v1（patch-based）和 v2（DCT encoding）编码方式失败，v3（image-like smooth/contrast encoding）成功。
- 数据平衡非常重要：从原始 284807 条（欺诈仅 0.17%）下采样为 500+500 balanced。
