# SPU batch=16 批量吞吐验证

## 目标与适用范围

该档面向已经排队的批量医疗推理，不面向单张图片的实时上传。它不改变模型权重、剪枝算法、秘密参数、party-local 输入分片或 final-logits-only reveal，仅把同形安全图从 `batch=8` 扩大到 `batch=16`，减少 32 样本任务的 chunk 数量。

低延迟剪枝参数保持为 `base_rate=0.655`、token 保留数 `128/84/55`，并继续使用从 524 样本校准得到的公开阈值 `0.688152923150007`。正式单图展示默认值和 `results/final/` 均未修改。

## VPS 结果

环境：16 vCPU Intel Xeon Platinum、约 61.4 GiB、Python 3.9.25、SPU 0.9.3b0、JAX/JAXLIB 0.4.30、CHEETAH + YACL Ferret、FM64/fxp16。模型测试全部在 VPS 执行。

| 指标 | 报告原始口径 | 当前 r=0.7/batch8 | r=0.655/batch8 | r=0.655/batch16 |
|---|---:|---:|---:|---:|
| medical32 安全前向 | 2849.945 s | 1463.214 s | 1151.815 s | **1060.968 s** |
| 每样本安全前向 | 89.061 s | 45.725 s | 35.994 s | **33.155 s** |
| loopback 通信 | 84.466 GiB | 53.036 GiB | 50.407 GiB | **49.504 GiB** |
| 阈值准确率 | 93.75% | 96.875% | 93.75% | **93.75%** |
| AUC | — | 0.9844 | 0.9766 | **0.9805** |

相对同一 `r=0.655/batch8` 图，batch16 的安全前向减少 7.89%，通信减少 1.79%；相对当前 `r=0.7/batch8` 主线，安全前向减少 27.49%，通信减少 6.66%；相对报告原始口径，安全前向减少 62.77%，通信减少 41.39%。

两个 batch16 chunk 分别耗时 527.946 秒和 532.419 秒，结果不是依赖单个最快 chunk。32 个输出均为有限值，预注册阈值准确率保持 93.75%。batch16 与 batch8 的 argmax 完全一致，但阈值预测匹配率为 93.75%，最大概率差为 0.0304，说明批次形状会造成固定点执行漂移；不能表述为逐样本数值完全一致。AUC 的小幅变化也不应被当作模型精度提升。

## 复现命令

在已启动的 2PC/SPU 节点上运行：

```bash
python integrations/transshield_runtime/e2e_secure_vit/transshield_e2e_secure_vit.py run \
  --runtime spu \
  --bundle-dir artifacts/frozen_bundle_medical_dynamic_mainline \
  --input-share-public-manifest-json results/vps_report_tests/report_regression_20260801_v1/medical32_public.json \
  --input-p1-share-manifest-json results/vps_report_tests/report_regression_20260801_v1/medical32_party_manifests/p1_share_manifest.json \
  --input-p2-share-manifest-json results/vps_report_tests/report_regression_20260801_v1/medical32_party_manifests/p2_share_manifest.json \
  --party-local-share-load \
  --redact-private-input-paths \
  --max-samples 32 \
  --static-depth-limit 10 \
  --spu-batch-size 16 \
  --spu-params-mode secret \
  --spu-layer-norm-policy exact \
  --spu-attention-policy uniform \
  --spu-activation-override fixed_square \
  --spu-secure-pruning-mode compact \
  --spu-secure-pruning-network unpadded_selection \
  --spu-final-block-cls-only \
  --spu-uniform-attention-value-fusion \
  --token-ratio-base-override 0.655 \
  --output-pt /path/to/medical32_batch16.pt \
  --output-json /path/to/medical32_batch16.json
```

`batch=16` 只适合至少已有 16 个排队样本、且样本数最好能被 16 整除的任务。runner 会把不足一个 batch 的末块补齐；对单张图片使用 batch16 会计算 15 个无效填充位置，延迟和资源占用都会显著恶化。因此医疗 live upload 仍保持 `batch=1`。

完整 candidate JSON、逐样本结果、网络快照和 time log 位于 `results/vps_optimization/batch_scaling_20260802_v1/`。

## 本轮否决项

- 剪枝预测器 `[-10, 10]` 裁剪消除：524 样本原始 logits 均在约 `[-0.003, 0.003]`，但单样本热缓存反而慢 0.23%，通信只减少 0.013%，已撤销代码开关。
- SPU `enable_lower_accuracy_rsqrt=true`：24 个 rsqrt 节点的热缓存耗时只减少 0.07%，通信减少 0.07%，不足以承担低精度运行时风险。

否决候选的原始结果和结构化结论位于 `results/vps_optimization/pruning_clip_elision_20260802_v1/`。
