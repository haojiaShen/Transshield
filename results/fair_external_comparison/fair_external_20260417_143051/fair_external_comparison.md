# Transshield 公平外部对比

## 公平口径

- 数据集：`PneumoniaMNIST imagefolder subset`
- 训练集：`/data/wyb/pneumoniamnist_imagefolder_subset/train`
- 验证集：`/data/wyb/pneumoniamnist_imagefolder_subset/val`
- 准确率对比是否同数据集：`True`
- 准确率口径说明：Transshield 与 MPCViT 都指向同一组 train/val 路径，样本量也一致，可做同数据集效果对比。
- secure 通信量是否同协议可比：`False`
- 通信量说明：外部 MPCViT 当前没有同数据集、同输入、同 SPU/2PC 协议路径的 secure 通信结果；因此通信量只能展示 Transshield 自身 live/full-run 结果，不能和外部模型硬比。

## 公平性自检

| 检查项 | 当前值 | 结果 |
|---|---|---|
| 请求的 train 样本量 | 4708 | 基准 |
| 请求的 val 样本量 | 524 | 基准 |
| Transshield val 路径 | `/data/wyb/pneumoniamnist_imagefolder_subset/val` | true |
| Transshield val 样本量 | 524 | true |
| Transshield val 文件列表哈希 | `6bd7a08763ddecd63308cc7e5848df35aeaa8c3430c7278b6e8db77af4f3c610` | true |
| MPCViT train 路径 | `/data/wyb/pneumoniamnist_imagefolder_subset/train` | true |
| MPCViT val 路径 | `/data/wyb/pneumoniamnist_imagefolder_subset/val` | true |
| MPCViT train 样本量 | 4708 | true |
| MPCViT val 样本量 | 524 | true |
| 多 seed 口径一致 | `{'all_runs_same_train_dir': True, 'all_runs_same_val_dir': True, 'all_runs_same_train_sample_count': True, 'all_runs_same_val_sample_count': True, 'unique_train_dirs': ['/data/wyb/pneumoniamnist_imagefolder_subset/train'], 'unique_val_dirs': ['/data/wyb/pneumoniamnist_imagefolder_subset/val'], 'unique_train_sample_counts': [4708], 'unique_val_sample_counts': [524]}` | 仅多 seed 有意义 |

## 准确率 / AUC 结果表（仅公平性通过时可用于主对比）

| 方法 | 样本数 | Argmax Acc (%) | Threshold Acc (%) | AUC | 来源 |
|---|---:|---:|---:|---:|---|
| Transshield modified + secure replay | 524 | 93.702292 | 94.083971 | 0.972313 | `/data/wyb/Transshield_final/artifacts/server_pipeline_run/fair_external_20260417_143051/plaintext_modified_eval.json` |
| MPCViT | 524 | 96.660305 | 96.946565 | 0.993449 | `/data/wyb/Transshield_final/results/fair_external_comparison/fair_external_20260417_143051/mpcvit/summary_multiseed.json` |

## 差值

| 指标 | Transshield - MPCViT |
|---|---:|
| Argmax Acc | -2.958013 pt |
| Threshold Acc | -2.862594 pt |
| AUC | -0.021137 |

## Secure 通信 / 运行开销（当前不做外部硬比）

| 方法 | 样本数 | Runtime | Total Pipeline Sec | RPC Bytes | 状态 | 是否外部可比 | 原因 |
|---|---:|---|---:|---:|---|---|---|
| Transshield SPU secure sidecar | 524 | spu | 16.532587 | 7686293 | available_python_fastpath | false | 当前没有外部模型在同数据集、同输入、同协议路径下的 secure 通信结果。 |
| MPCViT | 524 | N/A | N/A | N/A | not_run_secure | false | 当前 MPCViT 只有同数据集明文训练/评估结果，没有同协议 secure 通信结果。 |

## 结论

- 当前仅在公平性自检通过时，才能把准确率 / AUC 当作正式外部对比口径。
- 当前不能公平展示：外部模型 secure 通信量对比。
- 如果要补 secure 通信外部对比，需要让外部模型也走同输入、同样本量、同 SPU/2PC 协议路径后再统计。
