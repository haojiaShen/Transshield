# 外部基线对比

最后更新：`2026-04-23`

本页只保留当前**公平且仍然有效**的外部对比。

## 当前推荐工作流

- 运行入口：`artifacts/server_inference_friendly_pack/run_fair_external_comparison.sh`
- 报告输出：`results/fair_external_comparison/<run>/fair_external_comparison.json`
- Markdown 摘要：`results/fair_external_comparison/<run>/fair_external_comparison.md`
- 作用：把 `Transshield` 当前 bundle 的 full-val 结果，与 `MPCViT` 在**同一 train/val 路径**上的结果汇总到同一份报告中。

## 最新已完成公平对比

当前最新服务器公平报告：

- `results/fair_external_comparison/fair_external_20260423_113217/fair_external_comparison.json`
- `results/fair_external_comparison/fair_external_20260423_113217/fair_external_comparison.md`
- `accuracy_comparison_is_fair = true`
- `Transshield rpc_total_bytes = 10020639`

本轮使用 `MPCVIT_SEEDS="1 2"`，与旧公平报告主表 seed 口径一致；注意这里必须使用空格分隔，外部 `MPCViT` wrapper 会按 shell words 遍历 seed，不能写成逗号分隔。

## 当前主对比对象

- 对比对象：`MPCViT`
- 对比性质：同数据集明文基线
- 作用：衡量本项目在当前数据集上与强明文参考模型的效果差距

## 当前可直接引用的对比

来源：`results/fair_external_comparison/fair_external_20260423_113217/fair_external_comparison.json`

前端静态展示文件 `artifacts/web_demo_assets/best_demo_content.json` 已同步更新到这组来源：`external_comparison.source_run = fair_external_20260423_113217`。

| 指标 | 本项目 Transshield | 外部基线 MPCViT | 差值 |
|---|---:|---:|---:|
| Argmax 准确率 | `93.702292%` | `96.660305%` | `-2.958013 pt` |
| Threshold 准确率 | `94.083971%` | `96.946565%` | `-2.862594 pt` |
| AUC | `0.972313` | `0.993449` | `-0.021137` |

## 如需重跑的直接可执行命令

```bash
export REPO_ROOT=/data/wyb/Transshield_final
export PYTHON_BIN=/data/wyb/conda_envs/transshield/bin/python
mkdir -p /data/wyb/tmp
export TMPDIR=/data/wyb/tmp
export TMP=/data/wyb/tmp
export TEMP=/data/wyb/tmp

export TRAIN_DATA_PATH=/data/wyb/pneumoniamnist_imagefolder_subset/train
export VAL_DATA_PATH=/data/wyb/pneumoniamnist_imagefolder_subset/val
export BUNDLE_DIR="$REPO_ROOT/artifacts/frozen_bundle_verified_tracka_lr3e5_20260414"

run_fair=fair_external_$(date +%Y%m%d_%H%M%S)
export RUN_NAME="$run_fair"
export FAIR_OUTPUT_DIR="$REPO_ROOT/results/fair_external_comparison/$RUN_NAME"
export SECURE_RUN_DIR="$REPO_ROOT/artifacts/server_pipeline_run/${RUN_NAME}_transshield"

export MPCVIT_SEEDS="1 2"
export MPCVIT_DEVICE=cuda

cd "$REPO_ROOT"
bash artifacts/server_inference_friendly_pack/run_fair_external_comparison.sh

grep -nE '"accuracy_comparison_is_fair"|\"argmax_accuracy\"|\"threshold_accuracy\"|\"auc\"|\"rpc_total_bytes\"' \
  "$FAIR_OUTPUT_DIR/fair_external_comparison.json" || true
```

如果回贴结果满足：

- `accuracy_comparison_is_fair = true`
- `Transshield` 与 `MPCViT` 使用同一组 `train/val`
- 验证集样本量一致

那么下一步就更新：

- `docs/external_baseline_comparison.md`
- `docs/current_work_status.md`
- `docs/handoff-next.md`

## 这组对比该怎么讲

- 这是一组**同数据集、同任务**的模型效果对比；
- 它回答的是“本项目效果是否接近强明文参考模型”；
- 它**不是** secure 通信量对比，也不是协议层速度对比。

## 公平性要求

只有同时满足以下条件，才可以把结果写进答辩主表或前端对比区：

- `Transshield` 与 `MPCViT` 使用同一组 `train/val` 路径；
- 验证集样本量一致；
- 若是多 seed 汇总，所有 seed 的 `train/val` 路径与样本量都一致；
- `Transshield` 的验证集文件列表哈希与本次指定 `VAL_DATA_PATH` 一致。

如果上述任一条件不满足，则只能把该结果视为“参考跑通”，不能当作严格公平对比。

## 为什么不用旧 secure profile 做主对比

因为旧文档中的以下数字不是同口径、同样本量、同展示目标的数据：

- `1.90 MB`：历史 fastpath 8 样本记录
- `979.9903s` / `975.1174s` / `3.21 GB`：旧 archived SPU profile

这些数字已经从当前前端和主文档中移除。

## 外部 secure benchmark 现在怎么用

当前仓库已经补了一个**统一入口**，专门用来在同一个 secure benchmark harness 下重跑外部模型 proxy：

- 运行入口：`artifacts/server_inference_friendly_pack/run_standardized_secure_external_benchmark.sh`
- 报告输出：`results/standardized_secure_benchmark/<run>/standardized_secure_benchmark.json`
- Markdown 摘要：`results/standardized_secure_benchmark/<run>/standardized_secure_benchmark.md`

它的作用是：

- 把 `Transshield` 当前最终模型 proxy 和外部模型 proxy 都放进同一个 `MPCFormer local 2PC configurable transformer benchmark`；
- 在同一个 benchmark harness 下统计时间与模块通信量；
- 明确区分：
  - 这是 **统一 secure transformer benchmark**
  - 不是 **full-val 医学图像 pipeline**
  - 也不是 **网页单图 live run**

当前建议把它分成两类口径：

1. `architecture_proxy`
   - 各自使用自己的结构参数；
   - 适合回答“在同一个 2PC benchmark harness 下，不同模型 proxy 的 secure 开销大致怎样”。
2. `same_shape_operator_proxy`
   - 固定同一模型形状，只替换算子配置；
   - 适合回答“算子替换本身会让 secure 开销怎么变化”。

### 最新已完成 benchmark

当前最新服务器 benchmark 报告：

- `results/standardized_secure_benchmark/standardized_secure_benchmark_20260423_130435/standardized_secure_benchmark.json`
- `results/standardized_secure_benchmark/standardized_secure_benchmark_20260423_130435/standardized_secure_benchmark.md`

其中 `architecture_proxy` 对比结果为：

| 对象 | 模块通信均值 | 时间均值 |
|---|---:|---:|
| `Transshield 当前最终模型 proxy` | `4422.72 MiB` | `13.4821 s` |
| `MPCViT vit_7_4_32 proxy` | `262.56 MiB` | `4.5622 s` |

对应比值：

- 通信：`16.8447x`
- 时间：`2.9552x`

这组结果的解释必须固定为：

- 它说明在**同一 MPCFormer local 2PC benchmark harness** 下，当前 `Transshield` proxy 结构比 `MPCViT vit_7_4_32` proxy 更重；
- 它使用的是当前正式 bundle 的**结构 proxy**，不是加载历史最优 checkpoint 后跑完整模型；
- 它不表示 full-val 医学图像 pipeline 的真实总通信；
- 它也不表示网页单图 live run 的真实总耗时。

### 最新已完成 same-shape operator benchmark

当前最新 same-shape 报告：

- `results/standardized_secure_benchmark/standardized_secure_benchmark_20260423_132121_same_shape/standardized_secure_benchmark.json`
- `results/standardized_secure_benchmark/standardized_secure_benchmark_20260423_132121_same_shape/standardized_secure_benchmark.md`

其中 `same_shape_operator_proxy` 对比结果为：

| 对象 | 模块通信均值 | 时间均值 |
|---|---:|---:|
| `Transshield secure-friendly ops same-shape proxy` | `881.05 MiB` | `8.1045 s` |
| `External baseline ops same-shape proxy` | `5918.69 MiB` | `15.3365 s` |

对应比值：

- 通信：`0.1489x`
- 时间：`0.5284x`

这组结果的解释必须固定为：

- 它比较的是**固定 DeiT-S 形状**下的算子配置差异；
- 它说明 `quad + softmax_2QUAD` 这类 secure-friendly ops replacement 本身是明显正向的；
- 因此前一条 `architecture_proxy` 的高开销，不能简单归咎为 secure-friendly ops 本身更重，而应更多理解为**模型结构尺度差异**导致。

## 这条 benchmark 线的价值和边界

这条线的目标是验证本项目的**算法/算子替换是否真的对 secure 计算友好**。它的结论可以这样讲：

- 当前正式模型在外部公平精度上接近强明文基线，但结构尺度更大；
- 当用 `architecture_proxy` 按各自结构进入同一 MPCFormer harness 时，`Transshield` proxy 更重，说明结构尺度是主要开销来源；
- 当固定同一 DeiT-S 形状，只比较算子配置时，`Transshield secure-friendly ops` 明显降低通信与时间，说明算法替换本身有效。

它和“历史最优正式模型”的关系必须说清楚：

- 它不是重新加载历史最优 checkpoint 做 full-val secure inference；
- 它不会复现正式模型的验证集准确率，也不会替代 `secure sidecar + replay + compare`；
- 它只把当前正式 bundle 的结构/算子设计抽象成 benchmark proxy，用来证明“结构尺度”和“算子替换”两个因素分别带来的 secure 开销影响。

因此，当前仓库里**可以**新增外部 secure benchmark 数字，但必须满足：

- 同一个 benchmark harness；
- 同一批运行参数（batch / warmup / repeats / world size）；
- 报告里明确写明这不是 full-val image pipeline。

仍然禁止：

- 把 benchmark 数字写成网页单图通信量；
- 把 benchmark 数字写成 full-val sidecar 总通信量；
- 把不同样本量、不同协议路径的数据直接混在同一张主表里。
