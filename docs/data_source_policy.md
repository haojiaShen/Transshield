# 数据来源与展示口径规范

最后更新：`2026-04-26`

本文件是当前仓库关于“哪些数据可以展示、展示给谁看、能不能互相比”的唯一口径说明。

## 0. 当前默认数据集与目录格式

- 当前默认数据集是 `pneumoniamnist_imagefolder_subset`
- `baseline`、`modified`、legacy `secure sidecar + replay + compare`、当前 E2E approximate eval 默认共用这一套 `train/val`
- 默认目录结构遵循 `torchvision.datasets.ImageFolder`：

```text
pneumoniamnist_imagefolder_subset/
  train/
    0/
    1/
  val/
    0/
    1/
```

- 当前任务是二分类，因此默认要求：
  - `nb_classes=2`
  - `0/1` 标签语义不交换
  - baseline、modified、secure 使用同一验证集

- 如果后续要做公平外部对比或重新跑 full-val，首先要保证：
  - `TRAIN_DATA_PATH` / `VAL_DATA_PATH` 明确且同口径
  - 验证集文件列表与当前正式口径一致
  - 不把单图结果和全验证集结果混讲

## 1. 当前浏览器选择图片的即时结果

- **来源**
  - Web demo 运行时的 `/api/e2e/analyze_private_shares`
  - 浏览器本地读取原图、预处理并生成 `share0/share1`
  - 后端只接收两份 share，不接收原图或完整 plaintext `pixel_values`
- **数据内容**
  - 当前图片的 E2E approximate SPU `argmax` 类别
  - 当前图片的阈值判定类别
  - 当前图片的类别概率
  - 当前图片本次 E2E SPU live run 的耗时与通信证据
  - 隐私边界字段：`input_pt=null`、`host_plaintext_pixel_values_materialized=false`、`host_private_share_tensors_loaded=false` 等
- **适用范围**
  - 页面顶部总览
  - `交互演示`
  - `本次运行结果`
- **禁止用途**
  - 不能把它写成数据集级准确率
  - 不能拿当前一张图的通信量去和历史批量运行结果直接比
  - 不能把当前 demo 的“单进程同时接收两份 debug share”写成生产级 P1/P2 独立部署

旧 `/api/upload` 与 `/api/run_secure` 只保留为 legacy sidecar 调试路径，默认不属于当前最终 Web 主口径。

## 2. 离线验证集最佳成绩

- **来源**
  - `artifacts/web_demo_assets/best_demo_content.json`
  - 当前字段：`default_bundle`、`external_comparison`
- **数据内容**
  - `best_epoch = 8`
  - `argmax_accuracy = 93.702292%`
  - `threshold_accuracy = 94.083971%`
  - `auc = 0.972313`
  - `argmax_match_ratio = 100%`
  - `threshold_match_ratio = 100%`
- **适用范围**
  - 页面中的统一对比区
  - `docs/transshield_master_plan_20260505.md`
  - `results/delivery_acceptance/delivery_acceptance_20260510_full/delivery_acceptance_report.md`
  - `results/fair_external_comparison/fair_external_secure_static_20260505_clean/fair_external_comparison.md`
- **禁止用途**
  - 不能冒充当前浏览器选择图片的即时结果
  - 不能直接写成“本次上传图片准确率”

## 3. 外部基线对比

- **来源**
  - `artifacts/web_demo_assets/best_demo_content.json` 中的 `external_comparison`
- **补充来源**
  - `results/fair_external_comparison/<run>/fair_external_comparison.json`
  - `results/fair_external_comparison/<run>/fair_external_comparison.md`
- **当前主对比对象**
  - `MPCViT` 同数据集明文基线
- **适用范围**
  - 准确率 / AUC 的外部效果对比
- **推荐用途**
  - 前端静态展示可继续使用 `best_demo_content.json`
  - 如果要做“同路径、同样本量、同服务器”的最新公平对比，优先使用 `fair_external_comparison.*`
- **禁止用途**
  - 不能写成“外部 secure baseline 已与本项目同口径复现”
  - 不能把外部论文或历史日志中的通信量拿来和当前单图 live run 直接比较

## 3.1 公平外部对比报告

- **运行入口**
  - `artifacts/server_inference_friendly_pack/run_fair_external_comparison.sh`
- **报告生成器**
  - `tools/transshield_fair_external_comparison.py`
- **当前口径**
  - 会自动检查 `Transshield` 与 `MPCViT` 是否使用同一组 `train/val` 路径
  - 会自动检查验证集样本量是否一致
  - 对 `Transshield` 会额外校验验证集文件列表哈希
- **允许展示**
  - 同数据集准确率 / AUC
- **仍然禁止展示**
  - 任何未经同协议 secure 复现的外部通信量“硬比较”

## 4. 外部 secure benchmark

- **来源**
  - `artifacts/server_inference_friendly_pack/run_standardized_secure_external_benchmark.sh`
  - `results/standardized_secure_benchmark/<run>/standardized_secure_benchmark.json`
  - `results/standardized_secure_benchmark/<run>/standardized_secure_benchmark.md`
- **适用范围**
  - 解释在**同一个 secure transformer benchmark harness**下，不同模型 proxy / 算子 proxy 的通信与时间差异
  - 解释为什么 attention / 非线性 / 选择边界通常是 secure 推理热点
- **前提**
  - 必须明确标注这不是 full-val 医学图像 pipeline
  - 必须明确标注这不是网页单图 live run
- **禁止用途**
  - 不能冒充当前页面的单图通信量
  - 不能冒充 full-val sidecar 总通信量
- 不能把 benchmark 数字和不同样本量的真实 pipeline 数字直接硬比

## 4.1 研究性 ablation 结果

- **来源**
  - `margin-aware pruning` 相关服务器实验报告
- **当前已确认案例**
  - `w10` 候选能显著拉开 Stage 2 剪枝边界，并保持 secure replay `100%` 一致
- **适用范围**
  - 算法升级路线图
  - 研究性证据
  - 后续训练改造依据
- **禁止用途**
  - 不能写进当前 Web demo 主成绩卡
  - 不能替代 `artifacts/web_demo_assets/best_demo_content.json` 中的正式展示成绩
  - 不能写成“当前正式模型已经切换到 margin-aware 版本”

## 5. 明确禁用的数据

以下内容已经不允许继续出现在当前前端主展示、答辩结论或新文档中：

- 历史 fastpath 8 样本通信量 `1.90 MB`
- 旧 archived SPU profile：
  - `979.9903s`
  - `975.1174s`
  - `3.21 GB`
- 相对旧正式展示模型的训练收益
- 已删除 dated 文档中的任何数字

## 6. 页面展示规则

- **顶部与交互区**：只放当前图片即时结果
- **统一对比区**：只放离线验证集最佳成绩与外部明文基线对比
- **通信量**：只放本次 E2E SPU live run，或同口径 `run_e2e_secure_approx_eval.sh` 输出
- **统一 secure benchmark**：只能在明确写明 benchmark 口径时单独展示
- **附录**：只保留当前仍有复现价值的工程说明，不保留历史不公平数字

## 7. 更新规则

如果以后要加入新数据，必须同时满足：

1. 能明确写出来源文件或运行接口；
2. 能说明是单图、批量、还是全验证集；
3. 能说明是否与对比对象同口径；
4. 不会和本文件第 5 节的禁用数据冲突。
