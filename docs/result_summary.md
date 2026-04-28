# 当前结果摘要

最后更新：`2026-04-22`

本页只记录**当前离线验证集权威成绩**，不记录单图 live run 的即时结果。

## 当前展示包

- 展示名称：`当前主展示模型（已验证）`
- 目录：`artifacts/frozen_bundle_verified_tracka_lr3e5_20260414`
- 最佳轮次：`epoch 8`

说明：最佳轮次表示训练过程中验证集效果最好的 epoch，当前冻结展示包使用这一轮导出的权重与阈值。

## 当前权威 provenance

当前正式展示包的 provenance 以以下文件为准：

- `artifacts/frozen_bundle_verified_tracka_lr3e5_20260414/source_commands.sh`
- `artifacts/frozen_bundle_verified_tracka_lr3e5_20260414/source_manifest.json`
- `artifacts/frozen_bundle_verified_tracka_lr3e5_20260414/args_snapshot.json`
- `artifacts/frozen_bundle_verified_tracka_lr3e5_20260414/threshold_best.json`
- `artifacts/frozen_candidates/tracka_lr3e5_timm_best_20260414/train_stdout.log`

当前 TrackA 排障、drift audit 或新的重训诊断，不会反向替换这里的正式展示 provenance。

## 当前权威指标

来源：`artifacts/web_demo_assets/best_demo_content.json`

| 指标 | 数值 | 说明 |
|---|---:|---|
| Argmax 准确率 | `93.702292%` | 离线验证集分类效果 |
| Threshold 准确率 | `94.083971%` | 二分类阈值判定效果 |
| AUC | `0.972313` | 判别能力指标 |
| Argmax 一致率 | `100%` | secure 输出与明文输出一致 |
| Threshold 一致率 | `100%` | secure 输出与明文输出一致 |
| Secure pipeline / replay | `均通过` | 说明当前 secure 闭环已验证通过 |

## 如何正确使用这些数字

- 可以用于：
  - 页面统一对比区
  - 答辩中的总体效果总结
  - 与外部明文基线的效果对比
- 不可以用于：
  - 冒充当前上传图片的即时结果
  - 与某次单图 live run 的通信量混着讲

## 关于通信量

当前页面与当前文档都不再使用固定历史字节数作为主展示通信指标。

正确做法是：

- 对单图演示：看本次 `SPU live run`
- 对批量 / 全验证集：重新按同口径复跑，再单独出报告

## 当前页面结论

当前作品的展示口径已经分成两层：

1. **当前上传图片**：只展示即时预测结果与本次 secure 开销；
2. **离线验证集最佳成绩**：只展示总体准确率、AUC 与外部基线差值。
