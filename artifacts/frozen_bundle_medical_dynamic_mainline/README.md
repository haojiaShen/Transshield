# frozen_bundle_medical_dynamic_mainline

医疗正式主线冻结 bundle。

## 当前保留文件

- `args_snapshot.json`：训练 / 导出参数快照
- `manifest.json`：导出元数据与阈值摘要
- `modified_plaintext_model_state_dict.pth`：正式推理权重

## 说明

- 当前精简交付仓只保留正式推理资产，不再保留原训练 wrapper。
- `manifest.json` 中的 `primary.threshold_metrics` 保留的是 bundle 导出时的历史/静态阈值摘要。
- 当前报告和展示站使用的正式医疗动态阈值以 `results/final/medical_dynamic_threshold_calibration_final.json` 为准；该文件记录 `best_threshold=0.6619606018066406`、`best_threshold_accuracy=0.927480936050415` 和 `static_depth_limit=10`。
- 历史阈值搜索命令说明已移入 `archive/deprecated/artifacts/frozen_bundle_medical_dynamic_mainline/commands.sh`。
