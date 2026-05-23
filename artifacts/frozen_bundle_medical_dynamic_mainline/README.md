# frozen_bundle_medical_dynamic_mainline

医疗正式主线冻结 bundle。

## 当前保留文件

- `args_snapshot.json`：训练 / 导出参数快照
- `manifest.json`：导出元数据与阈值摘要
- `modified_plaintext_model_state_dict.pth`：正式推理权重

## 说明

- 当前精简交付仓只保留正式推理资产，不再保留原训练 wrapper。
- 本 bundle 的正式阈值信息由 `manifest.json` 中的 `primary.threshold_metrics` 提供。
- 历史阈值搜索命令说明已移入 `archive/deprecated/artifacts/frozen_bundle_medical_dynamic_mainline/commands.sh`。
