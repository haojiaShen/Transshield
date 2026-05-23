# artifacts

本目录现在只保留最终交付仍直接有价值的模型资产与结果证据。

## 当前保留

- `artifacts/frozen_bundle_medical_dynamic_mainline/`
  - 医疗正式主线冻结 bundle
- `artifacts/frozen_bundle_finance_boundary_stress/`
  - 金融边界压力验证冻结 bundle
- `artifacts/server_pipeline_run/`
  - 仅保留最终通信与主线对比所需的小体量运行证据
- `artifacts/train_runs/cnn_plaintext_baseline_final/`
  - 仅保留正式图件仍会使用的 CNN baseline 结果

## 已归档

- 原 `artifacts/server_inference_friendly_pack/`
  - 已移入 `archive/deprecated/artifacts/server_inference_friendly_pack/`
  - 原因：依赖已移除的前端、旧 sidecar 工具和外部基线环境，不再属于当前最终交付主链路
- `artifacts/server_pipeline_run/` 中的大体量准备目录与静态参考目录
  - 已移入 `archive/old_runs/artifacts/server_pipeline_run/`
- bundle / baseline 中仅作历史说明的辅助文件
  - 已移入 `archive/deprecated/artifacts/`
