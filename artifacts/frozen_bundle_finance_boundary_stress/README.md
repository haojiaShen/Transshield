# frozen_bundle_finance_boundary_stress

这是当前金融边界压力验证使用的冻结 bundle 目录。

## 当前用途

- 对应最终报告中的金融压力验证口径
- 用于金融领域完整隐私推理压力样本展示

## 目录说明

- `args_snapshot.json`：训练 / 导出参数快照
- `threshold_best.json`：金融二分类阈值文件
- `modified_plaintext_model_state_dict.pth`：最终推理权重

## 说明

本目录只保留当前最终压力验证所需的冻结资产。
历史 EMA 备份权重已移入 `archive/deprecated/artifacts/frozen_bundle_finance_boundary_stress/modified_plaintext_model_state_dict_ema.pth`。
