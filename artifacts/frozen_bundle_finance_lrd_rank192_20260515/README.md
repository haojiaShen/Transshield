# Finance LRD Rank192 Bundle

这是当前金融主展示线使用的冻结 bundle 元数据目录。

## 当前用途

- 对应最终报告中的金融模型主线：`depth12 + LRD rank192 merged`
- 用于金融领域完整隐私推理展示
- 默认被 Web demo 金融 live run 使用

## 目录说明

- `args_snapshot.json`：训练 / 导出参数快照
- `threshold_best.json`：金融二分类阈值文件
- `modified_plaintext_model_state_dict.pth`：最终推理权重（已通过 `.gitignore` 排除，不进 Git）
- `modified_plaintext_model_state_dict_ema.pth`：EMA 权重备份（已通过 `.gitignore` 排除，不进 Git）

## 说明

本仓只跟踪元数据文件；大权重文件保留在本地或服务器运行环境中。
