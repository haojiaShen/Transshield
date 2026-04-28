# archive 目录说明

本目录用于存放“默认比赛展示与复现流程不再直接使用，但仍建议保留以便追溯”的资产。

当前已归档内容包括：

- `baselines/baseline_plaintext_training_checkpoint_full.pth`  
  baseline 的完整服务器 checkpoint 原件，仅用于追溯与恢复训练，不是默认展示流程的运行入口。它就是原始 `checkpoint-best.pth` 的归档重命名版，保留了完整 checkpoint 语义。

- `frozen_bundle_full/modified_plaintext_training_checkpoint_full.pth`  
  modified 模型的完整服务器 checkpoint 原件，用于恢复训练、核对训练来源与补充强复现说明，不是默认比赛展示链路的运行入口。

- `logs/frozen_bundle_full_train_stdout.log`  
  modified 模型训练 stdout 日志归档。

- `logs/frozen_bundle_full_eval_threshold_stdout.log`  
  modified 模型阈值评估 stdout 日志归档。

- `logs/frozen_bundle_full_log.txt`  
  modified 模型训练过程的结构化日志归档。

使用原则：

- 默认比赛展示、默认对比脚本、默认 secure pipeline **不依赖** 本目录。
- 若后续需要追溯训练来源、核对完整训练状态或恢复非轻量流程，可人工查阅本目录；其中 baseline 恢复训练应优先使用 `baselines/baseline_plaintext_training_checkpoint_full.pth`。
