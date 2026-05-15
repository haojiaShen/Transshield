# Selection mode runtime inputs

这组文件用于 `run_secure_selection_mode_profile_compare.sh`。

- 用途：给 `flat_odd_even` / `phase3_lower_tail` 两种 `network-kth` 选择模式提供同一份 runtime inputs。
- 来源：从当前 `delivery_line_suite_20260505_clean` 中抽取的最小输入集合
- 保留原因：避免脚本继续依赖整目录历史 `server_pipeline_run/` 产物。
- 说明：这里只保留运行对比所需的最小输入，不代表完整 secure run 结果。
