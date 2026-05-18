# 数据口径说明

最后更新：`2026-05-18`

本文件只约束 `Transshield_final/` 当前最终交付仓的**正式展示口径**，避免旧实验数字、旧 bundle 或旧服务器产物再次混入页面、报告和答辩材料。

## 1. 当前主口径

- 最终作品报告：`docs/transshield_竞赛作品报告_最终版.docx`
- 主计划文档：`docs/transshield_master_plan_20260505.md`
- 当前状态摘要：`docs/current_work_status.md`
- 当前交接摘要：`docs/handoff-next.md`
- 创新点说明：`docs/transshield_innovation.md`
- Web 展示静态内容：`artifacts/web_demo_assets/best_demo_content.json`

发生冲突时，优先级按上面顺序执行。

## 2. 允许直接展示的数据

- 医疗主展示线：
  - 固定结构优化后的正式结果
  - `threshold accuracy = 91.98%`
  - `69.57s/sample`
- 金融主展示线：
  - `depth12 + LRD rank192 merged`
  - 完整隐私推理 `117.80s/sample`
  - `argmax_match = 100%`
- 动态裁剪保留能力：
  - keep-mask replay 的 exact 一致性
  - secure pruning 在 SPU 中的可执行性与隐私边界

## 3. 允许作为“过程证据”引用的数据

- `results/` 中被当前文档明确点名的报告
- `artifacts/server_inference_friendly_pack/README.md` 中仍被保留的正式入口说明
- 与最终报告一致的 smoke / heldout / compare 结果

这些数据可以用于解释来源链，但不能替代“最终展示线”的主数字。

## 4. 不再作为正式口径的数据

- 已移除或已放弃的旧 bundle
- 旧公开校准 / 旧 clip3 路线
- 已判负的蒸馏、BLB+LRD 分解式、无收益混合注意力等实验
- 任何只在历史服务器目录中存在、但当前报告没有继续引用的中间产物

## 5. 服务器最小保留原则

服务器重建后，只保留两类内容：

- Git 仓库内当前正式代码、文档、前端、脚本
- 运行最终展示所必需的最小非 Git 资产：
  - 最终 bundle 权重
  - 金融演示样本集
  - 当前页面 live demo 必需的校准 JSON

除此之外的旧训练产物、旧 `results/` 大目录、旧 `artifacts/server_pipeline_run/`、旧日志统一清理。
