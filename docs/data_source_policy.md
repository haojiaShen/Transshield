# 数据口径说明

最后更新：`2026-05-20`

本文件约束 `Transshield_final/` 当前最终交付仓的**正式展示口径**，用于统一作品报告、网页展示、答辩材料和演示后端摘要，避免旧实验数字、旧 bundle 或旧服务器产物再次混入正式叙事。

## 1. 当前主口径来源

- 当前主口径作品报告：`docs/transshield_竞赛作品报告_第二次修订版.docx`
- 当前报告生成器：`tools/generate_competition_report.py`
- 主线重建证据目录：`docs/report_evidence/`
- 实验总账：`docs/report_evidence/experiment_manifest.json`
- Web 展示静态内容：`artifacts/web_demo_assets/demo_content_summary.json`
- 创新点说明：`docs/transshield_innovation.md`

发生冲突时，优先级按上面顺序执行。

## 2. 允许直接展示的数据

### 2.1 医疗正式主线

- 正式主线：`dynamic secure pruning + full privacy`
- 正式精度：全量验证集阈值准确率 `92.7481%`
- 正式效率：`32` 样本复核集、批次规模 `8`、深度 `10` 的正式复核配置为 `86.91s/sample`
- 正式部署阈值：`0.6619606018`
- 正式隐私边界：
  - `host_plaintext_pixel_values_materialized = false`
  - `host_model_params_materialized = false`
  - `reveal_policy = final_logits_only`

### 2.2 金融边界压力验证

- 角色定位：`boundary stress validation only`
- 当前压力结果：`8` 样本压力验证集 `argmax / threshold = 100% / 100%`
- 当前压力效率：`105.16s/sample`
- 压缩比例：`68.39%`
- 当前隐私边界：
  - `host_plaintext_pixel_values_materialized = false`
  - `host_model_params_materialized = false`
  - `reveal_policy = final_logits_only`

### 2.3 前半部分统一写法

- 报告摘要、项目概述、最终模型介绍、网页 hero 和首页说明，只能把**医疗**写成正式落地主线。
- 金融只能写成：
  - 边界压力验证
  - 极端分布输入稳定性验证
  - 动态路由鲁棒性验证
- 这些位置不得把金融写成“第二正式场景”“正式双域落地”“跨领域正式验证”。

## 3. 只能放在“详细验证 / 采用原因 / 工程优化”里的数据

- 医疗小样本最快工程配置：`12` 样本工程复核集、批次规模 `12`、深度 `10`，`69.57s/sample`
- 医疗大样本语义基线：`32` 样本复核集、批次规模 `8`、深度 `12`，`102.77s/sample`
- 医疗 dynamic-path 阈值细节：
  - depth12：`0.6226428151`
  - depth10：`0.6619606018`
  - `32` 样本复核集在回代 dynamic threshold 后的阈值准确率为 `93.75%`
- 金融 true static fallback：
  - `103.64s/sample`
  - 与 dynamic `argmax / threshold match = 1.0 / 1.0`
- 各类 rejected / fallback / 历史链实验：
  - distillation
  - public-calibrated LN + clip0
  - clip3
  - token recycle
  - decomposed LRD
  - 历史保留掩码（keep-mask）金融对照链

## 4. 不再允许写成正式主线的数据

- 医疗 static control line：
  - `91.9847%`
  - `0.3577311039`
- 金融旧静态展示线：
  - `117.80s/sample`
- “医疗最终主线 = 保留掩码回放链（keep-mask replay）”
- “金融最终主线 = static whole-forward”
- “医疗与金融两条正式主线并列落地”
- 任何只存在于历史服务器目录、但当前 `docs/report_evidence/` 未继续引用的中间产物

## 5. 正式写作与展示同步规则

- 摘要和最终模型介绍章节：
  - 只写正式 adopted 主线
  - 不写 fallback、对照、失败项、旧内部运行标识
- 创新点章节：
  - 只总结**已采用技术**以及它们带来的实际收益
  - 不把失败实验包装成创新
- 详细验证章节：
  - 才允许解释为什么不用静态、为什么要做 dynamic threshold calibration、为什么金融被降级为压力验证区
- 网页展示与报告前半部分必须同口径：
  - 医疗 = 全量验证集阈值准确率 `92.7481%` + 正式复核配置 `86.91s/sample`
  - 金融 = `100%` + `105.16s/sample` + `68.39%`，但只能作为压力验证数字出现

## 6. 服务器最小保留原则

服务器重建后，只保留两类内容：

- Git 仓库内当前正式代码、文档、前端、脚本
- 运行最终展示所必需的最小非 Git 资产：
  - 最终 bundle 权重
  - 金融压力样本集
  - 当前页面 live demo 必需的校准 JSON

除此之外的旧训练产物、旧 `results/` 大目录、旧 `artifacts/server_pipeline_run/`、旧日志统一清理。
