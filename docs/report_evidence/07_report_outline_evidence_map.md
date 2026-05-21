# 长报告提纲与证据映射

## 1. 总规则

- 最终作品报告只能从 `docs/report_evidence/` 和 `results/report_evidence/` 取一级证据。
- 旧 `docs/...`、历史 `artifacts/server_pipeline_run/...` 只能作为 provenance 附注，不得直接拼接成当前主线。
- 报告前半部分只写**最终 adopted 主线**；过程性对照、fallback、失败项统一放到后半部分。

## 2. 推荐写作顺序

1. 摘要
2. 项目概述与双向隐私目标
3. 最终落地模型（医疗、金融）
4. 系统运行链与 OpenBumbleBee / SPU 集成边界
5. 创新点与实际收益
6. 详细验证与采用原因
7. 场景适配与部署建议
8. 结论
9. 代码附录

## 3. 章节与证据映射

| 报告章节 | 本目录取证文件 | 对应实验 / 证据重点 |
|---|---|---|
| 摘要与项目概述 | `00_method_inventory.md`、`01_openbumblebee_usage_map.md` | 两域共同隐私边界、统一方法主线 |
| 医疗最终模型 | `02_medical_dynamic_mainline.md`、`results/report_evidence/mainline_communication_profile.json` | 医疗动态安全推理正式主线、动态路径全量验证集阈值、同配置通信复核 |
| 金融最终模型 | `04_finance_dynamic_mainline.md`、`results/report_evidence/mainline_communication_profile.json` | 金融动态安全推理正式主线、同配置通信复核 |
| 创新点与收益 | `03_medical_ablation_matrix.md`、`05_finance_ablation_matrix.md`、`06_adoption_decisions.md` | adopted innovations 与定量收益 |
| 详细验证与采用原因 | `02_medical_dynamic_mainline.md`、`03_medical_ablation_matrix.md`、`04_finance_dynamic_mainline.md`、`05_finance_ablation_matrix.md`、`08_rerun_validation_checklist.md` | 动态阈值、dynamic-vs-static pair、rejected items |
| 场景适配与部署建议 | `09_context_adaptation_profiles.md` | 城市 / 县域 / 乡村三档适配建议 |
| 网页展示与工程交付 | `10_web_demo_change_audit.md`、`01_openbumblebee_usage_map.md` | 前后端口径、展示链与正式主线对齐 |
| 附录：实验总账 | `experiment_manifest.json` | 全部 experiment id 与 adopted / rejected 归档 |

## 4. 前半部分禁止混入的内容

以下内容不得出现在摘要、项目概述和“最终落地模型”章节：

- 医疗 `69.57s/sample`
- 金融 `103.64s/sample`
- 医疗 static control line `91.9847%`
- 金融旧静态展示线 `117.80s/sample`
- 保留掩码回放链（keep-mask replay）作为最终系统
- `true static no-pruning` 作为金融正式主线
- 失败项和 rejected 方向的具体 run 名

## 5. 后半部分应该解释的内容

以下内容应放到“详细验证与采用原因”章节：

- 为什么医疗 dynamic path 不能继续套 static threshold `0.3577311039`
- 为什么医疗正式复核门选择“32 样本复核集 / 批次规模 8 / 深度 10”配置，且其速度为 `86.91s/sample`
- 为什么通信量采用同配置复核运行的 `Link details` 计数器口径
- 为什么 `69.57s/sample` 只保留为小样本最快工程配置
- 为什么金融 dynamic 与 static 要做 same-condition pair study
- 为什么静态只保留为 fallback 而不是并列正式主线
- 为什么 token recycle、clip3、public-calibrated LN + clip0、distillation 等方向最终未采用
