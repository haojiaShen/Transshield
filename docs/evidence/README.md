# 证据链索引

本目录提供正式交付所依赖的语义化证据入口。

| 证据类型 | 文件位置 | 支撑报告内容 |
|---|---|---|
| 协议 fuzz 最终结果 | `results/fuzzing/protocol_fuzz_final.json` | 鲁棒性矩阵、协议层异常输入阻断 |
| 协议 fuzz 元数据 | `results/fuzzing/protocol_fuzz_final.meta.md` | 原始来源、保留理由、引用位置 |
| 控制面 guard 最终结果 | `results/guard_stress/guard_stress_final.json` | 控制面防护、重放与并发守卫 |
| 控制面 guard 元数据 | `results/guard_stress/guard_stress_final.meta.md` | 原始来源、保留理由、引用位置 |
| 通信量正式结果 | `results/communication/mainline_communication_profile_final.json` | 性能分析、通信量统计 |
| 通信量说明 | `results/communication/mainline_communication_profile_final.meta.md` | 原始来源、重命名说明 |
| 医疗阈值正式结果 | `results/final/medical_dynamic_threshold_calibration_final.json` | 医疗主线精度、阈值校准 |
| 医疗阈值元数据 | `results/final/medical_dynamic_threshold_calibration_final.meta.md` | 原始来源、引用位置 |
| 医疗 AUC 正式结果 | `results/final/medical_dynamic_auc_reference_final.json` | 医疗主线 AUC 证据 |
| 医疗 AUC 元数据 | `results/final/medical_dynamic_auc_reference_final.meta.md` | 原始来源、引用位置 |
| 演示摘要正式结果 | `results/final/demo_content_summary_final.json` | 展示页摘要、评委演示数据 |
| 演示摘要元数据 | `results/final/demo_content_summary_final.meta.md` | 原始来源、引用位置 |
| 控制面代码审计 | `docs/evidence/web_demo_control_plane_audit.md` | 工程安全特色、控制面约束 |
| SPU 本地修改说明 | `docs/evidence/spu_bumblebee_local_modifications.md` | 第三方许可、vendored 修改说明 |
| 第三方许可与局部改动 | `spu_vendored/MODIFICATIONS.md` | vendored 文件改动入口 |
| 第三方许可文本索引 | `licenses/README.md` | 许可分发完整性、报告知识产权说明 |
| VPS 报告口径回归说明 | `docs/evidence/vps_report_regression.md` | 固定 524/32/8 样本、VPS-only 执行、逐文件与逐样本新证据 |
| VPS 报告测试矩阵 | `configs/report_vps_test_matrix.json` | 报告环境、数据、预处理、模型配置、指标、验收阈值与历史数字 |
| 2026-08-01 VPS candidate 总证据 | `results/vps_report_tests/report_regression_20260801_v1/report_regression_aggregate.json` | 524/32/8、同机 A/B、13+4、隐私事实、代码测试与逐门槛判定 |

## 说明

- `docs/密捷竞赛作品报告.docx` 是当前正式报告主文件。
- 当前正式证据代码入口以 `tools/fuzzing/` 和 `results/` 为准。
- `docs/evidence/web_demo_control_plane_audit.md` 记录的是上一版正式前端实现的审计快照；该实现路径已移除，仅作为报告证据保留。
- 当前新增的 `showcase/` 与 `showcase_api/` 是对既有控制面证据的**可运行重建**，不是新增第二套正式结果口径。
- `tools/showcase_protocol_fuzz.py` 与 `tools/showcase_guard_stress.py` 面向新展示站接口做运行时验收；正式报告中的最终鲁棒性数字仍以 `results/fuzzing/` 与 `results/guard_stress/` 为准。
- 后续测试只在 VPS 执行，并将 candidate 结果写入 `results/vps_report_tests/`；除非另行审核批准，不回写或覆盖本页列出的正式结果。
