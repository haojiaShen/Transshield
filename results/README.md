# `results/`

当前 `results/` 只保留作品报告直接引用的正式结果。

## 正式统计

| 指标 | 数值 | 文件 |
|---|---:|---|
| 医疗阈值精度 | 92.7481% | `final/medical_dynamic_threshold_calibration_final.json` |
| 医疗 AUC | 0.9639 | `final/medical_dynamic_auc_reference_final.json` |
| 医疗端到端时延 | 89.06 秒/样本 | `communication/mainline_communication_profile_final.json` |
| 医疗双向通信量 | 84.47 GiB | `communication/mainline_communication_profile_final.json` |
| 协议 fuzz | 13 / 13 通过 | `fuzzing/protocol_fuzz_final.json` |
| guard stress | 4 / 4 通过 | `guard_stress/guard_stress_final.json` |
| 总鲁棒性 | 17 / 17 通过 | `fuzzing/protocol_fuzz_final.json` + `guard_stress/guard_stress_final.json` |

## 目录职责

- `final/`：正式摘要和保留的原始校准记录
- `communication/`：最终通信量与配套说明
- `fuzzing/`：协议层异常输入最终结果
- `guard_stress/`：控制面守卫最终结果

## 口径说明

正式报告、根 README 和展示站指标以本目录下的最终 JSON 为准。部分 `artifacts/frozen_bundle_*/manifest.json` 会保留导出时的历史阈值或训练摘要，不能直接替代 `results/final/` 中的正式结果口径。
